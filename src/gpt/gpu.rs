use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, device, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

use crate::{Result, mesh::validate_finite};

use super::{
    Attention, Features, GptConfig, GptTrainOptions, NormCache, Vocabulary, config::Layout,
};

#[cuda_module]
// Explicit tensor indices keep the channel, head, and neighborhood math visible.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod kernels {
    use super::*;

    #[device]
    fn slope(value: f32, activated: usize) -> f32 {
        if activated == 0 || value >= 0.0 {
            1.0
        } else {
            value + 1.0
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, sequence >= 1, sequence <= 128,
        out.len() == tokens.len(), state.len() >= width * sequence))]
    pub fn gpt_embed(
        width: usize,
        sequence: usize,
        vocab: usize,
        tokens: &[u32],
        state: &[[f32; 3]],
        mut out: DisjointSlice<Features>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            for channel in 0..width {
                output[channel] = state[tokens[row] as usize * width + channel][0]
                    + state[(vocab + row % sequence) * width + channel][0];
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, state.len() == 2 * width,
        out.len() == input.len(), cache.len() == input.len()))]
    pub fn gpt_norm(
        width: usize,
        input: &[Features],
        state: &[[f32; 3]],
        mut out: DisjointSlice<Features>,
        mut cache: DisjointSlice<NormCache>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index)
            && let Some(saved) = cache.get_mut(thread::index_1d())
        {
            let mut mean = 0.0;
            for channel in 0..width {
                mean += input[row][channel];
            }
            mean /= width as f32;
            let mut variance = 0.0;
            for channel in 0..width {
                let x = input[row][channel] - mean;
                variance += x * x;
            }
            let inverse = 1.0 / (variance / width as f32 + 1e-5).sqrt();
            saved[64] = inverse;
            for channel in 0..width {
                saved[channel] = (input[row][channel] - mean) * inverse;
                output[channel] = saved[channel] * state[channel][0] + state[width + channel][0];
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, offset + 3 * width <= state.len(), activated <= 1,
        out.len() == input.len()))]
    pub fn gpt_mesh(
        width: usize,
        offset: usize,
        activated: usize,
        input: &[Features],
        state: &[[f32; 3]],
        mut out: DisjointSlice<Features>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            for channel in 0..width {
                let p = offset + 3 * channel;
                let value = input[row][channel] * state[p][0]
                    + input[row][(channel + 1) % width] * state[p + 1][0]
                    + state[p + 2][0];
                output[channel] = if activated == 1 && value < 0.0 {
                    value.exp() - 1.0
                } else {
                    value
                };
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, out.len() == left.len(), left.len() == right.len()))]
    pub fn gpt_add(
        width: usize,
        left: &[Features],
        right: &[Features],
        mut out: DisjointSlice<Features>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            for channel in 0..width {
                output[channel] = left[row][channel] + right[row][channel];
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, out.len() == input.len()))]
    pub fn gpt_copy(width: usize, input: &[Features], mut out: DisjointSlice<Features>) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            output[..width].copy_from_slice(&input[row][..width]);
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, heads >= 1, heads <= 8, sequence >= 1, sequence <= 128,
        out.len() == query.len(), key.len() == query.len(), value.len() == query.len(), probabilities.len() == query.len()))]
    pub fn gpt_attention(
        width: usize,
        heads: usize,
        sequence: usize,
        query: &[Features],
        key: &[Features],
        value: &[Features],
        mut out: DisjointSlice<Features>,
        mut probabilities: DisjointSlice<Attention>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index)
            && let Some(probability) = probabilities.get_mut(thread::index_1d())
        {
            let position = row % sequence;
            let first = row - position;
            let head_width = width / heads;
            let scale = 1.0 / (head_width as f32).sqrt();
            for head in 0..heads {
                let begin = head * head_width;
                let mut maximum = f32::NEG_INFINITY;
                for previous in 0..=position {
                    let mut score = 0.0;
                    for channel in begin..begin + head_width {
                        score += query[row][channel] * key[first + previous][channel];
                    }
                    score *= scale;
                    probability[head * 128 + previous] = score;
                    if score > maximum {
                        maximum = score;
                    }
                }
                let mut sum = 0.0;
                for previous in 0..=position {
                    let p = head * 128 + previous;
                    probability[p] = (probability[p] - maximum).exp();
                    sum += probability[p];
                }
                for previous in 0..=position {
                    probability[head * 128 + previous] /= sum;
                }
                for channel in begin..begin + head_width {
                    let mut total = 0.0;
                    for previous in 0..=position {
                        total +=
                            probability[head * 128 + previous] * value[first + previous][channel];
                    }
                    output[channel] = total;
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, vocab >= 2, vocab <= 256,
        embedding.len() >= vocab * width, bias.len() == vocab, out.len() == input.len()))]
    pub fn gpt_head(
        width: usize,
        vocab: usize,
        input: &[Features],
        embedding: &[[f32; 3]],
        bias: &[[f32; 3]],
        mut out: DisjointSlice<Vocabulary>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            for token in 0..vocab {
                let mut value = 0.0;
                for channel in 0..width {
                    value += input[row][channel] * embedding[token * width + channel][0];
                }
                output[token] = value + bias[token][0];
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        vocab >= 2, vocab <= 256, logits.len() >= 1, targets.len() == logits.len(),
        gradient.len() == logits.len(), loss.len() == logits.len()))]
    pub fn gpt_cross_entropy(
        vocab: usize,
        logits: &[Vocabulary],
        targets: &[u32],
        mut gradient: DisjointSlice<Vocabulary>,
        mut loss: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = gradient.get_mut(index)
            && let Some(value) = loss.get_mut(thread::index_1d())
        {
            let mut maximum = f32::NEG_INFINITY;
            for token in 0..vocab {
                if logits[row][token] > maximum {
                    maximum = logits[row][token];
                }
            }
            let mut sum = 0.0;
            for token in 0..vocab {
                output[token] = (logits[row][token] - maximum).exp();
                sum += output[token];
            }
            *value = sum.ln() + maximum - logits[row][targets[row] as usize];
            for token in 0..vocab {
                let target = if token == targets[row] as usize {
                    1.0
                } else {
                    0.0
                };
                output[token] = (output[token] / sum - target) / logits.len() as f32;
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, vocab >= 2, vocab <= 256, embedding.len() >= vocab * width, out.len() == gradient.len()))]
    pub fn gpt_head_backward(
        width: usize,
        vocab: usize,
        gradient: &[Vocabulary],
        embedding: &[[f32; 3]],
        mut out: DisjointSlice<Features>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            for channel in 0..width {
                let mut total = 0.0;
                for token in 0..vocab {
                    total += gradient[row][token] * embedding[token * width + channel][0];
                }
                output[channel] = total;
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (vocab >= 2, vocab <= 256, out.len() == vocab))]
    pub fn gpt_bias_gradient(vocab: usize, gradient: &[Vocabulary], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let token = index.get();
        if token >= vocab {
            return;
        }
        if let Some(output) = out.get_mut(index) {
            let mut total = 0.0;
            for row in gradient {
                total += row[token];
            }
            *output = total;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, state.len() == 2 * width,
        gradient.len() == cache.len(), out.len() == cache.len()))]
    pub fn gpt_norm_backward(
        width: usize,
        gradient: &[Features],
        cache: &[NormCache],
        state: &[[f32; 3]],
        mut out: DisjointSlice<Features>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            let mut sum = 0.0;
            let mut centered = 0.0;
            for channel in 0..width {
                let value = gradient[row][channel] * state[channel][0];
                sum += value;
                centered += value * cache[row][channel];
            }
            for channel in 0..width {
                let value = gradient[row][channel] * state[channel][0];
                output[channel] = cache[row][64]
                    * (value - sum / width as f32 - cache[row][channel] * centered / width as f32);
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, gradient.len() == cache.len(), out.len() == 2 * width))]
    pub fn gpt_norm_gradient(
        width: usize,
        gradient: &[Features],
        cache: &[NormCache],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let parameter = index.get();
        if let Some(output) = out.get_mut(index) {
            let channel = parameter % width;
            let mut total = 0.0;
            for row in 0..gradient.len() {
                total += gradient[row][channel]
                    * if parameter < width {
                        cache[row][channel]
                    } else {
                        1.0
                    };
            }
            *output = total;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, offset + 3 * width <= state.len(), activated <= 1,
        gradient.len() == activation.len(), out.len() == activation.len()))]
    pub fn gpt_mesh_backward(
        width: usize,
        offset: usize,
        activated: usize,
        gradient: &[Features],
        activation: &[Features],
        state: &[[f32; 3]],
        mut out: DisjointSlice<Features>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            for channel in 0..width {
                let previous = (channel + width - 1) % width;
                output[channel] = gradient[row][channel]
                    * slope(activation[row][channel], activated)
                    * state[offset + 3 * channel][0]
                    + gradient[row][previous]
                        * slope(activation[row][previous], activated)
                        * state[offset + 3 * previous + 1][0];
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, offset + 3 * width <= out.len(), activated <= 1,
        input.len() == activation.len(), gradient.len() == activation.len()))]
    pub fn gpt_mesh_gradient(
        width: usize,
        offset: usize,
        activated: usize,
        input: &[Features],
        activation: &[Features],
        gradient: &[Features],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let parameter = index.get();
        if parameter >= offset
            && parameter < offset + 3 * width
            && let Some(output) = out.get_mut(index)
        {
            let channel = (parameter - offset) / 3;
            let edge = (parameter - offset) % 3;
            let mut total = 0.0;
            for row in 0..gradient.len() {
                let x = if edge == 0 {
                    input[row][channel]
                } else if edge == 1 {
                    input[row][(channel + 1) % width]
                } else {
                    1.0
                };
                total += gradient[row][channel] * slope(activation[row][channel], activated) * x;
            }
            *output = total;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, heads >= 1, heads <= 8, sequence >= 1, sequence <= 128,
        probabilities.len() == gradient.len(), value.len() == gradient.len(), out.len() == gradient.len()))]
    pub fn gpt_score_backward(
        width: usize,
        heads: usize,
        sequence: usize,
        gradient: &[Features],
        value: &[Features],
        probabilities: &[Attention],
        mut out: DisjointSlice<Attention>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            let position = row % sequence;
            let first = row - position;
            let head_width = width / heads;
            for head in 0..heads {
                let begin = head * head_width;
                let mut mean = 0.0;
                for previous in 0..=position {
                    let mut dot = 0.0;
                    for channel in begin..begin + head_width {
                        dot += gradient[row][channel] * value[first + previous][channel];
                    }
                    output[head * 128 + previous] = dot;
                    mean += probabilities[row][head * 128 + previous] * dot;
                }
                for previous in 0..=position {
                    let p = head * 128 + previous;
                    output[p] = probabilities[row][p] * (output[p] - mean);
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, heads >= 1, heads <= 8, sequence >= 1, sequence <= 128,
        key.len() == query.len(), gradient.len() == query.len(), scores.len() == query.len(), probabilities.len() == query.len(),
        dq.len() == query.len(), dk.len() == query.len(), dv.len() == query.len()))]
    pub fn gpt_attention_backward(
        width: usize,
        heads: usize,
        sequence: usize,
        query: &[Features],
        key: &[Features],
        gradient: &[Features],
        scores: &[Attention],
        probabilities: &[Attention],
        mut dq: DisjointSlice<Features>,
        mut dk: DisjointSlice<Features>,
        mut dv: DisjointSlice<Features>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(q_gradient) = dq.get_mut(index)
            && let Some(k_gradient) = dk.get_mut(thread::index_1d())
            && let Some(v_gradient) = dv.get_mut(thread::index_1d())
        {
            let position = row % sequence;
            let first = row - position;
            let head_width = width / heads;
            let scale = 1.0 / (head_width as f32).sqrt();
            for channel in 0..width {
                let head = channel / head_width;
                let mut q_sum = 0.0;
                let mut k_sum = 0.0;
                let mut v_sum = 0.0;
                for previous in 0..=position {
                    q_sum += scores[row][head * 128 + previous] * key[first + previous][channel];
                }
                for future in position..sequence {
                    k_sum += scores[first + future][head * 128 + position]
                        * query[first + future][channel];
                    v_sum += probabilities[first + future][head * 128 + position]
                        * gradient[first + future][channel];
                }
                q_gradient[channel] = q_sum * scale;
                k_gradient[channel] = k_sum * scale;
                v_gradient[channel] = v_sum;
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, vocab >= 2, vocab <= 256, sequence >= 1, sequence <= 128,
        input_gradient.len() == tokens.len(), head_gradient.len() == tokens.len(), final_features.len() == tokens.len(),
        out.len() >= (vocab + sequence) * width))]
    pub fn gpt_embedding_gradient(
        width: usize,
        vocab: usize,
        sequence: usize,
        tokens: &[u32],
        input_gradient: &[Features],
        head_gradient: &[Vocabulary],
        final_features: &[Features],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let parameter = index.get();
        if let Some(output) = out.get_mut(index) {
            let token_or_position = parameter / width;
            let channel = parameter % width;
            let mut total = 0.0;
            for row in 0..tokens.len() {
                if token_or_position < vocab {
                    if tokens[row] as usize == token_or_position {
                        total += input_gradient[row][channel];
                    }
                    total += head_gradient[row][token_or_position] * final_features[row][channel];
                } else if row % sequence == token_or_position - vocab {
                    total += input_gradient[row][channel];
                }
            }
            *output = total;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (gradient.len() == state.len()))]
    pub fn gpt_adam(
        learning_rate: f32,
        weight_decay: f32,
        correction1: f32,
        correction2: f32,
        gradient: &[f32],
        mut state: DisjointSlice<[f32; 3]>,
    ) {
        let index = thread::index_1d();
        let parameter = index.get();
        if let Some(slot) = state.get_mut(index) {
            let g = gradient[parameter];
            slot[1] = 0.9 * slot[1] + 0.1 * g;
            slot[2] = 0.999 * slot[2] + 0.001 * g * g;
            let m = slot[1] / correction1;
            let v = slot[2] / correction2;
            slot[0] -= learning_rate * (m / (v.sqrt() + 1e-8) + weight_decay * slot[0]);
        }
    }
}

#[derive(Clone, Copy)]
enum Op {
    Embed {
        out: usize,
    },
    Norm {
        group: usize,
        input: usize,
        out: usize,
        cache: usize,
    },
    Mesh {
        group: usize,
        input: usize,
        out: usize,
        layer: usize,
        activated: usize,
    },
    Add {
        left: usize,
        right: usize,
        out: usize,
    },
    Attention {
        query: usize,
        key: usize,
        value: usize,
        out: usize,
        cache: usize,
    },
}

struct Plan {
    ops: Vec<Op>,
    features: usize,
    norms: usize,
    attentions: usize,
    final_features: usize,
}

impl Plan {
    fn new(config: GptConfig, layout: &Layout) -> Self {
        let mut plan = Self {
            ops: vec![Op::Embed { out: 0 }],
            features: 1,
            norms: 0,
            attentions: 0,
            final_features: 0,
        };
        let mut x = 0;
        for block in &layout.blocks {
            let normalized = plan.norm(x, block.norm1);
            let query = plan.mesh(normalized, block.query, config.attention_mesh_depth);
            let key = plan.mesh(normalized, block.key, config.attention_mesh_depth);
            let value = plan.mesh(normalized, block.value, config.attention_mesh_depth);
            let out = plan.feature();
            let cache = plan.attentions;
            plan.attentions += 1;
            plan.ops.push(Op::Attention {
                query,
                key,
                value,
                out,
                cache,
            });
            let projected = plan.mesh(out, block.projection, config.attention_mesh_depth);
            x = plan.add(x, projected);
            let normalized = plan.norm(x, block.norm2);
            let ffn = plan.mesh(normalized, block.ffn, config.ffn_mesh_depth);
            x = plan.add(x, ffn);
        }
        plan.final_features = plan.norm(x, layout.final_norm);
        plan
    }
    fn feature(&mut self) -> usize {
        let id = self.features;
        self.features += 1;
        id
    }
    fn norm(&mut self, input: usize, group: usize) -> usize {
        let out = self.feature();
        let cache = self.norms;
        self.norms += 1;
        self.ops.push(Op::Norm {
            group,
            input,
            out,
            cache,
        });
        out
    }
    fn mesh(&mut self, mut input: usize, group: usize, depth: usize) -> usize {
        for layer in 0..depth {
            let out = self.feature();
            self.ops.push(Op::Mesh {
                group,
                input,
                out,
                layer,
                activated: usize::from(layer + 1 != depth),
            });
            input = out;
        }
        input
    }
    fn add(&mut self, left: usize, right: usize) -> usize {
        let out = self.feature();
        self.ops.push(Op::Add { left, right, out });
        out
    }
}

struct Parameters {
    state: DeviceBuffer<[f32; 3]>,
    gradient: DeviceBuffer<f32>,
}

struct Workspace {
    rows: usize,
    sequence: usize,
    tokens: DeviceBuffer<u32>,
    targets: DeviceBuffer<u32>,
    features: Vec<DeviceBuffer<Features>>,
    adjoints: Vec<DeviceBuffer<Features>>,
    norms: Vec<DeviceBuffer<NormCache>>,
    probabilities: Vec<DeviceBuffer<Attention>>,
    scores: DeviceBuffer<Attention>,
    scratch: Vec<DeviceBuffer<Features>>,
    initialized: Vec<bool>,
    logits: DeviceBuffer<Vocabulary>,
    head_gradient: DeviceBuffer<Vocabulary>,
    loss: DeviceBuffer<f32>,
}

impl Workspace {
    fn new(stream: &Arc<CudaStream>, plan: &Plan, rows: usize, sequence: usize) -> Result<Self> {
        fn buffers<T: cuda_core::DeviceCopy>(
            stream: &Arc<CudaStream>,
            count: usize,
            rows: usize,
        ) -> Result<Vec<DeviceBuffer<T>>> {
            (0..count)
                .map(|_| Ok(DeviceBuffer::zeroed(stream, rows)?))
                .collect()
        }
        Ok(Self {
            rows,
            sequence,
            tokens: DeviceBuffer::zeroed(stream, rows)?,
            targets: DeviceBuffer::zeroed(stream, rows)?,
            features: buffers(stream, plan.features, rows)?,
            adjoints: buffers(stream, plan.features, rows)?,
            norms: buffers(stream, plan.norms, rows)?,
            probabilities: buffers(stream, plan.attentions, rows)?,
            scores: DeviceBuffer::zeroed(stream, rows)?,
            scratch: buffers(stream, 4, rows)?,
            initialized: vec![false; plan.features],
            logits: DeviceBuffer::zeroed(stream, rows)?,
            head_gradient: DeviceBuffer::zeroed(stream, rows)?,
            loss: DeviceBuffer::zeroed(stream, rows)?,
        })
    }
}

/// A small GPT decoder whose Q/K/V, attention output, and feed-forward channel
/// projections are two-parent ELU meshes. Embeddings and the tied output head
/// retain their usual learned tables. Training operators and AdamW run on CUDA.
pub struct GpuMeshGpt {
    config: GptConfig,
    layout: Layout,
    plan: Plan,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    parameters: Vec<Parameters>,
    workspace: Option<Workspace>,
    beta1_power: f32,
    beta2_power: f32,
}

impl GpuMeshGpt {
    pub fn new(device: usize, config: GptConfig, seed: u64) -> Result<Self> {
        let layout = Layout::new(config)?;
        let plan = Plan::new(config, &layout);
        let initial = layout.initialize(config, seed);
        let context = CudaContext::new(device)?;
        let stream = context.default_stream();
        // SAFETY: the package embeds its own combined library artifact. All
        // GPT entries have unique names, and this constructor accepts no CUDA
        // artifact supplied by the caller.
        let module = unsafe { kernels::load(&context)? };
        let mut parameters = Vec::new();
        for group in &layout.groups {
            let state: Vec<_> = initial[group.offset..group.offset + group.size]
                .iter()
                .map(|&p| [p, 0.0, 0.0])
                .collect();
            parameters.push(Parameters {
                state: DeviceBuffer::from_host(&stream, &state)?,
                gradient: DeviceBuffer::zeroed(&stream, group.size)?,
            });
        }
        Ok(Self {
            config,
            layout,
            plan,
            stream,
            module,
            parameters,
            workspace: None,
            beta1_power: 1.0,
            beta2_power: 1.0,
        })
    }

    pub fn config(&self) -> GptConfig {
        self.config
    }

    pub fn parameters(&self) -> Result<Vec<f32>> {
        let mut values = Vec::with_capacity(self.layout.count());
        for group in &self.parameters {
            values.extend(group.state.to_host_vec(&self.stream)?.iter().map(|s| s[0]));
        }
        validate_finite(&values, "parameters")?;
        Ok(values)
    }

    /// Sets weights and resets AdamW's moments and step count.
    pub fn set_parameters(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.layout.count() {
            return Err("parameter count does not match GPT configuration".into());
        }
        validate_finite(values, "parameters")?;
        for (group, parameters) in self.layout.groups.iter().zip(&mut self.parameters) {
            let state: Vec<_> = values[group.offset..group.offset + group.size]
                .iter()
                .map(|&p| [p, 0.0, 0.0])
                .collect();
            parameters.state.copy_from_host(&self.stream, &state)?;
        }
        self.beta1_power = 1.0;
        self.beta2_power = 1.0;
        Ok(())
    }

    /// Returns logits for every token of a flat batch with equal sequence lengths.
    pub fn logits(&self, tokens: &[u32], sequence: usize) -> Result<Vec<Vec<f32>>> {
        self.config.validate_tokens(tokens, sequence)?;
        let mut workspace = Workspace::new(&self.stream, &self.plan, tokens.len(), sequence)?;
        workspace.tokens.copy_from_host(&self.stream, tokens)?;
        self.forward(&mut workspace)?;
        workspace
            .logits
            .to_host_vec(&self.stream)?
            .iter()
            .map(|row| {
                validate_finite(&row[..self.config.vocab_size], "logits")?;
                Ok(row[..self.config.vocab_size].to_vec())
            })
            .collect()
    }

    pub fn cross_entropy(&self, inputs: &[u32], targets: &[u32], sequence: usize) -> Result<f64> {
        self.config.validate_batch(inputs, targets, sequence)?;
        let mut workspace = Workspace::new(&self.stream, &self.plan, inputs.len(), sequence)?;
        workspace.tokens.copy_from_host(&self.stream, inputs)?;
        workspace.targets.copy_from_host(&self.stream, targets)?;
        self.forward(&mut workspace)?;
        self.loss(&mut workspace)
    }

    /// Diagnostic gradients are copied to the host; optimizer updates never are.
    pub fn loss_and_gradient(
        &mut self,
        inputs: &[u32],
        targets: &[u32],
        sequence: usize,
    ) -> Result<(f64, Vec<f64>)> {
        let loss = self.gradients(inputs, targets, sequence)?;
        let mut gradient = Vec::with_capacity(self.layout.count());
        for group in &self.parameters {
            let values = group.gradient.to_host_vec(&self.stream)?;
            validate_finite(&values, "gradients")?;
            gradient.extend(values.into_iter().map(f64::from));
        }
        Ok((loss, gradient))
    }

    /// One minibatch step. Repeated calls continue the same AdamW state.
    /// Returns the mean next-token cross-entropy before the update.
    pub fn train_batch(
        &mut self,
        inputs: &[u32],
        targets: &[u32],
        sequence: usize,
        learning_rate: f32,
        weight_decay: f32,
    ) -> Result<f64> {
        GptTrainOptions {
            steps: 1,
            learning_rate,
            weight_decay,
            report_every: 1,
        }
        .validate()?;
        let loss = self.gradients(inputs, targets, sequence)?;
        self.beta1_power *= 0.9;
        self.beta2_power *= 0.999;
        for group in &mut self.parameters {
            let launch = self.module.prepare_gpt_adam(config(group.state.len()))?;
            self.module.gpt_adam(
                &self.stream,
                &launch,
                learning_rate,
                weight_decay,
                1.0 - self.beta1_power,
                1.0 - self.beta2_power,
                &group.gradient,
                &mut group.state,
            )?;
        }
        Ok(loss)
    }

    pub fn generate(
        &self,
        tokenizer: &super::CharacterTokenizer,
        prompt: &str,
        count: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<String> {
        if tokenizer.vocab_size() != self.config.vocab_size {
            return Err("tokenizer does not match model vocabulary size".into());
        }
        if !temperature.is_finite() || temperature < 0.0 {
            return Err("temperature must be finite and nonnegative (0 is greedy)".into());
        }
        let mut tokens = tokenizer.encode(prompt)?;
        if tokens.is_empty() {
            return Err("generation prompt must not be empty".into());
        }
        let mut random = crate::mesh::Random::new(seed);
        for _ in 0..count {
            let context = &tokens[tokens.len().saturating_sub(self.config.context)..];
            let logits = self.logits(context, context.len())?;
            let last = logits.last().ok_or("missing generation logits")?;
            let maximum = last.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let token = if temperature == 0.0 {
                last.iter()
                    .position(|&p| p == maximum)
                    .ok_or("invalid generation logits")?
            } else {
                let probabilities: Vec<_> = last
                    .iter()
                    .map(|&p| ((p - maximum) / temperature).exp())
                    .collect();
                let mut sample = random.unit() * probabilities.iter().sum::<f32>();
                let mut selected = probabilities.len() - 1;
                for (token, probability) in probabilities.iter().enumerate() {
                    sample -= probability;
                    if sample < 0.0 {
                        selected = token;
                        break;
                    }
                }
                selected
            };
            tokens.push(token as u32);
        }
        tokenizer.decode(&tokens)
    }

    fn forward(&self, workspace: &mut Workspace) -> Result<()> {
        let c = self.config;
        let row_config = config(workspace.rows);
        for op in &self.plan.ops {
            match *op {
                Op::Embed { out } => {
                    let launch = self.module.prepare_gpt_embed(row_config)?;
                    self.module.gpt_embed(
                        &self.stream,
                        &launch,
                        c.width,
                        workspace.sequence,
                        c.vocab_size,
                        &workspace.tokens,
                        &self.parameters[0].state,
                        &mut workspace.features[out],
                    )?;
                }
                Op::Norm {
                    group,
                    input,
                    out,
                    cache,
                } => {
                    let launch = self.module.prepare_gpt_norm(row_config)?;
                    let (before, after) = workspace.features.split_at_mut(out);
                    self.module.gpt_norm(
                        &self.stream,
                        &launch,
                        c.width,
                        &before[input],
                        &self.parameters[group].state,
                        &mut after[0],
                        &mut workspace.norms[cache],
                    )?;
                }
                Op::Mesh {
                    group,
                    input,
                    out,
                    layer,
                    activated,
                } => {
                    let launch = self.module.prepare_gpt_mesh(row_config)?;
                    let (before, after) = workspace.features.split_at_mut(out);
                    self.module.gpt_mesh(
                        &self.stream,
                        &launch,
                        c.width,
                        layer * 3 * c.width,
                        activated,
                        &before[input],
                        &self.parameters[group].state,
                        &mut after[0],
                    )?;
                }
                Op::Add { left, right, out } => {
                    let launch = self.module.prepare_gpt_add(row_config)?;
                    let (before, after) = workspace.features.split_at_mut(out);
                    self.module.gpt_add(
                        &self.stream,
                        &launch,
                        c.width,
                        &before[left],
                        &before[right],
                        &mut after[0],
                    )?;
                }
                Op::Attention {
                    query,
                    key,
                    value,
                    out,
                    cache,
                } => {
                    let launch = self.module.prepare_gpt_attention(row_config)?;
                    let (before, after) = workspace.features.split_at_mut(out);
                    self.module.gpt_attention(
                        &self.stream,
                        &launch,
                        c.width,
                        c.heads,
                        workspace.sequence,
                        &before[query],
                        &before[key],
                        &before[value],
                        &mut after[0],
                        &mut workspace.probabilities[cache],
                    )?;
                }
            }
        }
        let launch = self.module.prepare_gpt_head(row_config)?;
        self.module.gpt_head(
            &self.stream,
            &launch,
            c.width,
            c.vocab_size,
            &workspace.features[self.plan.final_features],
            &self.parameters[0].state,
            &self.parameters[self.layout.head_bias].state,
            &mut workspace.logits,
        )?;
        Ok(())
    }

    fn loss(&self, workspace: &mut Workspace) -> Result<f64> {
        let launch = self
            .module
            .prepare_gpt_cross_entropy(config(workspace.rows))?;
        self.module.gpt_cross_entropy(
            &self.stream,
            &launch,
            self.config.vocab_size,
            &workspace.logits,
            &workspace.targets,
            &mut workspace.head_gradient,
            &mut workspace.loss,
        )?;
        let loss = workspace.loss.to_host_vec(&self.stream)?;
        validate_finite(&loss, "cross-entropy")?;
        Ok(loss.iter().map(|&value| f64::from(value)).sum::<f64>() / loss.len() as f64)
    }

    fn gradients(&mut self, inputs: &[u32], targets: &[u32], sequence: usize) -> Result<f64> {
        self.config.validate_batch(inputs, targets, sequence)?;
        let mut workspace = match self.workspace.take() {
            Some(workspace) if workspace.rows == inputs.len() && workspace.sequence == sequence => {
                workspace
            }
            _ => Workspace::new(&self.stream, &self.plan, inputs.len(), sequence)?,
        };
        let result = (|| {
            workspace.tokens.copy_from_host(&self.stream, inputs)?;
            workspace.targets.copy_from_host(&self.stream, targets)?;
            self.forward(&mut workspace)?;
            let loss = self.loss(&mut workspace)?;
            self.backward(&mut workspace)?;
            Ok(loss)
        })();
        self.workspace = Some(workspace);
        result
    }

    fn backward(&mut self, workspace: &mut Workspace) -> Result<()> {
        let c = self.config;
        let row_config = config(workspace.rows);
        workspace.initialized.fill(false);
        let launch = self.module.prepare_gpt_head_backward(row_config)?;
        self.module.gpt_head_backward(
            &self.stream,
            &launch,
            c.width,
            c.vocab_size,
            &workspace.head_gradient,
            &self.parameters[0].state,
            &mut workspace.scratch[0],
        )?;
        self.accumulate_scratch(workspace, self.plan.final_features, 0)?;
        let group = &mut self.parameters[self.layout.head_bias];
        let launch = self
            .module
            .prepare_gpt_bias_gradient(config(group.gradient.len()))?;
        self.module.gpt_bias_gradient(
            &self.stream,
            &launch,
            c.vocab_size,
            &workspace.head_gradient,
            &mut group.gradient,
        )?;

        for index in (0..self.plan.ops.len()).rev() {
            match self.plan.ops[index] {
                Op::Embed { .. } => {}
                Op::Norm {
                    group,
                    input,
                    out,
                    cache,
                } => {
                    let parameters = &mut self.parameters[group];
                    let launch = self
                        .module
                        .prepare_gpt_norm_gradient(config(parameters.gradient.len()))?;
                    self.module.gpt_norm_gradient(
                        &self.stream,
                        &launch,
                        c.width,
                        &workspace.adjoints[out],
                        &workspace.norms[cache],
                        &mut parameters.gradient,
                    )?;
                    let launch = self.module.prepare_gpt_norm_backward(row_config)?;
                    self.module.gpt_norm_backward(
                        &self.stream,
                        &launch,
                        c.width,
                        &workspace.adjoints[out],
                        &workspace.norms[cache],
                        &parameters.state,
                        &mut workspace.scratch[0],
                    )?;
                    self.accumulate_scratch(workspace, input, 0)?;
                }
                Op::Mesh {
                    group,
                    input,
                    out,
                    layer,
                    activated,
                } => {
                    let parameters = &mut self.parameters[group];
                    let launch = self
                        .module
                        .prepare_gpt_mesh_gradient(config(parameters.gradient.len()))?;
                    self.module.gpt_mesh_gradient(
                        &self.stream,
                        &launch,
                        c.width,
                        layer * 3 * c.width,
                        activated,
                        &workspace.features[input],
                        &workspace.features[out],
                        &workspace.adjoints[out],
                        &mut parameters.gradient,
                    )?;
                    let launch = self.module.prepare_gpt_mesh_backward(row_config)?;
                    self.module.gpt_mesh_backward(
                        &self.stream,
                        &launch,
                        c.width,
                        layer * 3 * c.width,
                        activated,
                        &workspace.adjoints[out],
                        &workspace.features[out],
                        &parameters.state,
                        &mut workspace.scratch[0],
                    )?;
                    self.accumulate_scratch(workspace, input, 0)?;
                }
                Op::Add { left, right, out } => {
                    self.accumulate_node(workspace, left, out)?;
                    self.accumulate_node(workspace, right, out)?;
                }
                Op::Attention {
                    query,
                    key,
                    value,
                    out,
                    cache,
                } => {
                    let launch = self.module.prepare_gpt_score_backward(row_config)?;
                    self.module.gpt_score_backward(
                        &self.stream,
                        &launch,
                        c.width,
                        c.heads,
                        workspace.sequence,
                        &workspace.adjoints[out],
                        &workspace.features[value],
                        &workspace.probabilities[cache],
                        &mut workspace.scores,
                    )?;
                    let launch = self.module.prepare_gpt_attention_backward(row_config)?;
                    let (dq, rest) = workspace.scratch.split_at_mut(1);
                    let (dk, rest) = rest.split_at_mut(1);
                    self.module.gpt_attention_backward(
                        &self.stream,
                        &launch,
                        c.width,
                        c.heads,
                        workspace.sequence,
                        &workspace.features[query],
                        &workspace.features[key],
                        &workspace.adjoints[out],
                        &workspace.scores,
                        &workspace.probabilities[cache],
                        &mut dq[0],
                        &mut dk[0],
                        &mut rest[0],
                    )?;
                    self.accumulate_scratch(workspace, query, 0)?;
                    self.accumulate_scratch(workspace, key, 1)?;
                    self.accumulate_scratch(workspace, value, 2)?;
                }
            }
        }
        let group = &mut self.parameters[0];
        let launch = self
            .module
            .prepare_gpt_embedding_gradient(config(group.gradient.len()))?;
        self.module.gpt_embedding_gradient(
            &self.stream,
            &launch,
            c.width,
            c.vocab_size,
            workspace.sequence,
            &workspace.tokens,
            &workspace.adjoints[0],
            &workspace.head_gradient,
            &workspace.features[self.plan.final_features],
            &mut group.gradient,
        )?;
        Ok(())
    }

    fn accumulate_scratch(
        &self,
        workspace: &mut Workspace,
        target: usize,
        source: usize,
    ) -> Result<()> {
        if workspace.initialized[target] {
            let launch = self.module.prepare_gpt_add(config(workspace.rows))?;
            let (before, after) = workspace.scratch.split_at_mut(3);
            self.module.gpt_add(
                &self.stream,
                &launch,
                self.config.width,
                &workspace.adjoints[target],
                &before[source],
                &mut after[0],
            )?;
            std::mem::swap(&mut workspace.adjoints[target], &mut workspace.scratch[3]);
        } else {
            std::mem::swap(
                &mut workspace.adjoints[target],
                &mut workspace.scratch[source],
            );
            workspace.initialized[target] = true;
        }
        Ok(())
    }

    fn accumulate_node(
        &self,
        workspace: &mut Workspace,
        target: usize,
        source: usize,
    ) -> Result<()> {
        if workspace.initialized[target] {
            let launch = self.module.prepare_gpt_add(config(workspace.rows))?;
            self.module.gpt_add(
                &self.stream,
                &launch,
                self.config.width,
                &workspace.adjoints[target],
                &workspace.adjoints[source],
                &mut workspace.scratch[3],
            )?;
            std::mem::swap(&mut workspace.adjoints[target], &mut workspace.scratch[3]);
        } else {
            let launch = self.module.prepare_gpt_copy(config(workspace.rows))?;
            let (before, after) = workspace.adjoints.split_at_mut(source);
            self.module.gpt_copy(
                &self.stream,
                &launch,
                self.config.width,
                &after[0],
                &mut before[target],
            )?;
            workspace.initialized[target] = true;
        }
        Ok(())
    }
}

fn config(elements: usize) -> LaunchConfig1D {
    LaunchConfig1D::new((elements as u32).div_ceil(128), 128, 0)
}
