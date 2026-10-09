use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, device, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

use crate::{Result, mesh::validate_finite};

use super::{
    Attention, Features, GptConfig, GptTrainOptions, NormCache, RMSNORM_EPSILON, ROPE_BASE,
    Vocabulary, XSA_EPSILON, config::Layout,
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
        width >= 2, width <= 64, _vocab >= 2, _vocab <= 256,
        out.len() == tokens.len(), state.len() == _vocab * width))]
    pub fn gpt_embed(
        width: usize,
        _vocab: usize,
        tokens: &[u32],
        state: &[[f32; 3]],
        mut out: DisjointSlice<Features>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            for channel in 0..width {
                output[channel] = state[tokens[row] as usize * width + channel][0];
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        heads >= 1, heads <= 8, head_width >= 2, head_width <= 64,
        heads * head_width <= 64,
        out.len() == input.len(), cache.len() == input.len()))]
    pub fn gpt_rmsnorm(
        heads: usize,
        head_width: usize,
        input: &[Features],
        mut out: DisjointSlice<Features>,
        mut cache: DisjointSlice<NormCache>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index)
            && let Some(saved) = cache.get_mut(thread::index_1d())
        {
            for head in 0..heads {
                let begin = head * head_width;
                let mut squares = 0.0;
                for channel in begin..begin + head_width {
                    let x = input[row][channel];
                    squares += x * x;
                }
                let inverse = 1.0 / (squares / head_width as f32 + RMSNORM_EPSILON).sqrt();
                saved[64 + head] = inverse;
                for channel in begin..begin + head_width {
                    saved[channel] = input[row][channel] * inverse;
                    output[channel] = saved[channel];
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, offset + 2 * width <= state.len(), activated <= 1,
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
                let p = offset + 2 * channel;
                let value = input[row][channel] * state[p][0]
                    + input[row][(channel + 1) % width] * state[p + 1][0];
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
        heads >= 1, heads <= 8, half_width >= 1, half_width <= 32,
        2 * heads * half_width <= 64, sequence >= 1, sequence <= 128,
        inverse <= 1, out.len() == input.len()))]
    pub fn gpt_rope(
        heads: usize,
        half_width: usize,
        sequence: usize,
        inverse: usize,
        input: &[Features],
        mut out: DisjointSlice<Features>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            let position = (row % sequence) as f32;
            for pair in 0..half_width {
                let frequency = ROPE_BASE.powf(-(pair as f32) / half_width as f32);
                let angle = if inverse == 0 { position } else { -position } * frequency;
                let cosine = angle.cos();
                let sine = angle.sin();
                for head in 0..heads {
                    let left = head * 2 * half_width + pair;
                    let right = left + half_width;
                    let a = input[row][left];
                    let b = input[row][right];
                    output[left] = a * cosine - b * sine;
                    output[right] = b * cosine + a * sine;
                }
            }
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
        width >= 2, width <= 64, heads >= 1, heads <= 8, heads <= width,
        value.len() == input.len(), out.len() == input.len()))]
    pub fn gpt_exclusive(
        width: usize,
        heads: usize,
        input: &[Features],
        value: &[Features],
        mut out: DisjointSlice<Features>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            let head_width = width / heads;
            for head in 0..heads {
                let begin = head * head_width;
                let end = begin + head_width;
                let mut squared_norm = 0.0;
                for channel in begin..end {
                    squared_norm += value[row][channel] * value[row][channel];
                }
                let inverse = 1.0 / squared_norm.sqrt().max(XSA_EPSILON);
                let mut projection = 0.0;
                for channel in begin..end {
                    projection += input[row][channel] * (value[row][channel] * inverse);
                }
                for channel in begin..end {
                    output[channel] =
                        input[row][channel] - projection * (value[row][channel] * inverse);
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, heads >= 1, heads <= 8, heads <= width,
        value.len() == input.len(), gradient.len() == input.len(),
        dy.len() == input.len(), dv.len() == input.len()))]
    pub fn gpt_exclusive_backward(
        width: usize,
        heads: usize,
        input: &[Features],
        value: &[Features],
        gradient: &[Features],
        mut dy: DisjointSlice<Features>,
        mut dv: DisjointSlice<Features>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(y_gradient) = dy.get_mut(index)
            && let Some(v_gradient) = dv.get_mut(thread::index_1d())
        {
            let head_width = width / heads;
            for head in 0..heads {
                let begin = head * head_width;
                let end = begin + head_width;
                let mut squared_norm = 0.0;
                for channel in begin..end {
                    squared_norm += value[row][channel] * value[row][channel];
                }
                let norm = squared_norm.sqrt();
                let inverse = 1.0 / norm.max(XSA_EPSILON);
                let mut projection = 0.0;
                let mut gradient_projection = 0.0;
                for channel in begin..end {
                    let normalized = value[row][channel] * inverse;
                    projection += input[row][channel] * normalized;
                    gradient_projection += gradient[row][channel] * normalized;
                }
                for channel in begin..end {
                    let normalized = value[row][channel] * inverse;
                    y_gradient[channel] = gradient[row][channel] - gradient_projection * normalized;
                    // Differentiate both uses of normalized V. Above the norm
                    // floor, also differentiate the normalization denominator.
                    let mut derivative = -gradient_projection * input[row][channel]
                        - projection * gradient[row][channel];
                    if norm >= XSA_EPSILON {
                        derivative += 2.0 * projection * gradient_projection * normalized;
                    }
                    v_gradient[channel] = derivative * inverse;
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, vocab >= 2, vocab <= 256,
        embedding.len() == vocab * width, out.len() == input.len()))]
    pub fn gpt_head(
        width: usize,
        vocab: usize,
        input: &[Features],
        embedding: &[[f32; 3]],
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
                output[token] = value;
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
        width >= 2, width <= 64, vocab >= 2, vocab <= 256, embedding.len() == vocab * width, out.len() == gradient.len()))]
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
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        heads >= 1, heads <= 8, head_width >= 2, head_width <= 64,
        heads * head_width <= 64,
        gradient.len() == cache.len(), out.len() == cache.len()))]
    pub fn gpt_rmsnorm_backward(
        heads: usize,
        head_width: usize,
        gradient: &[Features],
        cache: &[NormCache],
        mut out: DisjointSlice<Features>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            for head in 0..heads {
                let begin = head * head_width;
                let mut dot = 0.0;
                for channel in begin..begin + head_width {
                    dot += gradient[row][channel] * cache[row][channel];
                }
                for channel in begin..begin + head_width {
                    output[channel] = cache[row][64 + head]
                        * (gradient[row][channel] - cache[row][channel] * dot / head_width as f32);
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, offset + 2 * width <= state.len(), activated <= 1,
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
                    * state[offset + 2 * channel][0]
                    + gradient[row][previous]
                        * slope(activation[row][previous], activated)
                        * state[offset + 2 * previous + 1][0];
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= 64, offset + 2 * width <= out.len(), activated <= 1,
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
            && parameter < offset + 2 * width
            && let Some(output) = out.get_mut(index)
        {
            let channel = (parameter - offset) / 2;
            let edge = (parameter - offset) % 2;
            let mut total = 0.0;
            for row in 0..gradient.len() {
                let x = if edge == 0 {
                    input[row][channel]
                } else {
                    input[row][(channel + 1) % width]
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
        width >= 2, width <= 64, _vocab >= 2, _vocab <= 256,
        input_gradient.len() == tokens.len(), head_gradient.len() == tokens.len(), final_features.len() == tokens.len(),
        out.len() == _vocab * width))]
    pub fn gpt_embedding_gradient(
        width: usize,
        _vocab: usize,
        tokens: &[u32],
        input_gradient: &[Features],
        head_gradient: &[Vocabulary],
        final_features: &[Features],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let parameter = index.get();
        if let Some(output) = out.get_mut(index) {
            let token = parameter / width;
            let channel = parameter % width;
            let mut total = 0.0;
            for row in 0..tokens.len() {
                if tokens[row] as usize == token {
                    total += input_gradient[row][channel];
                }
                total += head_gradient[row][token] * final_features[row][channel];
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
    RmsNorm {
        heads: usize,
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
    Rotary {
        input: usize,
        out: usize,
    },
    Attention {
        query: usize,
        key: usize,
        value: usize,
        out: usize,
        cache: usize,
    },
    Exclusive {
        input: usize,
        value: usize,
        out: usize,
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
            let normalized = plan.rmsnorm(x, 1);
            let query = plan.mesh(normalized, block.query, config.attention_mesh_depth);
            let key = plan.mesh(normalized, block.key, config.attention_mesh_depth);
            let value = plan.mesh(normalized, block.value, config.attention_mesh_depth);
            let query = plan.rmsnorm(query, config.heads);
            let key = plan.rmsnorm(key, config.heads);
            let query = plan.rope(query);
            let key = plan.rope(key);
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
            let exclusive = plan.feature();
            plan.ops.push(Op::Exclusive {
                input: out,
                value,
                out: exclusive,
            });
            let projected = plan.mesh(exclusive, block.projection, config.attention_mesh_depth);
            x = plan.add(x, projected);
            let normalized = plan.rmsnorm(x, 1);
            let ffn = plan.mesh(normalized, block.ffn, config.ffn_mesh_depth);
            x = plan.add(x, ffn);
        }
        plan.final_features = plan.rmsnorm(x, 1);
        plan
    }
    fn feature(&mut self) -> usize {
        let id = self.features;
        self.features += 1;
        id
    }
    fn rmsnorm(&mut self, input: usize, heads: usize) -> usize {
        let out = self.feature();
        let cache = self.norms;
        self.norms += 1;
        self.ops.push(Op::RmsNorm {
            heads,
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
    fn rope(&mut self, input: usize) -> usize {
        let out = self.feature();
        self.ops.push(Op::Rotary { input, out });
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

/// A small GPT decoder with slice RoPE and exclusive self attention (XSA). Q/K/V, attention
/// output, and feed-forward projections are two-parent ELU meshes.
/// Embeddings and the tied output head
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
                        c.vocab_size,
                        &workspace.tokens,
                        &self.parameters[0].state,
                        &mut workspace.features[out],
                    )?;
                }
                Op::RmsNorm {
                    heads,
                    input,
                    out,
                    cache,
                } => {
                    let launch = self.module.prepare_gpt_rmsnorm(row_config)?;
                    let (before, after) = workspace.features.split_at_mut(out);
                    self.module.gpt_rmsnorm(
                        &self.stream,
                        &launch,
                        heads,
                        c.width / heads,
                        &before[input],
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
                        layer * 2 * c.width,
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
                Op::Rotary { input, out } => {
                    let launch = self.module.prepare_gpt_rope(row_config)?;
                    let (before, after) = workspace.features.split_at_mut(out);
                    self.module.gpt_rope(
                        &self.stream,
                        &launch,
                        c.heads,
                        c.width / c.heads / 2,
                        workspace.sequence,
                        0,
                        &before[input],
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
                Op::Exclusive { input, value, out } => {
                    let launch = self.module.prepare_gpt_exclusive(row_config)?;
                    let (before, after) = workspace.features.split_at_mut(out);
                    self.module.gpt_exclusive(
                        &self.stream,
                        &launch,
                        c.width,
                        c.heads,
                        &before[input],
                        &before[value],
                        &mut after[0],
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

        for index in (0..self.plan.ops.len()).rev() {
            match self.plan.ops[index] {
                Op::Embed { .. } => {}
                Op::RmsNorm {
                    heads,
                    input,
                    out,
                    cache,
                } => {
                    let launch = self.module.prepare_gpt_rmsnorm_backward(row_config)?;
                    self.module.gpt_rmsnorm_backward(
                        &self.stream,
                        &launch,
                        heads,
                        c.width / heads,
                        &workspace.adjoints[out],
                        &workspace.norms[cache],
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
                        layer * 2 * c.width,
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
                        layer * 2 * c.width,
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
                Op::Rotary { input, out } => {
                    let launch = self.module.prepare_gpt_rope(row_config)?;
                    self.module.gpt_rope(
                        &self.stream,
                        &launch,
                        c.heads,
                        c.width / c.heads / 2,
                        workspace.sequence,
                        1,
                        &workspace.adjoints[out],
                        &mut workspace.scratch[0],
                    )?;
                    self.accumulate_scratch(workspace, input, 0)?;
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
                Op::Exclusive { input, value, out } => {
                    let launch = self.module.prepare_gpt_exclusive_backward(row_config)?;
                    let (dy, rest) = workspace.scratch.split_at_mut(1);
                    self.module.gpt_exclusive_backward(
                        &self.stream,
                        &launch,
                        c.width,
                        c.heads,
                        &workspace.features[input],
                        &workspace.features[value],
                        &workspace.adjoints[out],
                        &mut dy[0],
                        &mut rest[0],
                    )?;
                    self.accumulate_scratch(workspace, input, 0)?;
                    // V receives this local contribution and the subsequent
                    // attention contribution gathered from all causal queries.
                    self.accumulate_scratch(workspace, value, 1)?;
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

mod verification {
    use super::*;
    use crate::gpt::{
        MAX_WIDTH,
        reference::{exclusive_reference, rmsnorm_reference, rotary_reference},
    };

    impl GpuMeshGpt {
        /// Checks parameter-free full-width RMSNorm and per-head QK-Norm against f64 autodiff.
        pub fn verify_rmsnorm(device: usize) -> Result<()> {
            let model = GpuMeshGpt::new(device, GptConfig::default(), 7)?;
            let rows = 129;
            for (width, heads) in [(2, 1), (4, 2), (6, 1), (12, 2), (32, 4), (64, 1), (64, 8)] {
                let head_width = width / heads;
                let mut input = vec![[0.0; MAX_WIDTH]; rows];
                let mut gradient = input.clone();
                for row in 0..rows {
                    for channel in 0..width {
                        let scale = (channel / head_width + 1) as f32;
                        let x = (((row + 3 * channel) % 19) as f32 * 0.2 - 1.7) * scale;
                        input[row][channel] = match row % 5 {
                            0 => 0.0,
                            1 => 2.0 * scale,
                            2 => x * 1e-7,
                            3 => x * RMSNORM_EPSILON.sqrt(),
                            _ => x,
                        };
                        gradient[row][channel] = ((row * 5 + channel) % 13) as f32 * 0.1 - 0.6;
                    }
                }
                let input_gpu = DeviceBuffer::from_host(&model.stream, &input)?;
                let gradient_gpu = DeviceBuffer::from_host(&model.stream, &gradient)?;
                let mut output_gpu = DeviceBuffer::zeroed(&model.stream, rows)?;
                let mut derivative_gpu = DeviceBuffer::zeroed(&model.stream, rows)?;
                let mut cache = DeviceBuffer::zeroed(&model.stream, rows)?;
                let launch = model.module.prepare_gpt_rmsnorm(config(rows))?;
                model.module.gpt_rmsnorm(
                    &model.stream,
                    &launch,
                    heads,
                    head_width,
                    &input_gpu,
                    &mut output_gpu,
                    &mut cache,
                )?;
                let backward = model.module.prepare_gpt_rmsnorm_backward(config(rows))?;
                model.module.gpt_rmsnorm_backward(
                    &model.stream,
                    &backward,
                    heads,
                    head_width,
                    &gradient_gpu,
                    &cache,
                    &mut derivative_gpu,
                )?;
                let output = output_gpu.to_host_vec(&model.stream)?;
                let derivative = derivative_gpu.to_host_vec(&model.stream)?;
                let doubles = |x: &[f32]| x.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
                for row in 0..rows {
                    let (expected, gradients) = rmsnorm_reference(
                        &doubles(&input[row][..width]),
                        &doubles(&gradient[row][..width]),
                        heads,
                    );
                    for channel in 0..width {
                        for (name, cpu, gpu) in [
                            ("output", expected[channel], output[row][channel]),
                            ("backward", gradients[channel], derivative[row][channel]),
                        ] {
                            let gpu = f64::from(gpu);
                            assert!(
                                gpu.is_finite() && (cpu - gpu).abs() < 4e-5 * (1.0 + cpu.abs()),
                                "RMSNorm {name} width={width} heads={heads} row={row} channel={channel}: {cpu} vs {gpu}"
                            );
                        }
                        if row % 5 == 0 {
                            assert_eq!(output[row][channel], 0.0);
                        }
                    }
                }
                for (invalid_heads, invalid_width) in [
                    (0, head_width),
                    (9, head_width),
                    (heads, 0),
                    (heads, 1),
                    (heads, 65),
                    (8, 16),
                ] {
                    assert!(
                        model
                            .module
                            .gpt_rmsnorm(
                                &model.stream,
                                &launch,
                                invalid_heads,
                                invalid_width,
                                &input_gpu,
                                &mut output_gpu,
                                &mut cache,
                            )
                            .is_err()
                    );
                    assert!(
                        model
                            .module
                            .gpt_rmsnorm_backward(
                                &model.stream,
                                &backward,
                                invalid_heads,
                                invalid_width,
                                &gradient_gpu,
                                &cache,
                                &mut derivative_gpu,
                            )
                            .is_err()
                    );
                }
                let mut wrong_cache = DeviceBuffer::zeroed(&model.stream, rows - 1)?;
                assert!(
                    model
                        .module
                        .gpt_rmsnorm(
                            &model.stream,
                            &launch,
                            heads,
                            head_width,
                            &input_gpu,
                            &mut output_gpu,
                            &mut wrong_cache,
                        )
                        .is_err()
                );
                assert!(
                    model
                        .module
                        .gpt_rmsnorm_backward(
                            &model.stream,
                            &backward,
                            heads,
                            head_width,
                            &gradient_gpu,
                            &wrong_cache,
                            &mut derivative_gpu,
                        )
                        .is_err()
                );
            }
            Ok(())
        }

        /// Checks slice RoPE, its inverse, and its CUDA backward against f64 autodiff.
        pub fn verify_rotary_embeddings(device: usize) -> Result<()> {
            let model = GpuMeshGpt::new(device, GptConfig::default(), 7)?;
            let rows = 129;
            for (width, heads) in [(4, 1), (4, 2), (12, 2), (64, 8)] {
                let head_width = width / heads;
                let half_width = head_width / 2;
                let mut input = vec![[0.0; MAX_WIDTH]; rows];
                let mut gradient = input.clone();
                for row in 0..rows {
                    for channel in 0..width {
                        input[row][channel] = ((row + 3 * channel) % 19) as f32 * 0.2 - 1.7;
                        gradient[row][channel] = ((row * 5 + channel) % 13) as f32 * 0.1 - 0.6;
                    }
                }
                let input_gpu = DeviceBuffer::from_host(&model.stream, &input)?;
                let gradient_gpu = DeviceBuffer::from_host(&model.stream, &gradient)?;
                let mut output_gpu = DeviceBuffer::zeroed(&model.stream, rows)?;
                let mut derivative_gpu = DeviceBuffer::zeroed(&model.stream, rows)?;
                let mut inverse_gpu = DeviceBuffer::zeroed(&model.stream, rows)?;
                let launch = model.module.prepare_gpt_rope(config(rows))?;
                for sequence in [1, 3, 128] {
                    for (source, target, inverse) in [
                        (&input_gpu, &mut output_gpu, 0),
                        (&gradient_gpu, &mut derivative_gpu, 1),
                    ] {
                        model.module.gpt_rope(
                            &model.stream,
                            &launch,
                            heads,
                            half_width,
                            sequence,
                            inverse,
                            source,
                            target,
                        )?;
                    }
                    model.module.gpt_rope(
                        &model.stream,
                        &launch,
                        heads,
                        half_width,
                        sequence,
                        1,
                        &output_gpu,
                        &mut inverse_gpu,
                    )?;
                    let output = output_gpu.to_host_vec(&model.stream)?;
                    let derivative = derivative_gpu.to_host_vec(&model.stream)?;
                    let inverse = inverse_gpu.to_host_vec(&model.stream)?;
                    for row in 0..rows {
                        let doubles =
                            |x: &[f32]| x.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
                        let (expected, expected_gradient) = rotary_reference(
                            &doubles(&input[row][..width]),
                            &doubles(&gradient[row][..width]),
                            row % sequence,
                            heads,
                        );
                        for channel in 0..width {
                            for (name, expected, actual) in [
                                ("output", expected[channel], output[row][channel]),
                                (
                                    "backward",
                                    expected_gradient[channel],
                                    derivative[row][channel],
                                ),
                                (
                                    "inverse",
                                    f64::from(input[row][channel]),
                                    inverse[row][channel],
                                ),
                            ] {
                                let actual = f64::from(actual);
                                assert!(
                                    actual.is_finite()
                                        && (actual - expected).abs()
                                            < 3e-5 * (1.0 + expected.abs()),
                                    "RoPE {name} width={width} heads={heads} sequence={sequence} row={row} channel={channel}: {actual} vs {expected}"
                                );
                            }
                            if row % sequence == 0 {
                                assert_eq!(output[row][channel], input[row][channel]);
                                assert_eq!(derivative[row][channel], gradient[row][channel]);
                            }
                        }
                        for head in 0..heads {
                            let begin = head * head_width;
                            let norm =
                                |x: &[f32]| x.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
                            let before = norm(&input[row][begin..begin + head_width]);
                            let after = norm(&output[row][begin..begin + head_width]);
                            assert!((before - after).abs() < 3e-5 * (1.0 + before));
                        }
                    }
                }
                for (half, sequence, inverse) in [
                    (0, 3, 0),
                    (33, 3, 0),
                    (half_width, 0, 0),
                    (half_width, 129, 0),
                    (half_width, 3, 2),
                ] {
                    assert!(
                        model
                            .module
                            .gpt_rope(
                                &model.stream,
                                &launch,
                                heads,
                                half,
                                sequence,
                                inverse,
                                &input_gpu,
                                &mut output_gpu,
                            )
                            .is_err()
                    );
                }
            }
            Ok(())
        }

        /// Checks the isolated CUDA XSA operators against independent f64 autodiff.
        /// Includes batch tails, one-channel heads, and zero/tiny/ordinary values.
        pub fn verify_exclusive_attention(device: usize) -> Result<()> {
            let model = GpuMeshGpt::new(device, GptConfig::default(), 7)?;
            let rows = 129;
            for (width, heads) in [(4, 2), (8, 2), (64, 8), (4, 4)] {
                let head_width = width / heads;
                let mut input = vec![[0.0; MAX_WIDTH]; rows];
                let mut value = input.clone();
                let mut gradient = input.clone();
                for row in 0..rows {
                    for channel in 0..width {
                        let y = ((row + channel * 3) % 17) as f32 * 0.2 - 1.1;
                        input[row][channel] = y;
                        gradient[row][channel] = ((row * 3 + channel) % 11) as f32 * 0.1 - 0.6;
                        value[row][channel] = match row % 5 {
                            0 => 0.0,
                            1 => 0.3 * (channel + 1) as f32,
                            2 => 0.25 * XSA_EPSILON / (head_width as f32).sqrt(),
                            3 => 2.0 * XSA_EPSILON / (head_width as f32).sqrt(),
                            _ => 2.0 * y,
                        };
                    }
                }
                let input_gpu = DeviceBuffer::from_host(&model.stream, &input)?;
                let value_gpu = DeviceBuffer::from_host(&model.stream, &value)?;
                let gradient_gpu = DeviceBuffer::from_host(&model.stream, &gradient)?;
                let mut output = DeviceBuffer::zeroed(&model.stream, rows)?;
                let mut dy = DeviceBuffer::zeroed(&model.stream, rows)?;
                let mut dv = DeviceBuffer::zeroed(&model.stream, rows)?;
                let launch = model.module.prepare_gpt_exclusive(config(rows))?;
                model.module.gpt_exclusive(
                    &model.stream,
                    &launch,
                    width,
                    heads,
                    &input_gpu,
                    &value_gpu,
                    &mut output,
                )?;
                let launch = model.module.prepare_gpt_exclusive_backward(config(rows))?;
                model.module.gpt_exclusive_backward(
                    &model.stream,
                    &launch,
                    width,
                    heads,
                    &input_gpu,
                    &value_gpu,
                    &gradient_gpu,
                    &mut dy,
                    &mut dv,
                )?;
                let output = output.to_host_vec(&model.stream)?;
                let dy = dy.to_host_vec(&model.stream)?;
                let dv = dv.to_host_vec(&model.stream)?;
                for row in 0..rows {
                    let doubles =
                        |values: &[f32]| values.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
                    let (expected, derivatives) = exclusive_reference(
                        &doubles(&input[row][..width]),
                        &doubles(&value[row][..width]),
                        &doubles(&gradient[row][..width]),
                        heads,
                    );
                    for channel in 0..width {
                        let begin = channel / head_width * head_width;
                        let norm = value[row][begin..begin + head_width]
                            .iter()
                            .map(|&v| f64::from(v).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        // Scale dV by its normalization denominator: its magnitude
                        // can be 1e12 for tiny V, including cancellation residuals.
                        for (name, cpu, gpu, scale) in [
                            ("output", expected[channel], output[row][channel], 1.0),
                            ("dY", derivatives[channel], dy[row][channel], 1.0),
                            (
                                "dV",
                                derivatives[width + channel],
                                dv[row][channel],
                                norm.max(f64::from(XSA_EPSILON)),
                            ),
                        ] {
                            let gpu = f64::from(gpu);
                            assert!(
                                cpu.is_finite()
                                    && gpu.is_finite()
                                    && (cpu - gpu).abs() * scale < 3e-5 * (1.0 + cpu.abs() * scale),
                                "{name} width={width} heads={heads} row={row} channel={channel}: {cpu} vs {gpu}"
                            );
                        }
                        if row % 5 == 0 {
                            assert_eq!(output[row][channel], input[row][channel]);
                            assert_eq!(dy[row][channel], gradient[row][channel]);
                            assert_eq!(dv[row][channel], 0.0);
                        }
                    }
                    for head in 0..heads {
                        let begin = head * head_width;
                        let dot = (begin..begin + head_width)
                            .map(|c| f64::from(output[row][c]) * f64::from(value[row][c]))
                            .sum::<f64>();
                        if row % 5 == 1 || row % 5 == 4 {
                            assert!(dot.abs() < 3e-5, "headwise orthogonality: {dot}");
                        }
                    }
                }
            }
            Ok(())
        }
    }
}
