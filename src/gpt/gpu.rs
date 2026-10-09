use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, device, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;

use crate::{Result, mesh::validate_finite};

use super::{
    AdamWOptions, Attention, Features, GptConfig, IGNORE_TARGET, NormCache, RMSNORM_EPSILON,
    ROPE_BASE, Vocabulary, XSA_EPSILON,
    config::{Layout, token_codes},
};

const EMBEDDING_GRADIENT_ROWS: usize = 128;

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
        width >= 2, width <= W, _vocab >= 2,
        out.len() == tokens.len(), embedding.len() == _vocab))]
    pub fn gpt_embed<const W: usize>(
        width: usize,
        _vocab: usize,
        tokens: &[u32],
        embedding: &[Features<W>],
        mut out: DisjointSlice<Features<W>>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            output[..width].copy_from_slice(&embedding[tokens[row] as usize][..width]);
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        heads >= 1, heads <= W, head_width >= 2, head_width <= W,
        heads * head_width <= W,
        out.len() == input.len(), cache.len() == input.len()))]
    pub fn gpt_rmsnorm<const W: usize>(
        heads: usize,
        head_width: usize,
        input: &[Features<W>],
        mut out: DisjointSlice<Features<W>>,
        mut cache: DisjointSlice<NormCache<W>>,
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
                for channel in begin..begin + head_width {
                    saved[channel] = [input[row][channel] * inverse, inverse];
                    output[channel] = saved[channel][0];
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        _capacity_warps >= 1, W == _capacity_warps * 32, heads >= 1,
        head_warps >= 1, heads * head_warps <= _capacity_warps,
        out.len() == input.len() * W, cache.len() == out.len()))]
    pub fn gpt_rmsnorm_parallel<const W: usize>(
        _capacity_warps: usize,
        heads: usize,
        head_warps: usize,
        input: &[Features<W>],
        mut out: DisjointSlice<f32>,
        mut cache: DisjointSlice<[f32; 2]>,
    ) {
        let head_width = head_warps * 32;
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        // Row, head, and capacity boundaries align with complete hardware warps.
        if row >= input.len() || channel >= heads * head_width {
            return;
        }
        let begin = channel / head_width * head_width;
        let lane = channel % 32;
        let mut squares = 0.0;
        for local in (lane..head_width).step_by(32) {
            let x = input[row][begin + local];
            squares += x * x;
        }
        let squares = warp::reduce_sum_f32(squares);
        let inverse = 1.0 / (squares / head_width as f32 + RMSNORM_EPSILON).sqrt();
        if let Some(output) = out.get_mut(index)
            && let Some(saved) = cache.get_mut(thread::index_1d())
        {
            *output = input[row][channel] * inverse;
            *saved = [*output, inverse];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, offset + 2 * width <= state.len(), activated <= 1,
        out.len() == input.len() * W))]
    pub fn gpt_mesh<const W: usize>(
        width: usize,
        offset: usize,
        activated: usize,
        input: &[Features<W>],
        state: &[[f32; 3]],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if let Some(output) = out.get_mut(index) {
            if channel >= width {
                return;
            }
            let p = offset + 2 * channel;
            let value = input[row][channel] * state[p][0]
                + input[row][(channel + 1) % width] * state[p + 1][0];
            *output = if activated == 1 && value < 0.0 {
                value.exp() - 1.0
            } else {
                value
            };
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, out.len() == left.len() * W, left.len() == right.len()))]
    pub fn gpt_add<const W: usize>(
        width: usize,
        left: &[Features<W>],
        right: &[Features<W>],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if let Some(output) = out.get_mut(index) {
            if channel >= width {
                return;
            }
            *output = left[row][channel] + right[row][channel];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, out.len() == input.len() * W))]
    pub fn gpt_copy<const W: usize>(
        width: usize,
        input: &[Features<W>],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if let Some(output) = out.get_mut(index) {
            if channel >= width {
                return;
            }
            *output = input[row][channel];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        heads >= 1, heads <= W, half_width >= 1, 2 * half_width <= W,
        2 * heads * half_width <= W, sequence >= 1, sequence <= C,
        inverse <= 1, out.len() == input.len() * W))]
    pub fn gpt_rope<const W: usize, const C: usize>(
        heads: usize,
        half_width: usize,
        sequence: usize,
        inverse: usize,
        input: &[Features<W>],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if let Some(output) = out.get_mut(index) {
            let position = (row % sequence) as f32;
            if channel >= 2 * heads * half_width {
                return;
            }
            let head = channel / (2 * half_width);
            let local = channel % (2 * half_width);
            let pair = local % half_width;
            let frequency = ROPE_BASE.powf(-(pair as f32) / half_width as f32);
            let angle = if inverse == 0 { position } else { -position } * frequency;
            let cosine = angle.cos();
            let sine = angle.sin();
            let left = head * 2 * half_width + pair;
            let right = left + half_width;
            let a = input[row][left];
            let b = input[row][right];
            *output = if local < half_width {
                a * cosine - b * sine
            } else {
                b * cosine + a * sine
            };
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, heads >= 1, heads <= H, sequence >= 1, sequence <= C,
        out.len() == query.len(), key.len() == query.len(), value.len() == query.len(), probabilities.len() == query.len()))]
    pub fn gpt_attention<const W: usize, const C: usize, const H: usize>(
        width: usize,
        heads: usize,
        sequence: usize,
        query: &[Features<W>],
        key: &[Features<W>],
        value: &[Features<W>],
        mut out: DisjointSlice<Features<W>>,
        mut probabilities: DisjointSlice<Attention<C, H>>,
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
                    probability[head][previous] = score;
                    if score > maximum {
                        maximum = score;
                    }
                }
                let mut sum = 0.0;
                for previous in 0..=position {
                    let p = previous;
                    probability[head][p] = (probability[head][p] - maximum).exp();
                    sum += probability[head][p];
                }
                for previous in 0..=position {
                    probability[head][previous] /= sum;
                }
                for channel in begin..begin + head_width {
                    let mut total = 0.0;
                    for previous in 0..=position {
                        total += probability[head][previous] * value[first + previous][channel];
                    }
                    output[channel] = total;
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, heads >= 1, heads <= W, heads <= width,
        value.len() == input.len(), out.len() == input.len()))]
    pub fn gpt_exclusive<const W: usize>(
        width: usize,
        heads: usize,
        input: &[Features<W>],
        value: &[Features<W>],
        mut out: DisjointSlice<Features<W>>,
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
        width >= 2, width <= W, heads >= 1, heads <= W, heads <= width,
        value.len() == input.len(), gradient.len() == input.len(),
        dy.len() == input.len(), dv.len() == input.len()))]
    pub fn gpt_exclusive_backward<const W: usize>(
        width: usize,
        heads: usize,
        input: &[Features<W>],
        value: &[Features<W>],
        gradient: &[Features<W>],
        mut dy: DisjointSlice<Features<W>>,
        mut dv: DisjointSlice<Features<W>>,
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
        width >= 2, width <= W, vocab >= 2, vocab <= V,
        embedding.len() == vocab, out.len() == width * V))]
    pub fn gpt_embedding_transpose<const W: usize, const V: usize>(
        width: usize,
        vocab: usize,
        embedding: &[Features<W>],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let channel = index.get() / V;
        let token = index.get() % V;
        if let Some(output) = out.get_mut(index) {
            if token >= vocab {
                return;
            }
            *output = embedding[token][channel];
        }
        let _ = width;
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, vocab >= 2, vocab <= V,
        embedding.len() == width, out.len() == input.len() * V))]
    pub fn gpt_head<const W: usize, const V: usize>(
        width: usize,
        vocab: usize,
        input: &[Features<W>],
        embedding: &[Vocabulary<V>],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / V;
        let token = index.get() % V;
        if let Some(output) = out.get_mut(index) {
            if token >= vocab {
                return;
            }
            let mut value = 0.0;
            for channel in 0..width {
                value += input[row][channel] * embedding[channel][token];
            }
            *output = value;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        vocab >= 2, vocab <= V, logits.len() >= 1, targets.len() == logits.len(),
        supervised >= 1, supervised <= logits.len(),
        gradient.len() == logits.len(), loss.len() == logits.len()))]
    pub fn gpt_cross_entropy<const V: usize>(
        vocab: usize,
        supervised: usize,
        logits: &[Vocabulary<V>],
        targets: &[u32],
        mut gradient: DisjointSlice<Vocabulary<V>>,
        mut loss: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = gradient.get_mut(index)
            && let Some(value) = loss.get_mut(thread::index_1d())
        {
            if targets[row] == IGNORE_TARGET {
                output[..vocab].fill(0.0);
                *value = 0.0;
                return;
            }
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
                output[token] = (output[token] / sum - target) / supervised as f32;
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, vocab >= 2, vocab <= V,
        embedding.len() == vocab, out.len() == gradient.len() * W))]
    pub fn gpt_head_backward<const W: usize, const V: usize>(
        width: usize,
        vocab: usize,
        gradient: &[Vocabulary<V>],
        embedding: &[Features<W>],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if let Some(output) = out.get_mut(index) {
            if channel >= width {
                return;
            }
            let mut total = 0.0;
            for token in 0..vocab {
                total += gradient[row][token] * embedding[token][channel];
            }
            *output = total;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        heads >= 1, heads <= W, head_width >= 2, head_width <= W,
        heads * head_width <= W,
        gradient.len() == cache.len(), out.len() == cache.len()))]
    pub fn gpt_rmsnorm_backward<const W: usize>(
        heads: usize,
        head_width: usize,
        gradient: &[Features<W>],
        cache: &[NormCache<W>],
        mut out: DisjointSlice<Features<W>>,
    ) {
        let index = thread::index_1d();
        let row = index.get();
        if let Some(output) = out.get_mut(index) {
            for head in 0..heads {
                let begin = head * head_width;
                let mut dot = 0.0;
                for channel in begin..begin + head_width {
                    dot += gradient[row][channel] * cache[row][channel][0];
                }
                for channel in begin..begin + head_width {
                    output[channel] = cache[row][channel][1]
                        * (gradient[row][channel]
                            - cache[row][channel][0] * dot / head_width as f32);
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        _capacity_warps >= 1, W == _capacity_warps * 32, heads >= 1,
        head_warps >= 1, heads * head_warps <= _capacity_warps,
        gradient.len() == cache.len(), out.len() == cache.len() * W))]
    pub fn gpt_rmsnorm_backward_parallel<const W: usize>(
        _capacity_warps: usize,
        heads: usize,
        head_warps: usize,
        gradient: &[Features<W>],
        cache: &[NormCache<W>],
        mut out: DisjointSlice<f32>,
    ) {
        let head_width = head_warps * 32;
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if row >= gradient.len() || channel >= heads * head_width {
            return;
        }
        let begin = channel / head_width * head_width;
        let lane = channel % 32;
        let mut dot = 0.0;
        for local in (lane..head_width).step_by(32) {
            dot += gradient[row][begin + local] * cache[row][begin + local][0];
        }
        let dot = warp::reduce_sum_f32(dot);
        if let Some(output) = out.get_mut(index) {
            *output = cache[row][channel][1]
                * (gradient[row][channel] - cache[row][channel][0] * dot / head_width as f32);
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, offset + 2 * width <= state.len(), activated <= 1,
        gradient.len() == activation.len(), out.len() == activation.len() * W))]
    pub fn gpt_mesh_backward<const W: usize>(
        width: usize,
        offset: usize,
        activated: usize,
        gradient: &[Features<W>],
        activation: &[Features<W>],
        state: &[[f32; 3]],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if let Some(output) = out.get_mut(index) {
            if channel >= width {
                return;
            }
            let previous = (channel + width - 1) % width;
            *output = gradient[row][channel]
                * slope(activation[row][channel], activated)
                * state[offset + 2 * channel][0]
                + gradient[row][previous]
                    * slope(activation[row][previous], activated)
                    * state[offset + 2 * previous + 1][0];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, offset + 2 * width <= out.len(), activated <= 1,
        input.len() == activation.len(), gradient.len() == activation.len()))]
    pub fn gpt_mesh_gradient<const W: usize>(
        width: usize,
        offset: usize,
        activated: usize,
        input: &[Features<W>],
        activation: &[Features<W>],
        gradient: &[Features<W>],
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
        width >= 2, width <= W, activated <= 1, chunks >= 1, chunks <= 128,
        input.len() <= chunks * 32, input.len() + 32 > chunks * 32,
        input.len() == activation.len(), gradient.len() == activation.len(),
        out.len() == chunks * 2 * width))]
    pub fn gpt_mesh_gradient_partials<const W: usize>(
        width: usize,
        activated: usize,
        chunks: usize,
        input: &[Features<W>],
        activation: &[Features<W>],
        gradient: &[Features<W>],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let parameter = item % (2 * width);
            let chunk = item / (2 * width);
            let channel = parameter / 2;
            let parent = (channel + parameter % 2) % width;
            let start = chunk * 32;
            let end = (start + 32).min(gradient.len());
            let mut total = 0.0;
            for row in start..end {
                total += gradient[row][channel]
                    * slope(activation[row][channel], activated)
                    * input[row][parent];
            }
            *output = total;
        }
        let _ = chunks;
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, offset + 2 * width <= out.len(),
        chunks >= 1, chunks <= 128, partials.len() == chunks * 2 * width))]
    pub fn gpt_mesh_gradient_reduce<const W: usize>(
        width: usize,
        offset: usize,
        chunks: usize,
        partials: &[f32],
        mut out: DisjointSlice<f32, thread::WarpIndex>,
    ) {
        let index = thread::index_1d();
        let parameter = index.get() / 32;
        // Each complete warp owns one parameter, including untouched ranges.
        if parameter < offset || parameter >= offset + 2 * width {
            return;
        }
        let local = parameter - offset;
        let lane = index.get() % 32;
        let mut total = 0.0;
        for chunk in (lane..chunks).step_by(32) {
            total += partials[chunk * 2 * width + local];
        }
        let total = warp::reduce_sum_f32(total);
        if let Some(warp) = thread::warp_index()
            && let Some(output) = out.get_mut(warp)
        {
            *output = total;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, heads >= 1, heads <= H, sequence >= 1, sequence <= C,
        probabilities.len() == gradient.len(), value.len() == gradient.len(), out.len() == gradient.len()))]
    pub fn gpt_score_backward<const W: usize, const C: usize, const H: usize>(
        width: usize,
        heads: usize,
        sequence: usize,
        gradient: &[Features<W>],
        value: &[Features<W>],
        probabilities: &[Attention<C, H>],
        mut out: DisjointSlice<Attention<C, H>>,
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
                    output[head][previous] = dot;
                    mean += probabilities[row][head][previous] * dot;
                }
                for previous in 0..=position {
                    let p = previous;
                    output[head][p] = probabilities[row][head][p] * (output[head][p] - mean);
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, heads >= 1, heads <= H, sequence >= 1, sequence <= C,
        key.len() == query.len(), gradient.len() == query.len(), scores.len() == query.len(), probabilities.len() == query.len(),
        dq.len() == query.len() * W, dk.len() == dq.len(), dv.len() == dq.len()))]
    pub fn gpt_attention_backward<const W: usize, const C: usize, const H: usize>(
        width: usize,
        heads: usize,
        sequence: usize,
        query: &[Features<W>],
        key: &[Features<W>],
        gradient: &[Features<W>],
        scores: &[Attention<C, H>],
        probabilities: &[Attention<C, H>],
        mut dq: DisjointSlice<f32>,
        mut dk: DisjointSlice<f32>,
        mut dv: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if let Some(q_gradient) = dq.get_mut(index)
            && let Some(k_gradient) = dk.get_mut(thread::index_1d())
            && let Some(v_gradient) = dv.get_mut(thread::index_1d())
        {
            let position = row % sequence;
            let first = row - position;
            let head_width = width / heads;
            let scale = 1.0 / (head_width as f32).sqrt();
            if channel >= width {
                return;
            }
            let head = channel / head_width;
            let mut q_sum = 0.0;
            let mut k_sum = 0.0;
            let mut v_sum = 0.0;
            for previous in 0..=position {
                q_sum += scores[row][head][previous] * key[first + previous][channel];
            }
            for future in position..sequence {
                k_sum += scores[first + future][head][position] * query[first + future][channel];
                v_sum += probabilities[first + future][head][position]
                    * gradient[first + future][channel];
            }
            *q_gradient = q_sum * scale;
            *k_gradient = k_sum * scale;
            *v_gradient = v_sum;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, _vocab >= 2, _vocab <= V,
        input_gradient.len() == tokens.len(), head_gradient.len() == tokens.len(), final_features.len() == tokens.len(),
        out.len() == _vocab * width))]
    pub fn gpt_embedding_backward<const W: usize, const V: usize>(
        width: usize,
        _vocab: usize,
        tokens: &[u32],
        input_gradient: &[Features<W>],
        head_gradient: &[Vocabulary<V>],
        final_features: &[Features<W>],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let token = index.get() / width;
        let channel = index.get() % width;
        if let Some(output) = out.get_mut(index) {
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
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, vocab >= 2, vocab <= V,
        R >= 1, R <= 4096, chunks >= 1, chunks <= 4096,
        tokens.len() <= chunks * R, tokens.len() + R > chunks * R,
        input_gradient.len() == tokens.len(), head_gradient.len() == tokens.len(),
        final_features.len() == tokens.len(), out.len() == chunks * vocab * width))]
    pub fn gpt_embedding_gradient_partials<const W: usize, const V: usize, const R: usize>(
        width: usize,
        vocab: usize,
        chunks: usize,
        tokens: &[u32],
        input_gradient: &[Features<W>],
        head_gradient: &[Vocabulary<V>],
        final_features: &[Features<W>],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let parameter = item % (vocab * width);
            let token = parameter / width;
            let channel = parameter % width;
            let start = item / (vocab * width) * R;
            let end = (start + R).min(tokens.len());
            let mut total = 0.0;
            for row in start..end {
                if tokens[row] as usize == token {
                    total += input_gradient[row][channel];
                }
                total += head_gradient[row][token] * final_features[row][channel];
            }
            *output = total;
        }
        let _ = chunks;
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, vocab >= 2, chunks >= 1, chunks <= 4096,
        partials.len() == chunks * vocab * width, out.len() == vocab * W))]
    pub fn gpt_embedding_gradient_reduce<const W: usize>(
        width: usize,
        vocab: usize,
        chunks: usize,
        partials: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let token = index.get() / W;
        let channel = index.get() % W;
        if let Some(output) = out.get_mut(index) {
            if channel >= width {
                return;
            }
            let mut total = 0.0;
            for chunk in 0..chunks {
                total += partials[(chunk * vocab + token) * width + channel];
            }
            *output = total;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, input.len() == out.len() * width,
        out.len() >= 2))]
    pub fn gpt_embedding_gradient_rows<const W: usize>(
        width: usize,
        input: &[f32],
        mut out: DisjointSlice<Features<W>>,
    ) {
        let index = thread::index_1d();
        let token = index.get();
        if let Some(output) = out.get_mut(index) {
            output[..width].copy_from_slice(&input[token * width..(token + 1) * width]);
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (gradient.len() == state.len()))]
    pub fn gpt_adam<const W: usize>(
        learning_rate: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        correction1: f32,
        correction2: f32,
        gradient: &[f32],
        mut state: DisjointSlice<[f32; 3]>,
    ) {
        let _ = W;
        let index = thread::index_1d();
        let parameter = index.get();
        if let Some(slot) = state.get_mut(index) {
            let g = gradient[parameter];
            slot[1] = beta1 * slot[1] + (1.0 - beta1) * g;
            slot[2] = beta2 * slot[2] + (1.0 - beta2) * g * g;
            let m = slot[1] / correction1;
            let v = slot[2] / correction2;
            slot[0] -= learning_rate * (m / (v.sqrt() + epsilon) + weight_decay * slot[0]);
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

struct Workspace<const W: usize, const V: usize, const C: usize, const H: usize> {
    rows: usize,
    sequence: usize,
    supervised: usize,
    tokens: DeviceBuffer<u32>,
    targets: DeviceBuffer<u32>,
    embeddings: Vec<DeviceBuffer<Features<W>>>,
    head_embedding: DeviceBuffer<Vocabulary<V>>,
    embedding_adjoints: Vec<DeviceBuffer<Features<W>>>,
    embedding_gradient: DeviceBuffer<f32>,
    embedding_gradient_partials: DeviceBuffer<f32>,
    mesh_gradient_partials: DeviceBuffer<f32>,
    features: Vec<DeviceBuffer<Features<W>>>,
    adjoints: Vec<DeviceBuffer<Features<W>>>,
    norms: Vec<DeviceBuffer<NormCache<W>>>,
    probabilities: Vec<DeviceBuffer<Attention<C, H>>>,
    scores: DeviceBuffer<Attention<C, H>>,
    scratch: Vec<DeviceBuffer<Features<W>>>,
    initialized: Vec<bool>,
    logits: DeviceBuffer<Vocabulary<V>>,
    head_gradient: DeviceBuffer<Vocabulary<V>>,
    loss: DeviceBuffer<f32>,
}

impl<const W: usize, const V: usize, const C: usize, const H: usize> Workspace<W, V, C, H> {
    fn new(
        stream: &Arc<CudaStream>,
        plan: &Plan,
        config: GptConfig,
        rows: usize,
        sequence: usize,
    ) -> Result<Self> {
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
            supervised: rows,
            tokens: DeviceBuffer::zeroed(stream, rows)?,
            targets: DeviceBuffer::zeroed(stream, rows)?,
            embeddings: buffers(stream, config.attention_mesh_depth, config.vocab_size)?,
            head_embedding: DeviceBuffer::zeroed(stream, config.width)?,
            embedding_adjoints: buffers(stream, config.attention_mesh_depth, config.vocab_size)?,
            embedding_gradient: DeviceBuffer::zeroed(stream, config.vocab_size * config.width)?,
            embedding_gradient_partials: DeviceBuffer::zeroed(
                stream,
                if use_chunked_embedding(config, rows) {
                    rows.div_ceil(EMBEDDING_GRADIENT_ROWS) * config.vocab_size * config.width
                } else {
                    0
                },
            )?,
            mesh_gradient_partials: DeviceBuffer::zeroed(
                stream,
                rows.div_ceil(32) * 2 * config.width,
            )?,
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
/// A shared mesh maps fixed token codes to both input and output embeddings.
/// Every learned parameter is a mesh edge. Training operators and AdamW run on CUDA.
pub type GpuMeshGpt = SpecializedMeshGpt<128, 256, 128, 8>;

/// Const-generic tensor capacities. Runtime configuration selects active dimensions
/// within these capacities; layers and mesh depths require no kernel rewriting.
pub struct SpecializedMeshGpt<const W: usize, const V: usize, const C: usize, const H: usize> {
    config: GptConfig,
    layout: Layout,
    plan: Plan,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    parameters: Vec<Parameters>,
    codes: DeviceBuffer<Features<W>>,
    workspace: Option<Workspace<W, V, C, H>>,
    beta1_power: f32,
    beta2_power: f32,
}

impl<const W: usize, const V: usize, const C: usize, const H: usize>
    SpecializedMeshGpt<W, V, C, H>
{
    pub fn new(device: usize, config: GptConfig, seed: u64) -> Result<Self> {
        const {
            assert!(W >= 2 && V >= 2 && C >= 1 && H >= 1);
        }
        config.validate()?;
        if config.width > W || config.vocab_size > V || config.context > C || config.heads > H {
            return Err(format!(
                "model exceeds kernel capacities: width {W}, vocabulary {V}, context {C}, heads {H}"
            )
            .into());
        }
        let layout = Layout::new(config)?;
        let plan = Plan::new(config, &layout);
        let initial = layout.initialize(config, seed);
        let context = CudaContext::new(device)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
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
        let codes: Vec<Features<W>> = token_codes(config)
            .into_iter()
            .map(|code| {
                let mut row = [0.0; W];
                row[..config.width].copy_from_slice(&code);
                row
            })
            .collect();
        let codes = DeviceBuffer::from_host(&stream, &codes)?;
        Ok(Self {
            config,
            layout,
            plan,
            stream,
            module,
            parameters,
            codes,
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
        let mut workspace = Workspace::new(
            &self.stream,
            &self.plan,
            self.config,
            tokens.len(),
            sequence,
        )?;
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
        let supervised = self.config.validate_batch(inputs, targets, sequence)?;
        let mut workspace = Workspace::new(
            &self.stream,
            &self.plan,
            self.config,
            inputs.len(),
            sequence,
        )?;
        workspace.tokens.copy_from_host(&self.stream, inputs)?;
        workspace.targets.copy_from_host(&self.stream, targets)?;
        workspace.supervised = supervised;
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
        self.train_batch_with_optimizer(
            inputs,
            targets,
            sequence,
            AdamWOptions {
                learning_rate,
                weight_decay,
                ..AdamWOptions::default()
            },
        )
    }

    /// Trains on non-ignored targets with configurable AdamW hyperparameters.
    pub fn train_batch_with_optimizer(
        &mut self,
        inputs: &[u32],
        targets: &[u32],
        sequence: usize,
        optimizer: AdamWOptions,
    ) -> Result<f64> {
        optimizer.validate()?;
        let loss = self.gradients(inputs, targets, sequence)?;
        self.beta1_power *= optimizer.beta1;
        self.beta2_power *= optimizer.beta2;
        for group in &mut self.parameters {
            let launch = self
                .module
                .prepare_gpt_adam::<W>(config(group.state.len()))?;
            self.module.gpt_adam(
                &self.stream,
                &launch,
                optimizer.learning_rate,
                optimizer.weight_decay,
                optimizer.beta1,
                optimizer.beta2,
                optimizer.epsilon,
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

    /// Returns the generated vectors shared by input lookup and the output head.
    pub fn token_embeddings(&self) -> Result<Vec<Vec<f32>>> {
        let mut workspace = Workspace::new(&self.stream, &self.plan, self.config, 1, 1)?;
        self.forward_embeddings(&mut workspace)?;
        workspace.embeddings[self.config.attention_mesh_depth - 1]
            .to_host_vec(&self.stream)?
            .iter()
            .map(|row| {
                validate_finite(&row[..self.config.width], "token embeddings")?;
                Ok(row[..self.config.width].to_vec())
            })
            .collect()
    }

    fn forward_embeddings(&self, workspace: &mut Workspace<W, V, C, H>) -> Result<()> {
        let c = self.config;
        // Generate the entire vocabulary once so the input and head share one
        // differentiable set of vectors. Recompute after every weight change.
        let embedding = self.layout.embedding;
        for layer in 0..c.attention_mesh_depth {
            let (before, after) = workspace.embeddings.split_at_mut(layer);
            let input = if layer == 0 {
                &self.codes
            } else {
                &before[layer - 1]
            };
            with_array_elements(&self.stream, &mut after[0], |output| {
                let launch = self.module.prepare_gpt_mesh::<W>(config(output.len()))?;
                self.module
                    .gpt_mesh(
                        &self.stream,
                        &launch,
                        c.width,
                        layer * 2 * c.width,
                        usize::from(layer + 1 != c.attention_mesh_depth),
                        input,
                        &self.parameters[embedding].state,
                        output,
                    )
                    .map_err(Into::into)
            })?;
        }
        Ok(())
    }

    fn forward(&self, workspace: &mut Workspace<W, V, C, H>) -> Result<()> {
        let c = self.config;
        let row_config = config(workspace.rows);
        self.forward_embeddings(workspace)?;
        let embedding_output = c.attention_mesh_depth - 1;
        // Store tokens contiguously for each channel so neighboring head threads
        // read adjacent vocabulary values. The tied vectors are still generated once.
        with_array_elements(&self.stream, &mut workspace.head_embedding, |output| {
            let launch = self
                .module
                .prepare_gpt_embedding_transpose::<W, V>(config(output.len()))?;
            self.module
                .gpt_embedding_transpose(
                    &self.stream,
                    &launch,
                    c.width,
                    c.vocab_size,
                    &workspace.embeddings[embedding_output],
                    output,
                )
                .map_err(Into::into)
        })?;
        for op in &self.plan.ops {
            match *op {
                Op::Embed { out } => {
                    let launch = self.module.prepare_gpt_embed::<W>(row_config)?;
                    self.module.gpt_embed(
                        &self.stream,
                        &launch,
                        c.width,
                        c.vocab_size,
                        &workspace.tokens,
                        &workspace.embeddings[embedding_output],
                        &mut workspace.features[out],
                    )?;
                }
                Op::RmsNorm {
                    heads,
                    input,
                    out,
                    cache,
                } => {
                    let (before, after) = workspace.features.split_at_mut(out);
                    if W.is_multiple_of(32) && (c.width / heads).is_multiple_of(32) {
                        with_array_elements(&self.stream, &mut after[0], |output| {
                            with_buffer_chunks(&self.stream, &mut workspace.norms[cache], |saved| {
                                let launch = self
                                    .module
                                    .prepare_gpt_rmsnorm_parallel::<W>(config(output.len()))?;
                                self.module
                                    .gpt_rmsnorm_parallel(
                                        &self.stream,
                                        &launch,
                                        W / 32,
                                        heads,
                                        c.width / heads / 32,
                                        &before[input],
                                        output,
                                        saved,
                                    )
                                    .map_err(Into::into)
                            })
                        })?;
                    } else {
                        let launch = self.module.prepare_gpt_rmsnorm::<W>(row_config)?;
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
                }
                Op::Mesh {
                    group,
                    input,
                    out,
                    layer,
                    activated,
                } => {
                    let (before, after) = workspace.features.split_at_mut(out);
                    with_array_elements(&self.stream, &mut after[0], |output| {
                        let launch = self.module.prepare_gpt_mesh::<W>(config(output.len()))?;
                        self.module
                            .gpt_mesh(
                                &self.stream,
                                &launch,
                                c.width,
                                layer * 2 * c.width,
                                activated,
                                &before[input],
                                &self.parameters[group].state,
                                output,
                            )
                            .map_err(Into::into)
                    })?;
                }
                Op::Add { left, right, out } => {
                    let (before, after) = workspace.features.split_at_mut(out);
                    with_array_elements(&self.stream, &mut after[0], |output| {
                        let launch = self.module.prepare_gpt_add::<W>(config(output.len()))?;
                        self.module
                            .gpt_add(
                                &self.stream,
                                &launch,
                                c.width,
                                &before[left],
                                &before[right],
                                output,
                            )
                            .map_err(Into::into)
                    })?;
                }
                Op::Rotary { input, out } => {
                    let (before, after) = workspace.features.split_at_mut(out);
                    with_array_elements(&self.stream, &mut after[0], |output| {
                        let launch = self.module.prepare_gpt_rope::<W, C>(config(output.len()))?;
                        self.module
                            .gpt_rope(
                                &self.stream,
                                &launch,
                                c.heads,
                                c.width / c.heads / 2,
                                workspace.sequence,
                                0,
                                &before[input],
                                output,
                            )
                            .map_err(Into::into)
                    })?;
                }
                Op::Attention {
                    query,
                    key,
                    value,
                    out,
                    cache,
                } => {
                    let launch = self.module.prepare_gpt_attention::<W, C, H>(row_config)?;
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
                    let launch = self.module.prepare_gpt_exclusive::<W>(row_config)?;
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
        with_array_elements(&self.stream, &mut workspace.logits, |output| {
            let launch = self.module.prepare_gpt_head::<W, V>(config(output.len()))?;
            self.module
                .gpt_head(
                    &self.stream,
                    &launch,
                    c.width,
                    c.vocab_size,
                    &workspace.features[self.plan.final_features],
                    &workspace.head_embedding,
                    output,
                )
                .map_err(Into::into)
        })?;
        Ok(())
    }

    fn loss(&self, workspace: &mut Workspace<W, V, C, H>) -> Result<f64> {
        let launch = self
            .module
            .prepare_gpt_cross_entropy::<V>(config(workspace.rows))?;
        self.module.gpt_cross_entropy(
            &self.stream,
            &launch,
            self.config.vocab_size,
            workspace.supervised,
            &workspace.logits,
            &workspace.targets,
            &mut workspace.head_gradient,
            &mut workspace.loss,
        )?;
        let loss = workspace.loss.to_host_vec(&self.stream)?;
        validate_finite(&loss, "cross-entropy")?;
        Ok(loss.iter().map(|&value| f64::from(value)).sum::<f64>() / workspace.supervised as f64)
    }

    fn gradients(&mut self, inputs: &[u32], targets: &[u32], sequence: usize) -> Result<f64> {
        let supervised = self.config.validate_batch(inputs, targets, sequence)?;
        let mut workspace = match self.workspace.take() {
            Some(workspace) if workspace.rows == inputs.len() && workspace.sequence == sequence => {
                workspace
            }
            _ => Workspace::new(
                &self.stream,
                &self.plan,
                self.config,
                inputs.len(),
                sequence,
            )?,
        };
        let result = (|| {
            workspace.tokens.copy_from_host(&self.stream, inputs)?;
            workspace.targets.copy_from_host(&self.stream, targets)?;
            workspace.supervised = supervised;
            self.forward(&mut workspace)?;
            let loss = self.loss(&mut workspace)?;
            self.backward(&mut workspace)?;
            Ok(loss)
        })();
        self.workspace = Some(workspace);
        result
    }

    fn backward(&mut self, workspace: &mut Workspace<W, V, C, H>) -> Result<()> {
        let c = self.config;
        let row_config = config(workspace.rows);
        workspace.initialized.fill(false);
        with_array_elements(&self.stream, &mut workspace.scratch[0], |output| {
            let launch = self
                .module
                .prepare_gpt_head_backward::<W, V>(config(output.len()))?;
            self.module
                .gpt_head_backward(
                    &self.stream,
                    &launch,
                    c.width,
                    c.vocab_size,
                    &workspace.head_gradient,
                    &workspace.embeddings[c.attention_mesh_depth - 1],
                    output,
                )
                .map_err(Into::into)
        })?;
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
                    if W.is_multiple_of(32) && (c.width / heads).is_multiple_of(32) {
                        with_array_elements(&self.stream, &mut workspace.scratch[0], |output| {
                            let launch = self
                                .module
                                .prepare_gpt_rmsnorm_backward_parallel::<W>(config(output.len()))?;
                            self.module
                                .gpt_rmsnorm_backward_parallel(
                                    &self.stream,
                                    &launch,
                                    W / 32,
                                    heads,
                                    c.width / heads / 32,
                                    &workspace.adjoints[out],
                                    &workspace.norms[cache],
                                    output,
                                )
                                .map_err(Into::into)
                        })?;
                    } else {
                        let launch = self.module.prepare_gpt_rmsnorm_backward::<W>(row_config)?;
                        self.module.gpt_rmsnorm_backward(
                            &self.stream,
                            &launch,
                            heads,
                            c.width / heads,
                            &workspace.adjoints[out],
                            &workspace.norms[cache],
                            &mut workspace.scratch[0],
                        )?;
                    }
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
                    if workspace.rows > 128 {
                        let chunks = workspace.rows.div_ceil(32);
                        let launch = self.module.prepare_gpt_mesh_gradient_partials::<W>(
                            config(workspace.mesh_gradient_partials.len()),
                        )?;
                        self.module.gpt_mesh_gradient_partials(
                            &self.stream,
                            &launch,
                            c.width,
                            activated,
                            chunks,
                            &workspace.features[input],
                            &workspace.features[out],
                            &workspace.adjoints[out],
                            &mut workspace.mesh_gradient_partials,
                        )?;
                        let launch = self.module.prepare_gpt_mesh_gradient_reduce::<W>(config(
                            parameters.gradient.len() * 32,
                        ))?;
                        self.module.gpt_mesh_gradient_reduce(
                            &self.stream,
                            &launch,
                            c.width,
                            layer * 2 * c.width,
                            chunks,
                            &workspace.mesh_gradient_partials,
                            &mut parameters.gradient,
                        )?;
                    } else {
                        let launch = self
                            .module
                            .prepare_gpt_mesh_gradient::<W>(config(parameters.gradient.len()))?;
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
                    }
                    with_array_elements(&self.stream, &mut workspace.scratch[0], |output| {
                        let launch = self
                            .module
                            .prepare_gpt_mesh_backward::<W>(config(output.len()))?;
                        self.module
                            .gpt_mesh_backward(
                                &self.stream,
                                &launch,
                                c.width,
                                layer * 2 * c.width,
                                activated,
                                &workspace.adjoints[out],
                                &workspace.features[out],
                                &parameters.state,
                                output,
                            )
                            .map_err(Into::into)
                    })?;
                    self.accumulate_scratch(workspace, input, 0)?;
                }
                Op::Add { left, right, out } => {
                    self.accumulate_node(workspace, left, out)?;
                    self.accumulate_node(workspace, right, out)?;
                }
                Op::Rotary { input, out } => {
                    with_array_elements(&self.stream, &mut workspace.scratch[0], |output| {
                        let launch = self.module.prepare_gpt_rope::<W, C>(config(output.len()))?;
                        self.module
                            .gpt_rope(
                                &self.stream,
                                &launch,
                                c.heads,
                                c.width / c.heads / 2,
                                workspace.sequence,
                                1,
                                &workspace.adjoints[out],
                                output,
                            )
                            .map_err(Into::into)
                    })?;
                    self.accumulate_scratch(workspace, input, 0)?;
                }
                Op::Attention {
                    query,
                    key,
                    value,
                    out,
                    cache,
                } => {
                    let launch = self
                        .module
                        .prepare_gpt_score_backward::<W, C, H>(row_config)?;
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
                    let (dq, rest) = workspace.scratch.split_at_mut(1);
                    let (dk, rest) = rest.split_at_mut(1);
                    with_array_elements(&self.stream, &mut dq[0], |dq| {
                        with_array_elements(&self.stream, &mut dk[0], |dk| {
                            with_array_elements(&self.stream, &mut rest[0], |dv| {
                                let launch = self
                                    .module
                                    .prepare_gpt_attention_backward::<W, C, H>(config(dq.len()))?;
                                self.module
                                    .gpt_attention_backward(
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
                                        dq,
                                        dk,
                                        dv,
                                    )
                                    .map_err(Into::into)
                            })
                        })
                    })?;
                    self.accumulate_scratch(workspace, query, 0)?;
                    self.accumulate_scratch(workspace, key, 1)?;
                    self.accumulate_scratch(workspace, value, 2)?;
                }
                Op::Exclusive { input, value, out } => {
                    let launch = self
                        .module
                        .prepare_gpt_exclusive_backward::<W>(row_config)?;
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
        let embedding_output = c.attention_mesh_depth - 1;
        if use_chunked_embedding(c, workspace.rows) {
            let chunks = workspace.rows.div_ceil(EMBEDDING_GRADIENT_ROWS);
            let launch = self
                .module
                .prepare_gpt_embedding_gradient_partials::<W, V, EMBEDDING_GRADIENT_ROWS>(
                    config(workspace.embedding_gradient_partials.len()),
                )?;
            self.module.gpt_embedding_gradient_partials(
                &self.stream,
                &launch,
                c.width,
                c.vocab_size,
                chunks,
                &workspace.tokens,
                &workspace.adjoints[0],
                &workspace.head_gradient,
                &workspace.features[self.plan.final_features],
                &mut workspace.embedding_gradient_partials,
            )?;
            with_array_elements(
                &self.stream,
                &mut workspace.embedding_adjoints[embedding_output],
                |output| {
                    let launch = self
                        .module
                        .prepare_gpt_embedding_gradient_reduce::<W>(config(output.len()))?;
                    self.module
                        .gpt_embedding_gradient_reduce(
                            &self.stream,
                            &launch,
                            c.width,
                            c.vocab_size,
                            chunks,
                            &workspace.embedding_gradient_partials,
                            output,
                        )
                        .map_err(Into::into)
                },
            )?;
        } else {
            let launch = self
                .module
                .prepare_gpt_embedding_backward::<W, V>(config(c.vocab_size * c.width))?;
            self.module.gpt_embedding_backward(
                &self.stream,
                &launch,
                c.width,
                c.vocab_size,
                &workspace.tokens,
                &workspace.adjoints[0],
                &workspace.head_gradient,
                &workspace.features[self.plan.final_features],
                &mut workspace.embedding_gradient,
            )?;
            let launch = self
                .module
                .prepare_gpt_embedding_gradient_rows::<W>(config(c.vocab_size))?;
            self.module.gpt_embedding_gradient_rows(
                &self.stream,
                &launch,
                c.width,
                &workspace.embedding_gradient,
                &mut workspace.embedding_adjoints[embedding_output],
            )?;
        }
        // Combine repeated input lookups with the dense head contribution
        // before traversing the shared embedding mesh exactly once.
        let group = &mut self.parameters[self.layout.embedding];
        for layer in (0..c.attention_mesh_depth).rev() {
            let activated = usize::from(layer + 1 != c.attention_mesh_depth);
            let input = if layer == 0 {
                &self.codes
            } else {
                &workspace.embeddings[layer - 1]
            };
            let launch = self
                .module
                .prepare_gpt_mesh_gradient::<W>(config(group.gradient.len()))?;
            self.module.gpt_mesh_gradient(
                &self.stream,
                &launch,
                c.width,
                layer * 2 * c.width,
                activated,
                input,
                &workspace.embeddings[layer],
                &workspace.embedding_adjoints[layer],
                &mut group.gradient,
            )?;
            if layer > 0 {
                let (before, after) = workspace.embedding_adjoints.split_at_mut(layer);
                with_array_elements(&self.stream, &mut before[layer - 1], |output| {
                    let launch = self
                        .module
                        .prepare_gpt_mesh_backward::<W>(config(output.len()))?;
                    self.module
                        .gpt_mesh_backward(
                            &self.stream,
                            &launch,
                            c.width,
                            layer * 2 * c.width,
                            activated,
                            &after[0],
                            &workspace.embeddings[layer],
                            &group.state,
                            output,
                        )
                        .map_err(Into::into)
                })?;
            }
        }
        Ok(())
    }

    fn accumulate_scratch(
        &self,
        workspace: &mut Workspace<W, V, C, H>,
        target: usize,
        source: usize,
    ) -> Result<()> {
        if workspace.initialized[target] {
            let (before, after) = workspace.scratch.split_at_mut(3);
            with_array_elements(&self.stream, &mut after[0], |output| {
                let launch = self.module.prepare_gpt_add::<W>(config(output.len()))?;
                self.module
                    .gpt_add(
                        &self.stream,
                        &launch,
                        self.config.width,
                        &workspace.adjoints[target],
                        &before[source],
                        output,
                    )
                    .map_err(Into::into)
            })?;
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
        workspace: &mut Workspace<W, V, C, H>,
        target: usize,
        source: usize,
    ) -> Result<()> {
        if workspace.initialized[target] {
            with_array_elements(&self.stream, &mut workspace.scratch[3], |output| {
                let launch = self.module.prepare_gpt_add::<W>(config(output.len()))?;
                self.module
                    .gpt_add(
                        &self.stream,
                        &launch,
                        self.config.width,
                        &workspace.adjoints[target],
                        &workspace.adjoints[source],
                        output,
                    )
                    .map_err(Into::into)
            })?;
            std::mem::swap(&mut workspace.adjoints[target], &mut workspace.scratch[3]);
        } else {
            let (before, after) = workspace.adjoints.split_at_mut(source);
            with_array_elements(&self.stream, &mut before[target], |output| {
                let launch = self.module.prepare_gpt_copy::<W>(config(output.len()))?;
                self.module
                    .gpt_copy(&self.stream, &launch, self.config.width, &after[0], output)
                    .map_err(Into::into)
            })?;
            workspace.initialized[target] = true;
        }
        Ok(())
    }
}

/// Reuse the same allocation through checked, owning array casts. No aliasing
/// view, copy, or device allocation is introduced; restore rows even on launch error.
fn with_array_elements<const N: usize>(
    stream: &CudaStream,
    buffer: &mut DeviceBuffer<[f32; N]>,
    run: impl FnOnce(&mut DeviceBuffer<f32>) -> Result<()>,
) -> Result<()> {
    with_buffer_chunks(stream, buffer, run)
}

fn with_buffer_chunks<Row: cuda_core::DeviceCopy, Element: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    buffer: &mut DeviceBuffer<Row>,
    run: impl FnOnce(&mut DeviceBuffer<Element>) -> Result<()>,
) -> Result<()> {
    let empty = DeviceBuffer::zeroed(stream, 0)?;
    let rows = std::mem::replace(buffer, empty);
    let mut elements = match rows.cast_chunks::<Element>() {
        Ok(elements) => elements,
        Err(rows) => {
            *buffer = rows;
            return Err("allocation cannot be grouped into the requested elements".into());
        }
    };
    let result = run(&mut elements);
    *buffer = elements
        .cast_chunks::<Row>()
        .map_err(|_| "allocation cannot be restored to its original rows")?;
    result
}

fn config(elements: usize) -> LaunchConfig1D {
    LaunchConfig1D::new((elements as u32).div_ceil(128), 128, 0)
}

fn use_chunked_embedding(config: GptConfig, rows: usize) -> bool {
    rows > 128 && config.width >= 32
}

mod verification {
    use super::*;
    use crate::gpt::{
        MAX_WIDTH,
        reference::{exclusive_reference, rmsnorm_reference, rotary_reference},
    };

    impl GpuMeshGpt {
        /// Checks both tied-embedding gradient contributions, chunk tails, and contracts.
        pub fn verify_embedding_gradient_reduction(device: usize) -> Result<()> {
            let model = GpuMeshGpt::new(device, GptConfig::default(), 7)?;
            for (width, vocab) in [(2, 3), (18, 17), (32, 17), (128, 99)] {
                for rows in [
                    1_usize, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 2048, 4096,
                ] {
                    let tokens: Vec<_> = (0..rows).map(|row| (row % vocab) as u32).collect();
                    let mut input = vec![[0.0; MAX_WIDTH]; rows];
                    let mut features = input.clone();
                    let mut gradient = vec![[0.0; super::super::MAX_VOCAB]; rows];
                    for row in 0..rows {
                        for channel in 0..width {
                            input[row][channel] =
                                ((row + 3 * channel) % 19) as f32 * 0.0317 / rows as f32;
                            features[row][channel] =
                                ((3 * row + channel) % 23) as f32 * 0.1137 - 1.01;
                        }
                        for (token, value) in gradient[row][..vocab].iter_mut().enumerate() {
                            *value =
                                (((row + 5 * token) % 29) as f32 * 0.0637 - 0.891) / rows as f32;
                        }
                    }
                    let tokens_gpu = DeviceBuffer::from_host(&model.stream, &tokens)?;
                    let input_gpu = DeviceBuffer::from_host(&model.stream, &input)?;
                    let features_gpu = DeviceBuffer::from_host(&model.stream, &features)?;
                    let gradient_gpu = DeviceBuffer::from_host(&model.stream, &gradient)?;
                    let chunks = rows.div_ceil(EMBEDDING_GRADIENT_ROWS);
                    let mut partials = DeviceBuffer::zeroed(&model.stream, chunks * vocab * width)?;
                    let launch = model
                        .module
                        .prepare_gpt_embedding_gradient_partials::<
                            128,
                            256,
                            EMBEDDING_GRADIENT_ROWS,
                        >(config(partials.len()))?;
                    model.module.gpt_embedding_gradient_partials(
                        &model.stream,
                        &launch,
                        width,
                        vocab,
                        chunks,
                        &tokens_gpu,
                        &input_gpu,
                        &gradient_gpu,
                        &features_gpu,
                        &mut partials,
                    )?;
                    let mut output =
                        DeviceBuffer::from_host(&model.stream, &vec![[42.0; MAX_WIDTH]; vocab])?;
                    with_array_elements(&model.stream, &mut output, |output| {
                        let reduce = model
                            .module
                            .prepare_gpt_embedding_gradient_reduce::<128>(config(output.len()))?;
                        model.module.gpt_embedding_gradient_reduce(
                            &model.stream,
                            &reduce,
                            width,
                            vocab,
                            chunks,
                            &partials,
                            output,
                        )?;
                        let mut short = DeviceBuffer::zeroed(&model.stream, output.len() - 1)?;
                        assert!(
                            model
                                .module
                                .gpt_embedding_gradient_reduce(
                                    &model.stream,
                                    &reduce,
                                    width,
                                    vocab,
                                    chunks,
                                    &partials,
                                    &mut short,
                                )
                                .is_err()
                        );
                        for count in [0, chunks + 1, usize::MAX] {
                            assert!(
                                model
                                    .module
                                    .gpt_embedding_gradient_reduce(
                                        &model.stream,
                                        &reduce,
                                        width,
                                        vocab,
                                        count,
                                        &partials,
                                        output,
                                    )
                                    .is_err()
                            );
                        }
                        Ok(())
                    })?;
                    let output = output.to_host_vec(&model.stream)?;
                    for token in 0..vocab {
                        for channel in 0..width {
                            let expected: f64 = (0..rows)
                                .map(|row| {
                                    let lookup = if tokens[row] as usize == token {
                                        f64::from(input[row][channel])
                                    } else {
                                        0.0
                                    };
                                    lookup
                                        + f64::from(gradient[row][token])
                                            * f64::from(features[row][channel])
                                })
                                .sum();
                            let actual = f64::from(output[token][channel]);
                            assert!(
                                actual.is_finite()
                                    && (expected - actual).abs() < 3e-6 * (1.0 + expected.abs()),
                                "embedding gradient width={width} vocab={vocab} rows={rows} token={token} channel={channel}: {expected} vs {actual}"
                            );
                        }
                        assert!(output[token][width..].iter().all(|&value| value == 42.0));
                    }
                    for count in [0, chunks + 1, usize::MAX] {
                        assert!(
                            model
                                .module
                                .gpt_embedding_gradient_partials(
                                    &model.stream,
                                    &launch,
                                    width,
                                    vocab,
                                    count,
                                    &tokens_gpu,
                                    &input_gpu,
                                    &gradient_gpu,
                                    &features_gpu,
                                    &mut partials,
                                )
                                .is_err()
                        );
                    }
                    let mut short = DeviceBuffer::zeroed(&model.stream, partials.len() - 1)?;
                    assert!(
                        model
                            .module
                            .gpt_embedding_gradient_partials(
                                &model.stream,
                                &launch,
                                width,
                                vocab,
                                chunks,
                                &tokens_gpu,
                                &input_gpu,
                                &gradient_gpu,
                                &features_gpu,
                                &mut short,
                            )
                            .is_err()
                    );
                }
            }
            Ok(())
        }

        /// Checks chunked reductions, row tails, untouched layers, and launch contracts.
        pub fn verify_mesh_gradient_reduction(device: usize) -> Result<()> {
            let model = GpuMeshGpt::new(device, GptConfig::default(), 7)?;
            for width in [2, 4, 128] {
                for rows in [1_usize, 31, 32, 33, 127, 128, 129, 257, 2048, 4096] {
                    let mut input = vec![[0.0; MAX_WIDTH]; rows];
                    let mut activation = input.clone();
                    let mut gradient = input.clone();
                    for row in 0..rows {
                        for channel in 0..width {
                            input[row][channel] = ((row + channel * 3) % 19) as f32 * 0.1 - 0.7;
                            activation[row][channel] =
                                ((row * 3 + channel) % 7) as f32 * 0.25 - 0.5;
                            gradient[row][channel] =
                                (((row + channel * 5) % 17) as f32 * 0.07 - 0.5) / rows as f32;
                        }
                    }
                    let input_gpu = DeviceBuffer::from_host(&model.stream, &input)?;
                    let activation_gpu = DeviceBuffer::from_host(&model.stream, &activation)?;
                    let gradient_gpu = DeviceBuffer::from_host(&model.stream, &gradient)?;
                    let chunks = rows.div_ceil(32);
                    let mut partials = DeviceBuffer::zeroed(&model.stream, chunks * 2 * width)?;
                    let partial_launch = model
                        .module
                        .prepare_gpt_mesh_gradient_partials::<128>(config(partials.len()))?;
                    let offset = 2 * width;
                    let sentinel = vec![42.0_f32; 6 * width];
                    for activated in [0, 1] {
                        let mut serial = DeviceBuffer::from_host(&model.stream, &sentinel)?;
                        let mut reduced = DeviceBuffer::from_host(&model.stream, &sentinel)?;
                        let serial_launch = model
                            .module
                            .prepare_gpt_mesh_gradient::<128>(config(serial.len()))?;
                        model.module.gpt_mesh_gradient(
                            &model.stream,
                            &serial_launch,
                            width,
                            offset,
                            activated,
                            &input_gpu,
                            &activation_gpu,
                            &gradient_gpu,
                            &mut serial,
                        )?;
                        model.module.gpt_mesh_gradient_partials(
                            &model.stream,
                            &partial_launch,
                            width,
                            activated,
                            chunks,
                            &input_gpu,
                            &activation_gpu,
                            &gradient_gpu,
                            &mut partials,
                        )?;
                        let reduce_launch = model
                            .module
                            .prepare_gpt_mesh_gradient_reduce::<128>(config(reduced.len() * 32))?;
                        model.module.gpt_mesh_gradient_reduce(
                            &model.stream,
                            &reduce_launch,
                            width,
                            offset,
                            chunks,
                            &partials,
                            &mut reduced,
                        )?;
                        let serial = serial.to_host_vec(&model.stream)?;
                        let reduced = reduced.to_host_vec(&model.stream)?;
                        for parameter in 0..sentinel.len() {
                            let expected = if (offset..offset + 2 * width).contains(&parameter) {
                                let local = parameter - offset;
                                let channel = local / 2;
                                let parent = (channel + local % 2) % width;
                                (0..rows)
                                    .map(|row| {
                                        let value = f64::from(activation[row][channel]);
                                        let slope = if activated == 1 && value < 0.0 {
                                            value + 1.0
                                        } else {
                                            1.0
                                        };
                                        f64::from(gradient[row][channel])
                                            * slope
                                            * f64::from(input[row][parent])
                                    })
                                    .sum::<f64>()
                            } else {
                                42.0
                            };
                            for actual in [serial[parameter], reduced[parameter]] {
                                assert!(
                                    (f64::from(actual) - expected).abs()
                                        < 3e-5 * (1.0 + expected.abs()),
                                    "mesh reduction width={width} rows={rows} activated={activated} parameter={parameter}: {actual} vs {expected}"
                                );
                            }
                        }
                    }
                    for bad_chunks in [0, 129, chunks + 1] {
                        assert!(
                            model
                                .module
                                .gpt_mesh_gradient_partials(
                                    &model.stream,
                                    &partial_launch,
                                    width,
                                    0,
                                    bad_chunks,
                                    &input_gpu,
                                    &activation_gpu,
                                    &gradient_gpu,
                                    &mut partials,
                                )
                                .is_err()
                        );
                    }
                    let mut too_short = DeviceBuffer::zeroed(&model.stream, partials.len() - 1)?;
                    assert!(
                        model
                            .module
                            .gpt_mesh_gradient_partials(
                                &model.stream,
                                &partial_launch,
                                width,
                                0,
                                chunks,
                                &input_gpu,
                                &activation_gpu,
                                &gradient_gpu,
                                &mut too_short,
                            )
                            .is_err()
                    );
                }
            }
            Ok(())
        }

        /// Checks parameter-free full-width RMSNorm and per-head QK-Norm against f64 autodiff.
        pub fn verify_rmsnorm(device: usize) -> Result<()> {
            let model = GpuMeshGpt::new(device, GptConfig::default(), 7)?;
            let rows = 129;
            for (width, heads) in [
                (2, 1),
                (4, 2),
                (6, 1),
                (12, 2),
                (32, 1),
                (32, 4),
                (64, 1),
                (64, 8),
                (128, 1),
                (128, 2),
                (128, 4),
                (128, 8),
            ] {
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
                let launch = model.module.prepare_gpt_rmsnorm::<128>(config(rows))?;
                model.module.gpt_rmsnorm(
                    &model.stream,
                    &launch,
                    heads,
                    head_width,
                    &input_gpu,
                    &mut output_gpu,
                    &mut cache,
                )?;
                let backward = model
                    .module
                    .prepare_gpt_rmsnorm_backward::<128>(config(rows))?;
                model.module.gpt_rmsnorm_backward(
                    &model.stream,
                    &backward,
                    heads,
                    head_width,
                    &gradient_gpu,
                    &cache,
                    &mut derivative_gpu,
                )?;
                if head_width.is_multiple_of(32) {
                    with_array_elements(&model.stream, &mut output_gpu, |output| {
                        with_buffer_chunks(&model.stream, &mut cache, |saved| {
                            let launch = model
                                .module
                                .prepare_gpt_rmsnorm_parallel::<128>(config(output.len()))?;
                            model.module.gpt_rmsnorm_parallel(
                                &model.stream,
                                &launch,
                                4,
                                heads,
                                head_width / 32,
                                &input_gpu,
                                output,
                                saved,
                            )?;
                            for (capacity, count, head_warps) in [
                                (0, heads, head_width / 32),
                                (3, heads, head_width / 32),
                                (4, 0, head_width / 32),
                                (4, heads, 0),
                                (4, 5, 1),
                                (4, 1, usize::MAX),
                            ] {
                                assert!(
                                    model
                                        .module
                                        .gpt_rmsnorm_parallel(
                                            &model.stream,
                                            &launch,
                                            capacity,
                                            count,
                                            head_warps,
                                            &input_gpu,
                                            output,
                                            saved,
                                        )
                                        .is_err()
                                );
                            }
                            Ok(())
                        })
                    })?;
                    with_array_elements(&model.stream, &mut derivative_gpu, |output| {
                        let launch = model
                            .module
                            .prepare_gpt_rmsnorm_backward_parallel::<128>(config(output.len()))?;
                        model.module.gpt_rmsnorm_backward_parallel(
                            &model.stream,
                            &launch,
                            4,
                            heads,
                            head_width / 32,
                            &gradient_gpu,
                            &cache,
                            output,
                        )?;
                        for (capacity, count, head_warps) in
                            [(3, heads, 1), (4, 0, 1), (4, 1, 0), (4, 5, 1)]
                        {
                            assert!(
                                model
                                    .module
                                    .gpt_rmsnorm_backward_parallel(
                                        &model.stream,
                                        &launch,
                                        capacity,
                                        count,
                                        head_warps,
                                        &gradient_gpu,
                                        &cache,
                                        output,
                                    )
                                    .is_err()
                            );
                        }
                        Ok(())
                    })?;
                }
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
                    (MAX_WIDTH + 1, head_width),
                    (heads, 0),
                    (heads, 1),
                    (heads, MAX_WIDTH + 1),
                    (8, MAX_WIDTH / 8 + 1),
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
            for (width, heads) in [(4, 1), (4, 2), (12, 2), (64, 8), (128, 1), (128, 4)] {
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
                let mut inverse_gpu: DeviceBuffer<Features> =
                    DeviceBuffer::zeroed(&model.stream, rows)?;
                let launch = model
                    .module
                    .prepare_gpt_rope::<128, 128>(config(rows * MAX_WIDTH))?;
                for sequence in [1, 3, 128] {
                    for (source, target, inverse) in [
                        (&input_gpu, &mut output_gpu, 0),
                        (&gradient_gpu, &mut derivative_gpu, 1),
                    ] {
                        with_array_elements(&model.stream, target, |target| {
                            model
                                .module
                                .gpt_rope(
                                    &model.stream,
                                    &launch,
                                    heads,
                                    half_width,
                                    sequence,
                                    inverse,
                                    source,
                                    target,
                                )
                                .map_err(Into::into)
                        })?;
                    }
                    with_array_elements(&model.stream, &mut inverse_gpu, |target| {
                        model
                            .module
                            .gpt_rope(
                                &model.stream,
                                &launch,
                                heads,
                                half_width,
                                sequence,
                                1,
                                &output_gpu,
                                target,
                            )
                            .map_err(Into::into)
                    })?;
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
                    (MAX_WIDTH / 2 + 1, 3, 0),
                    (half_width, 0, 0),
                    (half_width, 129, 0),
                    (half_width, 3, 2),
                ] {
                    assert!(
                        with_array_elements(&model.stream, &mut output_gpu, |target| {
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
                                    target,
                                )
                                .map_err(Into::into)
                        })
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
            for (width, heads) in [(4, 2), (8, 2), (64, 8), (4, 4), (128, 4)] {
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
                let launch = model.module.prepare_gpt_exclusive::<128>(config(rows))?;
                model.module.gpt_exclusive(
                    &model.stream,
                    &launch,
                    width,
                    heads,
                    &input_gpu,
                    &value_gpu,
                    &mut output,
                )?;
                let launch = model
                    .module
                    .prepare_gpt_exclusive_backward::<128>(config(rows))?;
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
                            let norm = |x: &[f32]| {
                                x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt()
                            };
                            let scale = norm(&input[row][begin..begin + head_width])
                                * norm(&value[row][begin..begin + head_width]);
                            assert!(
                                dot.abs() < 3e-5 * (1.0 + scale),
                                "headwise orthogonality width={width} heads={heads}: {dot} / {scale}"
                            );
                        }
                    }
                }
            }
            Ok(())
        }
    }
}
