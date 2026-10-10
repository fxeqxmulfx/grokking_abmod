use std::sync::Arc;

use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, DeviceViewMut, LaunchConfig1D, LaunchContractError,
    LaunchDimension,
};
use cuda_device::{DisjointSlice, device, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;

use crate::{Result, mesh::validate_finite};

use super::{
    AdamWOptions, Features, GptConfig, IGNORE_TARGET, MAX_ROWS, NormStats, RMSNORM_EPSILON,
    ROPE_BASE, Vocabulary, XSA_EPSILON,
    config::{Layout, token_codes},
};

const EMBEDDING_GRADIENT_ROWS: usize = 256;

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

    #[device]
    pub fn causal_blocks(sequence: usize, tile: usize) -> usize {
        let bands = sequence / tile;
        tile * bands * (bands + 1) / 2 + sequence % tile * (bands + 1)
    }

    #[device]
    pub(super) fn causal_position(block: usize, tile: usize) -> (usize, usize) {
        if block < tile {
            return (block, 0);
        }
        let mut band = ((2.0 * block as f32 / tile as f32 + 0.25).sqrt() - 0.5) as usize;
        // Correct either rounding direction before deriving integer coordinates.
        while tile * band * (band + 1) / 2 > block {
            band -= 1;
        }
        while tile * (band + 1) * (band + 2) / 2 <= block {
            band += 1;
        }
        let local = block - tile * band * (band + 1) / 2;
        (band * tile + local / (band + 1), local % (band + 1) * tile)
    }

    #[device]
    pub(super) fn causal_offset(
        position: usize,
        group: usize,
        blocks: usize,
        tile: usize,
    ) -> usize {
        let band = position / tile;
        (group * blocks + tile * band * (band + 1) / 2 + position % tile * (band + 1)) * tile
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, _vocab >= 2,
        out.len() == tokens.len() * W, embedding.len() == _vocab))]
    pub fn gpt_embed<const W: usize>(
        width: usize,
        _vocab: usize,
        tokens: &[u32],
        embedding: &[Features<W>],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if let Some(output) = out.get_mut(index)
            && channel < width
        {
            *output = embedding[tokens[row] as usize][channel];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        heads >= 1, heads <= W, head_width >= 2, head_width <= W,
        heads * head_width <= W, backward <= 1,
        gradient.len() == input.len(), normalized.len() == input.len(),
        out.len() >= input.len() * heads))]
    pub fn gpt_rmsnorm_statistics<const W: usize>(
        heads: usize,
        head_width: usize,
        backward: usize,
        input: &[Features<W>],
        gradient: &[Features<W>],
        normalized: &[Features<W>],
        mut out: DisjointSlice<NormStats, thread::WarpIndex>,
    ) {
        let index = thread::index_1d();
        let slot = index.get() / 32;
        let row = slot / heads;
        let begin = slot % heads * head_width;
        let lane = index.get() % 32;
        if row >= input.len() {
            return;
        }
        let squares = attention_dot(begin, head_width, lane, &input[row], &input[row]);
        let dot = if backward == 1 {
            attention_dot(begin, head_width, lane, &gradient[row], &normalized[row])
        } else {
            0.0
        };
        if let Some(output) = warp_output(&mut out) {
            *output = [
                1.0 / (squares / head_width as f32 + RMSNORM_EPSILON).sqrt(),
                dot,
            ];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, offset + 2 * width <= state.len(),
        BACKWARD <= 1, BACKWARD * add_residual == 0,
        activated + add_residual <= 1, residual.len() == input.len(), activation.len() == input.len(),
        out.len() == input.len() * W))]
    pub fn gpt_mesh<const W: usize, const BACKWARD: usize>(
        width: usize,
        offset: usize,
        activated: usize,
        add_residual: usize,
        input: &[Features<W>],
        residual: &[Features<W>],
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
            if BACKWARD == 1 {
                let previous = (channel + width - 1) % width;
                *output = input[row][channel]
                    * slope(activation[row][channel], activated)
                    * state[offset + 2 * channel][0]
                    + input[row][previous]
                        * slope(activation[row][previous], activated)
                        * state[offset + 2 * previous + 1][0];
                return;
            }
            let p = offset + 2 * channel;
            let value = input[row][channel] * state[p][0]
                + input[row][(channel + 1) % width] * state[p + 1][0];
            let value = if activated == 1 && value < 0.0 {
                value.exp() - 1.0
            } else {
                value
            };
            *output = if add_residual == 1 {
                residual[row][channel] + value
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

    #[device]
    fn warp_output<'a, T>(
        output: &'a mut DisjointSlice<'_, T, thread::WarpIndex>,
    ) -> Option<&'a mut T> {
        if let Some(warp) = thread::warp_index() {
            output.get_mut(warp)
        } else {
            None
        }
    }

    #[device]
    fn attention_dot<const W: usize>(
        begin: usize,
        head_width: usize,
        lane: usize,
        left: &Features<W>,
        right: &Features<W>,
    ) -> f32 {
        let mut total = 0.0;
        if head_width <= 32 {
            if lane < head_width {
                total = left[begin + lane] * right[begin + lane];
            }
        } else {
            for local in (lane..head_width).step_by(32) {
                total += left[begin + local] * right[begin + local];
            }
        }
        warp::reduce_sum_f32(total)
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        heads >= 1, sequence >= 1, TILE >= 1, _tail < TILE,
        sequence == _bands * TILE + _tail, _sequences >= 1,
        2 * blocks == TILE * _bands * (_bands + 1) + 2 * _tail * (_bands + 1),
        scores.len() == _sequences * heads * blocks * TILE,
        out.len() == _sequences * sequence * heads))]
    pub fn gpt_attention_statistics<const TILE: usize>(
        heads: usize,
        sequence: usize,
        _bands: usize,
        _tail: usize,
        _sequences: usize,
        blocks: usize,
        scores: &[f32],
        mut out: DisjointSlice<NormStats, thread::WarpIndex>,
    ) {
        let index = thread::index_1d();
        let slot = index.get() / 32;
        let lane = index.get() % 32;
        if slot >= out.len() {
            return;
        }
        let row = slot / heads;
        let position = row % sequence;
        let offset = causal_offset(
            position,
            row / sequence * heads + slot % heads,
            blocks,
            TILE,
        );
        let mut maximum = f32::NEG_INFINITY;
        for previous in (lane..=position).step_by(32) {
            maximum = maximum.max(scores[offset + previous]);
        }
        let maximum = warp::reduce_max_f32(maximum);
        let mut sum = 0.0;
        for previous in (lane..=position).step_by(32) {
            sum += (scores[offset + previous] - maximum).exp();
        }
        let sum = warp::reduce_sum_f32(sum);
        if let Some(output) = warp_output(&mut out) {
            *output = [maximum, sum];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        heads >= 1, sequence >= 1, TILE >= 1, _tail < TILE,
        sequence == _bands * TILE + _tail, _sequences >= 1,
        2 * blocks == TILE * _bands * (_bands + 1) + 2 * _tail * (_bands + 1),
        out.len() == _sequences * heads * blocks * TILE,
        statistics.len() == _sequences * sequence * heads))]
    pub fn gpt_attention_probabilities<const TILE: usize>(
        heads: usize,
        sequence: usize,
        _bands: usize,
        _tail: usize,
        _sequences: usize,
        blocks: usize,
        statistics: &[NormStats],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let scalar = index.get();
        if let Some(output) = out.get_mut(index) {
            let item = scalar / TILE;
            let group = item / blocks;
            let (position, previous) = causal_position(item % blocks, TILE);
            if previous + scalar % TILE > position {
                return;
            }
            let row = group / heads * sequence + position;
            let [maximum, sum] = statistics[row * heads + group % heads];
            *output = (*output - maximum).exp() / sum;
        }
    }

    #[inline(always)]
    fn attention_sum<const W: usize, const TILE: usize, const COLUMN: bool>(
        start: usize,
        end: usize,
        first: usize,
        channel: usize,
        mut offset: usize,
        coefficients: &[f32],
        values: &[Features<W>],
    ) -> f32 {
        let mut sum = 0.0;
        let whole = end - (end - start) % 4;
        for row in (start..whole).step_by(4) {
            let mut products = [0.0; 4];
            for local in 0..4 {
                let index = if COLUMN {
                    offset + start
                } else {
                    offset + row + local
                };
                products[local] = coefficients[index] * values[first + row + local][channel];
                if COLUMN {
                    offset += ((row + local) / TILE + 1) * TILE;
                }
            }
            // Issue independent loads before preserving the original addition order.
            for local in 0..4 {
                sum += products[local];
            }
        }
        for row in whole..end {
            let index = if COLUMN { offset + start } else { offset + row };
            sum += coefficients[index] * values[first + row][channel];
            if COLUMN {
                offset += (row / TILE + 1) * TILE;
            }
        }
        sum
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, heads >= 1, heads <= width, sequence >= 1, TILE >= 1,
        _tail < TILE, sequence == _bands * TILE + _tail, _sequences >= 1,
        value.len() == _sequences * sequence,
        2 * blocks == TILE * _bands * (_bands + 1) + 2 * _tail * (_bands + 1),
        probabilities.len() == _sequences * heads * blocks * TILE, out.len() == value.len() * W))]
    pub fn gpt_attention_context<const W: usize, const TILE: usize, const COLUMN: bool>(
        width: usize,
        heads: usize,
        sequence: usize,
        _bands: usize,
        _tail: usize,
        _sequences: usize,
        blocks: usize,
        value: &[Features<W>],
        probabilities: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if let Some(output) = out.get_mut(index) {
            if channel >= width / heads * heads {
                return;
            }
            let position = row % sequence;
            let head = channel / (width / heads);
            let offset = causal_offset(position, row / sequence * heads + head, blocks, TILE);
            *output = attention_sum::<W, TILE, COLUMN>(
                if COLUMN { position } else { 0 },
                if COLUMN { sequence } else { position + 1 },
                row - position,
                channel,
                offset,
                probabilities,
                value,
            );
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, input.len() >= 1, out.len() == input.len() * W))]
    pub fn gpt_feature_transpose<const W: usize>(
        width: usize,
        input: &[Features<W>],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let channel = index.get() / input.len();
        let row = index.get() % input.len();
        if let Some(output) = out.get_mut(index) {
            *output = if channel < width {
                input[row][channel]
            } else {
                0.0
            };
        }
    }

    #[inline(always)]
    fn serial_score_dot<const W: usize, const H: usize, const FULL: bool, const PACK: usize>(
        begin: usize,
        head_width: usize,
        key: usize,
        rows: usize,
        count: usize,
        query: &Features<W>,
        transposed: &[f32],
    ) -> [f32; PACK] {
        let length = if FULL { W / H } else { head_width };
        let query = &query[begin..begin + length];
        let mut totals = [0.0; PACK];
        for local in 0..length {
            let channel = begin + local;
            for slot in 0..PACK {
                if slot < count {
                    totals[slot] += query[local] * transposed[channel * rows + key + slot];
                }
            }
        }
        totals
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, heads >= 1, heads <= H, heads <= width,
        sequence >= 1, sequence <= C, BACKWARD <= 1,
        PACK >= 1, PACK <= 2,
        TILE >= 1, _tail < TILE, sequence == _bands * TILE + _tail,
        _sequences >= 1, gradient.len() == _sequences * sequence,
        2 * blocks == TILE * _bands * (_bands + 1) + 2 * _tail * (_bands + 1),
        value.len() == gradient.len() * W, statistics.len() == gradient.len() * heads,
        out.len() * PACK == _sequences * heads * blocks * TILE))]
    pub fn gpt_attention_scores_serial<
        const W: usize,
        const C: usize,
        const H: usize,
        const TILE: usize,
        const BACKWARD: usize,
        const PACK: usize,
    >(
        width: usize,
        heads: usize,
        sequence: usize,
        _bands: usize,
        _tail: usize,
        _sequences: usize,
        blocks: usize,
        gradient: &[Features<W>],
        value: &[f32],
        statistics: &[NormStats],
        mut out: DisjointSlice<[f32; PACK]>,
    ) {
        const {
            assert!(PACK == 1 || PACK == 2 && TILE.is_multiple_of(64));
        }
        let index = thread::index_1d();
        let packet = index.get();
        let scalar = packet * PACK;
        if let Some(output) = out.get_mut(index) {
            let mut coordinates = [0; 5];
            if !TILE.is_multiple_of(32 * PACK) || packet.is_multiple_of(32) {
                let item = scalar / TILE;
                let group = item / blocks;
                let (position, previous) = causal_position(item % blocks, TILE);
                coordinates = [
                    group / heads * sequence,
                    group % heads,
                    position,
                    previous,
                    width / heads,
                ];
            }
            if TILE.is_multiple_of(32 * PACK) {
                // The checked output length keeps every participating warp complete.
                for local in 0..5 {
                    coordinates[local] = warp::shuffle_u64(coordinates[local] as u64, 0) as usize;
                }
            }
            let [first, head, position, previous, head_width] = coordinates;
            let previous = previous + scalar % TILE;
            if previous > position {
                return;
            }
            let row = first + position;
            let count = PACK.min(position - previous + 1);
            let dot = if width == W && heads == H {
                serial_score_dot::<W, H, true, PACK>(
                    head * head_width,
                    head_width,
                    first + previous,
                    gradient.len(),
                    count,
                    &gradient[row],
                    value,
                )
            } else {
                serial_score_dot::<W, H, false, PACK>(
                    head * head_width,
                    head_width,
                    first + previous,
                    gradient.len(),
                    count,
                    &gradient[row],
                    value,
                )
            };
            for slot in 0..count {
                output[slot] = if BACKWARD == 1 {
                    output[slot] * (dot[slot] - statistics[row * heads + head][1])
                } else {
                    dot[slot] * (1.0 / (head_width as f32).sqrt())
                };
            }
        }
    }

    #[device]
    fn exclusive_terms<const W: usize, const BACKWARD: bool>(
        width: usize,
        heads: usize,
        channel: usize,
        input: &Features<W>,
        value: &Features<W>,
        gradient: &Features<W>,
    ) -> (f32, f32, f32, f32) {
        let head_width = width / heads;
        let begin = channel / head_width * head_width;
        // These conditions align row/head boundaries and active lanes with
        // complete warps. Other shapes perform the same serial head reduction.
        let parallel = W.is_multiple_of(32) && head_width.is_multiple_of(32);
        let (first, stride) = if parallel { (channel % 32, 32) } else { (0, 1) };
        let mut squares = 0.0;
        for local in (first..head_width).step_by(stride) {
            let v = value[begin + local];
            squares += v * v;
        }
        if parallel {
            squares = warp::reduce_sum_f32(squares);
        }
        let norm = squares.sqrt();
        let inverse = 1.0 / norm.max(XSA_EPSILON);
        let (mut projection, mut gradient_projection) = (0.0, 0.0);
        for local in (first..head_width).step_by(stride) {
            let c = begin + local;
            let normalized = value[c] * inverse;
            projection += input[c] * normalized;
            if BACKWARD {
                gradient_projection += gradient[c] * normalized;
            }
        }
        if parallel {
            projection = warp::reduce_sum_f32(projection);
            if BACKWARD {
                gradient_projection = warp::reduce_sum_f32(gradient_projection);
            }
        }
        (norm, inverse, projection, gradient_projection)
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, heads >= 1, heads <= width,
        value.len() == input.len(), out.len() == input.len() * W))]
    pub fn gpt_exclusive<const W: usize>(
        width: usize,
        heads: usize,
        input: &[Features<W>],
        value: &[Features<W>],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if row >= input.len() || channel >= width / heads * heads {
            return;
        }
        let (_, inverse, projection, _) = exclusive_terms::<W, false>(
            width,
            heads,
            channel,
            &input[row],
            &value[row],
            &input[row],
        );
        if let Some(output) = out.get_mut(index) {
            *output = input[row][channel] - projection * (value[row][channel] * inverse);
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, heads >= 1, heads <= width,
        value.len() == input.len(), gradient.len() == input.len(),
        dy.len() == input.len() * W, dv.len() == dy.len()))]
    pub fn gpt_exclusive_backward<const W: usize>(
        width: usize,
        heads: usize,
        input: &[Features<W>],
        value: &[Features<W>],
        gradient: &[Features<W>],
        mut dy: DisjointSlice<f32>,
        mut dv: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if row >= input.len() || channel >= width / heads * heads {
            return;
        }
        let (norm, inverse, projection, gradient_projection) = exclusive_terms::<W, true>(
            width,
            heads,
            channel,
            &input[row],
            &value[row],
            &gradient[row],
        );
        if let Some(y_gradient) = dy.get_mut(index)
            && let Some(v_gradient) = dv.get_mut(thread::index_1d())
        {
            let normalized = value[row][channel] * inverse;
            *y_gradient = gradient[row][channel] - gradient_projection * normalized;
            // Differentiate both uses of normalized V and its norm above the floor.
            let mut derivative =
                -gradient_projection * input[row][channel] - projection * gradient[row][channel];
            if norm >= XSA_EPSILON {
                derivative += 2.0 * projection * gradient_projection * normalized;
            }
            *v_gradient = derivative * inverse;
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

    #[inline(always)]
    pub(super) fn fused_norm<const W: usize>(head_width: usize) -> bool {
        W.is_multiple_of(32) && head_width.is_multiple_of(32) && head_width <= 128
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        heads >= 1, heads <= W, head_width >= 2, head_width <= W,
        heads * head_width <= W, backward <= 1,
        source.len() == input.len(), normalized.len() == input.len(),
        statistics.len() >= input.len() * heads, out.len() == input.len() * W))]
    pub fn gpt_rmsnorm_apply<const W: usize>(
        heads: usize,
        head_width: usize,
        backward: usize,
        source: &[Features<W>],
        input: &[Features<W>],
        normalized: &[Features<W>],
        statistics: &[NormStats],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if let Some(output) = out.get_mut(index) {
            if channel >= heads * head_width {
                return;
            }
            let [inverse, dot] = if fused_norm::<W>(head_width) {
                // Row and active-head boundaries preserve complete participating warps.
                let begin = channel / head_width * head_width;
                let lane = channel % 32;
                let squares = attention_dot(begin, head_width, lane, &source[row], &source[row]);
                let dot = if backward == 1 {
                    attention_dot(begin, head_width, lane, &input[row], &normalized[row])
                } else {
                    0.0
                };
                [
                    1.0 / (squares / head_width as f32 + RMSNORM_EPSILON).sqrt(),
                    dot,
                ]
            } else {
                statistics[row * heads + channel / head_width]
            };
            let value = input[row][channel];
            *output = inverse
                * if backward == 1 {
                    value - normalized[row][channel] * dot / head_width as f32
                } else {
                    value
                };
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, activated <= 1, ROWS >= 1, ROWS <= 131072,
        chunks >= 1, chunks <= 4096, offset + chunks * width <= out.len(),
        input.len() <= chunks * ROWS, input.len() + ROWS > chunks * ROWS,
        input.len() == activation.len(), gradient.len() == activation.len()))]
    pub fn gpt_mesh_gradient<const W: usize, const ROWS: usize>(
        width: usize,
        offset: usize,
        activated: usize,
        chunks: usize,
        input: &[Features<W>],
        activation: &[Features<W>],
        gradient: &[Features<W>],
        mut out: DisjointSlice<[f32; 2]>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if item < offset || item >= offset + chunks * width {
            return;
        }
        if let Some(output) = out.get_mut(index) {
            let local = item - offset;
            let channel = local % width;
            let parent = (channel + 1) % width;
            let start = local / width * ROWS;
            let end = (start + ROWS).min(gradient.len());
            let mut totals = [0.0; 2];
            for row in start..end {
                let left = input[row][channel];
                let right = input[row][parent];
                let delta = gradient[row][channel] * slope(activation[row][channel], activated);
                totals[0] += delta * left;
                totals[1] += delta * right;
            }
            *output = totals;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, offset + 2 * width <= out.len(),
        chunks >= 1, chunks <= 4096, partials.len() == chunks * 2 * width))]
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
        if let Some(output) = warp_output(&mut out) {
            *output = total;
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, heads >= 1, heads <= H, heads <= width,
        sequence >= 1, sequence <= C,
        BACKWARD <= 1, value.len() == gradient.len(),
        TILE >= 1, _tail < TILE, sequence == _bands * TILE + _tail,
        _sequences >= 1, gradient.len() == _sequences * sequence,
        2 * blocks == TILE * _bands * (_bands + 1) + 2 * _tail * (_bands + 1),
        attention.len() == gradient.len(), out.len() == _sequences * heads * blocks))]
    pub fn gpt_attention_scores<
        const W: usize,
        const C: usize,
        const H: usize,
        const TILE: usize,
        const BACKWARD: usize,
    >(
        width: usize,
        heads: usize,
        sequence: usize,
        _bands: usize,
        _tail: usize,
        _sequences: usize,
        blocks: usize,
        gradient: &[Features<W>],
        value: &[Features<W>],
        attention: &[Features<W>],
        mut out: DisjointSlice<[f32; TILE], thread::WarpIndex>,
    ) {
        let index = thread::index_1d();
        let item = index.get() / 32;
        let lane = index.get() % 32;
        if item >= out.len() {
            return;
        }
        let (group, block) = if sequence == C {
            let full_blocks = causal_blocks(C, TILE);
            (item / full_blocks, item % full_blocks)
        } else {
            (item / blocks, item % blocks)
        };
        let (position, previous) = causal_position(block, TILE);
        let (row, head) = if heads == H {
            (group / H * sequence + position, group % H)
        } else {
            (group / heads * sequence + position, group % heads)
        };
        let head_width = width / heads;
        // sum(P * dot(dY, V)) = dot(dY, Y), with the raw attention Y.
        let mean = if BACKWARD == 1 {
            attention_dot(
                head * head_width,
                head_width,
                lane,
                &gradient[row],
                &attention[row],
            )
        } else {
            0.0
        };
        let scale = if BACKWARD == 0 {
            1.0 / (head_width as f32).sqrt()
        } else {
            0.0
        };
        let mut output = warp_output(&mut out);
        for local in 0..TILE.min(row % sequence - previous + 1) {
            let dot = attention_dot(
                head * head_width,
                head_width,
                lane,
                &gradient[row],
                &value[row - row % sequence + previous + local],
            );
            if let Some(output) = &mut output {
                output[local] = if BACKWARD == 1 {
                    output[local] * (dot - mean)
                } else {
                    dot * scale
                };
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        width >= 2, width <= W, heads >= 1, heads <= H, heads <= width, sequence >= 1, sequence <= C,
        TILE >= 1, _tail < TILE, sequence == _bands * TILE + _tail,
        _sequences >= 1, query.len() == _sequences * sequence,
        2 * blocks == TILE * _bands * (_bands + 1) + 2 * _tail * (_bands + 1),
        key.len() == query.len(),
        scores.len() == _sequences * heads * blocks * TILE,
        dq.len() == query.len() * W, dk.len() == dq.len()))]
    pub fn gpt_attention_backward<
        const W: usize,
        const C: usize,
        const H: usize,
        const TILE: usize,
    >(
        width: usize,
        heads: usize,
        sequence: usize,
        _bands: usize,
        _tail: usize,
        _sequences: usize,
        blocks: usize,
        query: &[Features<W>],
        key: &[Features<W>],
        scores: &[f32],
        mut dq: DisjointSlice<f32>,
        mut dk: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / W;
        let channel = index.get() % W;
        if let Some(q_gradient) = dq.get_mut(index)
            && let Some(k_gradient) = dk.get_mut(thread::index_1d())
        {
            let position = row % sequence;
            let first = row - position;
            let head_width = width / heads;
            let scale = 1.0 / (head_width as f32).sqrt();
            if channel >= width / heads * heads {
                return;
            }
            let head = channel / head_width;
            let q_offset = causal_offset(position, row / sequence * heads + head, blocks, TILE);
            let q_sum = attention_sum::<W, TILE, false>(
                0,
                position + 1,
                first,
                channel,
                q_offset,
                scores,
                key,
            );
            let k_sum = attention_sum::<W, TILE, true>(
                position, sequence, first, channel, q_offset, scores, query,
            );
            *q_gradient = q_sum * scale;
            *k_gradient = k_sum * scale;
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
            let tokens = &tokens[start..end];
            let input_gradient = &input_gradient[start..end];
            let head_gradient = &head_gradient[start..end];
            let final_features = &final_features[start..end];
            let mut total = 0.0;
            for row in 0..tokens.len() {
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
    Embed,
    RmsNorm {
        heads: usize,
        input: usize,
    },
    Mesh {
        group: usize,
        input: usize,
        layer: usize,
        activated: usize,
        residual: Option<usize>,
    },
    Rotary {
        input: usize,
    },
    Attention {
        query: usize,
        key: usize,
        value: usize,
        cache: usize,
    },
    Exclusive {
        input: usize,
        value: usize,
    },
}

impl Op {
    fn inputs(self) -> [Option<usize>; 3] {
        match self {
            Self::Embed => [None; 3],
            Self::RmsNorm { input, .. } | Self::Rotary { input } => [Some(input), None, None],
            Self::Mesh {
                input, residual, ..
            } => [Some(input), residual, None],
            Self::Exclusive { input, value } => [Some(input), Some(value), None],
            Self::Attention {
                query, key, value, ..
            } => [Some(query), Some(key), Some(value)],
        }
    }
}

struct Plan {
    ops: Vec<Op>,
    gradient_slots: Vec<usize>,
    gradient_buffers: usize,
    attentions: usize,
    final_features: usize,
}

impl Plan {
    fn new(config: GptConfig, layout: &Layout) -> Self {
        let mut plan = Self {
            ops: vec![Op::Embed],
            gradient_slots: Vec::new(),
            gradient_buffers: 0,
            attentions: 0,
            final_features: 0,
        };
        let mut x = 0;
        for block in &layout.blocks {
            let normalized = plan.rmsnorm(x, 1);
            let query = plan.mesh(normalized, block.query, config.attention_mesh_depth, None);
            let key = plan.mesh(normalized, block.key, config.attention_mesh_depth, None);
            let value = plan.mesh(normalized, block.value, config.attention_mesh_depth, None);
            let query = plan.rmsnorm(query, config.heads);
            let key = plan.rmsnorm(key, config.heads);
            let query = plan.rope(query);
            let key = plan.rope(key);
            let cache = plan.attentions;
            plan.attentions += 1;
            let out = plan.push(Op::Attention {
                query,
                key,
                value,
                cache,
            });
            let exclusive = plan.push(Op::Exclusive { input: out, value });
            x = plan.mesh(
                exclusive,
                block.projection,
                config.attention_mesh_depth,
                Some(x),
            );
            let normalized = plan.rmsnorm(x, 1);
            x = plan.mesh(normalized, block.ffn, config.ffn_mesh_depth, Some(x));
        }
        plan.final_features = plan.rmsnorm(x, 1);
        // A node's gradient is live from its first contribution until its op runs.
        // Keep the output live while allocating all inputs, then reuse its slot.
        plan.gradient_slots = vec![usize::MAX; plan.ops.len()];
        plan.gradient_slots[plan.final_features] = 0;
        plan.gradient_buffers = 1;
        let mut available = Vec::new();
        for (out, op) in plan.ops.iter().copied().enumerate().rev() {
            let inputs = op.inputs();
            for input in inputs.into_iter().flatten() {
                if plan.gradient_slots[input] == usize::MAX {
                    plan.gradient_slots[input] = available.pop().unwrap_or_else(|| {
                        let slot = plan.gradient_buffers;
                        plan.gradient_buffers += 1;
                        slot
                    });
                }
            }
            available.push(plan.gradient_slots[out]);
        }
        plan
    }
    fn push(&mut self, op: Op) -> usize {
        let out = self.ops.len();
        self.ops.push(op);
        out
    }
    fn rmsnorm(&mut self, input: usize, heads: usize) -> usize {
        self.push(Op::RmsNorm { heads, input })
    }
    fn mesh(
        &mut self,
        mut input: usize,
        group: usize,
        depth: usize,
        residual: Option<usize>,
    ) -> usize {
        for layer in 0..depth {
            input = self.push(Op::Mesh {
                group,
                input,
                layer,
                activated: usize::from(layer + 1 != depth),
                residual: if layer + 1 == depth { residual } else { None },
            });
        }
        input
    }
    fn rope(&mut self, input: usize) -> usize {
        self.push(Op::Rotary { input })
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn causal_tiles_cover_every_past_token_once() {
        for tile in [1, 4, 32, 64] {
            for sequence in [1, 3, 4, 7, 32, 63, 64, 65, 127, 128, 129, 1024, 4095, 4096] {
                let blocks = kernels::causal_blocks(sequence, tile);
                let mut counts = vec![0; sequence];
                for block in 0..blocks {
                    let (position, previous) = kernels::causal_position(block, tile);
                    assert!(position < sequence && previous <= position);
                    assert!(previous.is_multiple_of(tile));
                    assert_eq!(
                        kernels::causal_offset(position, 0, blocks, tile) + previous,
                        block * tile
                    );
                    counts[position] += 1;
                }
                for (position, count) in counts.into_iter().enumerate() {
                    assert_eq!(count, position / tile + 1);
                }
                assert_eq!(
                    kernels::causal_offset(0, 2, blocks, tile),
                    2 * blocks * tile
                );
            }
        }
    }

    #[test]
    fn launch_geometry_handles_large_buffers_and_rejects_overflow() {
        let blocks = u32::MAX as usize;
        assert_eq!(
            config::<32>(blocks * 4).unwrap(),
            LaunchConfig1D::new(u32::MAX, 128, 0)
        );
        assert_eq!(config::<32>(5).unwrap(), LaunchConfig1D::new(2, 128, 0));
        assert!(config::<32>(blocks * 4 + 1).is_err());
        assert!(config::<32>(usize::MAX).is_err());
        assert_eq!(
            config::<1>(u32::MAX as usize + 1).unwrap(),
            LaunchConfig1D::new(33_554_432, 128, 0)
        );
        assert_eq!(
            config::<1>(blocks * 128).unwrap(),
            LaunchConfig1D::new(u32::MAX, 128, 0)
        );
        assert!(config::<1>(blocks * 128 + 1).is_err());
        assert!(config::<1>(usize::MAX).is_err());
    }

    #[test]
    fn shared_gradient_slots_never_overlap_live_nodes() {
        for (layers, attention_mesh_depth, ffn_mesh_depth) in
            [(1, 1, 1), (2, 4, 6), (20, 4, 6), (64, 1, 2), (1, 256, 256)]
        {
            let c = GptConfig {
                layers,
                attention_mesh_depth,
                ffn_mesh_depth,
                ..GptConfig::default()
            };
            let plan = Plan::new(c, &Layout::new(c).unwrap());
            let mut live = HashSet::from([plan.final_features]);
            let mut peak = 1;
            for (out, op) in plan.ops.iter().copied().enumerate().rev() {
                let inputs = op.inputs();
                assert!(live.contains(&out));
                live.extend(inputs.into_iter().flatten());
                let slots: HashSet<_> =
                    live.iter().map(|&node| plan.gradient_slots[node]).collect();
                assert_eq!(
                    live.len(),
                    slots.len(),
                    "live gradients must own distinct buffers"
                );
                peak = peak.max(live.len());
                live.remove(&out);
            }
            assert!(live.is_empty());
            assert_eq!(peak, plan.gradient_buffers);
            assert!(plan.gradient_buffers < plan.ops.len());
        }
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
    embedding_gradient_partials: DeviceBuffer<f32>,
    mesh_gradient_partials: DeviceBuffer<f32>,
    features: Vec<DeviceBuffer<Features<W>>>,
    adjoints: Vec<DeviceBuffer<Features<W>>>,
    norm_statistics: DeviceBuffer<NormStats>,
    probabilities: Vec<DeviceBuffer<f32>>,
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
        let cells = rows / sequence
            * config.heads
            * kernels::causal_blocks(sequence, score_tile(C))
            * score_tile(C);
        Ok(Self {
            rows,
            sequence,
            supervised: rows,
            tokens: DeviceBuffer::zeroed(stream, rows)?,
            targets: DeviceBuffer::zeroed(stream, rows)?,
            embeddings: buffers(stream, config.attention_mesh_depth, config.vocab_size)?,
            head_embedding: DeviceBuffer::zeroed(stream, config.width)?,
            embedding_adjoints: buffers(stream, config.attention_mesh_depth, config.vocab_size)?,
            embedding_gradient_partials: DeviceBuffer::zeroed(
                stream,
                rows.div_ceil(EMBEDDING_GRADIENT_ROWS) * config.vocab_size * config.width,
            )?,
            mesh_gradient_partials: DeviceBuffer::zeroed(
                stream,
                rows.div_ceil(32) * 2 * config.width,
            )?,
            features: buffers(stream, plan.ops.len(), rows)?,
            adjoints: buffers(stream, plan.gradient_buffers, rows)?,
            norm_statistics: DeviceBuffer::zeroed(stream, rows * config.heads)?,
            probabilities: buffers(stream, plan.attentions, cells)?,
            scratch: buffers(stream, 4, rows)?,
            initialized: vec![false; plan.ops.len()],
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

    /// Bytes owned by device buffers, excluding CUDA context and module overhead.
    pub fn allocated_device_bytes(&self) -> usize {
        let mut bytes = self.codes.num_bytes()
            + self
                .parameters
                .iter()
                .map(|p| p.state.num_bytes() + p.gradient.num_bytes())
                .sum::<usize>();
        if let Some(w) = &self.workspace {
            bytes += [
                &w.embeddings,
                &w.embedding_adjoints,
                &w.features,
                &w.adjoints,
                &w.scratch,
            ]
            .into_iter()
            .flatten()
            .map(DeviceBuffer::num_bytes)
            .sum::<usize>()
                + w.norm_statistics.num_bytes()
                + w.probabilities
                    .iter()
                    .map(DeviceBuffer::num_bytes)
                    .sum::<usize>()
                + w.tokens.num_bytes()
                + w.targets.num_bytes()
                + w.head_embedding.num_bytes()
                + w.embedding_gradient_partials.num_bytes()
                + w.mesh_gradient_partials.num_bytes()
                + w.logits.num_bytes()
                + w.head_gradient.num_bytes()
                + w.loss.num_bytes();
        }
        bytes
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
                .prepare_gpt_adam::<W>(config::<1>(group.state.len())?)?;
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
            after[0].with_chunks(|output| {
                let launch = self
                    .module
                    .prepare_gpt_mesh::<W, 0>(config::<1>(output.len())?)?;
                self.module.gpt_mesh(
                    &self.stream,
                    &launch,
                    c.width,
                    layer * 2 * c.width,
                    usize::from(layer + 1 != c.attention_mesh_depth),
                    0,
                    input,
                    input,
                    input,
                    &self.parameters[embedding].state,
                    output,
                )
            })??;
        }
        Ok(())
    }

    fn forward(&self, workspace: &mut Workspace<W, V, C, H>) -> Result<()> {
        let c = self.config;
        self.forward_embeddings(workspace)?;
        let embedding_output = c.attention_mesh_depth - 1;
        // Store tokens contiguously for each channel so neighboring head threads
        // read adjacent vocabulary values. The tied vectors are still generated once.
        workspace.head_embedding.with_chunks(|output| {
            let launch = self
                .module
                .prepare_gpt_embedding_transpose::<W, V>(config::<1>(output.len())?)?;
            self.module.gpt_embedding_transpose(
                &self.stream,
                &launch,
                c.width,
                c.vocab_size,
                &workspace.embeddings[embedding_output],
                output,
            )
        })??;
        for (out, op) in self.plan.ops.iter().enumerate() {
            match *op {
                Op::Embed => {
                    workspace.features[out].with_chunks(|output| {
                        let launch = self
                            .module
                            .prepare_gpt_embed::<W>(config::<1>(output.len())?)?;
                        self.module.gpt_embed(
                            &self.stream,
                            &launch,
                            c.width,
                            c.vocab_size,
                            &workspace.tokens,
                            &workspace.embeddings[embedding_output],
                            output,
                        )
                    })??;
                }
                Op::RmsNorm { heads, input } => {
                    let (before, after) = workspace.features.split_at_mut(out);
                    self.rmsnorm(
                        heads,
                        c.width / heads,
                        0,
                        [&before[input]; 3],
                        &mut workspace.norm_statistics,
                        &mut after[0],
                    )?;
                }
                Op::Mesh {
                    group,
                    input,
                    layer,
                    activated,
                    residual,
                } => {
                    let (before, after) = workspace.features.split_at_mut(out);
                    after[0].with_chunks(|output| {
                        let launch = self
                            .module
                            .prepare_gpt_mesh::<W, 0>(config::<1>(output.len())?)?;
                        self.module.gpt_mesh(
                            &self.stream,
                            &launch,
                            c.width,
                            layer * 2 * c.width,
                            activated,
                            usize::from(residual.is_some()),
                            &before[input],
                            &before[residual.unwrap_or(input)],
                            &before[input],
                            &self.parameters[group].state,
                            output,
                        )
                    })??;
                }
                Op::Rotary { input } => {
                    let (before, after) = workspace.features.split_at_mut(out);
                    after[0].with_chunks(|output| {
                        let launch = self
                            .module
                            .prepare_gpt_rope::<W, C>(config::<1>(output.len())?)?;
                        self.module.gpt_rope(
                            &self.stream,
                            &launch,
                            c.heads,
                            c.width / c.heads / 2,
                            workspace.sequence,
                            0,
                            &before[input],
                            output,
                        )
                    })??;
                }
                Op::Attention {
                    query,
                    key,
                    value,
                    cache,
                } => {
                    self.attention::<0>(workspace, 0, [query, key, value, out], cache)?;
                }
                Op::Exclusive { input, value } => {
                    let (before, after) = workspace.features.split_at_mut(out);
                    after[0].with_chunks(|output| {
                        let launch = self
                            .module
                            .prepare_gpt_exclusive::<W>(config::<1>(output.len())?)?;
                        self.module.gpt_exclusive(
                            &self.stream,
                            &launch,
                            c.width,
                            c.heads,
                            &before[input],
                            &before[value],
                            output,
                        )
                    })??;
                }
            }
        }
        workspace.logits.with_chunks(|output| {
            let launch = self
                .module
                .prepare_gpt_head::<W, V>(config::<1>(output.len())?)?;
            self.module.gpt_head(
                &self.stream,
                &launch,
                c.width,
                c.vocab_size,
                &workspace.features[self.plan.final_features],
                &workspace.head_embedding,
                output,
            )
        })??;
        Ok(())
    }

    fn loss(&self, workspace: &mut Workspace<W, V, C, H>) -> Result<f64> {
        let launch = self
            .module
            .prepare_gpt_cross_entropy::<V>(config::<1>(workspace.rows)?)?;
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
        workspace.initialized.fill(false);
        workspace.scratch[0].with_chunks(|output| {
            let launch = self
                .module
                .prepare_gpt_head_backward::<W, V>(config::<1>(output.len())?)?;
            self.module.gpt_head_backward(
                &self.stream,
                &launch,
                c.width,
                c.vocab_size,
                &workspace.head_gradient,
                &workspace.embeddings[c.attention_mesh_depth - 1],
                output,
            )
        })??;
        self.accumulate_scratch(workspace, self.plan.final_features, 0)?;

        for out in (0..self.plan.ops.len()).rev() {
            let op = self.plan.ops[out];
            let gradient_slot = self.plan.gradient_slots[out];
            match op {
                Op::Embed => {}
                Op::RmsNorm { heads, input } => {
                    self.rmsnorm(
                        heads,
                        c.width / heads,
                        1,
                        [
                            &workspace.features[input],
                            &workspace.adjoints[gradient_slot],
                            &workspace.features[out],
                        ],
                        &mut workspace.norm_statistics,
                        &mut workspace.scratch[0],
                    )?;
                    self.accumulate_scratch(workspace, input, 0)?;
                }
                Op::Mesh {
                    group,
                    input,
                    layer,
                    activated,
                    residual,
                } => {
                    let parameters = &mut self.parameters[group];
                    if workspace.rows > 128 {
                        let chunks = workspace.rows.div_ceil(32);
                        mesh_gradient::<W, 32>(
                            &self.module,
                            &self.stream,
                            [c.width, 0, chunks, activated],
                            [
                                &workspace.features[input],
                                &workspace.features[out],
                                &workspace.adjoints[gradient_slot],
                            ],
                            &mut workspace.mesh_gradient_partials,
                        )?;
                        let launch =
                            self.module
                                .prepare_gpt_mesh_gradient_reduce::<W>(config::<1>(
                                    parameters.gradient.len() * 32,
                                )?)?;
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
                        mesh_gradient::<W, MAX_ROWS>(
                            &self.module,
                            &self.stream,
                            [c.width, layer * c.width, 1, activated],
                            [
                                &workspace.features[input],
                                &workspace.features[out],
                                &workspace.adjoints[gradient_slot],
                            ],
                            &mut parameters.gradient,
                        )?;
                    }
                    workspace.scratch[0].with_chunks(|output| {
                        let launch = self
                            .module
                            .prepare_gpt_mesh::<W, 1>(config::<1>(output.len())?)?;
                        self.module.gpt_mesh(
                            &self.stream,
                            &launch,
                            c.width,
                            layer * 2 * c.width,
                            activated,
                            0,
                            &workspace.adjoints[gradient_slot],
                            &workspace.features[out],
                            &workspace.features[out],
                            &parameters.state,
                            output,
                        )
                    })??;
                    self.accumulate_scratch(workspace, input, 0)?;
                    if let Some(residual) = residual {
                        // Linear projection backward is finished; transfer its gradient ownership.
                        self.consume_node(workspace, residual, out)?;
                    }
                }
                Op::Rotary { input } => {
                    workspace.scratch[0].with_chunks(|output| {
                        let launch = self
                            .module
                            .prepare_gpt_rope::<W, C>(config::<1>(output.len())?)?;
                        self.module.gpt_rope(
                            &self.stream,
                            &launch,
                            c.heads,
                            c.width / c.heads / 2,
                            workspace.sequence,
                            1,
                            &workspace.adjoints[gradient_slot],
                            output,
                        )
                    })??;
                    self.accumulate_scratch(workspace, input, 0)?;
                }
                Op::Attention {
                    query,
                    key,
                    value,
                    cache,
                } => {
                    self.attention::<1>(workspace, gradient_slot, [query, key, value, out], cache)?;
                    self.accumulate_scratch(workspace, query, 0)?;
                    self.accumulate_scratch(workspace, key, 1)?;
                    self.accumulate_scratch(workspace, value, 2)?;
                }
                Op::Exclusive { input, value } => {
                    let (dy, rest) = workspace.scratch.split_at_mut(1);
                    dy[0].with_chunks(|dy| {
                        rest[0]
                            .with_chunks(|dv| {
                                let launch = self
                                    .module
                                    .prepare_gpt_exclusive_backward::<W>(config::<1>(dy.len())?)?;
                                self.module.gpt_exclusive_backward(
                                    &self.stream,
                                    &launch,
                                    c.width,
                                    c.heads,
                                    &workspace.features[input],
                                    &workspace.features[value],
                                    &workspace.adjoints[gradient_slot],
                                    dy,
                                    dv,
                                )
                            })?
                            .map_err(Box::<dyn std::error::Error>::from)
                    })??;
                    self.accumulate_scratch(workspace, input, 0)?;
                    // V receives this local contribution and the subsequent
                    // attention contribution gathered from all causal queries.
                    self.accumulate_scratch(workspace, value, 1)?;
                }
            }
        }
        let embedding_output = c.attention_mesh_depth - 1;
        let chunks = workspace.rows.div_ceil(EMBEDDING_GRADIENT_ROWS);
        let launch = self
            .module
            .prepare_gpt_embedding_gradient_partials::<W, V, EMBEDDING_GRADIENT_ROWS>(
                config::<1>(workspace.embedding_gradient_partials.len())?,
            )?;
        self.module.gpt_embedding_gradient_partials(
            &self.stream,
            &launch,
            c.width,
            c.vocab_size,
            chunks,
            &workspace.tokens,
            &workspace.adjoints[self.plan.gradient_slots[0]],
            &workspace.head_gradient,
            &workspace.features[self.plan.final_features],
            &mut workspace.embedding_gradient_partials,
        )?;
        workspace.embedding_adjoints[embedding_output].with_chunks(|output| {
            let launch = self
                .module
                .prepare_gpt_embedding_gradient_reduce::<W>(config::<1>(output.len())?)?;
            self.module.gpt_embedding_gradient_reduce(
                &self.stream,
                &launch,
                c.width,
                c.vocab_size,
                chunks,
                &workspace.embedding_gradient_partials,
                output,
            )
        })??;
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
            mesh_gradient::<W, MAX_ROWS>(
                &self.module,
                &self.stream,
                [c.width, layer * c.width, 1, activated],
                [
                    input,
                    &workspace.embeddings[layer],
                    &workspace.embedding_adjoints[layer],
                ],
                &mut group.gradient,
            )?;
            if layer > 0 {
                let (before, after) = workspace.embedding_adjoints.split_at_mut(layer);
                before[layer - 1].with_chunks(|output| {
                    let launch = self
                        .module
                        .prepare_gpt_mesh::<W, 1>(config::<1>(output.len())?)?;
                    self.module.gpt_mesh(
                        &self.stream,
                        &launch,
                        c.width,
                        layer * 2 * c.width,
                        activated,
                        0,
                        &after[0],
                        &workspace.embeddings[layer],
                        &workspace.embeddings[layer],
                        &group.state,
                        output,
                    )
                })??;
            }
        }
        Ok(())
    }

    fn rmsnorm(
        &self,
        heads: usize,
        head_width: usize,
        backward: usize,
        inputs: [&DeviceBuffer<Features<W>>; 3],
        statistics: &mut DeviceBuffer<NormStats>,
        output: &mut DeviceBuffer<Features<W>>,
    ) -> Result<()> {
        let [input, gradient, normalized] = inputs;
        if !kernels::fused_norm::<W>(head_width) {
            let launch = self
                .module
                .prepare_gpt_rmsnorm_statistics::<W>(config::<32>(input.len() * heads)?)?;
            self.module.gpt_rmsnorm_statistics(
                &self.stream,
                &launch,
                heads,
                head_width,
                backward,
                input,
                gradient,
                normalized,
                statistics,
            )?;
        }
        output
            .with_chunks(|output| {
                let launch = self
                    .module
                    .prepare_gpt_rmsnorm_apply::<W>(config::<1>(output.len())?)?;
                self.module.gpt_rmsnorm_apply(
                    &self.stream,
                    &launch,
                    heads,
                    head_width,
                    backward,
                    input,
                    if backward == 1 { gradient } else { input },
                    normalized,
                    statistics,
                    output,
                )
            })?
            .map_err(Box::<dyn std::error::Error>::from)
    }

    fn attention<const BACKWARD: usize>(
        &self,
        workspace: &mut Workspace<W, V, C, H>,
        gradient_slot: usize,
        inputs: [usize; 4],
        cache: usize,
    ) -> Result<()> {
        if C.is_multiple_of(64) {
            self.attention_tile::<64, BACKWARD, 2>(workspace, gradient_slot, inputs, cache)
        } else if C.is_multiple_of(32) {
            self.attention_tile::<32, BACKWARD, 1>(workspace, gradient_slot, inputs, cache)
        } else if C.is_multiple_of(4) {
            self.attention_tile::<4, BACKWARD, 1>(workspace, gradient_slot, inputs, cache)
        } else {
            self.attention_tile::<1, BACKWARD, 1>(workspace, gradient_slot, inputs, cache)
        }
    }

    fn serial_scores<const TILE: usize, const BACKWARD: usize, const PACK: usize>(
        &self,
        workspace: &mut Workspace<W, V, C, H>,
        gradient_slot: usize,
        inputs: [usize; 4],
        cache: usize,
    ) -> Result<()> {
        let [query, key, value, attention] = inputs;
        let c = self.config;
        if BACKWARD == 1 {
            let launch = self
                .module
                .prepare_gpt_rmsnorm_statistics::<W>(config::<32>(
                    workspace.norm_statistics.len(),
                )?)?;
            self.module.gpt_rmsnorm_statistics(
                &self.stream,
                &launch,
                c.heads,
                c.width / c.heads,
                1,
                &workspace.adjoints[gradient_slot],
                &workspace.adjoints[gradient_slot],
                &workspace.features[attention],
                &mut workspace.norm_statistics,
            )?;
        }
        workspace.scratch[0].with_chunks(|output| {
            let launch = self
                .module
                .prepare_gpt_feature_transpose::<W>(config::<1>(output.len())?)?;
            self.module.gpt_feature_transpose(
                &self.stream,
                &launch,
                c.width,
                &workspace.features[if BACKWARD == 1 { value } else { key }],
                output,
            )
        })??;
        workspace.scratch[0].with_chunks(|transposed| {
            workspace.probabilities[cache]
                .with_chunks(|output: &mut DeviceViewMut<'_, [f32; PACK]>| {
                    let launch = self
                        .module
                        .prepare_gpt_attention_scores_serial::<W, C, H, TILE, BACKWARD, PACK>(
                            config::<1>(output.len())?,
                        )?;
                    self.module.gpt_attention_scores_serial(
                        &self.stream,
                        &launch,
                        c.width,
                        c.heads,
                        workspace.sequence,
                        workspace.sequence / TILE,
                        workspace.sequence % TILE,
                        workspace.rows / workspace.sequence,
                        kernels::causal_blocks(workspace.sequence, TILE),
                        if BACKWARD == 1 {
                            &workspace.adjoints[gradient_slot]
                        } else {
                            &workspace.features[query]
                        },
                        transposed,
                        &workspace.norm_statistics,
                        output,
                    )
                })?
                .map_err(Box::<dyn std::error::Error>::from)
        })??;
        Ok(())
    }

    fn attention_tile<const TILE: usize, const BACKWARD: usize, const PACK: usize>(
        &self,
        workspace: &mut Workspace<W, V, C, H>,
        gradient_slot: usize,
        inputs: [usize; 4],
        cache: usize,
    ) -> Result<()> {
        let [query, key, value, attention] = inputs;
        let c = self.config;
        if BACKWARD == 1 {
            // Probabilities are needed only for dV; then their storage can hold dScore.
            self.attention_context::<TILE, true>(
                workspace.sequence,
                &workspace.adjoints[gradient_slot],
                &workspace.probabilities[cache],
                &mut workspace.scratch[2],
            )?;
        }
        if workspace.sequence >= 32 {
            self.serial_scores::<TILE, BACKWARD, PACK>(workspace, gradient_slot, inputs, cache)?;
        } else {
            workspace.probabilities[cache].with_chunks(
                |output: &mut DeviceViewMut<'_, [f32; TILE]>| {
                    let launch = self
                        .module
                        .prepare_gpt_attention_scores::<W, C, H, TILE, BACKWARD>(config::<32>(
                            output.len(),
                        )?)?;
                    self.module.gpt_attention_scores(
                        &self.stream,
                        &launch,
                        self.config.width,
                        self.config.heads,
                        workspace.sequence,
                        workspace.sequence / TILE,
                        workspace.sequence % TILE,
                        workspace.rows / workspace.sequence,
                        kernels::causal_blocks(workspace.sequence, TILE),
                        if BACKWARD == 1 {
                            &workspace.adjoints[gradient_slot]
                        } else {
                            &workspace.features[query]
                        },
                        &workspace.features[if BACKWARD == 1 { value } else { key }],
                        &workspace.features[attention],
                        output,
                    )?;
                    Ok::<_, Box<dyn std::error::Error>>(())
                },
            )??;
        }
        if BACKWARD == 0 {
            let sequence = workspace.sequence;
            let bands = sequence / TILE;
            let tail = sequence % TILE;
            let sequences = workspace.rows / sequence;
            let blocks = kernels::causal_blocks(sequence, TILE);
            let launch = self
                .module
                .prepare_gpt_attention_statistics::<TILE>(config::<32>(
                    workspace.norm_statistics.len(),
                )?)?;
            self.module.gpt_attention_statistics(
                &self.stream,
                &launch,
                c.heads,
                sequence,
                bands,
                tail,
                sequences,
                blocks,
                &workspace.probabilities[cache],
                &mut workspace.norm_statistics,
            )?;
            let launch = self
                .module
                .prepare_gpt_attention_probabilities::<TILE>(config::<1>(
                    workspace.probabilities[cache].len(),
                )?)?;
            self.module.gpt_attention_probabilities(
                &self.stream,
                &launch,
                c.heads,
                sequence,
                bands,
                tail,
                sequences,
                blocks,
                &workspace.norm_statistics,
                &mut workspace.probabilities[cache],
            )?;
            let (before, after) = workspace.features.split_at_mut(attention);
            self.attention_context::<TILE, false>(
                sequence,
                &before[value],
                &workspace.probabilities[cache],
                &mut after[0],
            )?;
            return Ok(());
        }
        let (dq, rest) = workspace.scratch.split_at_mut(1);
        dq[0].with_chunks(|dq| {
            rest[0]
                .with_chunks(|dk| {
                    let launch = self
                        .module
                        .prepare_gpt_attention_backward::<W, C, H, TILE>(config::<1>(dq.len())?)?;
                    self.module.gpt_attention_backward(
                        &self.stream,
                        &launch,
                        c.width,
                        c.heads,
                        workspace.sequence,
                        workspace.sequence / TILE,
                        workspace.sequence % TILE,
                        workspace.rows / workspace.sequence,
                        kernels::causal_blocks(workspace.sequence, TILE),
                        &workspace.features[query],
                        &workspace.features[key],
                        &workspace.probabilities[cache],
                        dq,
                        dk,
                    )
                })?
                .map_err(Box::<dyn std::error::Error>::from)
        })??;
        Ok(())
    }

    fn attention_context<const TILE: usize, const COLUMN: bool>(
        &self,
        sequence: usize,
        value: &DeviceBuffer<Features<W>>,
        probabilities: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<Features<W>>,
    ) -> Result<()> {
        output
            .with_chunks(|output| {
                let launch = self
                    .module
                    .prepare_gpt_attention_context::<W, TILE, COLUMN>(config::<1>(output.len())?)?;
                self.module.gpt_attention_context(
                    &self.stream,
                    &launch,
                    self.config.width,
                    self.config.heads,
                    sequence,
                    sequence / TILE,
                    sequence % TILE,
                    value.len() / sequence,
                    kernels::causal_blocks(sequence, TILE),
                    value,
                    probabilities,
                    output,
                )
            })?
            .map_err(Box::<dyn std::error::Error>::from)
    }

    fn add_features(
        &self,
        left: &DeviceBuffer<Features<W>>,
        right: &DeviceBuffer<Features<W>>,
        output: &mut DeviceBuffer<Features<W>>,
    ) -> Result<()> {
        output
            .with_chunks(|output| {
                let launch = self
                    .module
                    .prepare_gpt_add::<W>(config::<1>(output.len())?)?;
                self.module.gpt_add(
                    &self.stream,
                    &launch,
                    self.config.width,
                    left,
                    right,
                    output,
                )
            })?
            .map_err(Box::<dyn std::error::Error>::from)
    }

    fn accumulate_scratch(
        &self,
        workspace: &mut Workspace<W, V, C, H>,
        target: usize,
        source: usize,
    ) -> Result<()> {
        let adjoint = &mut workspace.adjoints[self.plan.gradient_slots[target]];
        if workspace.initialized[target] {
            let (before, after) = workspace.scratch.split_at_mut(3);
            self.add_features(adjoint, &before[source], &mut after[0])?;
            std::mem::swap(adjoint, &mut workspace.scratch[3]);
        } else {
            std::mem::swap(adjoint, &mut workspace.scratch[source]);
            workspace.initialized[target] = true;
        }
        Ok(())
    }

    fn consume_node(
        &self,
        workspace: &mut Workspace<W, V, C, H>,
        target: usize,
        source: usize,
    ) -> Result<()> {
        let [output, input] = workspace.adjoints.get_disjoint_mut([
            self.plan.gradient_slots[target],
            self.plan.gradient_slots[source],
        ])?;
        if workspace.initialized[target] {
            self.add_features(output, input, &mut workspace.scratch[3])?;
            std::mem::swap(output, &mut workspace.scratch[3]);
        } else {
            std::mem::swap(output, input);
            workspace.initialized[target] = true;
        }
        Ok(())
    }
}

fn mesh_gradient<const W: usize, const ROWS: usize>(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    geometry: [usize; 4],
    features: [&DeviceBuffer<Features<W>>; 3],
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let [width, offset, chunks, activated] = geometry;
    output
        .with_chunks(|output: &mut DeviceViewMut<'_, [f32; 2]>| {
            let launch = module.prepare_gpt_mesh_gradient::<W, ROWS>(config::<1>(output.len())?)?;
            module.gpt_mesh_gradient(
                stream,
                &launch,
                width,
                offset,
                activated,
                chunks,
                features[0],
                features[1],
                features[2],
                output,
            )
        })?
        .map_err(Box::<dyn std::error::Error>::from)
}

fn config<const LANES: usize>(
    items: usize,
) -> std::result::Result<LaunchConfig1D, LaunchContractError> {
    const {
        assert!(LANES == 1 || LANES == 32);
    }
    // Divide before converting: legal tensors can exceed u32::MAX elements.
    let blocks = items.div_ceil(128 / LANES).try_into().map_err(|_| {
        LaunchContractError::DimensionProductOverflow {
            kernel: "Mesh GPT launch",
            dimension: LaunchDimension::Grid,
        }
    })?;
    Ok(LaunchConfig1D::new(blocks, 128, 0))
}

fn score_tile(capacity: usize) -> usize {
    if capacity.is_multiple_of(64) {
        64
    } else if capacity.is_multiple_of(32) {
        32
    } else if capacity.is_multiple_of(4) {
        4
    } else {
        1
    }
}

mod verification {
    use super::*;
    use crate::gpt::{
        MAX_WIDTH,
        reference::{exclusive_reference, rmsnorm_reference, rotary_reference},
    };

    impl GpuMeshGpt {
        /// Rejects incomplete, inconsistent, and overflowing causal attention geometry.
        pub fn verify_attention_contracts(device: usize) -> Result<()> {
            let model = GpuMeshGpt::new(device, GptConfig::default(), 7)?;
            let scores = DeviceBuffer::zeroed(&model.stream, 2 * 2 * 3 * 64)?;
            let value = DeviceBuffer::<Features>::zeroed(&model.stream, 6)?;
            let mut probabilities = DeviceBuffer::zeroed(&model.stream, scores.len())?;
            let mut statistics = DeviceBuffer::zeroed(&model.stream, 12)?;
            let mut output = DeviceBuffer::<f32>::zeroed(&model.stream, 6 * MAX_WIDTH)?;
            let stats_launch = model
                .module
                .prepare_gpt_attention_statistics::<64>(config::<32>(statistics.len())?)?;
            let probability_launch = model
                .module
                .prepare_gpt_attention_probabilities::<64>(config::<1>(probabilities.len())?)?;
            let context_launch = model
                .module
                .prepare_gpt_attention_context::<MAX_WIDTH, 64, false>(config::<1>(
                    output.len(),
                )?)?;
            let score_launch = model
                .module
                .prepare_gpt_attention_scores_serial::<MAX_WIDTH, 128, 8, 64, 0, 1>(config::<1>(
                    scores.len(),
                )?)?;
            let mut serial_scores = DeviceBuffer::zeroed(&model.stream, scores.len())?;
            let valid = [2, 3, 0, 3, 2, 3];
            for geometry in [
                valid,
                [0, 3, 0, 3, 2, 3],
                [2, 0, 0, 0, 2, 0],
                [2, 3, 0, 64, 2, 3],
                [2, 3, 1, 3, 2, 3],
                [2, 3, 0, 3, 0, 3],
                [2, 3, 0, 3, 2, 4],
                [2, 3, 0, 3, usize::MAX, 3],
                [2, 3, 0, 3, 2, usize::MAX],
                [usize::MAX, 3, 0, 3, 2, 3],
            ] {
                let [heads, sequence, bands, tail, sequences, blocks] = geometry;
                let results = [
                    model.module.gpt_attention_statistics(
                        &model.stream,
                        &stats_launch,
                        heads,
                        sequence,
                        bands,
                        tail,
                        sequences,
                        blocks,
                        &scores,
                        &mut statistics,
                    ),
                    model.module.gpt_attention_probabilities(
                        &model.stream,
                        &probability_launch,
                        heads,
                        sequence,
                        bands,
                        tail,
                        sequences,
                        blocks,
                        &statistics,
                        &mut probabilities,
                    ),
                    model.module.gpt_attention_context(
                        &model.stream,
                        &context_launch,
                        8,
                        heads,
                        sequence,
                        bands,
                        tail,
                        sequences,
                        blocks,
                        &value,
                        &probabilities,
                        &mut output,
                    ),
                    model.module.gpt_attention_scores_serial(
                        &model.stream,
                        &score_launch,
                        8,
                        heads,
                        sequence,
                        bands,
                        tail,
                        sequences,
                        blocks,
                        &value,
                        &output,
                        &statistics,
                        &mut serial_scores,
                    ),
                ];
                for result in results {
                    assert_eq!(result.is_ok(), geometry == valid, "geometry={geometry:?}");
                }
            }
            let actual = probabilities.to_host_vec(&model.stream)?;
            for group in 0..4 {
                for position in 0..3 {
                    for local in 0..64 {
                        let expected = if local <= position {
                            1.0 / (position + 1) as f32
                        } else {
                            0.0
                        };
                        assert_eq!(actual[(group * 3 + position) * 64 + local], expected);
                    }
                }
            }
            for cells in [0, probabilities.len() - 1, probabilities.len() + 1] {
                let mut invalid = DeviceBuffer::<f32>::zeroed(&model.stream, cells)?;
                assert!(
                    model
                        .module
                        .gpt_attention_probabilities(
                            &model.stream,
                            &probability_launch,
                            2,
                            3,
                            0,
                            3,
                            2,
                            3,
                            &statistics,
                            &mut invalid,
                        )
                        .is_err()
                );
            }
            let transpose_launch = model
                .module
                .prepare_gpt_feature_transpose::<MAX_WIDTH>(config::<1>(output.len())?)?;
            for width in [0, 1, 8, MAX_WIDTH, MAX_WIDTH + 1, usize::MAX] {
                assert_eq!(
                    model
                        .module
                        .gpt_feature_transpose(
                            &model.stream,
                            &transpose_launch,
                            width,
                            &value,
                            &mut output,
                        )
                        .is_ok(),
                    (2..=MAX_WIDTH).contains(&width)
                );
            }
            for rows in [0, 5, 7] {
                let invalid = DeviceBuffer::<Features>::zeroed(&model.stream, rows)?;
                assert!(
                    model
                        .module
                        .gpt_feature_transpose(
                            &model.stream,
                            &transpose_launch,
                            8,
                            &invalid,
                            &mut output,
                        )
                        .is_err()
                );
            }
            model.stream.synchronize()?;
            Ok(())
        }

        /// Checks both tied-embedding gradient contributions, chunk tails, and contracts.
        pub fn verify_embedding_gradient_reduction(device: usize) -> Result<()> {
            let model = GpuMeshGpt::new(device, GptConfig::default(), 7)?;
            for (width, vocab) in [(2, 3), (18, 17), (32, 17), (128, 99)] {
                for rows in [
                    1_usize,
                    31,
                    32,
                    33,
                    63,
                    64,
                    65,
                    127,
                    128,
                    129,
                    EMBEDDING_GRADIENT_ROWS - 1,
                    EMBEDDING_GRADIENT_ROWS,
                    EMBEDDING_GRADIENT_ROWS + 1,
                    2048,
                    4096,
                    4097,
                ]
                .into_iter()
                .chain((width == 2).then_some(super::super::MAX_ROWS))
                {
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
                        >(config::<1>(partials.len())?)?;
                    let run_partials =
                        |chunks, output: &mut dyn cuda_host::KernelSliceArgMut<Elem = f32>| {
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
                                output,
                            )
                        };
                    run_partials(chunks, &mut partials)?;
                    let mut output: DeviceBuffer<Features> =
                        DeviceBuffer::from_host(&model.stream, &vec![[42.0; MAX_WIDTH]; vocab])?;
                    output.with_chunks(|output| {
                        let reduce =
                            model
                                .module
                                .prepare_gpt_embedding_gradient_reduce::<128>(config::<1>(
                                    output.len(),
                                )?)?;
                        let run_reduce =
                            |chunks, output: &mut dyn cuda_host::KernelSliceArgMut<Elem = f32>| {
                                model.module.gpt_embedding_gradient_reduce(
                                    &model.stream,
                                    &reduce,
                                    width,
                                    vocab,
                                    chunks,
                                    &partials,
                                    output,
                                )
                            };
                        run_reduce(chunks, output)?;
                        let mut short = DeviceBuffer::zeroed(&model.stream, output.len() - 1)?;
                        assert!(run_reduce(chunks, &mut short).is_err());
                        for count in [0, chunks + 1, usize::MAX] {
                            assert!(run_reduce(count, output).is_err());
                        }
                        Ok::<_, cuda_core::LaunchContractError>(())
                    })??;
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
                        assert!(run_partials(count, &mut partials).is_err());
                    }
                    let mut short = DeviceBuffer::zeroed(&model.stream, partials.len() - 1)?;
                    assert!(run_partials(chunks, &mut short).is_err());
                }
            }
            Ok(())
        }

        /// Checks chunked reductions, row tails, untouched layers, and launch contracts.
        pub fn verify_mesh_gradient_reduction(device: usize) -> Result<()> {
            let model = GpuMeshGpt::new(device, GptConfig::default(), 7)?;
            for width in [2, 4, 32, 128] {
                for rows in [
                    1_usize,
                    31,
                    32,
                    33,
                    127,
                    128,
                    129,
                    257,
                    2048,
                    4096,
                    4097,
                    65535,
                    65536,
                    65537,
                    super::super::MAX_ROWS - 1,
                    super::super::MAX_ROWS,
                ] {
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
                    let run_mesh_partials = |activated, chunks, output: &mut DeviceBuffer<f32>| {
                        mesh_gradient::<128, 32>(
                            &model.module,
                            &model.stream,
                            [width, 0, chunks, activated],
                            [&input_gpu, &activation_gpu, &gradient_gpu],
                            output,
                        )
                    };
                    let sentinel = vec![42.0_f32; 6 * width];
                    for (activated, offset) in [(0, 2 * width), (1, 2 * width)]
                        .into_iter()
                        .chain((rows == 1).then_some((0, 2)))
                        .chain((rows == 1).then_some((1, 2)))
                    {
                        let mut serial = DeviceBuffer::from_host(&model.stream, &sentinel)?;
                        let mut reduced = DeviceBuffer::from_host(&model.stream, &sentinel)?;
                        mesh_gradient::<128, MAX_ROWS>(
                            &model.module,
                            &model.stream,
                            [width, offset / 2, 1, activated],
                            [&input_gpu, &activation_gpu, &gradient_gpu],
                            &mut serial,
                        )?;
                        run_mesh_partials(activated, chunks, &mut partials)?;
                        let reduce_launch =
                            model
                                .module
                                .prepare_gpt_mesh_gradient_reduce::<128>(config::<1>(
                                    reduced.len() * 32,
                                )?)?;
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
                    for bad_chunks in [0, 4097, chunks + 1, usize::MAX] {
                        assert!(run_mesh_partials(0, bad_chunks, &mut partials).is_err());
                    }
                    for length in [partials.len() - 1, partials.len() - 2] {
                        let mut too_short = DeviceBuffer::zeroed(&model.stream, length)?;
                        assert!(run_mesh_partials(0, chunks, &mut too_short).is_err());
                    }
                    assert!(run_mesh_partials(2, chunks, &mut partials).is_err());
                    assert!(
                        mesh_gradient::<128, 32>(
                            &model.module,
                            &model.stream,
                            [width, usize::MAX, chunks, 0],
                            [&input_gpu, &activation_gpu, &gradient_gpu],
                            &mut partials,
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
                let sentinel = vec![[42.0_f32; MAX_WIDTH]; rows];
                let mut output_gpu = DeviceBuffer::from_host(&model.stream, &sentinel)?;
                let mut derivative_gpu = DeviceBuffer::from_host(&model.stream, &sentinel)?;
                let mut statistics =
                    DeviceBuffer::from_host(&model.stream, &vec![[42.0_f32; 2]; rows * heads + 1])?;
                let run_norm =
                    |heads,
                     head_width,
                     output: &mut DeviceBuffer<Features>,
                     statistics: &mut DeviceBuffer<NormStats>| {
                        model.rmsnorm(heads, head_width, 0, [&input_gpu; 3], statistics, output)
                    };
                run_norm(heads, head_width, &mut output_gpu, &mut statistics)?;
                let run_backward =
                    |heads,
                     head_width,
                     normalized: &DeviceBuffer<Features>,
                     statistics: &mut DeviceBuffer<NormStats>,
                     output: &mut DeviceBuffer<Features>| {
                        model.rmsnorm(
                            heads,
                            head_width,
                            1,
                            [&input_gpu, &gradient_gpu, normalized],
                            statistics,
                            output,
                        )
                    };
                run_backward(
                    heads,
                    head_width,
                    &output_gpu,
                    &mut statistics,
                    &mut derivative_gpu,
                )?;
                let output = output_gpu.to_host_vec(&model.stream)?;
                let derivative = derivative_gpu.to_host_vec(&model.stream)?;
                assert_eq!(
                    statistics.to_host_vec(&model.stream)?[rows * heads],
                    [42.0; 2]
                );
                let doubles = |x: &[f32]| x.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
                for row in 0..rows {
                    assert!(
                        output[row][width..]
                            .iter()
                            .chain(&derivative[row][width..])
                            .all(|&v| v == 42.0)
                    );
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
                    (heads, usize::MAX),
                ] {
                    assert!(
                        run_norm(
                            invalid_heads,
                            invalid_width,
                            &mut output_gpu,
                            &mut statistics
                        )
                        .is_err()
                    );
                    assert!(
                        run_backward(
                            invalid_heads,
                            invalid_width,
                            &output_gpu,
                            &mut statistics,
                            &mut derivative_gpu
                        )
                        .is_err()
                    );
                }
                let mut short = DeviceBuffer::zeroed(&model.stream, rows * heads - 1)?;
                assert!(run_norm(heads, head_width, &mut output_gpu, &mut short).is_err());
                assert!(
                    run_backward(
                        heads,
                        head_width,
                        &output_gpu,
                        &mut short,
                        &mut derivative_gpu
                    )
                    .is_err()
                );
                let mut short_output = DeviceBuffer::zeroed(&model.stream, rows - 1)?;
                assert!(run_norm(heads, head_width, &mut short_output, &mut statistics).is_err());
                assert!(
                    model
                        .rmsnorm(
                            heads,
                            head_width,
                            2,
                            [&input_gpu; 3],
                            &mut statistics,
                            &mut output_gpu
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
                    .prepare_gpt_rope::<128, 128>(config::<1>(rows * MAX_WIDTH)?)?;
                let rotate = |half_width,
                              sequence,
                              inverse,
                              source: &DeviceBuffer<Features>,
                              target: &mut DeviceBuffer<Features>| {
                    target
                        .with_chunks(|target| {
                            model.module.gpt_rope(
                                &model.stream,
                                &launch,
                                heads,
                                half_width,
                                sequence,
                                inverse,
                                source,
                                target,
                            )
                        })?
                        .map_err(Box::<dyn std::error::Error>::from)
                };
                for sequence in [1, 3, 128] {
                    for (source, target, inverse) in [
                        (&input_gpu, &mut output_gpu, 0),
                        (&gradient_gpu, &mut derivative_gpu, 1),
                    ] {
                        rotate(half_width, sequence, inverse, source, target)?;
                    }
                    rotate(half_width, sequence, 1, &output_gpu, &mut inverse_gpu)?;
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
                    assert!(rotate(half, sequence, inverse, &input_gpu, &mut output_gpu).is_err());
                }
            }
            Ok(())
        }

        /// Checks the isolated CUDA XSA operators against independent f64 autodiff.
        /// Includes batch tails, one-channel heads, and zero/tiny/ordinary values.
        pub fn verify_exclusive_attention(device: usize) -> Result<()> {
            let model = GpuMeshGpt::new(device, GptConfig::default(), 7)?;
            let rows = 129;
            for (width, heads) in [
                (4, 2),
                (8, 2),
                (64, 8),
                (4, 4),
                (64, 2),
                (96, 1),
                (128, 4),
                (128, 2),
                (128, 1),
            ] {
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
                let sentinel = vec![[42.0_f32; MAX_WIDTH]; rows];
                let mut output = DeviceBuffer::from_host(&model.stream, &sentinel)?;
                let mut dy = DeviceBuffer::from_host(&model.stream, &sentinel)?;
                let mut dv = DeviceBuffer::from_host(&model.stream, &sentinel)?;
                output.with_chunks(|output| {
                    let launch = model
                        .module
                        .prepare_gpt_exclusive::<128>(config::<1>(output.len())?)?;
                    let run = |width, heads, output: &mut dyn cuda_host::KernelSliceArgMut<Elem = f32>| {
                        model.module.gpt_exclusive(
                            &model.stream,
                            &launch,
                            width,
                            heads,
                            &input_gpu,
                            &value_gpu,
                            output,
                        )
                    };
                    run(width, heads, output)?;
                    for (width, heads) in [
                        (0, heads),
                        (MAX_WIDTH + 1, heads),
                        (width, 0),
                        (width, usize::MAX),
                    ] {
                        assert!(run(width, heads, output).is_err());
                    }
                    let mut short = DeviceBuffer::zeroed(&model.stream, output.len() - 1)?;
                    assert!(run(width, heads, &mut short).is_err());
                    Ok::<_, Box<dyn std::error::Error>>(())
                })??;
                dy.with_chunks(|dy| {
                    dv.with_chunks(|dv| {
                        let launch = model
                            .module
                            .prepare_gpt_exclusive_backward::<128>(config::<1>(dy.len())?)?;
                        model.module.gpt_exclusive_backward(
                            &model.stream,
                            &launch,
                            width,
                            heads,
                            &input_gpu,
                            &value_gpu,
                            &gradient_gpu,
                            dy,
                            dv,
                        )
                    })?
                    .map_err(Box::<dyn std::error::Error>::from)
                })??;
                let output = output.to_host_vec(&model.stream)?;
                let dy = dy.to_host_vec(&model.stream)?;
                let dv = dv.to_host_vec(&model.stream)?;
                for row in 0..rows {
                    for padded in [&output, &dy, &dv] {
                        assert!(padded[row][width..].iter().all(|&v| v == 42.0));
                    }
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
