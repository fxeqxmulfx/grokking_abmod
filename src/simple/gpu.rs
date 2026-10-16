use std::{
    any::{Any, TypeId},
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, KernelLaunchContract, LaunchConfig1D,
    LaunchContractError, LaunchDimension, PinnedHostBuffer, PreparedLaunch,
};
use cuda_device::{DisjointSlice, device, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;

use crate::{
    Result,
    gpt::{AdamWOptions, RMSNORM_EPSILON},
    mesh::validate_finite,
};

use super::{Activation, Architecture, Dense, MAX_SIMPLE_ROWS, SimpleConfig, TwoParentLayer};

const IDENTITY: usize = 0;
const RELU: usize = 1;
const SQUARE: usize = 2;
const ADD: usize = 0;
const PRODUCT: usize = 1;
/// Training alternates full and tail minibatches; evaluation uses complete splits.
const WORKSPACE_CAPACITY: usize = 4;

#[path = "dense_tiled.rs"]
mod dense_tiled;

#[path = "regression.rs"]
pub(super) mod regression;

#[cuda_module]
// Flat row-major tensors keep every index explicit.
#[allow(clippy::too_many_arguments)]
// Every kernel is generic, so each launch resolves its own specialization by
// function-item identity and `kernels::load` is safe. Kernels without a
// natural compile-time parameter take an unused `S`, always 0.
mod kernels {
    use super::*;

    #[device]
    fn simple_activate(value: f32, activation: usize) -> f32 {
        if activation == RELU {
            if value > 0.0 { value } else { 0.0 }
        } else if activation == SQUARE {
            value * value
        } else {
            value
        }
    }

    /// The derivative of `simple_activate`, with ReLU's subgradient 0 at the kink.
    #[device]
    fn simple_slope(value: f32, activation: usize) -> f32 {
        if activation == RELU {
            if value > 0.0 { 1.0 } else { 0.0 }
        } else if activation == SQUARE {
            2.0 * value
        } else {
            1.0
        }
    }

    #[device]
    fn simple_warp_output<'a, T>(
        output: &'a mut DisjointSlice<'_, T, thread::WarpIndex>,
    ) -> Option<&'a mut T> {
        if let Some(warp) = thread::warp_index() {
            output.get_mut(warp)
        } else {
            None
        }
    }

    /// Copies table rows `indices[offset..offset + rows]`.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        columns >= 1, rows >= 1, offset + rows <= indices.len(),
        table.len() >= columns, out.len() == rows * columns))]
    pub fn simple_gather<const S: usize>(
        columns: usize,
        rows: usize,
        offset: usize,
        indices: &[u32],
        table: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let row = indices[offset + item / columns] as usize;
            *output = table[row * columns + item % columns];
        }
        let _ = rows;
    }

    /// Fixed, unary column copies: split operands, zero-pad logits for the
    /// transposed readout's backward pass, or select the class prefix.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, source >= 1, target >= 1, copied >= 1, copied <= target,
        offset + copied <= source, input.len() == rows * source,
        out.len() == rows * target))]
    pub fn simple_copy_columns<const S: usize>(
        rows: usize,
        source: usize,
        target: usize,
        offset: usize,
        copied: usize,
        input: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let (row, column) = (item / target, item % target);
            *output = if column < copied {
                input[row * source + offset + column]
            } else {
                0.0
            };
        }
        let _ = rows;
    }

    /// Adjacent-pair J rotation, or its transpose for backpropagation. Each
    /// output has one activation parent and a fixed sign.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, pairs >= 1, width == 2 * pairs,
        input.len() == rows * width, out.len() == input.len()))]
    pub fn simple_pair_rotation<const TRANSPOSE: usize>(
        rows: usize,
        pairs: usize,
        width: usize,
        input: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let even = item.is_multiple_of(2);
            let parent = if even { item + 1 } else { item - 1 };
            let negative = even != (TRANSPOSE == 1);
            *output = if negative {
                -input[parent]
            } else {
                input[parent]
            };
        }
        let _ = width;
        let _ = (rows, pairs);
    }

    /// Two operand embeddings share E; the output row is [E(a) | E(b)].
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, classes >= 2, width >= 1,
        input.len() == rows * 2 * classes, weights.len() == classes * width,
        out.len() == rows * 2 * width))]
    pub fn simple_concat_embedding<const S: usize>(
        rows: usize,
        classes: usize,
        width: usize,
        input: &[f32],
        weights: &[[f32; 3]],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let column = item % (2 * width);
            let start = item / (2 * width) * 2 * classes + column / width * classes;
            let channel = column % width;
            let mut total = 0.0;
            for residue in 0..classes {
                total += input[start + residue] * weights[residue * width + channel][0];
            }
            *output = total;
        }
        let _ = rows;
    }

    /// All three E uses contribute to one gradient and one AdamW state.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, classes >= 2, width >= 1,
        input.len() == rows * 2 * classes, embedded_gradient.len() == rows * 2 * width,
        logit_gradient.len() == rows * classes, hidden.len() == rows * width,
        out.len() == classes * width))]
    pub fn simple_shared_embedding_gradient<const ACTIVATION: usize>(
        rows: usize,
        classes: usize,
        width: usize,
        input: &[f32],
        embedded_gradient: &[f32],
        logit_gradient: &[f32],
        hidden: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let residue = item / width;
            let channel = item % width;
            let mut total = 0.0;
            for row in 0..rows {
                total += input[row * 2 * classes + residue]
                    * embedded_gradient[row * 2 * width + channel]
                    + input[row * 2 * classes + classes + residue]
                        * embedded_gradient[row * 2 * width + width + channel]
                    + logit_gradient[row * classes + residue]
                        * simple_activate(hidden[row * width + channel], ACTIVATION);
            }
            *output = total;
        }
    }

    /// `out[r][c] = W[c][0] f(input[r][c]) + W[c][1] f(input[r][(c + stride) % width])`,
    /// a mesh layer whose weights are stored unit by unit. With one parent per
    /// unit only the first term remains, and the stride is unused.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, parents >= 1, parents <= 2, stride + 1 >= parents, stride < width,
        input.len() == rows * width, weights.len() == parents * width,
        out.len() == rows * width))]
    pub fn simple_mesh<const ACTIVATION: usize>(
        rows: usize,
        width: usize,
        parents: usize,
        stride: usize,
        input: &[f32],
        weights: &[[f32; 3]],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let channel = item % width;
            // One expression per branch: accumulating across the branch fused
            // the second product instead of the first, so two-parent runs no
            // longer reproduced bitwise.
            *output = if parents == 2 {
                let parent = item - channel + (channel + stride) % width;
                weights[2 * channel][0] * simple_activate(input[item], ACTIVATION)
                    + weights[2 * channel + 1][0] * simple_activate(input[parent], ACTIVATION)
            } else {
                weights[channel][0] * simple_activate(input[item], ACTIVATION)
            };
        }
        let _ = rows;
    }

    /// `out[r][c] = f'(before[r][c]) (gradient[r][c] W[c][0] + gradient[r][u] W[u][1])`,
    /// where `u = (c - stride) mod width` is the unit that reads channel `c`
    /// as its second parent. With one parent per unit only the first term remains.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, parents >= 1, parents <= 2, stride + 1 >= parents, stride < width,
        gradient.len() == rows * width, weights.len() == parents * width,
        before.len() == rows * width, out.len() == rows * width))]
    pub fn simple_mesh_input_gradient<const ACTIVATION: usize>(
        rows: usize,
        width: usize,
        parents: usize,
        stride: usize,
        gradient: &[f32],
        weights: &[[f32; 3]],
        before: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let channel = item % width;
            // As in `simple_mesh`, one expression per branch fixes the fused product.
            let total = if parents == 2 {
                let reader = (channel + width - stride) % width;
                gradient[item] * weights[2 * channel][0]
                    + gradient[item - channel + reader] * weights[2 * reader + 1][0]
            } else {
                gradient[item] * weights[channel][0]
            };
            *output = total * simple_slope(before[item], ACTIVATION);
        }
        let _ = rows;
    }

    /// `out[parents c + k] = sum_r f(input[r][parent]) gradient[r][c]`, with the
    /// parent `c` for the self weight (`k = 0`) and `(c + stride) % width` for
    /// the other. One thread owns each weight and adds the rows in order.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, parents >= 1, parents <= 2, stride + 1 >= parents, stride < width,
        input.len() == rows * width, gradient.len() == rows * width,
        out.len() == parents * width))]
    pub fn simple_mesh_weight_gradient<const ACTIVATION: usize>(
        rows: usize,
        width: usize,
        parents: usize,
        stride: usize,
        input: &[f32],
        gradient: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let channel = item / parents;
            let parent = (channel + item % parents * stride) % width;
            let mut total = 0.0;
            for row in 0..rows {
                total += simple_activate(input[row * width + parent], ACTIVATION)
                    * gradient[row * width + channel];
            }
            *output = total;
        }
    }

    /// A complete sparse affine layer: two activation parents and a node bias.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, inputs >= 2, outputs >= 1, stride >= 1, stride < inputs,
        input.len() == rows * inputs, weights.len() == 3 * outputs,
        out.len() == rows * outputs))]
    pub fn simple_two_parent<const ACTIVATION: usize>(
        rows: usize,
        inputs: usize,
        outputs: usize,
        stride: usize,
        input: &[f32],
        weights: &[[f32; 3]],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let (row, channel) = (item / outputs, item % outputs);
            let first = channel * inputs / outputs;
            let second = (first + stride) % inputs;
            *output = weights[3 * channel][0]
                * simple_activate(input[row * inputs + first], ACTIVATION)
                + weights[3 * channel + 1][0]
                    * simple_activate(input[row * inputs + second], ACTIVATION)
                + weights[3 * channel + 2][0];
        }
        let _ = rows;
    }

    /// Each input owns its reverse accumulation. Integer intervals find all
    /// readers of the evenly distributed parents without atomics or dense scans.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, inputs >= 2, outputs >= 1, stride >= 1, stride < inputs,
        gradient.len() == rows * outputs, weights.len() == 3 * outputs,
        before.len() == rows * inputs, out.len() == rows * inputs))]
    pub fn simple_two_parent_input_gradient<const ACTIVATION: usize>(
        rows: usize,
        inputs: usize,
        outputs: usize,
        stride: usize,
        gradient: &[f32],
        weights: &[[f32; 3]],
        before: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let (row, channel) = (item / inputs, item % inputs);
            let mut total = 0.0;
            let mut parent = 0;
            while parent < 2 {
                let first = (channel + inputs - parent * stride) % inputs;
                let start = (first * outputs).div_ceil(inputs);
                let end = ((first + 1) * outputs).div_ceil(inputs);
                for reader in start..end {
                    total += gradient[row * outputs + reader] * weights[3 * reader + parent][0];
                }
                parent += 1;
            }
            *output = total * simple_slope(before[item], ACTIVATION);
        }
        let _ = rows;
    }

    /// One thread owns each edge or bias and accumulates the batch in row order.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, inputs >= 2, outputs >= 1, stride >= 1, stride < inputs,
        input.len() == rows * inputs, gradient.len() == rows * outputs,
        out.len() == 3 * outputs))]
    pub fn simple_two_parent_weight_gradient<const ACTIVATION: usize>(
        rows: usize,
        inputs: usize,
        outputs: usize,
        stride: usize,
        input: &[f32],
        gradient: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let (channel, kind) = (item / 3, item % 3);
            let first = channel * inputs / outputs;
            let parent = (first + kind % 2 * stride) % inputs;
            let mut total = 0.0;
            for row in 0..rows {
                let feature = if kind == 2 {
                    1.0
                } else {
                    simple_activate(input[row * inputs + parent], ACTIVATION)
                };
                total += feature * gradient[row * outputs + channel];
            }
            *output = total;
        }
    }

    /// Elementwise `left + right`, or `left * right` for the `PRODUCT` operation.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        left.len() == out.len(), right.len() == out.len()))]
    pub fn simple_combine<const OPERATION: usize>(
        left: &[f32],
        right: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            *output = if OPERATION == PRODUCT {
                left[item] * right[item]
            } else {
                left[item] + right[item]
            };
        }
    }

    /// Activate a branch, optionally adding the unchanged residual stream.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        stream.len() == out.len(), branch.len() == out.len()))]
    pub fn simple_activated_sum<const ACTIVATION: usize>(
        residual: bool,
        stream: &[f32],
        branch: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            *output = if residual {
                stream[item] + simple_activate(branch[item], ACTIVATION)
            } else {
                simple_activate(branch[item], ACTIVATION)
            };
        }
    }

    /// Differentiate the final branch activation without altering the skip gradient.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        branch.len() == out.len(), gradient.len() == out.len()))]
    pub fn simple_activation_gradient<const ACTIVATION: usize>(
        branch: &[f32],
        gradient: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            *output = gradient[item] * simple_slope(branch[item], ACTIVATION);
        }
    }

    /// One warp per row: the inverse RMS and, backward, `sum_c dn[c] n[c]`.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, width >= 1, input.len() == rows * width,
        gradient.len() == input.len(), normalized.len() == input.len(), out.len() == rows))]
    pub fn simple_rmsnorm_statistics<const BACKWARD: usize>(
        rows: usize,
        width: usize,
        input: &[f32],
        gradient: &[f32],
        normalized: &[f32],
        mut out: DisjointSlice<[f32; 2], thread::WarpIndex>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / 32;
        let lane = index.get() % 32;
        // Rows are whole warps, so every reduction below has all 32 lanes.
        if row >= rows {
            return;
        }
        let start = row * width;
        let mut squares = 0.0;
        let mut dot = 0.0;
        for channel in (lane..width).step_by(32) {
            let value = input[start + channel];
            squares += value * value;
            if BACKWARD == 1 {
                dot += gradient[start + channel] * normalized[start + channel];
            }
        }
        let squares = warp::reduce_sum_f32(squares);
        let dot = warp::reduce_sum_f32(dot);
        if let Some(output) = simple_warp_output(&mut out) {
            *output = [1.0 / (squares / width as f32 + RMSNORM_EPSILON).sqrt(), dot];
        }
    }

    /// Forward `inverse * x`; backward `inverse * (dn - n * dot / width)`.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, width >= 1, input.len() == rows * width,
        normalized.len() == input.len(), statistics.len() == rows,
        out.len() == input.len()))]
    pub fn simple_rmsnorm_apply<const BACKWARD: usize>(
        rows: usize,
        width: usize,
        input: &[f32],
        normalized: &[f32],
        statistics: &[[f32; 2]],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let [inverse, dot] = statistics[item / width];
            *output = inverse
                * if BACKWARD == 1 {
                    input[item] - normalized[item] * dot / width as f32
                } else {
                    input[item]
                };
        }
        let _ = rows;
    }

    /// `simple_rmsnorm_statistics` and `simple_rmsnorm_apply` in one pass
    /// over rows of `WIDTH` channels: each warp reduces its row as the
    /// statistics kernel does, then writes the row with the apply formula.
    /// The reduction loop and both divisions use the runtime `width`. A known
    /// trip count lets the compiler unroll the loop and pair products with
    /// other additions: it fused a lane's only square into the first shuffle
    /// sum, and the first square into the second instead of the second into
    /// the first. A known power-of-two width would let it fuse the division
    /// into the subtraction.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        WIDTH >= 1, width == WIDTH, rows >= 1, input.len() == rows * width,
        gradient.len() == input.len(), normalized.len() == input.len(),
        out.len() == input.len()))]
    pub fn simple_rmsnorm_rows<const BACKWARD: usize, const WIDTH: usize>(
        rows: usize,
        width: usize,
        input: &[f32],
        gradient: &[f32],
        normalized: &[f32],
        mut out: DisjointSlice<f32, thread::StripedWarp<WIDTH>>,
    ) {
        let index = thread::index_1d();
        let row = index.get() / 32;
        let lane = index.get() % 32;
        // Rows are whole warps, so every reduction below has all 32 lanes.
        if row >= rows {
            return;
        }
        let start = row * WIDTH;
        let mut squares = 0.0;
        let mut dot = 0.0;
        for channel in (lane..width).step_by(32) {
            let value = input[start + channel];
            squares += value * value;
            if BACKWARD == 1 {
                dot += gradient[start + channel] * normalized[start + channel];
            }
        }
        let squares = warp::reduce_sum_f32(squares);
        let dot = warp::reduce_sum_f32(dot);
        let inverse = 1.0 / (squares / width as f32 + RMSNORM_EPSILON).sqrt();
        // The lane's cells are channels `lane`, `lane + 32`, ... of its row.
        let mut cells = thread::striped_columns::<WIDTH>();
        for channel in (lane..WIDTH).step_by(32) {
            if let Some(cell) = cells.next()
                && let Some(output) = out.get_mut(cell)
            {
                let item = start + channel;
                *output = inverse
                    * if BACKWARD == 1 {
                        gradient[item] - normalized[item] * dot / width as f32
                    } else {
                        input[item]
                    };
            }
        }
    }

    /// The first maximum of `values[start..start + count]` and its index, as
    /// a sequential scan with `>` finds it, for every lane of a warp: lanes
    /// scan their strided elements in order, and equal maxima keep the lowest
    /// index, so the result keeps the sign of a zero maximum. NaN never wins;
    /// without other values the maximum stays -inf and the index `u32::MAX`.
    #[device]
    fn simple_first_maximum(values: &[f32], start: usize, count: usize, lane: usize) -> (f32, u32) {
        let mut maximum = f32::NEG_INFINITY;
        let mut first = u32::MAX;
        for index in (lane..count).step_by(32) {
            if values[start + index] > maximum {
                maximum = values[start + index];
                first = index as u32;
            }
        }
        let mut offset = 16;
        while offset > 0 {
            let other = warp::shuffle_xor_f32(maximum, offset);
            let index = warp::shuffle_xor(first, offset);
            if other > maximum || (other == maximum && index < first) {
                maximum = other;
                first = index;
            }
            offset /= 2;
        }
        (maximum, first)
    }

    /// The maximum logit of each row and the sum of `exp(logit - maximum)` in
    /// class order, one warp per row. Every lane adds all exponentials in
    /// class order, evaluating each itself so that the last multiply of the
    /// exponential fuses into the sum exactly as in a one-thread loop.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, classes >= 2, logits.len() == rows * classes, out.len() == rows))]
    pub fn simple_softmax_statistics<const S: usize>(
        rows: usize,
        classes: usize,
        logits: &[f32],
        mut out: DisjointSlice<[f32; 2], thread::WarpIndex>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let row = index.get() / 32;
        let lane = index.get() % 32;
        // Rows are whole warps, so every shuffle below has all 32 lanes.
        if row >= rows {
            return;
        }
        let start = row * classes;
        let (maximum, _) = simple_first_maximum(logits, start, classes, lane);
        let mut sum = 0.0;
        let mut base = 0;
        while base < classes {
            let class = base + lane;
            let value = if class < classes {
                logits[start + class]
            } else {
                0.0
            };
            let count = (classes - base).min(32);
            let mut source = 0;
            while source < count {
                sum += (warp::shuffle_f32(value, source as u32) - maximum).exp();
                source += 1;
            }
            base += 32;
        }
        if let Some(output) = simple_warp_output(&mut out) {
            *output = [maximum, sum];
        }
    }

    /// `(softmax(logits[r]) - onehot(target)) / rows` from the row statistics.
    /// Targets are read through the same minibatch indices as the gathered
    /// features.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, classes >= 2, offset + rows <= indices.len(), targets.len() >= 1,
        logits.len() == rows * classes, statistics.len() == rows,
        out.len() == logits.len()))]
    pub fn simple_cross_entropy_gradient<const S: usize>(
        rows: usize,
        classes: usize,
        offset: usize,
        indices: &[u32],
        targets: &[u32],
        logits: &[f32],
        statistics: &[[f32; 2]],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            let row = item / classes;
            let [maximum, sum] = statistics[row];
            let target = targets[indices[offset + row] as usize] as usize;
            let indicator = if item - row * classes == target {
                1.0
            } else {
                0.0
            };
            *output = ((logits[item] - maximum).exp() / sum - indicator) / rows as f32;
        }
    }

    /// The class with the highest logit of each row, the first one on ties,
    /// or `u32::MAX` when the row holds a value that is not finite. One warp
    /// scans each row.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, classes >= 2, logits.len() == rows * classes, out.len() == rows))]
    pub fn simple_predictions<const S: usize>(
        rows: usize,
        classes: usize,
        logits: &[f32],
        mut out: DisjointSlice<u32, thread::WarpIndex>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let row = index.get() / 32;
        let lane = index.get() % 32;
        // Rows are whole warps, so every shuffle and vote has all 32 lanes.
        if row >= rows {
            return;
        }
        let start = row * classes;
        let mut finite = true;
        for class in (lane..classes).step_by(32) {
            finite &= logits[start + class].is_finite();
        }
        let finite = warp::all(finite);
        let (_, prediction) = simple_first_maximum(logits, start, classes, lane);
        if let Some(output) = simple_warp_output(&mut out) {
            *output = if finite { prediction } else { u32::MAX };
        }
    }

    /// The weights of AdamW states without their moments.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (out.len() == state.len()))]
    pub fn simple_weights<const S: usize>(state: &[[f32; 3]], mut out: DisjointSlice<f32>) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        if let Some(output) = out.get_mut(index) {
            *output = state[item][0];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (gradient.len() == state.len()))]
    pub fn simple_adam<const S: usize>(
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
        let _ = S;
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

    /// `out[c] = max(s_c, MIN_POSITIVE)^exponent` for hidden channel `c` of a
    /// one-layer MLP, with `s_c = (sum_i E[i][c]^2 + sum_k R[c][k]^2) / 2` over
    /// its input-major input layer `E` and its readout `R`.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        inputs >= 1, width >= 1, classes >= 1, input.len() == inputs * width,
        readout.len() == width * classes, out.len() == width))]
    pub fn simple_channel_decay<const S: usize>(
        inputs: usize,
        width: usize,
        classes: usize,
        exponent: f32,
        input: &[[f32; 3]],
        readout: &[[f32; 3]],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let channel = index.get();
        if let Some(output) = out.get_mut(index) {
            let mut squares = 0.0;
            for row in 0..inputs {
                let weight = input[row * width + channel][0];
                squares += weight * weight;
            }
            for class in 0..classes {
                let weight = readout[channel * classes + class][0];
                squares += weight * weight;
            }
            *output = (0.5 * squares).max(f32::MIN_POSITIVE).powf(exponent);
        }
    }

    /// `simple_adam` with the decay of weight `p` multiplied by
    /// `scale[p / stride % width]` and capped at `1 / learning_rate`, so that
    /// the decay alone removes at most the whole weight.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        stride >= 1, width >= 1, scale.len() == width, state.len() >= stride * width,
        gradient.len() == state.len()))]
    pub fn simple_adam_channels<const S: usize>(
        learning_rate: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        correction1: f32,
        correction2: f32,
        stride: usize,
        width: usize,
        scale: &[f32],
        gradient: &[f32],
        mut state: DisjointSlice<[f32; 3]>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let parameter = index.get();
        if let Some(slot) = state.get_mut(index) {
            let g = gradient[parameter];
            slot[1] = beta1 * slot[1] + (1.0 - beta1) * g;
            slot[2] = beta2 * slot[2] + (1.0 - beta2) * g * g;
            let m = slot[1] / correction1;
            let v = slot[2] / correction2;
            let decay = (weight_decay * scale[parameter / stride % width]).min(1.0 / learning_rate);
            slot[0] -= learning_rate * (m / (v.sqrt() + epsilon) + decay * slot[0]);
        }
    }
}

/// Weights with their AdamW moments, and the latest gradient, for one layer.
struct Parameters {
    state: DeviceBuffer<[f32; 3]>,
    gradient: DeviceBuffer<f32>,
}

/// Encoded equations and answer classes resident on the device.
pub struct DeviceTable {
    features: DeviceBuffer<f32>,
    targets: DeviceBuffer<u32>,
    identity: DeviceBuffer<u32>,
    rows: usize,
}

impl DeviceTable {
    pub fn rows(&self) -> usize {
        self.rows
    }
}

/// Validated row indices into one table, such as every minibatch of an epoch.
pub struct DeviceBatches {
    indices: DeviceBuffer<u32>,
    table_rows: usize,
}

impl DeviceBatches {
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

struct Workspace {
    rows: usize,
    inputs: DeviceBuffer<f32>,
    values: Vec<DeviceBuffer<f32>>,
    adjoints: Vec<DeviceBuffer<f32>>,
    logits: DeviceBuffer<f32>,
    logit_gradient: DeviceBuffer<f32>,
    statistics: DeviceBuffer<[f32; 2]>,
}

/// One table's logits and predictions, held on the device while the rest of
/// an evaluation runs.
struct Readback {
    logits: DeviceBuffer<f32>,
    predictions: DeviceBuffer<u32>,
}

/// The logits and predicted class of every row of one table, in row order.
pub struct Evaluation {
    /// `modulus` logits per row, in page-locked memory that the device fills
    /// directly. `GpuSimpleModel::reuse` takes the buffer back.
    pub logits: PinnedHostBuffer<f32>,
    /// The class with the highest logit of each row, the first one on ties.
    pub predictions: Vec<u32>,
}

/// Page-locked host buffers, reused by length, so that copies back skip both
/// the staging copy of pageable memory and fresh allocations.
#[derive(Default)]
struct HostBuffers(Vec<PinnedHostBuffer<f32>>);

impl HostBuffers {
    fn take(&mut self, context: &Arc<CudaContext>, len: usize) -> Result<PinnedHostBuffer<f32>> {
        match self.0.iter().position(|buffer| buffer.len() == len) {
            Some(index) => Ok(self.0.swap_remove(index)),
            None => Ok(PinnedHostBuffer::zeroed(context, len)?),
        }
    }
}

impl Workspace {
    fn new(stream: &Arc<CudaStream>, config: SimpleConfig, rows: usize) -> Result<Self> {
        let (values, adjoints) = buffer_widths(config);
        let buffers = |widths: Vec<usize>| -> Result<Vec<DeviceBuffer<f32>>> {
            widths
                .into_iter()
                .map(|width| Ok(DeviceBuffer::zeroed(stream, rows * width)?))
                .collect()
        };
        Ok(Self {
            rows,
            inputs: DeviceBuffer::zeroed(stream, rows * config.input_features())?,
            values: buffers(values)?,
            adjoints: buffers(adjoints)?,
            logits: DeviceBuffer::zeroed(stream, rows * config.modulus)?,
            logit_gradient: DeviceBuffer::zeroed(stream, rows * config.modulus)?,
            statistics: DeviceBuffer::zeroed(stream, rows)?,
        })
    }
}

/// Row widths of the saved forward values and of the backward scratch buffers.
///
/// MLP and mesh-MLP values are the hidden pre-activations; their gradients alternate
/// between two buffers. A residual MLP saves the streams `h_0..=h_B`, their RMSNorms
/// `n_0..=n_B`, each block's pre-activation `z_b`, and one block output. A bilinear
/// model saves both embeddings and their product.
fn buffer_widths(config: SimpleConfig) -> (Vec<usize>, Vec<usize>) {
    let (width, depth) = (config.width, config.depth);
    match config.architecture {
        Architecture::SharedNeighborMeshMlp => (
            vec![width; 3 * config.embedding_mesh_depth + 3 * depth + 5],
            vec![width; 5],
        ),
        Architecture::Mlp | Architecture::MeshMlp | Architecture::FullMeshMlp => {
            (vec![width; depth], vec![width; 2])
        }
        Architecture::TiedConcatMeshMlp => {
            let mut values = vec![width; depth];
            values.push(2 * width);
            (values, vec![width, width, 2 * width])
        }
        Architecture::ResidualMlp => {
            let hidden = width * config.expansion;
            let mut values = vec![width; 2 * depth + 2];
            values.extend(std::iter::repeat_n(hidden, depth));
            values.push(width);
            (values, vec![width, width, hidden, width, width])
        }
        Architecture::ResidualMeshMlp
        | Architecture::TiedConcatResidualMeshMlp
        | Architecture::TiedRotaryResidualMeshMlp
        | Architecture::TiedRotaryResidualDenseMlp => {
            let mut values = vec![width; 2 * depth + 2 + depth * config.residual_branch_depth()];
            let mut adjoints = vec![width; 5];
            if config.architecture == Architecture::TiedConcatResidualMeshMlp {
                values.push(2 * width);
                adjoints.push(2 * width);
            }
            (values, adjoints)
        }
        Architecture::Bilinear => (vec![width; 3], vec![width; 3]),
    }
}

/// Prepared launches by kernel specialization and launch configuration.
/// Preparing a launch queries the driver and the module's function table;
/// a prepared launch is reused without either, while each launch still checks
/// its buffers against the kernel's contract.
#[derive(Default)]
struct Launches(Mutex<BTreeMap<TypeId, Vec<Prepared>>>);

/// A launch configuration and its `PreparedLaunch` for one specialization.
type Prepared = (LaunchConfig1D, Box<dyn Any + Send>);

impl Launches {
    fn get<C: KernelLaunchContract<Config = LaunchConfig1D> + 'static>(
        &self,
        config: LaunchConfig1D,
        prepare: impl FnOnce(
            LaunchConfig1D,
        ) -> std::result::Result<PreparedLaunch<C>, LaunchContractError>,
    ) -> Result<PreparedLaunch<C>> {
        let mut launches = self.0.lock().map_err(|_| "launch cache poisoned")?;
        let prepared = launches.entry(TypeId::of::<C>()).or_default();
        let cached = prepared
            .iter()
            .filter(|(prepared, _)| *prepared == config)
            .find_map(|(_, launch)| launch.downcast_ref::<PreparedLaunch<C>>());
        if let Some(launch) = cached {
            return Ok(launch.clone());
        }
        let launch = prepare(config)?;
        prepared.push((config, Box::new(launch.clone())));
        Ok(launch)
    }
}

struct Kernels {
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    tiled: dense_tiled::Kernels,
    launches: Launches,
}

// Launch wrappers mirror their kernels' argument lists.
#[allow(clippy::too_many_arguments)]
impl Kernels {
    fn copy_columns(
        &self,
        rows: usize,
        shape: [usize; 4],
        input: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [source, target, offset, copied] = shape;
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module.prepare_simple_copy_columns::<0>(c)
        })?;
        self.module.simple_copy_columns(
            &self.stream,
            &launch,
            rows,
            source,
            target,
            offset,
            copied,
            input,
            out,
        )?;
        Ok(())
    }

    fn pair_rotation<const TRANSPOSE: usize>(
        &self,
        width: usize,
        input: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if width < 2 {
            return Err("pair rotation needs a width of at least two".into());
        }
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module.prepare_simple_pair_rotation::<TRANSPOSE>(c)
        })?;
        self.module.simple_pair_rotation(
            &self.stream,
            &launch,
            input.len() / width,
            width / 2,
            width,
            input,
            out,
        )?;
        Ok(())
    }
    fn two_parent<const ACTIVATION: usize>(
        &self,
        rows: usize,
        layer: TwoParentLayer,
        input: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module.prepare_simple_two_parent::<ACTIVATION>(c)
        })?;
        self.module.simple_two_parent(
            &self.stream,
            &launch,
            rows,
            layer.inputs,
            layer.outputs,
            layer.stride,
            input,
            weights,
            out,
        )?;
        Ok(())
    }

    fn two_parent_input_gradient<const ACTIVATION: usize>(
        &self,
        rows: usize,
        layer: TwoParentLayer,
        gradient: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        before: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module
                .prepare_simple_two_parent_input_gradient::<ACTIVATION>(c)
        })?;
        self.module.simple_two_parent_input_gradient(
            &self.stream,
            &launch,
            rows,
            layer.inputs,
            layer.outputs,
            layer.stride,
            gradient,
            weights,
            before,
            out,
        )?;
        Ok(())
    }

    fn two_parent_weight_gradient<const ACTIVATION: usize>(
        &self,
        rows: usize,
        layer: TwoParentLayer,
        input: &DeviceBuffer<f32>,
        gradient: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module
                .prepare_simple_two_parent_weight_gradient::<ACTIVATION>(c)
        })?;
        self.module.simple_two_parent_weight_gradient(
            &self.stream,
            &launch,
            rows,
            layer.inputs,
            layer.outputs,
            layer.stride,
            input,
            gradient,
            out,
        )?;
        Ok(())
    }

    fn gather(
        &self,
        table: &DeviceTable,
        indices: &DeviceBuffer<u32>,
        offset: usize,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let rows = out.len() / (table.features.len() / table.rows);
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module.prepare_simple_gather::<0>(c)
        })?;
        self.module.simple_gather(
            &self.stream,
            &launch,
            table.features.len() / table.rows,
            rows,
            offset,
            indices,
            &table.features,
            out,
        )?;
        Ok(())
    }

    /// `columns` holds the input row stride and the first feature the layer
    /// reads; `ACTIVATION` applies to those inputs. Dense products of every
    /// size use the register-blocked kernels.
    fn dense<const ACTIVATION: usize>(
        &self,
        layer: Dense,
        rows: usize,
        input: &DeviceBuffer<f32>,
        columns: [usize; 2],
        weights: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.tiled
            .forward::<ACTIVATION>(layer, rows, input, columns, weights, out)
    }

    /// `before` holds the pre-activation values of the layer input.
    fn input_gradient<const ACTIVATION: usize>(
        &self,
        layer: Dense,
        rows: usize,
        gradient: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        before: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.tiled
            .input_gradient::<ACTIVATION>(layer, rows, gradient, weights, before, out)
    }

    fn weight_gradient<const ACTIVATION: usize>(
        &self,
        layer: Dense,
        rows: usize,
        input: &DeviceBuffer<f32>,
        columns: [usize; 2],
        gradient: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.tiled
            .weight_gradient::<ACTIVATION>(layer, rows, input, columns, gradient, out)
    }

    fn concat_embedding(
        &self,
        shape: [usize; 3],
        input: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [rows, classes, width] = shape;
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module.prepare_simple_concat_embedding::<0>(c)
        })?;
        self.module.simple_concat_embedding(
            &self.stream,
            &launch,
            rows,
            classes,
            width,
            input,
            weights,
            out,
        )?;
        Ok(())
    }

    /// E(a) + J E(b), where J rotates each adjacent pair by +90 degrees.
    fn rotary_embedding(
        &self,
        shape: [usize; 3],
        input: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.tiled.rotary_embedding(shape, input, weights, out)
    }

    /// Sum E's input-a, J^T input-b, and transposed-readout derivatives.
    fn rotary_embedding_gradient(
        &self,
        shape: [usize; 3],
        input: &DeviceBuffer<f32>,
        input_gradient: &DeviceBuffer<f32>,
        logit_gradient: &DeviceBuffer<f32>,
        hidden: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.tiled.rotary_embedding_gradient(
            shape,
            input,
            input_gradient,
            logit_gradient,
            hidden,
            out,
        )
    }

    /// z = f(h) E^T, reading the same E state as both operand embeddings.
    fn tied_readout<const ACTIVATION: usize>(
        &self,
        shape: [usize; 3],
        input: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.tiled
            .tied_readout::<ACTIVATION>(shape, input, weights, out)
    }

    fn tied_readout_input_gradient<const ACTIVATION: usize>(
        &self,
        shape: [usize; 3],
        gradient: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        before: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.tiled
            .tied_readout_input_gradient::<ACTIVATION>(shape, gradient, weights, before, out)
    }

    fn shared_embedding_gradient<const ACTIVATION: usize>(
        &self,
        shape: [usize; 3],
        input: &DeviceBuffer<f32>,
        embedded_gradient: &DeviceBuffer<f32>,
        logit_gradient: &DeviceBuffer<f32>,
        hidden: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [rows, classes, width] = shape;
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module
                .prepare_simple_shared_embedding_gradient::<ACTIVATION>(c)
        })?;
        self.module.simple_shared_embedding_gradient(
            &self.stream,
            &launch,
            rows,
            classes,
            width,
            input,
            embedded_gradient,
            logit_gradient,
            hidden,
            out,
        )?;
        Ok(())
    }

    fn combine<const OPERATION: usize>(
        &self,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module.prepare_simple_combine::<OPERATION>(c)
        })?;
        self.module
            .simple_combine(&self.stream, &launch, left, right, out)?;
        Ok(())
    }

    fn activated_sum<const ACTIVATION: usize>(
        &self,
        residual: bool,
        stream: &DeviceBuffer<f32>,
        branch: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module.prepare_simple_activated_sum::<ACTIVATION>(c)
        })?;
        self.module
            .simple_activated_sum(&self.stream, &launch, residual, stream, branch, out)?;
        Ok(())
    }

    fn activation_gradient<const ACTIVATION: usize>(
        &self,
        branch: &DeviceBuffer<f32>,
        gradient: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module
                .prepare_simple_activation_gradient::<ACTIVATION>(c)
        })?;
        self.module
            .simple_activation_gradient(&self.stream, &launch, branch, gradient, out)?;
        Ok(())
    }

    /// `mesh` holds the layer width, the parents per unit, and the stride to
    /// each unit's second parent; `ACTIVATION` applies to the inputs.
    fn mesh<const ACTIVATION: usize>(
        &self,
        rows: usize,
        mesh: [usize; 3],
        input: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [width, parents, stride] = mesh;
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module.prepare_simple_mesh::<ACTIVATION>(c)
        })?;
        self.module.simple_mesh(
            &self.stream,
            &launch,
            rows,
            width,
            parents,
            stride,
            input,
            weights,
            out,
        )?;
        Ok(())
    }

    /// `before` holds the pre-activation values of the mesh layer input.
    fn mesh_input_gradient<const ACTIVATION: usize>(
        &self,
        rows: usize,
        mesh: [usize; 3],
        gradient: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        before: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [width, parents, stride] = mesh;
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module
                .prepare_simple_mesh_input_gradient::<ACTIVATION>(c)
        })?;
        self.module.simple_mesh_input_gradient(
            &self.stream,
            &launch,
            rows,
            width,
            parents,
            stride,
            gradient,
            weights,
            before,
            out,
        )?;
        Ok(())
    }

    fn mesh_weight_gradient<const ACTIVATION: usize>(
        &self,
        rows: usize,
        mesh: [usize; 3],
        input: &DeviceBuffer<f32>,
        gradient: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [width, parents, stride] = mesh;
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module
                .prepare_simple_mesh_weight_gradient::<ACTIVATION>(c)
        })?;
        self.module.simple_mesh_weight_gradient(
            &self.stream,
            &launch,
            rows,
            width,
            parents,
            stride,
            input,
            gradient,
            out,
        )?;
        Ok(())
    }

    /// Forward reads `[x, x, x]`; backward reads `[x, dn, n]` and writes `dx`.
    /// Common widths take one pass; other widths keep `statistics` between
    /// two.
    fn rmsnorm<const BACKWARD: usize>(
        &self,
        rows: usize,
        width: usize,
        inputs: [&DeviceBuffer<f32>; 3],
        statistics: &mut DeviceBuffer<[f32; 2]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        match width {
            16 => self.rmsnorm_rows::<BACKWARD, 16>(rows, inputs, out),
            32 => self.rmsnorm_rows::<BACKWARD, 32>(rows, inputs, out),
            64 => self.rmsnorm_rows::<BACKWARD, 64>(rows, inputs, out),
            128 => self.rmsnorm_rows::<BACKWARD, 128>(rows, inputs, out),
            256 => self.rmsnorm_rows::<BACKWARD, 256>(rows, inputs, out),
            512 => self.rmsnorm_rows::<BACKWARD, 512>(rows, inputs, out),
            1024 => self.rmsnorm_rows::<BACKWARD, 1024>(rows, inputs, out),
            _ => self.rmsnorm_split::<BACKWARD>(rows, width, inputs, statistics, out),
        }
    }

    fn rmsnorm_rows<const BACKWARD: usize, const WIDTH: usize>(
        &self,
        rows: usize,
        inputs: [&DeviceBuffer<f32>; 3],
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [input, gradient, normalized] = inputs;
        let launch = self.launches.get(config::<32>(rows)?, |c| {
            self.module
                .prepare_simple_rmsnorm_rows::<BACKWARD, WIDTH>(c)
        })?;
        self.module.simple_rmsnorm_rows(
            &self.stream,
            &launch,
            rows,
            WIDTH,
            input,
            gradient,
            normalized,
            out,
        )?;
        Ok(())
    }

    fn rmsnorm_split<const BACKWARD: usize>(
        &self,
        rows: usize,
        width: usize,
        inputs: [&DeviceBuffer<f32>; 3],
        statistics: &mut DeviceBuffer<[f32; 2]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [input, gradient, normalized] = inputs;
        let launch = self.launches.get(config::<32>(rows)?, |c| {
            self.module.prepare_simple_rmsnorm_statistics::<BACKWARD>(c)
        })?;
        self.module.simple_rmsnorm_statistics(
            &self.stream,
            &launch,
            rows,
            width,
            input,
            gradient,
            normalized,
            statistics,
        )?;
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module.prepare_simple_rmsnorm_apply::<BACKWARD>(c)
        })?;
        self.module.simple_rmsnorm_apply(
            &self.stream,
            &launch,
            rows,
            width,
            if BACKWARD == 1 { gradient } else { input },
            normalized,
            statistics,
            out,
        )?;
        Ok(())
    }

    /// `statistics` receives each row's maximum logit and softmax denominator.
    fn cross_entropy_gradient(
        &self,
        classes: usize,
        offset: usize,
        indices: &DeviceBuffer<u32>,
        targets: &DeviceBuffer<u32>,
        logits: &DeviceBuffer<f32>,
        statistics: &mut DeviceBuffer<[f32; 2]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let rows = logits.len() / classes;
        let launch = self.launches.get(config::<32>(rows)?, |c| {
            self.module.prepare_simple_softmax_statistics::<0>(c)
        })?;
        self.module.simple_softmax_statistics(
            &self.stream,
            &launch,
            rows,
            classes,
            logits,
            statistics,
        )?;
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module.prepare_simple_cross_entropy_gradient::<0>(c)
        })?;
        self.module.simple_cross_entropy_gradient(
            &self.stream,
            &launch,
            rows,
            classes,
            offset,
            indices,
            targets,
            logits,
            statistics,
            out,
        )?;
        Ok(())
    }

    fn predictions(
        &self,
        classes: usize,
        logits: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        let launch = self.launches.get(config::<32>(out.len())?, |c| {
            self.module.prepare_simple_predictions::<0>(c)
        })?;
        self.module
            .simple_predictions(&self.stream, &launch, out.len(), classes, logits, out)?;
        Ok(())
    }

    fn weights(&self, state: &DeviceBuffer<[f32; 3]>, out: &mut DeviceBuffer<f32>) -> Result<()> {
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module.prepare_simple_weights::<0>(c)
        })?;
        self.module
            .simple_weights(&self.stream, &launch, state, out)?;
        Ok(())
    }

    fn adam(
        &self,
        optimizer: AdamWOptions,
        corrections: [f32; 2],
        group: &mut Parameters,
    ) -> Result<()> {
        let launch = self.launches.get(config::<1>(group.state.len())?, |c| {
            self.module.prepare_simple_adam::<0>(c)
        })?;
        self.module.simple_adam(
            &self.stream,
            &launch,
            optimizer.learning_rate,
            optimizer.weight_decay,
            optimizer.beta1,
            optimizer.beta2,
            optimizer.epsilon,
            corrections[0],
            corrections[1],
            &group.gradient,
            &mut group.state,
        )?;
        Ok(())
    }

    /// The decay multipliers of a one-layer MLP's hidden channels from its
    /// input layer and readout.
    fn channel_decay(
        &self,
        layers: [Dense; 2],
        exponent: f32,
        input: &DeviceBuffer<[f32; 3]>,
        readout: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self.launches.get(config::<1>(out.len())?, |c| {
            self.module.prepare_simple_channel_decay::<0>(c)
        })?;
        self.module.simple_channel_decay(
            &self.stream,
            &launch,
            layers[0].inputs,
            layers[0].outputs,
            layers[1].outputs,
            exponent,
            input,
            readout,
            out,
        )?;
        Ok(())
    }

    /// `channels` holds the parameters per channel step and the channel count:
    /// weight `p` decays with `scale[p / stride % width]`.
    fn adam_channels(
        &self,
        optimizer: AdamWOptions,
        corrections: [f32; 2],
        channels: [usize; 2],
        scale: &DeviceBuffer<f32>,
        group: &mut Parameters,
    ) -> Result<()> {
        let [stride, width] = channels;
        let launch = self.launches.get(config::<1>(group.state.len())?, |c| {
            self.module.prepare_simple_adam_channels::<0>(c)
        })?;
        self.module.simple_adam_channels(
            &self.stream,
            &launch,
            optimizer.learning_rate,
            optimizer.weight_decay,
            optimizer.beta1,
            optimizer.beta2,
            optimizer.epsilon,
            corrections[0],
            corrections[1],
            stride,
            width,
            scale,
            &group.gradient,
            &mut group.state,
        )?;
        Ok(())
    }
}

/// An attention-free model trained entirely on the device. Parameters,
/// AdamW moments, encoded splits, and epoch indices stay resident; a training
/// step copies nothing to or from the host.
pub struct GpuSimpleModel {
    config: SimpleConfig,
    layers: Vec<Dense>,
    kernels: Kernels,
    parameters: Vec<Parameters>,
    /// The running sum and one layer's gradient for tied mesh layers.
    mesh_gradients: Vec<DeviceBuffer<f32>>,
    /// Hidden-channel decay multipliers when weight decay imitates more factors.
    channel_decay: Option<DeviceBuffer<f32>>,
    /// Each group's weights without their moments, for copying back.
    weights: Vec<DeviceBuffer<f32>>,
    workspaces: Vec<Workspace>,
    /// The results of the latest evaluation, reused by the next one.
    readbacks: Vec<Readback>,
    host_buffers: HostBuffers,
    beta1_power: f32,
    beta2_power: f32,
}

impl GpuSimpleModel {
    pub fn new(device: usize, config: SimpleConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        let context = CudaContext::new(device)?;
        let stream = context.default_stream();
        let module = kernels::load(&context)?;
        let tiled = dense_tiled::Kernels::new(&context, &stream)?;
        let layers = config.layers();
        let initial = config.initial_parameters(seed);
        let groups = config.groups();
        let mut parameters = Vec::with_capacity(groups.len());
        let mut weights = Vec::with_capacity(groups.len());
        let mut offset = 0;
        for size in groups {
            let state: Vec<_> = initial[offset..offset + size]
                .iter()
                .map(|&p| [p, 0.0, 0.0])
                .collect();
            parameters.push(Parameters {
                state: DeviceBuffer::from_host(&stream, &state)?,
                gradient: DeviceBuffer::zeroed(&stream, size)?,
            });
            weights.push(DeviceBuffer::zeroed(&stream, size)?);
            offset += size;
        }
        let tied = config.stored_mesh_layers() < config.mesh_layers()
            || config.architecture == Architecture::SharedNeighborMeshMlp;
        let mesh_gradients = (0..if tied { 2 } else { 0 })
            .map(|_| {
                let size = config.connectivity.parents() * config.width;
                Ok(DeviceBuffer::zeroed(&stream, size)?)
            })
            .collect::<Result<_>>()?;
        let channel_decay = if config.decay_factors == 2 {
            None
        } else {
            Some(DeviceBuffer::zeroed(&stream, config.width)?)
        };
        Ok(Self {
            config,
            layers,
            kernels: Kernels {
                stream,
                module,
                tiled,
                launches: Launches::default(),
            },
            parameters,
            mesh_gradients,
            channel_decay,
            weights,
            workspaces: Vec::new(),
            readbacks: Vec::new(),
            host_buffers: HostBuffers::default(),
            beta1_power: 1.0,
            beta2_power: 1.0,
        })
    }

    pub fn config(&self) -> SimpleConfig {
        self.config
    }

    /// The weights alone, separated from their moments on the device so that
    /// a third of the state crosses the bus.
    pub fn parameters(&mut self) -> Result<Vec<f32>> {
        let mut values = vec![0.0; self.config.parameter_count()];
        self.copy_parameters(&mut values)?;
        Ok(values)
    }

    /// `parameters` in page-locked memory that `reuse` takes back.
    pub fn pinned_parameters(&mut self) -> Result<PinnedHostBuffer<f32>> {
        let context = self.kernels.stream.context();
        let mut values = self
            .host_buffers
            .take(context, self.config.parameter_count())?;
        self.copy_parameters(&mut values)?;
        Ok(values)
    }

    /// Takes back the page-locked logits or parameters of earlier results,
    /// so that later copies skip allocating them.
    pub fn reuse(&mut self, buffer: PinnedHostBuffer<f32>) {
        self.host_buffers.0.push(buffer);
    }

    fn copy_parameters(&mut self, values: &mut [f32]) -> Result<()> {
        for (group, weights) in self.parameters.iter().zip(&mut self.weights) {
            self.kernels.weights(&group.state, weights)?;
        }
        let mut offset = 0;
        for weights in &self.weights {
            let end = offset + weights.len();
            weights.copy_to_host(&self.kernels.stream, &mut values[offset..end])?;
            offset = end;
        }
        validate_finite(values, "parameters")
    }

    /// Sets weights and resets AdamW's moments and step count.
    pub fn set_parameters(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.config.parameter_count() {
            return Err("parameter count does not match the model".into());
        }
        validate_finite(values, "parameters")?;
        let mut offset = 0;
        for group in &mut self.parameters {
            let size = group.gradient.len();
            let state: Vec<_> = values[offset..offset + size]
                .iter()
                .map(|&p| [p, 0.0, 0.0])
                .collect();
            group.state.copy_from_host(&self.kernels.stream, &state)?;
            offset += size;
        }
        self.beta1_power = 1.0;
        self.beta2_power = 1.0;
        Ok(())
    }

    /// Uploads encoded rows from `SimpleConfig::encode` and their answer classes.
    pub fn upload_table(&self, features: &[f32], targets: &[u32]) -> Result<DeviceTable> {
        let rows = targets.len();
        if rows == 0 || rows > MAX_SIMPLE_ROWS {
            return Err(format!("tables need 1..={MAX_SIMPLE_ROWS} rows").into());
        }
        if features.len() != rows * self.config.input_features() {
            return Err("each table row needs the model's input features".into());
        }
        validate_finite(features, "features")?;
        if targets
            .iter()
            .any(|&target| target as usize >= self.config.modulus)
        {
            return Err("answer classes must be below the modulus".into());
        }
        let stream = &self.kernels.stream;
        let identity: Vec<_> = (0..rows as u32).collect();
        Ok(DeviceTable {
            features: DeviceBuffer::from_host(stream, features)?,
            targets: DeviceBuffer::from_host(stream, targets)?,
            identity: DeviceBuffer::from_host(stream, &identity)?,
            rows,
        })
    }

    /// Uploads row indices for later minibatches, checking each against `table`.
    pub fn upload_batches(&self, table: &DeviceTable, indices: &[u32]) -> Result<DeviceBatches> {
        if indices.is_empty() || indices.len() > MAX_SIMPLE_ROWS {
            return Err(format!("batches need 1..={MAX_SIMPLE_ROWS} indices").into());
        }
        if indices.iter().any(|&index| index as usize >= table.rows) {
            return Err("batch index is outside the table".into());
        }
        Ok(DeviceBatches {
            indices: DeviceBuffer::from_host(&self.kernels.stream, indices)?,
            table_rows: table.rows,
        })
    }

    /// Logits for every table row in order, `modulus` classes per row.
    pub fn logits(&mut self, table: &DeviceTable) -> Result<Vec<f32>> {
        let [evaluation] = self.evaluate([table])?;
        let logits = evaluation.logits.to_vec();
        self.reuse(evaluation.logits);
        Ok(logits)
    }

    /// Logits and predictions for every row of each table. All tables run on
    /// the device before the first copy back, so the host waits once.
    pub fn evaluate<const N: usize>(
        &mut self,
        tables: [&DeviceTable; N],
    ) -> Result<[Evaluation; N]> {
        let classes = self.config.modulus;
        let mut readbacks = Vec::with_capacity(N);
        for table in tables {
            let cached = self
                .readbacks
                .iter()
                .position(|readback| readback.predictions.len() == table.rows);
            readbacks.push(match cached {
                Some(index) => self.readbacks.remove(index),
                None => Readback {
                    logits: DeviceBuffer::zeroed(&self.kernels.stream, table.rows * classes)?,
                    predictions: DeviceBuffer::zeroed(&self.kernels.stream, table.rows)?,
                },
            });
        }
        let result = (|| -> Result<Vec<Evaluation>> {
            for (table, readback) in tables.into_iter().zip(&mut readbacks) {
                let mut workspace = self.take_workspace(table.rows)?;
                let queued = (|| -> Result<()> {
                    self.kernels
                        .gather(table, &table.identity, 0, &mut workspace.inputs)?;
                    self.forward(&mut workspace)?;
                    self.kernels.predictions(
                        classes,
                        &workspace.logits,
                        &mut readback.predictions,
                    )?;
                    // Every forward pass overwrites all logits, so the
                    // workspace may continue with the previous readback buffer.
                    std::mem::swap(&mut workspace.logits, &mut readback.logits);
                    Ok(())
                })();
                self.workspaces.push(workspace);
                queued?;
            }
            let stream = &self.kernels.stream;
            let mut evaluations = Vec::with_capacity(N);
            for readback in &readbacks {
                let predictions = readback.predictions.to_host_vec(stream)?;
                // The predictions kernel marks rows that hold a value that
                // is not finite.
                if predictions.contains(&u32::MAX) {
                    return Err("logits must contain only finite values".into());
                }
                let mut logits = self
                    .host_buffers
                    .take(stream.context(), readback.logits.len())?;
                readback.logits.copy_to_host(stream, &mut logits)?;
                evaluations.push(Evaluation {
                    logits,
                    predictions,
                });
            }
            Ok(evaluations)
        })();
        self.readbacks = readbacks;
        result?
            .try_into()
            .map_err(|_| "evaluation lost a table".into())
    }

    /// The mean answer cross-entropy and every parameter gradient on the
    /// selected rows. Diagnostic only: training never copies gradients.
    pub fn loss_and_gradient(
        &mut self,
        table: &DeviceTable,
        indices: &[u32],
    ) -> Result<(f64, Vec<f64>)> {
        let batches = self.upload_batches(table, indices)?;
        let logits = self.gradients(table, &batches, 0, indices.len(), true)?;
        validate_finite(&logits, "logits")?;
        let targets = table.targets.to_host_vec(&self.kernels.stream)?;
        let mut loss = 0.0;
        for (row, &index) in logits.chunks_exact(self.config.modulus).zip(indices) {
            let maximum = row
                .iter()
                .fold(f64::NEG_INFINITY, |a, &x| a.max(f64::from(x)));
            let sum = row
                .iter()
                .map(|&x| (f64::from(x) - maximum).exp())
                .sum::<f64>();
            loss += sum.ln() + maximum - f64::from(row[targets[index as usize] as usize]);
        }
        let mut gradient = Vec::with_capacity(self.config.parameter_count());
        for group in &self.parameters {
            let values = group.gradient.to_host_vec(&self.kernels.stream)?;
            validate_finite(&values, "gradients")?;
            gradient.extend(values.into_iter().map(f64::from));
        }
        Ok((loss / indices.len() as f64, gradient))
    }

    /// One AdamW step on rows `batches[offset..offset + rows]` of `table`.
    /// Repeated calls continue the same optimizer state.
    pub fn train_batch(
        &mut self,
        table: &DeviceTable,
        batches: &DeviceBatches,
        offset: usize,
        rows: usize,
        optimizer: AdamWOptions,
    ) -> Result<()> {
        optimizer.validate()?;
        self.gradients(table, batches, offset, rows, false)?;
        self.beta1_power *= optimizer.beta1;
        self.beta2_power *= optimizer.beta2;
        let corrections = [1.0 - self.beta1_power, 1.0 - self.beta2_power];
        match &mut self.channel_decay {
            // Mesh layers follow the dense layers and may train faster.
            None => {
                let mesh = AdamWOptions {
                    learning_rate: optimizer.learning_rate
                        * self.config.mesh_learning_rate_multiplier as f32,
                    ..optimizer
                };
                for (index, group) in self.parameters.iter_mut().enumerate() {
                    let options = if index < self.layers.len() {
                        optimizer
                    } else {
                        mesh
                    };
                    self.kernels.adam(options, corrections, group)?;
                }
            }
            // Both layers decay by multipliers of the weights before this step.
            Some(scale) => {
                let (width, classes) = (self.config.width, self.config.modulus);
                let [input, readout] = self.parameters.get_disjoint_mut([0, 1])?;
                self.kernels.channel_decay(
                    [self.layers[0], self.layers[1]],
                    self.config.decay_exponent(),
                    &input.state,
                    &readout.state,
                    scale,
                )?;
                let k = &self.kernels;
                k.adam_channels(optimizer, corrections, [1, width], scale, input)?;
                k.adam_channels(optimizer, corrections, [classes, width], scale, readout)?;
            }
        }
        Ok(())
    }

    fn take_workspace(&mut self, rows: usize) -> Result<Workspace> {
        if let Some(index) = self.workspaces.iter().position(|w| w.rows == rows) {
            return Ok(self.workspaces.remove(index));
        }
        if self.workspaces.len() == WORKSPACE_CAPACITY {
            self.workspaces.remove(0);
        }
        Workspace::new(&self.kernels.stream, self.config, rows)
    }

    /// Computes every layer's gradient; returns the logits only when requested.
    fn gradients(
        &mut self,
        table: &DeviceTable,
        batches: &DeviceBatches,
        offset: usize,
        rows: usize,
        read_logits: bool,
    ) -> Result<Vec<f32>> {
        if batches.table_rows != table.rows {
            return Err("batch indices belong to a table with another row count".into());
        }
        if rows == 0
            || rows > MAX_SIMPLE_ROWS
            || offset
                .checked_add(rows)
                .is_none_or(|end| end > batches.len())
        {
            return Err("minibatch is outside the uploaded indices".into());
        }
        let mut workspace = self.take_workspace(rows)?;
        let result = (|| {
            self.kernels
                .gather(table, &batches.indices, offset, &mut workspace.inputs)?;
            self.forward(&mut workspace)?;
            self.kernels.cross_entropy_gradient(
                self.config.modulus,
                offset,
                &batches.indices,
                &table.targets,
                &workspace.logits,
                &mut workspace.statistics,
                &mut workspace.logit_gradient,
            )?;
            self.backward(&mut workspace)?;
            if read_logits {
                Ok(workspace.logits.to_host_vec(&self.kernels.stream)?)
            } else {
                Ok(Vec::new())
            }
        })();
        self.workspaces.push(workspace);
        result
    }

    fn forward(&self, workspace: &mut Workspace) -> Result<()> {
        match self.config.activation {
            Activation::Relu => self.forward_with::<RELU>(workspace),
            Activation::Square => self.forward_with::<SQUARE>(workspace),
        }
    }

    fn backward(&mut self, workspace: &mut Workspace) -> Result<()> {
        match self.config.activation {
            Activation::Relu => self.backward_with::<RELU>(workspace),
            Activation::Square => self.backward_with::<SQUARE>(workspace),
        }
    }

    fn forward_with<const ACTIVATION: usize>(&self, workspace: &mut Workspace) -> Result<()> {
        let c = self.config;
        let rows = workspace.rows;
        let (k, layers) = (&self.kernels, &self.layers);
        let state = |layer: usize| &self.parameters[layer].state;
        let input = [c.input_features(), 0];
        let values = &mut workspace.values;
        match c.architecture {
            Architecture::SharedNeighborMeshMlp => {
                let embedding_depth = c.embedding_mesh_depth;
                let b = embedding_depth + 1;
                let (rotation, joined) = (2 * embedding_depth + 2, 2 * embedding_depth + 3);
                let normalized = joined + c.depth + 1;
                let branches = normalized + c.depth;
                let final_norm = branches + c.depth;
                let readout = final_norm + 1;
                let mesh = [c.width, 2, 1];
                k.copy_columns(
                    rows,
                    [2 * c.modulus, c.width, 0, c.modulus],
                    &workspace.inputs,
                    &mut values[0],
                )?;
                k.copy_columns(
                    rows,
                    [2 * c.modulus, c.width, c.modulus, c.modulus],
                    &workspace.inputs,
                    &mut values[b],
                )?;
                for layer in 0..embedding_depth {
                    for start in [0, b] {
                        let [before, after] =
                            values.get_disjoint_mut([start + layer, start + layer + 1])?;
                        k.mesh::<IDENTITY>(rows, mesh, before, state(layer), after)?;
                    }
                }
                let [before, after] = values.get_disjoint_mut([b + embedding_depth, rotation])?;
                k.pair_rotation::<0>(c.width, before, after)?;
                let [a, rb, out] = values.get_disjoint_mut([embedding_depth, rotation, joined])?;
                k.combine::<ADD>(a, rb, out)?;
                for (layer, shape) in c.two_parent_layers().into_iter().enumerate() {
                    let [before, after] =
                        values.get_disjoint_mut([joined + layer, normalized + layer])?;
                    k.rmsnorm::<0>(
                        rows,
                        c.width,
                        [before, before, before],
                        &mut workspace.statistics,
                        after,
                    )?;
                    let [before, after] =
                        values.get_disjoint_mut([normalized + layer, branches + layer])?;
                    k.two_parent::<IDENTITY>(
                        rows,
                        shape,
                        before,
                        state(embedding_depth + layer),
                        after,
                    )?;
                    let [stream, branch, after] = values.get_disjoint_mut([
                        joined + layer,
                        branches + layer,
                        joined + layer + 1,
                    ])?;
                    k.activated_sum::<ACTIVATION>(true, stream, branch, after)?;
                }
                let [before, after] = values.get_disjoint_mut([joined + c.depth, final_norm])?;
                k.rmsnorm::<0>(
                    rows,
                    c.width,
                    [before, before, before],
                    &mut workspace.statistics,
                    after,
                )?;
                for layer in 0..embedding_depth {
                    let previous = if layer == 0 {
                        final_norm
                    } else {
                        readout + layer - 1
                    };
                    let [before, after] = values.get_disjoint_mut([previous, readout + layer])?;
                    // The input-gradient kernel is exactly a linear mesh's transpose.
                    k.mesh_input_gradient::<IDENTITY>(
                        rows,
                        mesh,
                        before,
                        state(embedding_depth - layer - 1),
                        before,
                        after,
                    )?;
                }
                k.copy_columns(
                    rows,
                    [c.width, c.modulus, 0, c.modulus],
                    &values[readout + embedding_depth - 1],
                    &mut workspace.logits,
                )?;
            }
            Architecture::FullMeshMlp => {
                let sparse = c.two_parent_layers();
                k.two_parent::<IDENTITY>(
                    rows,
                    sparse[0],
                    &workspace.inputs,
                    state(0),
                    &mut values[0],
                )?;
                for (layer, &shape) in sparse.iter().enumerate().take(c.depth).skip(1) {
                    let [before, after] = values.get_disjoint_mut([layer - 1, layer])?;
                    k.two_parent::<ACTIVATION>(rows, shape, before, state(layer), after)?;
                }
                k.two_parent::<ACTIVATION>(
                    rows,
                    sparse[c.depth],
                    &values[c.depth - 1],
                    state(c.depth),
                    &mut workspace.logits,
                )?;
            }
            Architecture::Mlp => {
                let depth = c.depth;
                k.dense::<IDENTITY>(
                    layers[0],
                    rows,
                    &workspace.inputs,
                    input,
                    state(0),
                    &mut values[0],
                )?;
                for (layer, &shape) in layers.iter().enumerate().take(depth).skip(1) {
                    let [before, after] = values.get_disjoint_mut([layer - 1, layer])?;
                    k.dense::<ACTIVATION>(shape, rows, before, [c.width, 0], state(layer), after)?;
                }
                k.dense::<ACTIVATION>(
                    layers[depth],
                    rows,
                    &values[depth - 1],
                    [c.width, 0],
                    state(depth),
                    &mut workspace.logits,
                )?;
            }
            Architecture::MeshMlp | Architecture::TiedConcatMeshMlp => {
                // Groups: the input layer, the readout, then stored mesh layer l at 2 + l.
                let depth = c.depth;
                let tied = c.architecture == Architecture::TiedConcatMeshMlp;
                if tied {
                    k.concat_embedding(
                        [rows, c.modulus, c.width],
                        &workspace.inputs,
                        state(0),
                        &mut values[depth],
                    )?;
                    let [joined, hidden] = values.get_disjoint_mut([depth, 0])?;
                    k.dense::<IDENTITY>(
                        layers[1],
                        rows,
                        joined,
                        [2 * c.width, 0],
                        state(1),
                        hidden,
                    )?;
                } else {
                    k.dense::<IDENTITY>(
                        layers[0],
                        rows,
                        &workspace.inputs,
                        input,
                        state(0),
                        &mut values[0],
                    )?;
                }
                for layer in 1..depth {
                    let mesh = [c.width, c.connectivity.parents(), c.mesh_stride(layer - 1)];
                    let group = 2 + c.stored_mesh_layer(layer - 1);
                    let [before, after] = values.get_disjoint_mut([layer - 1, layer])?;
                    k.mesh::<ACTIVATION>(rows, mesh, before, state(group), after)?;
                }
                if tied {
                    k.tied_readout::<ACTIVATION>(
                        [rows, c.modulus, c.width],
                        &values[depth - 1],
                        state(0),
                        &mut workspace.logits,
                    )?;
                } else {
                    k.dense::<ACTIVATION>(
                        layers[1],
                        rows,
                        &values[depth - 1],
                        [c.width, 0],
                        state(1),
                        &mut workspace.logits,
                    )?;
                }
            }
            Architecture::ResidualMeshMlp
            | Architecture::TiedConcatResidualMeshMlp
            | Architecture::TiedRotaryResidualMeshMlp
            | Architecture::TiedRotaryResidualDenseMlp => {
                let blocks = c.depth;
                let branch_depth = c.residual_branch_depth();
                let (normalized, mesh_values) = (blocks + 1, 2 * blocks + 2);
                let tied = c.architecture.tied_embeddings();
                let mesh_group = layers.len();
                if c.architecture.rotary_embeddings() {
                    k.rotary_embedding(
                        [rows, c.modulus, c.width],
                        &workspace.inputs,
                        state(0),
                        &mut values[0],
                    )?;
                } else if tied {
                    let joined = mesh_values + blocks * branch_depth;
                    k.concat_embedding(
                        [rows, c.modulus, c.width],
                        &workspace.inputs,
                        state(0),
                        &mut values[joined],
                    )?;
                    let [joined, hidden] = values.get_disjoint_mut([joined, 0])?;
                    k.dense::<IDENTITY>(
                        layers[1],
                        rows,
                        joined,
                        [2 * c.width, 0],
                        state(1),
                        hidden,
                    )?;
                } else {
                    k.dense::<IDENTITY>(
                        layers[0],
                        rows,
                        &workspace.inputs,
                        input,
                        state(0),
                        &mut values[0],
                    )?;
                }
                for block in 0..blocks {
                    let [h, n] = values.get_disjoint_mut([block, normalized + block])?;
                    k.rmsnorm::<0>(rows, c.width, [&*h, &*h, &*h], &mut workspace.statistics, n)?;
                    for layer in 0..branch_depth {
                        let mesh_layer = block * branch_depth + layer;
                        let mesh = [c.width, c.connectivity.parents(), c.mesh_stride(mesh_layer)];
                        let after = mesh_values + mesh_layer;
                        let before = if layer == 0 {
                            normalized + block
                        } else {
                            after - 1
                        };
                        let [before, after] = values.get_disjoint_mut([before, after])?;
                        if c.architecture.residual_dense() {
                            k.dense::<IDENTITY>(
                                layers[1 + block],
                                rows,
                                before,
                                [c.width, 0],
                                state(1 + block),
                                after,
                            )?;
                        } else if layer == 0 {
                            k.mesh::<IDENTITY>(
                                rows,
                                mesh,
                                before,
                                state(mesh_group + mesh_layer),
                                after,
                            )?;
                        } else {
                            k.mesh::<ACTIVATION>(
                                rows,
                                mesh,
                                before,
                                state(mesh_group + mesh_layer),
                                after,
                            )?;
                        }
                    }
                    let branch = mesh_values + (block + 1) * branch_depth - 1;
                    let [h, branch, next] = values.get_disjoint_mut([block, branch, block + 1])?;
                    if c.residual_branch_activation() {
                        k.activated_sum::<ACTIVATION>(c.dense_residual, h, branch, next)?;
                    } else {
                        k.combine::<ADD>(h, branch, next)?;
                    }
                }
                let [h, n] = values.get_disjoint_mut([blocks, normalized + blocks])?;
                k.rmsnorm::<0>(rows, c.width, [&*h, &*h, &*h], &mut workspace.statistics, n)?;
                if tied {
                    k.tied_readout::<IDENTITY>(
                        [rows, c.modulus, c.width],
                        &values[normalized + blocks],
                        state(0),
                        &mut workspace.logits,
                    )?;
                } else {
                    k.dense::<IDENTITY>(
                        layers[1],
                        rows,
                        &values[normalized + blocks],
                        [c.width, 0],
                        state(1),
                        &mut workspace.logits,
                    )?;
                }
            }
            Architecture::ResidualMlp => {
                let blocks = c.depth;
                let hidden = c.width * c.expansion;
                // Value indices: h_k = k, n_k = blocks + 1 + k, z_b = 2 blocks + 2 + b.
                let (normalized, expanded) = (blocks + 1, 2 * blocks + 2);
                let branch = 3 * blocks + 2;
                k.dense::<IDENTITY>(
                    layers[0],
                    rows,
                    &workspace.inputs,
                    input,
                    state(0),
                    &mut values[0],
                )?;
                for block in 0..blocks {
                    let (up, down) = (1 + 2 * block, 2 + 2 * block);
                    let [h, n] = values.get_disjoint_mut([block, normalized + block])?;
                    k.rmsnorm::<0>(rows, c.width, [&*h, &*h, &*h], &mut workspace.statistics, n)?;
                    let [n, z] = values.get_disjoint_mut([normalized + block, expanded + block])?;
                    k.dense::<IDENTITY>(layers[up], rows, n, [c.width, 0], state(up), z)?;
                    let [z, y] = values.get_disjoint_mut([expanded + block, branch])?;
                    k.dense::<ACTIVATION>(layers[down], rows, z, [hidden, 0], state(down), y)?;
                    let [h, y, next] = values.get_disjoint_mut([block, branch, block + 1])?;
                    k.combine::<ADD>(h, y, next)?;
                }
                let [h, n] = values.get_disjoint_mut([blocks, normalized + blocks])?;
                k.rmsnorm::<0>(rows, c.width, [&*h, &*h, &*h], &mut workspace.statistics, n)?;
                let readout = 1 + 2 * blocks;
                k.dense::<IDENTITY>(
                    layers[readout],
                    rows,
                    &values[normalized + blocks],
                    [c.width, 0],
                    state(readout),
                    &mut workspace.logits,
                )?;
            }
            Architecture::Bilinear => {
                let features = c.operand_features();
                let [left, right, product] = values.get_disjoint_mut([0, 1, 2])?;
                k.dense::<IDENTITY>(
                    layers[0],
                    rows,
                    &workspace.inputs,
                    [2 * features, 0],
                    state(0),
                    left,
                )?;
                k.dense::<IDENTITY>(
                    layers[1],
                    rows,
                    &workspace.inputs,
                    [2 * features, features],
                    state(1),
                    right,
                )?;
                k.combine::<PRODUCT>(left, right, product)?;
                k.dense::<IDENTITY>(
                    layers[2],
                    rows,
                    product,
                    [c.width, 0],
                    state(2),
                    &mut workspace.logits,
                )?;
            }
        }
        Ok(())
    }

    fn backward_with<const ACTIVATION: usize>(&mut self, workspace: &mut Workspace) -> Result<()> {
        let c = self.config;
        let rows = workspace.rows;
        let (k, layers, parameters) = (&self.kernels, &self.layers, &mut self.parameters);
        let mesh_gradients = &mut self.mesh_gradients;
        let input = [c.input_features(), 0];
        let values = &workspace.values;
        let adjoints = &mut workspace.adjoints;
        let logits = &workspace.logit_gradient;
        match c.architecture {
            Architecture::SharedNeighborMeshMlp => {
                let embedding_depth = c.embedding_mesh_depth;
                let b = embedding_depth + 1;
                let joined = 2 * embedding_depth + 3;
                let normalized = joined + c.depth + 1;
                let branches = normalized + c.depth;
                let final_norm = branches + c.depth;
                let readout = final_norm + 1;
                let mesh = [c.width, 2, 1];
                let mut upstream = 0;
                k.copy_columns(
                    rows,
                    [c.modulus, c.width, 0, c.modulus],
                    logits,
                    &mut adjoints[0],
                )?;
                for layer in (0..embedding_depth).rev() {
                    let group = embedding_depth - layer - 1;
                    let before = if layer == 0 {
                        final_norm
                    } else {
                        readout + layer - 1
                    };
                    // dM for X M^T is the ordinary mesh weight gradient with
                    // its input/upstream operands exchanged. This initializes
                    // the shared gradient; a and b add to it below.
                    k.mesh_weight_gradient::<IDENTITY>(
                        rows,
                        mesh,
                        &adjoints[upstream],
                        &values[before],
                        &mut parameters[group].gradient,
                    )?;
                    let [before, after] = adjoints.get_disjoint_mut([upstream, 1 - upstream])?;
                    k.mesh::<IDENTITY>(rows, mesh, before, &parameters[group].state, after)?;
                    upstream = 1 - upstream;
                }
                let [before, after] = adjoints.get_disjoint_mut([upstream, 1 - upstream])?;
                k.rmsnorm::<1>(
                    rows,
                    c.width,
                    [&values[joined + c.depth], &*before, &values[final_norm]],
                    &mut workspace.statistics,
                    after,
                )?;
                upstream = 1 - upstream;
                let sparse = c.two_parent_layers();
                for layer in (0..c.depth).rev() {
                    let group = embedding_depth + layer;
                    let [gradient, dz] = adjoints.get_disjoint_mut([upstream, 2])?;
                    k.activation_gradient::<ACTIVATION>(&values[branches + layer], gradient, dz)?;
                    k.two_parent_weight_gradient::<IDENTITY>(
                        rows,
                        sparse[layer],
                        &values[normalized + layer],
                        &adjoints[2],
                        &mut parameters[group].gradient,
                    )?;
                    let [dz, dn] = adjoints.get_disjoint_mut([2, 3])?;
                    k.two_parent_input_gradient::<IDENTITY>(
                        rows,
                        sparse[layer],
                        dz,
                        &parameters[group].state,
                        &values[normalized + layer],
                        dn,
                    )?;
                    let [dn, dx] = adjoints.get_disjoint_mut([3, 4])?;
                    k.rmsnorm::<1>(
                        rows,
                        c.width,
                        [&values[joined + layer], &*dn, &values[normalized + layer]],
                        &mut workspace.statistics,
                        dx,
                    )?;
                    let [skip, branch, out] =
                        adjoints.get_disjoint_mut([upstream, 4, 1 - upstream])?;
                    k.combine::<ADD>(skip, branch, out)?;
                    upstream = 1 - upstream;
                }
                let [gradient, saved] = adjoints.get_disjoint_mut([upstream, 2])?;
                k.copy_columns(rows, [c.width, c.width, 0, c.width], gradient, saved)?;
                for start in [0, b] {
                    let [saved, out] = adjoints.get_disjoint_mut([2, 0])?;
                    if start == 0 {
                        k.copy_columns(rows, [c.width, c.width, 0, c.width], saved, out)?;
                    } else {
                        k.pair_rotation::<1>(c.width, saved, out)?;
                    }
                    upstream = 0;
                    for layer in (0..embedding_depth).rev() {
                        k.mesh_weight_gradient::<IDENTITY>(
                            rows,
                            mesh,
                            &values[start + layer],
                            &adjoints[upstream],
                            &mut mesh_gradients[0],
                        )?;
                        let [added, sum] = mesh_gradients.get_disjoint_mut([0, 1])?;
                        k.combine::<ADD>(&parameters[layer].gradient, added, sum)?;
                        std::mem::swap(&mut parameters[layer].gradient, sum);
                        if layer > 0 {
                            let [before, after] =
                                adjoints.get_disjoint_mut([upstream, 1 - upstream])?;
                            k.mesh_input_gradient::<IDENTITY>(
                                rows,
                                mesh,
                                before,
                                &parameters[layer].state,
                                &values[start + layer],
                                after,
                            )?;
                            upstream = 1 - upstream;
                        }
                    }
                }
            }
            Architecture::FullMeshMlp => {
                let sparse = c.two_parent_layers();
                let depth = c.depth;
                k.two_parent_weight_gradient::<ACTIVATION>(
                    rows,
                    sparse[depth],
                    &values[depth - 1],
                    logits,
                    &mut parameters[depth].gradient,
                )?;
                k.two_parent_input_gradient::<ACTIVATION>(
                    rows,
                    sparse[depth],
                    logits,
                    &parameters[depth].state,
                    &values[depth - 1],
                    &mut adjoints[(depth - 1) % 2],
                )?;
                for layer in (1..depth).rev() {
                    let [upstream, target] =
                        adjoints.get_disjoint_mut([layer % 2, (layer - 1) % 2])?;
                    k.two_parent_weight_gradient::<ACTIVATION>(
                        rows,
                        sparse[layer],
                        &values[layer - 1],
                        upstream,
                        &mut parameters[layer].gradient,
                    )?;
                    k.two_parent_input_gradient::<ACTIVATION>(
                        rows,
                        sparse[layer],
                        upstream,
                        &parameters[layer].state,
                        &values[layer - 1],
                        target,
                    )?;
                }
                k.two_parent_weight_gradient::<IDENTITY>(
                    rows,
                    sparse[0],
                    &workspace.inputs,
                    &adjoints[0],
                    &mut parameters[0].gradient,
                )?;
            }
            Architecture::Mlp => {
                // The gradient of hidden pre-activation z_k lives in adjoints[k % 2].
                let depth = c.depth;
                k.weight_gradient::<ACTIVATION>(
                    layers[depth],
                    rows,
                    &values[depth - 1],
                    [c.width, 0],
                    logits,
                    &mut parameters[depth].gradient,
                )?;
                k.input_gradient::<ACTIVATION>(
                    layers[depth],
                    rows,
                    logits,
                    &parameters[depth].state,
                    &values[depth - 1],
                    &mut adjoints[(depth - 1) % 2],
                )?;
                for layer in (1..depth).rev() {
                    let [upstream, target] =
                        adjoints.get_disjoint_mut([layer % 2, (layer - 1) % 2])?;
                    k.weight_gradient::<ACTIVATION>(
                        layers[layer],
                        rows,
                        &values[layer - 1],
                        [c.width, 0],
                        upstream,
                        &mut parameters[layer].gradient,
                    )?;
                    k.input_gradient::<ACTIVATION>(
                        layers[layer],
                        rows,
                        upstream,
                        &parameters[layer].state,
                        &values[layer - 1],
                        target,
                    )?;
                }
                k.weight_gradient::<IDENTITY>(
                    layers[0],
                    rows,
                    &workspace.inputs,
                    input,
                    &adjoints[0],
                    &mut parameters[0].gradient,
                )?;
            }
            Architecture::MeshMlp | Architecture::TiedConcatMeshMlp => {
                // As in the MLP, the gradient of z_k lives in adjoints[k % 2].
                let depth = c.depth;
                let tied_embeddings = c.architecture == Architecture::TiedConcatMeshMlp;
                if tied_embeddings {
                    k.tied_readout_input_gradient::<ACTIVATION>(
                        [rows, c.modulus, c.width],
                        logits,
                        &parameters[0].state,
                        &values[depth - 1],
                        &mut adjoints[(depth - 1) % 2],
                    )?;
                } else {
                    k.weight_gradient::<ACTIVATION>(
                        layers[1],
                        rows,
                        &values[depth - 1],
                        [c.width, 0],
                        logits,
                        &mut parameters[1].gradient,
                    )?;
                    k.input_gradient::<ACTIVATION>(
                        layers[1],
                        rows,
                        logits,
                        &parameters[1].state,
                        &values[depth - 1],
                        &mut adjoints[(depth - 1) % 2],
                    )?;
                }
                for layer in (1..depth).rev() {
                    let mesh = [c.width, c.connectivity.parents(), c.mesh_stride(layer - 1)];
                    let group = 2 + c.stored_mesh_layer(layer - 1);
                    let [upstream, target] =
                        adjoints.get_disjoint_mut([layer % 2, (layer - 1) % 2])?;
                    // Tied layers below the last add their gradients to its gradient.
                    let tied = c.tied_mesh && layer + 1 < depth;
                    k.mesh_weight_gradient::<ACTIVATION>(
                        rows,
                        mesh,
                        &values[layer - 1],
                        upstream,
                        if tied {
                            &mut mesh_gradients[0]
                        } else {
                            &mut parameters[group].gradient
                        },
                    )?;
                    if tied {
                        let [layer_gradient, sum] = mesh_gradients.get_disjoint_mut([0, 1])?;
                        k.combine::<ADD>(&parameters[group].gradient, layer_gradient, sum)?;
                        std::mem::swap(&mut parameters[group].gradient, sum);
                    }
                    k.mesh_input_gradient::<ACTIVATION>(
                        rows,
                        mesh,
                        upstream,
                        &parameters[group].state,
                        &values[layer - 1],
                        target,
                    )?;
                }
                if tied_embeddings {
                    k.weight_gradient::<IDENTITY>(
                        layers[1],
                        rows,
                        &values[depth],
                        [2 * c.width, 0],
                        &adjoints[0],
                        &mut parameters[1].gradient,
                    )?;
                    let [dh, joined_gradient] = adjoints.get_disjoint_mut([0, 2])?;
                    k.input_gradient::<IDENTITY>(
                        layers[1],
                        rows,
                        dh,
                        &parameters[1].state,
                        &values[depth],
                        joined_gradient,
                    )?;
                    k.shared_embedding_gradient::<ACTIVATION>(
                        [rows, c.modulus, c.width],
                        &workspace.inputs,
                        &adjoints[2],
                        logits,
                        &values[depth - 1],
                        &mut parameters[0].gradient,
                    )?;
                } else {
                    k.weight_gradient::<IDENTITY>(
                        layers[0],
                        rows,
                        &workspace.inputs,
                        input,
                        &adjoints[0],
                        &mut parameters[0].gradient,
                    )?;
                }
            }
            Architecture::ResidualMeshMlp
            | Architecture::TiedConcatResidualMeshMlp
            | Architecture::TiedRotaryResidualMeshMlp
            | Architecture::TiedRotaryResidualDenseMlp => {
                const DN: usize = 0;
                const DH: usize = 1;
                const DZ: usize = 2;
                const BRANCH: usize = 3;
                const SUM: usize = 4;
                let blocks = c.depth;
                let branch_depth = c.residual_branch_depth();
                let (normalized, mesh_values) = (blocks + 1, 2 * blocks + 2);
                let last = &values[normalized + blocks];
                let tied = c.architecture.tied_embeddings();
                let mesh_group = layers.len();
                if tied {
                    k.tied_readout_input_gradient::<IDENTITY>(
                        [rows, c.modulus, c.width],
                        logits,
                        &parameters[0].state,
                        last,
                        &mut adjoints[DN],
                    )?;
                } else {
                    k.weight_gradient::<IDENTITY>(
                        layers[1],
                        rows,
                        last,
                        [c.width, 0],
                        logits,
                        &mut parameters[1].gradient,
                    )?;
                    k.input_gradient::<IDENTITY>(
                        layers[1],
                        rows,
                        logits,
                        &parameters[1].state,
                        last,
                        &mut adjoints[DN],
                    )?;
                }
                let [dn, dh] = adjoints.get_disjoint_mut([DN, DH])?;
                k.rmsnorm::<1>(
                    rows,
                    c.width,
                    [&values[blocks], &*dn, last],
                    &mut workspace.statistics,
                    dh,
                )?;
                for block in (0..blocks).rev() {
                    let mut upstream = DH;
                    if c.residual_branch_activation() {
                        let branch = mesh_values + (block + 1) * branch_depth - 1;
                        let [gradient, out] = adjoints.get_disjoint_mut([DH, DZ])?;
                        k.activation_gradient::<ACTIVATION>(&values[branch], gradient, out)?;
                        upstream = DZ;
                    }
                    for layer in (0..branch_depth).rev() {
                        let mesh_layer = block * branch_depth + layer;
                        let group = if c.architecture.residual_dense() {
                            1 + block
                        } else {
                            mesh_group + mesh_layer
                        };
                        let mesh = [c.width, c.connectivity.parents(), c.mesh_stride(mesh_layer)];
                        let before = if layer == 0 {
                            normalized + block
                        } else {
                            mesh_values + mesh_layer - 1
                        };
                        let target = if upstream == DN { DZ } else { DN };
                        let [gradient, out] = adjoints.get_disjoint_mut([upstream, target])?;
                        if c.architecture.residual_dense() {
                            k.weight_gradient::<IDENTITY>(
                                layers[group],
                                rows,
                                &values[before],
                                [c.width, 0],
                                gradient,
                                &mut parameters[group].gradient,
                            )?;
                            k.input_gradient::<IDENTITY>(
                                layers[group],
                                rows,
                                gradient,
                                &parameters[group].state,
                                &values[before],
                                out,
                            )?;
                        } else if layer == 0 {
                            k.mesh_weight_gradient::<IDENTITY>(
                                rows,
                                mesh,
                                &values[before],
                                gradient,
                                &mut parameters[group].gradient,
                            )?;
                            k.mesh_input_gradient::<IDENTITY>(
                                rows,
                                mesh,
                                gradient,
                                &parameters[group].state,
                                &values[before],
                                out,
                            )?;
                        } else {
                            k.mesh_weight_gradient::<ACTIVATION>(
                                rows,
                                mesh,
                                &values[before],
                                gradient,
                                &mut parameters[group].gradient,
                            )?;
                            k.mesh_input_gradient::<ACTIVATION>(
                                rows,
                                mesh,
                                gradient,
                                &parameters[group].state,
                                &values[before],
                                out,
                            )?;
                        }
                        upstream = target;
                    }
                    let [dn, branch] = adjoints.get_disjoint_mut([upstream, BRANCH])?;
                    k.rmsnorm::<1>(
                        rows,
                        c.width,
                        [&values[block], &*dn, &values[normalized + block]],
                        &mut workspace.statistics,
                        branch,
                    )?;
                    if c.dense_residual {
                        let [dh, branch, sum] = adjoints.get_disjoint_mut([DH, BRANCH, SUM])?;
                        k.combine::<ADD>(dh, branch, sum)?;
                        adjoints.swap(DH, SUM);
                    } else {
                        adjoints.swap(DH, BRANCH);
                    }
                }
                if c.architecture.rotary_embeddings() {
                    k.rotary_embedding_gradient(
                        [rows, c.modulus, c.width],
                        &workspace.inputs,
                        &adjoints[DH],
                        logits,
                        last,
                        &mut parameters[0].gradient,
                    )?;
                } else if tied {
                    let joined = &values[mesh_values + blocks * branch_depth];
                    k.weight_gradient::<IDENTITY>(
                        layers[1],
                        rows,
                        joined,
                        [2 * c.width, 0],
                        &adjoints[DH],
                        &mut parameters[1].gradient,
                    )?;
                    let [dh, joined_gradient] = adjoints.get_disjoint_mut([DH, 5])?;
                    k.input_gradient::<IDENTITY>(
                        layers[1],
                        rows,
                        dh,
                        &parameters[1].state,
                        joined,
                        joined_gradient,
                    )?;
                    k.shared_embedding_gradient::<IDENTITY>(
                        [rows, c.modulus, c.width],
                        &workspace.inputs,
                        &adjoints[5],
                        logits,
                        last,
                        &mut parameters[0].gradient,
                    )?;
                } else {
                    k.weight_gradient::<IDENTITY>(
                        layers[0],
                        rows,
                        &workspace.inputs,
                        input,
                        &adjoints[DH],
                        &mut parameters[0].gradient,
                    )?;
                }
            }
            Architecture::ResidualMlp => {
                const DN: usize = 0;
                const DH: usize = 1;
                const DZ: usize = 2;
                const BRANCH: usize = 3;
                const SUM: usize = 4;
                let blocks = c.depth;
                let hidden = c.width * c.expansion;
                let (normalized, expanded) = (blocks + 1, 2 * blocks + 2);
                let readout = 1 + 2 * blocks;
                let last = &values[normalized + blocks];
                k.weight_gradient::<IDENTITY>(
                    layers[readout],
                    rows,
                    last,
                    [c.width, 0],
                    logits,
                    &mut parameters[readout].gradient,
                )?;
                k.input_gradient::<IDENTITY>(
                    layers[readout],
                    rows,
                    logits,
                    &parameters[readout].state,
                    last,
                    &mut adjoints[DN],
                )?;
                let [dn, dh] = adjoints.get_disjoint_mut([DN, DH])?;
                k.rmsnorm::<1>(
                    rows,
                    c.width,
                    [&values[blocks], &*dn, last],
                    &mut workspace.statistics,
                    dh,
                )?;
                for block in (0..blocks).rev() {
                    let (up, down) = (1 + 2 * block, 2 + 2 * block);
                    let (n, z) = (&values[normalized + block], &values[expanded + block]);
                    let [dh, dz] = adjoints.get_disjoint_mut([DH, DZ])?;
                    k.weight_gradient::<ACTIVATION>(
                        layers[down],
                        rows,
                        z,
                        [hidden, 0],
                        dh,
                        &mut parameters[down].gradient,
                    )?;
                    k.input_gradient::<ACTIVATION>(
                        layers[down],
                        rows,
                        dh,
                        &parameters[down].state,
                        z,
                        dz,
                    )?;
                    let [dz, dn] = adjoints.get_disjoint_mut([DZ, DN])?;
                    k.weight_gradient::<IDENTITY>(
                        layers[up],
                        rows,
                        n,
                        [c.width, 0],
                        dz,
                        &mut parameters[up].gradient,
                    )?;
                    k.input_gradient::<IDENTITY>(
                        layers[up],
                        rows,
                        dz,
                        &parameters[up].state,
                        n,
                        dn,
                    )?;
                    let [dn, branch] = adjoints.get_disjoint_mut([DN, BRANCH])?;
                    k.rmsnorm::<1>(
                        rows,
                        c.width,
                        [&values[block], &*dn, n],
                        &mut workspace.statistics,
                        branch,
                    )?;
                    // The residual stream receives both the skip and the branch gradient.
                    let [dh, branch, sum] = adjoints.get_disjoint_mut([DH, BRANCH, SUM])?;
                    k.combine::<ADD>(dh, branch, sum)?;
                    adjoints.swap(DH, SUM);
                }
                k.weight_gradient::<IDENTITY>(
                    layers[0],
                    rows,
                    &workspace.inputs,
                    input,
                    &adjoints[DH],
                    &mut parameters[0].gradient,
                )?;
            }
            Architecture::Bilinear => {
                let features = c.operand_features();
                let product = &values[2];
                k.weight_gradient::<IDENTITY>(
                    layers[2],
                    rows,
                    product,
                    [c.width, 0],
                    logits,
                    &mut parameters[2].gradient,
                )?;
                k.input_gradient::<IDENTITY>(
                    layers[2],
                    rows,
                    logits,
                    &parameters[2].state,
                    product,
                    &mut adjoints[0],
                )?;
                let [dp, dl, dr] = adjoints.get_disjoint_mut([0, 1, 2])?;
                k.combine::<PRODUCT>(dp, &values[1], dl)?;
                k.combine::<PRODUCT>(dp, &values[0], dr)?;
                k.weight_gradient::<IDENTITY>(
                    layers[0],
                    rows,
                    &workspace.inputs,
                    [2 * features, 0],
                    dl,
                    &mut parameters[0].gradient,
                )?;
                k.weight_gradient::<IDENTITY>(
                    layers[1],
                    rows,
                    &workspace.inputs,
                    [2 * features, features],
                    dr,
                    &mut parameters[1].gradient,
                )?;
            }
        }
        Ok(())
    }
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
            kernel: "simple model launch",
            dimension: LaunchDimension::Grid,
        }
    })?;
    Ok(LaunchConfig1D::new(blocks, 128, 0))
}

mod verification {
    use super::*;
    use crate::{
        gpt::{EquationDataset, ModularOperation},
        simple::{
            Activation, Connectivity, DenseEmbeddingInitialization, Encoding,
            FullMeshInitialization, ReferenceSimpleModel, reference::verification_parameters,
        },
    };

    impl GpuSimpleModel {
        /// Checks every kernel path against f64 automatic differentiation:
        /// logits, the loss, all layer gradients, gathered and offset
        /// minibatches with repeated rows, AdamW steps, bitwise repeatability,
        /// input validation, and launch contracts.
        pub fn verify(device: usize) -> Result<()> {
            Self::verify_selected(device, None)
        }

        pub fn verify_architecture(device: usize, architecture: Architecture) -> Result<()> {
            Self::verify_selected(device, Some(architecture))
        }

        fn verify_selected(device: usize, architecture: Option<Architecture>) -> Result<()> {
            dense_tiled::verify(device)?;
            verify_row_reductions(device)?;
            for config in configurations()
                .into_iter()
                .filter(|config| architecture.is_none_or(|model| config.architecture == model))
            {
                let [logits, gradients] = verify_model(device, config)?;
                let model = match config.architecture {
                    Architecture::FullMeshMlp => format!(
                        "{} {} {}",
                        config.architecture, config.connectivity, config.full_mesh_initialization
                    ),
                    Architecture::SharedNeighborMeshMlp => format!(
                        "{} {} embedding-depth={} embedding-init={} {}",
                        config.architecture,
                        config.connectivity,
                        config.embedding_mesh_depth,
                        config.shared_embedding_initialization,
                        config.full_mesh_initialization
                    ),
                    Architecture::TiedRotaryResidualDenseMlp => format!(
                        "{} embedding-init={}",
                        config.architecture, config.dense_embedding_initialization
                    ),
                    Architecture::MeshMlp | Architecture::TiedConcatMeshMlp if config.tied_mesh => {
                        format!("{} {} tied", config.architecture, config.connectivity)
                    }
                    Architecture::MeshMlp | Architecture::TiedConcatMeshMlp => {
                        format!("{} {}", config.architecture, config.connectivity)
                    }
                    Architecture::ResidualMeshMlp
                    | Architecture::TiedConcatResidualMeshMlp
                    | Architecture::TiedRotaryResidualMeshMlp => {
                        format!(
                            "{} {} mesh-depth={}{}",
                            config.architecture,
                            config.connectivity,
                            config.mesh_depth,
                            if config.mesh_output_activation {
                                " activated-output"
                            } else {
                                ""
                            }
                        )
                    }
                    _ if config.decay_factors != 2 => {
                        format!(
                            "{} decay-factors={}",
                            config.architecture, config.decay_factors
                        )
                    }
                    _ => config.architecture.to_string(),
                };
                let rate = match config.mesh_learning_rate_multiplier {
                    1 => String::new(),
                    k => format!(" mesh-rate={k}"),
                };
                println!(
                    "PASS {model}{rate} {} {} width={} depth={} expansion={} modulus={}: logits (max error {logits:.1e} of max), loss, gradients (max error {gradients:.1e} of max) for 1/37/130 gathered rows, offset AdamW steps, repeatability",
                    config.activation,
                    config.encoding,
                    config.width,
                    config.depth,
                    config.expansion,
                    config.modulus
                );
            }
            verify_validation(device)?;
            println!(
                "PASS simple-model validation: tables, indices, minibatch ranges, parameters, launch contracts"
            );
            Ok(())
        }
    }

    fn configurations() -> Vec<SimpleConfig> {
        let base = SimpleConfig {
            modulus: 7,
            width: 5,
            depth: 1,
            ..SimpleConfig::default()
        };
        let residual = SimpleConfig {
            architecture: Architecture::ResidualMlp,
            ..base
        };
        let bilinear = SimpleConfig {
            architecture: Architecture::Bilinear,
            ..base
        };
        let mesh = SimpleConfig {
            architecture: Architecture::MeshMlp,
            ..base
        };
        vec![
            SimpleConfig {
                architecture: Architecture::SharedNeighborMeshMlp,
                encoding: Encoding::OneHot,
                width: 8,
                depth: 2,
                embedding_mesh_depth: 2,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::SharedNeighborMeshMlp,
                encoding: Encoding::OneHot,
                width: 16,
                depth: 3,
                embedding_mesh_depth: 3,
                connectivity: Connectivity::Butterfly,
                full_mesh_initialization: FullMeshInitialization::IdentityHidden,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::SharedNeighborMeshMlp,
                encoding: Encoding::OneHot,
                width: 8,
                depth: 1,
                embedding_mesh_depth: 2,
                activation: Activation::Square,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::SharedNeighborMeshMlp,
                encoding: Encoding::OneHot,
                width: 128,
                modulus: 97,
                depth: 64,
                embedding_mesh_depth: 32,
                full_mesh_initialization: FullMeshInitialization::IdentityHidden,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::SharedNeighborMeshMlp,
                encoding: Encoding::OneHot,
                width: 128,
                modulus: 97,
                depth: 64,
                embedding_mesh_depth: 32,
                full_mesh_initialization: FullMeshInitialization::IdentityHidden,
                shared_embedding_initialization:
                    crate::simple::SharedEmbeddingInitialization::UnitNormWeak,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::SharedNeighborMeshMlp,
                encoding: Encoding::OneHot,
                width: 128,
                modulus: 97,
                depth: 64,
                embedding_mesh_depth: 32,
                full_mesh_initialization: FullMeshInitialization::IdentityHidden,
                shared_embedding_initialization:
                    crate::simple::SharedEmbeddingInitialization::UnitNormStrong,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::FullMeshMlp,
                encoding: Encoding::Binary,
                width: 8,
                depth: 3,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::FullMeshMlp,
                encoding: Encoding::Binary,
                width: 16,
                depth: 4,
                connectivity: Connectivity::Butterfly,
                full_mesh_initialization: FullMeshInitialization::IdentityHidden,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::FullMeshMlp,
                encoding: Encoding::Binary,
                activation: Activation::Square,
                width: 5,
                depth: 1,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::FullMeshMlp,
                encoding: Encoding::Binary,
                modulus: 97,
                width: 128,
                depth: 8,
                connectivity: Connectivity::Butterfly,
                ..base
            },
            base,
            SimpleConfig {
                activation: Activation::Square,
                encoding: Encoding::Binary,
                modulus: 11,
                width: 33,
                depth: 2,
                ..base
            },
            SimpleConfig {
                encoding: Encoding::Scalar,
                width: 4,
                depth: 3,
                ..base
            },
            SimpleConfig {
                activation: Activation::Square,
                encoding: Encoding::DiscreteLog,
                modulus: 13,
                width: 40,
                ..base
            },
            // The experiments' shapes: 194 one-hot inputs and 97 classes.
            SimpleConfig {
                modulus: 97,
                width: 12,
                depth: 2,
                ..base
            },
            SimpleConfig {
                width: 6,
                expansion: 4,
                ..residual
            },
            // Widths above one warp exercise strided RMSNorm reductions.
            SimpleConfig {
                activation: Activation::Square,
                encoding: Encoding::DiscreteLog,
                modulus: 11,
                width: 70,
                expansion: 1,
                ..residual
            },
            SimpleConfig {
                modulus: 97,
                width: 8,
                depth: 2,
                expansion: 4,
                ..residual
            },
            SimpleConfig {
                width: 3,
                ..bilinear
            },
            SimpleConfig {
                encoding: Encoding::Binary,
                modulus: 97,
                width: 17,
                ..bilinear
            },
            SimpleConfig {
                encoding: Encoding::Scalar,
                modulus: 5,
                width: 2,
                ..bilinear
            },
            SimpleConfig { depth: 3, ..mesh },
            // Butterfly strides 1, 2, 4, 1 on a width that is not a power of two.
            SimpleConfig {
                encoding: Encoding::Binary,
                modulus: 11,
                width: 12,
                depth: 5,
                connectivity: Connectivity::Butterfly,
                ..mesh
            },
            // The narrowest mesh. Squares keep one row's two units alive.
            SimpleConfig {
                activation: Activation::Square,
                encoding: Encoding::DiscreteLog,
                modulus: 13,
                width: 2,
                depth: 3,
                connectivity: Connectivity::Butterfly,
                ..mesh
            },
            SimpleConfig {
                depth: 3,
                connectivity: Connectivity::Own,
                ..mesh
            },
            // Own-channel meshes also run on a single channel.
            SimpleConfig {
                activation: Activation::Square,
                width: 1,
                depth: 3,
                connectivity: Connectivity::Own,
                ..mesh
            },
            // The experiments' shapes, including the depth-1 mesh MLP.
            SimpleConfig {
                modulus: 97,
                width: 12,
                depth: 1,
                ..mesh
            },
            SimpleConfig {
                modulus: 97,
                width: 16,
                depth: 9,
                ..mesh
            },
            SimpleConfig {
                modulus: 97,
                width: 16,
                depth: 9,
                connectivity: Connectivity::Own,
                ..mesh
            },
            // Tied mesh layers sum their gradients, with one or two parents.
            SimpleConfig {
                depth: 4,
                connectivity: Connectivity::Own,
                tied_mesh: true,
                ..mesh
            },
            SimpleConfig {
                encoding: Encoding::Binary,
                modulus: 11,
                width: 12,
                depth: 5,
                connectivity: Connectivity::Butterfly,
                tied_mesh: true,
                ..mesh
            },
            SimpleConfig {
                modulus: 97,
                width: 16,
                depth: 9,
                connectivity: Connectivity::Own,
                tied_mesh: true,
                ..mesh
            },
            // Mesh layers at a multiple of the learning rate, tied or not.
            SimpleConfig {
                depth: 4,
                connectivity: Connectivity::Own,
                mesh_learning_rate_multiplier: 3,
                ..mesh
            },
            SimpleConfig {
                encoding: Encoding::Binary,
                modulus: 11,
                width: 12,
                depth: 5,
                connectivity: Connectivity::Butterfly,
                tied_mesh: true,
                mesh_learning_rate_multiplier: 2,
                ..mesh
            },
            SimpleConfig {
                modulus: 97,
                width: 16,
                depth: 2,
                connectivity: Connectivity::Own,
                mesh_learning_rate_multiplier: 8,
                ..mesh
            },
            // Shared E participates in both operands and the transposed readout.
            SimpleConfig {
                architecture: Architecture::TiedConcatMeshMlp,
                connectivity: Connectivity::Own,
                depth: 3,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedConcatMeshMlp,
                activation: Activation::Square,
                width: 6,
                depth: 1,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedConcatMeshMlp,
                tied_mesh: true,
                depth: 4,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedConcatMeshMlp,
                modulus: 97,
                width: 64,
                depth: 5,
                connectivity: Connectivity::Own,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::ResidualMeshMlp,
                depth: 2,
                mesh_depth: 3,
                connectivity: Connectivity::Butterfly,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::ResidualMeshMlp,
                depth: 1,
                mesh_depth: 1,
                connectivity: Connectivity::Own,
                activation: Activation::Square,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::ResidualMeshMlp,
                modulus: 97,
                width: 64,
                depth: 2,
                mesh_depth: 4,
                connectivity: Connectivity::Butterfly,
                ..base
            },
            // The Neighbor plateau candidate: Square inside two three-layer branches.
            SimpleConfig {
                architecture: Architecture::ResidualMeshMlp,
                modulus: 97,
                width: 128,
                depth: 2,
                mesh_depth: 3,
                connectivity: Connectivity::Neighbor,
                activation: Activation::Square,
                ..base
            },
            // Shared-E concat residual ablations.
            SimpleConfig {
                architecture: Architecture::TiedConcatResidualMeshMlp,
                modulus: 97,
                width: 128,
                depth: 2,
                mesh_depth: 3,
                connectivity: Connectivity::Neighbor,
                activation: Activation::Square,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedConcatResidualMeshMlp,
                depth: 1,
                mesh_depth: 1,
                connectivity: Connectivity::Own,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedConcatResidualMeshMlp,
                depth: 2,
                mesh_depth: 2,
                connectivity: Connectivity::Butterfly,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedRotaryResidualMeshMlp,
                modulus: 97,
                width: 128,
                depth: 2,
                mesh_depth: 3,
                connectivity: Connectivity::Neighbor,
                activation: Activation::Square,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedRotaryResidualMeshMlp,
                width: 6,
                depth: 1,
                mesh_depth: 1,
                connectivity: Connectivity::Own,
                ..base
            },
            // The wide version is the plateau demonstration, with identical blocks.
            SimpleConfig {
                architecture: Architecture::TiedRotaryResidualMeshMlp,
                modulus: 97,
                width: 1024,
                depth: 2,
                mesh_depth: 3,
                connectivity: Connectivity::Neighbor,
                activation: Activation::Square,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedRotaryResidualMeshMlp,
                width: 6,
                depth: 2,
                mesh_depth: 2,
                connectivity: Connectivity::Butterfly,
                ..base
            },
            // Final activation preserves Square when a branch has only one mesh.
            SimpleConfig {
                architecture: Architecture::TiedRotaryResidualMeshMlp,
                modulus: 97,
                width: 128,
                depth: 1,
                mesh_depth: 1,
                mesh_output_activation: true,
                connectivity: Connectivity::Neighbor,
                activation: Activation::Square,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedRotaryResidualMeshMlp,
                modulus: 97,
                width: 1024,
                depth: 1,
                mesh_depth: 1,
                mesh_output_activation: true,
                connectivity: Connectivity::Neighbor,
                activation: Activation::Square,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedRotaryResidualMeshMlp,
                width: 6,
                depth: 1,
                mesh_depth: 1,
                mesh_output_activation: true,
                connectivity: Connectivity::Own,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::ResidualMeshMlp,
                width: 6,
                depth: 2,
                mesh_depth: 2,
                mesh_output_activation: true,
                connectivity: Connectivity::Butterfly,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedConcatResidualMeshMlp,
                width: 6,
                depth: 1,
                mesh_depth: 1,
                mesh_output_activation: true,
                connectivity: Connectivity::Neighbor,
                activation: Activation::Square,
                ..base
            },
            // Dense replacement: all matrix gradients and repeated residual blocks.
            SimpleConfig {
                architecture: Architecture::TiedRotaryResidualDenseMlp,
                width: 6,
                depth: 2,
                dense_embedding_initialization: DenseEmbeddingInitialization::NormalWidth,
                dense_residual: false,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedRotaryResidualDenseMlp,
                width: 6,
                depth: 2,
                dense_embedding_initialization: DenseEmbeddingInitialization::NormalWidth,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedRotaryResidualDenseMlp,
                modulus: 97,
                width: 128,
                depth: 1,
                dense_embedding_initialization: DenseEmbeddingInitialization::NormalWidth,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedRotaryResidualDenseMlp,
                width: 6,
                depth: 2,
                ..base
            },
            SimpleConfig {
                architecture: Architecture::TiedRotaryResidualDenseMlp,
                modulus: 97,
                width: 32,
                depth: 1,
                activation: Activation::Square,
                ..base
            },
            // Channel decay as ten factors, whose shrunken first channel hits the cap.
            SimpleConfig {
                modulus: 97,
                width: 12,
                decay_factors: 10,
                ..base
            },
            SimpleConfig {
                encoding: Encoding::Binary,
                modulus: 11,
                width: 9,
                decay_factors: 3,
                ..base
            },
        ]
    }

    /// AdamW's decay of every parameter: the weight decay, or for channel
    /// decay the capped multiple for the parameter's hidden channel.
    fn decays(config: SimpleConfig, parameters: &[f32], optimizer: AdamWOptions) -> Vec<f32> {
        if config.decay_factors == 2 {
            return vec![optimizer.weight_decay; parameters.len()];
        }
        let (width, classes) = (config.width, config.modulus);
        let (input, readout) = parameters.split_at(config.input_features() * width);
        let channels: Vec<_> = (0..width)
            .map(|channel| {
                let mut squares = 0.0_f32;
                for row in 0..config.input_features() {
                    squares += input[row * width + channel].powi(2);
                }
                for class in 0..classes {
                    squares += readout[channel * classes + class].powi(2);
                }
                let scale = (0.5 * squares)
                    .max(f32::MIN_POSITIVE)
                    .powf(config.decay_exponent());
                (optimizer.weight_decay * scale).min(1.0 / optimizer.learning_rate)
            })
            .collect();
        let mut decays: Vec<_> = (0..input.len()).map(|p| channels[p % width]).collect();
        decays.extend((0..readout.len()).map(|p| channels[p / classes]));
        decays
    }

    /// AdamW's learning rate for every parameter: the multiplied rate for
    /// mesh-layer weights, which follow the dense layers.
    fn rates(config: SimpleConfig, count: usize, optimizer: AdamWOptions) -> Vec<f32> {
        let dense: usize = config.layers().iter().map(|layer| layer.parameters()).sum();
        let mesh = optimizer.learning_rate * config.mesh_learning_rate_multiplier as f32;
        (0..count)
            .map(|p| {
                if p < dense {
                    optimizer.learning_rate
                } else {
                    mesh
                }
            })
            .collect()
    }

    /// The largest absolute difference as a fraction of the largest expected value.
    fn relative_error(name: &str, expected: &[f64], actual: &[f64], tolerance: f64) -> Result<f64> {
        if expected.len() != actual.len() || actual.iter().any(|x| !x.is_finite()) {
            return Err(format!("{name}: missing or nonfinite values").into());
        }
        let scale = expected.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        if scale == 0.0 {
            return Err(format!("{name}: the reference is identically zero").into());
        }
        let mut worst = 0.0_f64;
        for (index, (&e, &a)) in expected.iter().zip(actual).enumerate() {
            let error = (e - a).abs() / scale;
            if error > tolerance {
                return Err(format!(
                    "{name} {index}: expected {e:.9e}, got {a:.9e} ({error:.1e} of max)"
                )
                .into());
            }
            worst = worst.max(error);
        }
        Ok(worst)
    }

    fn verify_model(device: usize, config: SimpleConfig) -> Result<[f64; 2]> {
        let mut model = GpuSimpleModel::new(device, config, 11)?;
        let initial = ReferenceSimpleModel::new(config, 11)?;
        if model.parameters()? != initial.parameters() {
            return Err("GPU initialization differs from the reference".into());
        }
        let mut parameters = verification_parameters(config, 11);
        if config.decay_factors != 2 {
            // A nearly dead channel tests the cap on its decay.
            let (width, classes) = (config.width, config.modulus);
            let readout = config.input_features() * width;
            for row in 0..config.input_features() {
                parameters[row * width] *= 1e-3;
            }
            for class in 0..classes {
                parameters[readout + class] *= 1e-3;
            }
        }
        model.set_parameters(&parameters)?;
        let reference = ReferenceSimpleModel::from_parameters(config, parameters.clone())?;
        let dataset = EquationDataset::new(config.modulus, ModularOperation::Divide, 0.5, 7)?;
        let equations = dataset.split(false);
        let (features, targets) = config.encode(&equations[..equations.len().min(64)])?;
        let table = model.upload_table(&features, &targets)?;
        let columns = config.input_features();

        let logits = model.logits(&table)?;
        if logits != model.logits(&table)? {
            return Err("logits are not bitwise repeatable".into());
        }
        let expected: Vec<_> = reference.logits(&features)?.into_iter().flatten().collect();
        let actual: Vec<_> = logits.iter().map(|&x| f64::from(x)).collect();
        let logit_error = relative_error("logit", &expected, &actual, 1e-5)?;

        let mut gradient_error = 0.0_f64;
        for rows in [1, 37, 130] {
            // Repeated rows in an arbitrary order exercise the gather.
            let indices: Vec<_> = (0..rows)
                .map(|i| ((i * 7 + 3) % table.rows()) as u32)
                .collect();
            let mut gathered = Vec::with_capacity(rows * columns);
            for &index in &indices {
                let start = index as usize * columns;
                gathered.extend_from_slice(&features[start..start + columns]);
            }
            let answers: Vec<_> = indices.iter().map(|&i| targets[i as usize]).collect();
            let (cpu_loss, cpu_gradient) = reference.loss_and_gradient(&gathered, &answers)?;
            let (gpu_loss, gpu_gradient) = model.loss_and_gradient(&table, &indices)?;
            if (cpu_loss - gpu_loss).abs() > 1e-5 * (1.0 + cpu_loss.abs()) {
                return Err(
                    format!("loss rows={rows}: expected {cpu_loss}, got {gpu_loss}").into(),
                );
            }
            gradient_error = gradient_error.max(relative_error(
                &format!("gradient rows={rows} parameter"),
                &cpu_gradient,
                &gpu_gradient,
                2e-4,
            )?);
        }

        // The minibatch sits at an offset inside a longer index list, as in an epoch.
        let indices: Vec<_> = (0..37)
            .map(|i| ((i * 5 + 1) % table.rows()) as u32)
            .collect();
        let mut epoch: Vec<_> = (0..5).map(|i| (i % table.rows()) as u32).collect();
        epoch.extend_from_slice(&indices);
        epoch.extend_from_slice(&[0, 1 % table.rows() as u32]);
        let batches = model.upload_batches(&table, &epoch)?;
        let optimizer = AdamWOptions {
            learning_rate: 0.01,
            weight_decay: 0.1,
            beta2: 0.98,
            ..AdamWOptions::default()
        };
        model.set_parameters(&parameters)?;
        let mut moments = vec![[0.0_f32; 2]; parameters.len()];
        let (mut b1, mut b2) = (1.0_f32, 1.0_f32);
        let mut first_step = Vec::new();
        for step in 1..=3 {
            let mut expected = model.parameters()?;
            let decays = decays(config, &expected, optimizer);
            let rates = rates(config, expected.len(), optimizer);
            let (_, gradient) = model.loss_and_gradient(&table, &indices)?;
            // Check the optimizer with its actual FP32 inputs.
            b1 *= optimizer.beta1;
            b2 *= optimizer.beta2;
            for ((((parameter, moment), gradient), decay), rate) in expected
                .iter_mut()
                .zip(&mut moments)
                .zip(gradient)
                .zip(decays)
                .zip(rates)
            {
                let gradient = gradient as f32;
                moment[0] = optimizer.beta1 * moment[0] + (1.0 - optimizer.beta1) * gradient;
                moment[1] =
                    optimizer.beta2 * moment[1] + (1.0 - optimizer.beta2) * gradient * gradient;
                *parameter -= rate
                    * (moment[0]
                        / (1.0 - b1)
                        / ((moment[1] / (1.0 - b2)).sqrt() + optimizer.epsilon)
                        + decay * *parameter);
            }
            model.train_batch(&table, &batches, 5, indices.len(), optimizer)?;
            let actual = model.parameters()?;
            for (index, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
                if (e - a).abs() > 1e-6 * (1.0 + e.abs()) {
                    return Err(format!(
                        "AdamW step={step} parameter={index}: expected {e}, got {a}"
                    )
                    .into());
                }
            }
            if step == 1 {
                first_step = actual;
            }
        }
        // Resetting the parameters also resets AdamW, and the step repeats bitwise.
        model.set_parameters(&parameters)?;
        model.train_batch(&table, &batches, 5, indices.len(), optimizer)?;
        if model.parameters()? != first_step {
            return Err("AdamW steps are not bitwise repeatable after a reset".into());
        }
        Ok([logit_error, gradient_error])
    }

    /// The warp-per-row reductions against one-thread scans: predictions and
    /// softmax maxima bit for bit, softmax denominators against f64, on rows
    /// with ties, zeros of both signs and values that are not finite. One-pass
    /// RMSNorm must match the two-pass kernels bit for bit, which catches a
    /// compiler that fuses its products into other additions.
    fn verify_row_reductions(device: usize) -> Result<()> {
        let config = SimpleConfig {
            modulus: 7,
            width: 4,
            ..SimpleConfig::default()
        };
        let model = GpuSimpleModel::new(device, config, 1)?;
        let k = &model.kernels;
        for classes in [2, 7, 31, 32, 33, 97] {
            let with = |fill: f32, cells: &[(usize, f32)]| {
                let mut row = vec![fill; classes];
                for &(class, value) in cells {
                    row[class] = value;
                }
                row
            };
            let last = classes - 1;
            let mut rows: Vec<Vec<f32>> = (0..40)
                .map(|row| {
                    (0..classes)
                        .map(|class| ((row * 31 + class * 17) % 23) as f32 * 0.37 - 4.0)
                        .collect()
                })
                .collect();
            rows.extend([
                with(1.5, &[]),
                with(-1.0, &[(classes / 2, 2.0), (last, 2.0)]),
                with(-1.0, &[(0, -0.0), (last, 0.0)]),
                with(-1.0, &[(0, 0.0), (last, -0.0)]),
                with(0.5, &[(last, f32::NAN)]),
                with(0.5, &[(0, f32::NAN)]),
                with(0.5, &[(1, f32::INFINITY)]),
                with(0.5, &[(last, f32::NEG_INFINITY)]),
                with(f32::NEG_INFINITY, &[]),
            ]);
            let count = rows.len();
            let logits = DeviceBuffer::from_host(&k.stream, &rows.concat())?;
            let identity: Vec<_> = (0..count as u32).collect();
            let indices = DeviceBuffer::from_host(&k.stream, &identity)?;
            let targets = DeviceBuffer::from_host(&k.stream, &vec![0_u32; count])?;
            let mut predictions = DeviceBuffer::zeroed(&k.stream, count)?;
            let mut statistics = DeviceBuffer::zeroed(&k.stream, count)?;
            let mut gradient = DeviceBuffer::zeroed(&k.stream, count * classes)?;
            k.predictions(classes, &logits, &mut predictions)?;
            k.cross_entropy_gradient(
                classes,
                0,
                &indices,
                &targets,
                &logits,
                &mut statistics,
                &mut gradient,
            )?;
            let predictions = predictions.to_host_vec(&k.stream)?;
            let statistics = statistics.to_host_vec(&k.stream)?;
            for (row, (&prediction, &[maximum, sum])) in
                rows.iter().zip(predictions.iter().zip(&statistics))
            {
                let mut expected = 0;
                let mut expected_maximum = f32::NEG_INFINITY;
                for (class, &value) in row.iter().enumerate() {
                    if value > row[expected] {
                        expected = class;
                    }
                    if value > expected_maximum {
                        expected_maximum = value;
                    }
                }
                let finite = row.iter().all(|value| value.is_finite());
                let expected = if finite { expected as u32 } else { u32::MAX };
                let expected_sum: f64 = row
                    .iter()
                    .map(|&value| f64::from(value - expected_maximum).exp())
                    .sum();
                let sum_matches = if expected_sum.is_nan() {
                    sum.is_nan()
                } else {
                    (f64::from(sum) - expected_sum).abs() <= 1e-5 * expected_sum
                };
                if prediction != expected
                    || maximum.to_bits() != expected_maximum.to_bits()
                    || !sum_matches
                {
                    return Err(format!(
                        "row reductions over {classes} classes differ on {row:?}: prediction {prediction}, maximum {maximum}, sum {sum}"
                    )
                    .into());
                }
            }
        }
        for width in [16, 32, 64, 128, 256, 512, 1024] {
            // A different rounding in one lane's partial sum of squares
            // changes the inverse RMS of only about 0.4% of rows at width
            // 1024, so the check needs thousands of rows with full mantissas.
            let rows = 4096;
            let values = |seed: u32| -> Vec<f32> {
                (0..rows * width)
                    .map(|item| {
                        let scale = 10.0_f32.powi((item / width) as i32 % 7 - 3);
                        let hash = (item as u32)
                            .wrapping_mul(0x9E37_79B1)
                            .wrapping_add(seed.wrapping_mul(0x85EB_CA77));
                        ((hash >> 8) as f32 / 16_777_216.0 * 8.0 - 4.0) * scale
                    })
                    .collect()
            };
            let mut input = values(0);
            input[..width].fill(0.0);
            let upload = |values: &[f32]| DeviceBuffer::from_host(&k.stream, values);
            let x = upload(&input)?;
            let (dn, n) = (upload(&values(5))?, upload(&values(11))?);
            let mut statistics = DeviceBuffer::zeroed(&k.stream, rows)?;
            let mut one_pass = DeviceBuffer::zeroed(&k.stream, rows * width)?;
            let mut two_passes = DeviceBuffer::zeroed(&k.stream, rows * width)?;
            for backward in [false, true] {
                if backward {
                    let inputs = [&x, &dn, &n];
                    k.rmsnorm::<1>(rows, width, inputs, &mut statistics, &mut one_pass)?;
                    k.rmsnorm_split::<1>(rows, width, inputs, &mut statistics, &mut two_passes)?;
                } else {
                    let inputs = [&x, &x, &x];
                    k.rmsnorm::<0>(rows, width, inputs, &mut statistics, &mut one_pass)?;
                    k.rmsnorm_split::<0>(rows, width, inputs, &mut statistics, &mut two_passes)?;
                }
                let one_pass = one_pass.to_host_vec(&k.stream)?;
                let two_passes = two_passes.to_host_vec(&k.stream)?;
                if one_pass
                    .iter()
                    .zip(&two_passes)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    return Err(format!(
                        "one-pass RMSNorm (backward: {backward}) differs from two passes at width {width}"
                    )
                    .into());
                }
            }
        }
        println!(
            "PASS simple-model row reductions: predictions and softmax maxima match one-thread scans bit for bit, denominators match f64, with ties, signed zeros and values that are not finite; one-pass RMSNorm matches two passes bit for bit"
        );
        Ok(())
    }

    fn verify_validation(device: usize) -> Result<()> {
        let config = SimpleConfig {
            modulus: 7,
            width: 4,
            ..SimpleConfig::default()
        };
        for invalid in [
            SimpleConfig { width: 0, ..config },
            SimpleConfig {
                modulus: 1,
                ..config
            },
        ] {
            if GpuSimpleModel::new(device, invalid, 1).is_ok() {
                return Err("model accepted an invalid configuration".into());
            }
        }
        let mut model = GpuSimpleModel::new(device, config, 1)?;
        let columns = config.input_features();
        let features = vec![0.5; 3 * columns];
        let table = model.upload_table(&features, &[0, 6, 3])?;
        let mut nan = features.clone();
        nan[1] = f32::NAN;
        let parameters = model.parameters()?;
        let mut nonfinite = parameters.clone();
        nonfinite[0] = f32::INFINITY;
        let short = model.upload_table(&features[..columns], &[1])?;
        let foreign = model.upload_batches(&short, &[0])?;
        let batches = model.upload_batches(&table, &[2, 0, 1])?;
        let optimizer = AdamWOptions::default();
        if model.upload_table(&[], &[]).is_ok()
            || model.upload_table(&features[1..], &[0, 6, 3]).is_ok()
            || model.upload_table(&nan, &[0, 6, 3]).is_ok()
            || model.upload_table(&features, &[0, 7, 3]).is_ok()
            || model.upload_batches(&table, &[]).is_ok()
            || model.upload_batches(&table, &[3]).is_ok()
            || model.train_batch(&table, &batches, 2, 2, optimizer).is_ok()
            || model.train_batch(&table, &batches, 0, 0, optimizer).is_ok()
            || model
                .train_batch(&table, &batches, usize::MAX, 1, optimizer)
                .is_ok()
            || model.train_batch(&table, &foreign, 0, 1, optimizer).is_ok()
            || model
                .train_batch(
                    &table,
                    &batches,
                    0,
                    3,
                    AdamWOptions {
                        beta1: 1.0,
                        ..optimizer
                    },
                )
                .is_ok()
            || model.loss_and_gradient(&table, &[3]).is_ok()
            || model.set_parameters(&parameters[1..]).is_ok()
            || model.set_parameters(&nonfinite).is_ok()
        {
            return Err("simple-model validation accepted invalid input".into());
        }
        if model.parameters()? != parameters {
            return Err("rejected input changed the parameters".into());
        }
        // Prepared launches check buffer sizes before the kernels run. Each
        // call below breaks exactly one requirement.
        let k = &model.kernels;
        let layer = model.layers[0];
        let input = DeviceBuffer::<f32>::zeroed(&k.stream, 2 * columns)?;
        let state = &model.parameters[0].state;
        let mut exact = DeviceBuffer::<f32>::zeroed(&k.stream, 2 * layer.outputs)?;
        let mut long = DeviceBuffer::<f32>::zeroed(&k.stream, 2 * layer.outputs + 1)?;
        let mut gradient = DeviceBuffer::<f32>::zeroed(&k.stream, layer.parameters())?;
        let mut statistics = DeviceBuffer::<[f32; 2]>::zeroed(&k.stream, 1)?;
        let mut normalized = DeviceBuffer::<f32>::zeroed(&k.stream, 2 * columns)?;
        // Mesh layers as wide as the dense layer's output: two parents with
        // stride width - 1, and one parent.
        let width = layer.outputs;
        let mesh = [width, 2, width - 1];
        let own = [width, 1, 0];
        let hidden = DeviceBuffer::<f32>::zeroed(&k.stream, 2 * width)?;
        let weights = DeviceBuffer::<[f32; 3]>::zeroed(&k.stream, 2 * width)?;
        let gains = DeviceBuffer::<[f32; 3]>::zeroed(&k.stream, width)?;
        let triples = DeviceBuffer::<[f32; 3]>::zeroed(&k.stream, 3 * width)?;
        let mut mesh_gradient = DeviceBuffer::<f32>::zeroed(&k.stream, 2 * width)?;
        let mut gain_gradient = DeviceBuffer::<f32>::zeroed(&k.stream, width)?;
        // Channel decay of this one-layer MLP, and an input layer to update.
        let dense = [layer, model.layers[1]];
        let readout = &model.parameters[1].state;
        let mut scale = DeviceBuffer::<f32>::zeroed(&k.stream, width)?;
        let mut long_scale = DeviceBuffer::<f32>::zeroed(&k.stream, width + 1)?;
        let mut group = Parameters {
            state: DeviceBuffer::zeroed(&k.stream, layer.parameters())?,
            gradient: DeviceBuffer::zeroed(&k.stream, layer.parameters())?,
        };
        k.dense::<IDENTITY>(layer, 2, &input, [columns, 0], state, &mut exact)?;
        k.weight_gradient::<IDENTITY>(layer, 2, &input, [columns, 0], &exact, &mut gradient)?;
        k.gather(&table, &batches.indices, 1, &mut normalized)?;
        k.mesh::<RELU>(2, mesh, &hidden, &weights, &mut exact)?;
        k.mesh_input_gradient::<RELU>(2, mesh, &hidden, &weights, &hidden, &mut exact)?;
        k.mesh_weight_gradient::<RELU>(2, mesh, &hidden, &exact, &mut mesh_gradient)?;
        k.mesh::<RELU>(2, own, &hidden, &gains, &mut exact)?;
        k.mesh_input_gradient::<RELU>(2, own, &hidden, &gains, &hidden, &mut exact)?;
        k.mesh_weight_gradient::<RELU>(2, own, &hidden, &exact, &mut gain_gradient)?;
        k.activated_sum::<SQUARE>(true, &hidden, &hidden, &mut exact)?;
        k.activation_gradient::<SQUARE>(&hidden, &hidden, &mut exact)?;
        let sparse = TwoParentLayer {
            inputs: width,
            outputs: width,
            stride: 1,
        };
        k.copy_columns(2, [width, width, 0, width], &hidden, &mut exact)?;
        k.pair_rotation::<0>(width, &hidden, &mut exact)?;
        k.pair_rotation::<1>(width, &hidden, &mut exact)?;
        let odd = DeviceBuffer::<f32>::zeroed(&k.stream, 6)?;
        let mut odd_out = DeviceBuffer::<f32>::zeroed(&k.stream, 6)?;
        if k.copy_columns(2, [width, width, 1, width], &hidden, &mut exact)
            .is_ok()
            || k.copy_columns(2, [width, width, 0, width + 1], &hidden, &mut exact)
                .is_ok()
            || k.copy_columns(2, [width, width, 0, width], &hidden, &mut long)
                .is_ok()
            || k.pair_rotation::<0>(3, &odd, &mut odd_out).is_ok()
            || k.pair_rotation::<1>(width, &hidden, &mut long).is_ok()
        {
            return Err(
                "a shared Neighbor launch contract accepted an invalid copy or rotation".into(),
            );
        }
        let mut sparse_gradient = DeviceBuffer::<f32>::zeroed(&k.stream, 3 * width)?;
        k.two_parent::<RELU>(2, sparse, &hidden, &triples, &mut exact)?;
        k.two_parent_input_gradient::<RELU>(2, sparse, &hidden, &triples, &hidden, &mut exact)?;
        k.two_parent_weight_gradient::<RELU>(2, sparse, &hidden, &exact, &mut sparse_gradient)?;
        if k.two_parent::<RELU>(2, sparse, &hidden, &weights, &mut exact)
            .is_ok()
            || k.two_parent::<RELU>(2, sparse, &hidden, &triples, &mut long)
                .is_ok()
            || k.two_parent::<RELU>(
                2,
                TwoParentLayer {
                    stride: 0,
                    ..sparse
                },
                &hidden,
                &triples,
                &mut exact,
            )
            .is_ok()
            || k.two_parent::<RELU>(
                2,
                TwoParentLayer {
                    stride: width,
                    ..sparse
                },
                &hidden,
                &triples,
                &mut exact,
            )
            .is_ok()
            || k.two_parent_input_gradient::<RELU>(2, sparse, &long, &triples, &hidden, &mut exact)
                .is_ok()
            || k.two_parent_input_gradient::<RELU>(2, sparse, &hidden, &triples, &long, &mut exact)
                .is_ok()
            || k.two_parent_input_gradient::<RELU>(
                2, sparse, &hidden, &weights, &hidden, &mut exact,
            )
            .is_ok()
            || k.two_parent_input_gradient::<RELU>(2, sparse, &hidden, &triples, &hidden, &mut long)
                .is_ok()
            || k.two_parent_weight_gradient::<RELU>(2, sparse, &hidden, &long, &mut sparse_gradient)
                .is_ok()
            || k.two_parent_weight_gradient::<RELU>(2, sparse, &long, &hidden, &mut sparse_gradient)
                .is_ok()
            || k.two_parent_weight_gradient::<RELU>(2, sparse, &hidden, &exact, &mut mesh_gradient)
                .is_ok()
        {
            return Err("a two-parent launch contract accepted an invalid shape or parent".into());
        }
        k.channel_decay(dense, -0.5, state, readout, &mut scale)?;
        k.adam_channels(optimizer, [0.5, 0.5], [1, width], &scale, &mut group)?;
        if k.dense::<IDENTITY>(layer, 2, &input, [columns, 0], state, &mut long)
            .is_ok()
            || k.dense::<IDENTITY>(layer, 2, &input, [columns, 1], state, &mut exact)
                .is_ok()
            || k.weight_gradient::<IDENTITY>(layer, 2, &input, [columns, 0], &long, &mut gradient)
                .is_ok()
            || k.gather(&table, &batches.indices, 2, &mut normalized)
                .is_ok()
            || k.activated_sum::<SQUARE>(true, &long, &hidden, &mut exact)
                .is_ok()
            || k.activated_sum::<SQUARE>(true, &hidden, &long, &mut exact)
                .is_ok()
            || k.activation_gradient::<SQUARE>(&long, &hidden, &mut exact)
                .is_ok()
            || k.activation_gradient::<SQUARE>(&hidden, &long, &mut exact)
                .is_ok()
            || k.rmsnorm::<0>(
                2,
                columns,
                [&input, &input, &input],
                &mut statistics,
                &mut normalized,
            )
            .is_ok()
            || k.mesh::<RELU>(2, [width, 2, width], &hidden, &weights, &mut exact)
                .is_ok()
            || k.mesh::<RELU>(2, [width, 2, 0], &hidden, &weights, &mut exact)
                .is_ok()
            || k.mesh::<RELU>(2, [width, 3, 2], &hidden, &triples, &mut exact)
                .is_ok()
            || k.mesh::<RELU>(2, own, &hidden, &weights, &mut exact)
                .is_ok()
            || k.mesh::<RELU>(2, mesh, &hidden, &weights, &mut long)
                .is_ok()
            || k.mesh_input_gradient::<RELU>(2, mesh, &hidden, state, &hidden, &mut exact)
                .is_ok()
            || k.mesh_weight_gradient::<RELU>(2, mesh, &hidden, &exact, &mut gradient)
                .is_ok()
            || k.mesh_weight_gradient::<RELU>(2, own, &hidden, &exact, &mut mesh_gradient)
                .is_ok()
            || k.channel_decay(dense, -0.5, state, readout, &mut long_scale)
                .is_ok()
            || k.channel_decay(dense, -0.5, readout, state, &mut scale)
                .is_ok()
            || k.adam_channels(optimizer, [0.5, 0.5], [1, width], &long_scale, &mut group)
                .is_ok()
            || k.adam_channels(optimizer, [0.5, 0.5], [0, width], &scale, &mut group)
                .is_ok()
        {
            return Err("a launch contract accepted mismatched buffers".into());
        }
        // Odd widths cannot form complete coordinate pairs. Buffer shapes are
        // otherwise valid, so both rejections exercise the pairing contract.
        let odd_width = 3;
        let classes = config.modulus;
        let rotary_input = DeviceBuffer::<f32>::zeroed(&k.stream, 2 * classes)?;
        let rotary_state = DeviceBuffer::<[f32; 3]>::zeroed(&k.stream, classes * odd_width)?;
        let rotary_hidden = DeviceBuffer::<f32>::zeroed(&k.stream, odd_width)?;
        let rotary_logits = DeviceBuffer::<f32>::zeroed(&k.stream, classes)?;
        let mut rotary_out = DeviceBuffer::<f32>::zeroed(&k.stream, odd_width)?;
        let mut rotary_gradient = DeviceBuffer::<f32>::zeroed(&k.stream, classes * odd_width)?;
        if k.rotary_embedding(
            [1, classes, odd_width],
            &rotary_input,
            &rotary_state,
            &mut rotary_out,
        )
        .is_ok()
            || k.rotary_embedding_gradient(
                [1, classes, odd_width],
                &rotary_input,
                &rotary_hidden,
                &rotary_logits,
                &rotary_hidden,
                &mut rotary_gradient,
            )
            .is_ok()
        {
            return Err("a rotary launch contract accepted incomplete coordinate pairs".into());
        }
        Ok(())
    }
}
