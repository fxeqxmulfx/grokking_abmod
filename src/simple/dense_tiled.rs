//! Register-blocked CUDA matrix products for Dense layers, the tied readout
//! and the rotary embedding with its gradient.
//!
//! Operands are packed into K-major panels. Each of 256 threads accumulates a
//! `TM × TN` block of adjacent outputs from +0 in ascending reduction order
//! with fused multiply-adds, the order of a sequential loop. Tile shapes
//! therefore change speed but never a bit of the result. Reduction padding
//! multiplies -0 by +0, which leaves every partial sum unchanged, including an
//! underflowed -0. The compiler stages read-only tiles and supplies both block
//! barriers.

use super::{Dense, IDENTITY, Launches, RELU, SQUARE};
use crate::Result;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::shared::BlockTile;
use cuda_device::{DisjointSlice, device, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::sync::{Arc, Mutex};

/// Register blocks, largest first. Thread `16 y + x` of a block owns rows
/// `TM y + i` and columns `TN x + j`, so each reduction step reads its `TM`
/// and `TN` operand values with vector loads from shared memory.
const TILES: [[usize; 2]; 5] = [[8, 8], [8, 4], [4, 4], [2, 2], [1, 1]];
/// The rotary gradient reads three values per operand cell, so its stages
/// are three times wider and its register blocks smaller.
const ROTARY_TILES: [[usize; 2]; 3] = [[4, 4], [2, 2], [1, 1]];

#[cuda_module]
#[allow(clippy::too_many_arguments)]
// Kernels without a natural compile-time parameter take an unused `S`,
// always 0, so that each launch resolves its specialization by identity.
mod kernels {
    use super::*;

    #[device]
    fn dense_tiled_activate(value: f32, code: usize) -> f32 {
        if code == RELU {
            if value > 0.0 { value } else { 0.0 }
        } else if code == SQUARE {
            value * value
        } else {
            value
        }
    }

    #[device]
    fn dense_tiled_slope(value: f32, code: usize) -> f32 {
        if code == RELU {
            if value > 0.0 { 1.0 } else { 0.0 }
        } else if code == SQUARE {
            2.0 * value
        } else {
            1.0
        }
    }

    /// Where `blocked_product` stores output `[row][column]`: thread `16 y + x`
    /// of a block owns rows `tm y + i` and columns `tn x + j`, and each warp
    /// stores its `32 tm tn` products contiguously, lane-interleaved.
    #[device]
    fn dense_tiled_address(
        row: usize,
        column: usize,
        tm: usize,
        tn: usize,
        n_tiles: usize,
    ) -> usize {
        let (tile_row, r) = (row / (16 * tm), row % (16 * tm));
        let (tile_column, c) = (column / (16 * tn), column % (16 * tn));
        let thread = r / tm * 16 + c / tn;
        let warp = (tile_row * n_tiles + tile_column) * 8 + thread / 32;
        let cell = r % tm * tn + c % tn;
        (warp * tm * tn + cell) * 32 + thread % 32
    }

    /// Packs `f(input[r][offset + c])` into K-major panels of `width` columns
    /// and `depth` reduction rows. With `TRANSPOSE`, input rows become panel
    /// columns and features the reduction; otherwise rows are the reduction.
    /// Reduction padding holds `padding`; other missing cells hold zero.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        TRANSPOSE <= 1, rows >= 1, columns >= 1, stride >= 1,
        offset + columns <= stride, input.len() == rows * stride,
        width >= 1, depth >= 1, panels >= 1, out.len() == panels * depth * width))]
    pub fn pack_panels<const ACTIVATION: usize, const TRANSPOSE: usize>(
        rows: usize,
        columns: usize,
        stride: usize,
        offset: usize,
        width: usize,
        depth: usize,
        panels: usize,
        padding: f32,
        input: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let k = item / width % depth;
        let m = item / (width * depth) * width + item % width;
        let (row, feature) = if TRANSPOSE == 1 { (m, k) } else { (k, m) };
        let reduction = if TRANSPOSE == 1 { columns } else { rows };
        if let Some(value) = out.get_mut(index) {
            *value = if k >= reduction {
                padding
            } else if row < rows && feature < columns {
                dense_tiled_activate(input[row * stride + offset + feature], ACTIVATION)
            } else {
                0.0
            };
        }
        let _ = panels;
    }

    /// `pack_panels` for the weights of AdamW states, always the right
    /// operand, so reduction padding is +0.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        TRANSPOSE <= 1, rows >= 1, columns >= 1, input.len() == rows * columns,
        width >= 1, depth >= 1, panels >= 1, out.len() == panels * depth * width))]
    pub fn pack_weight_panels<const TRANSPOSE: usize>(
        rows: usize,
        columns: usize,
        width: usize,
        depth: usize,
        panels: usize,
        input: &[[f32; 3]],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let k = item / width % depth;
        let m = item / (width * depth) * width + item % width;
        let (row, feature) = if TRANSPOSE == 1 { (m, k) } else { (k, m) };
        if let Some(value) = out.get_mut(index) {
            *value = if row < rows && feature < columns {
                input[row * columns + feature][0]
            } else {
                0.0
            };
        }
        let _ = panels;
    }

    /// `out[m][n] = sum_k left[k][m] right[k][n]` for one `(16 TM) × (16 TN)`
    /// block tile per thread block, in the `dense_tiled_address` layout.
    /// Unrolling each stage's reduction loop eight times rather than fully
    /// keeps the 8x8 block in registers and is no slower for smaller ones.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (
        TM >= 1, TN >= 1, STRIPE == 32 * TM * TN,
        depth == 16 * depth_tiles, depth_tiles >= 1, m_tiles >= 1, n_tiles >= 1,
        left.len() == m_tiles * depth * 16 * TM, right.len() == n_tiles * depth * 16 * TN,
        out.len() == m_tiles * n_tiles * 256 * TM * TN))]
    #[block_pipeline(rows = 16, iterations = depth_tiles)]
    pub fn blocked_product<const TM: usize, const TN: usize, const STRIPE: usize>(
        depth: usize,
        depth_tiles: usize,
        m_tiles: usize,
        n_tiles: usize,
        #[tile_source(columns = 16 * TM, stride = 16 * TM, offset = (block / n_tiles) * depth * 16 * TM)]
        left: &[f32],
        #[tile_source(columns = 16 * TN, stride = 16 * TN, offset = (block % n_tiles) * depth * 16 * TN)]
        right: &[f32],
        mut out: DisjointSlice<f32, thread::StripedWarp<STRIPE>>,
    ) {
        let init = || {
            let _ = (depth, m_tiles);
            [[0.0_f32; TN]; TM]
        };
        let step = |sums: &mut [[f32; TN]; TM],
                    _iteration: usize,
                    a: BlockTile<'_, f32>,
                    b: BlockTile<'_, f32>| {
            let lane = (thread::threadIdx_x() % 256) as usize;
            let row = lane / 16 * TM;
            let column = lane % 16 * TN;
            let mut k = 0;
            #[unroll(8)]
            while k < 16 {
                let mut x = [0.0_f32; TM];
                let mut y = [0.0_f32; TN];
                let mut i = 0;
                #[unroll]
                while i < TM {
                    x[i] = a.load(k, row + i);
                    i += 1;
                }
                let mut j = 0;
                #[unroll]
                while j < TN {
                    y[j] = b.load(k, column + j);
                    j += 1;
                }
                let mut i = 0;
                #[unroll]
                while i < TM {
                    let mut j = 0;
                    #[unroll]
                    while j < TN {
                        sums[i][j] += x[i] * y[j];
                        j += 1;
                    }
                    i += 1;
                }
                k += 1;
            }
        };
        let mut finish = |sums: [[f32; TN]; TM]| {
            let mut cells = thread::striped_columns::<STRIPE>();
            let mut i = 0;
            #[unroll]
            while i < TM {
                let mut j = 0;
                #[unroll]
                while j < TN {
                    if let Some(index) = cells.next()
                        && let Some(value) = out.get_mut(index)
                    {
                        *value = sums[i][j];
                    }
                    j += 1;
                }
                i += 1;
            }
        };
    }

    /// Reorders blocked products into row-major cells, optionally multiplying
    /// each by the activation slope at `before`. Compile-time register blocks
    /// turn the address arithmetic into shifts.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        SLOPE <= 1, TM >= 1, TN >= 1, rows >= 1, columns >= 1, m_tiles >= 1, n_tiles >= 1,
        16 * TM * m_tiles >= rows, 16 * TN * n_tiles >= columns,
        packed.len() == m_tiles * n_tiles * 256 * TM * TN, out.len() == rows * columns,
        SLOPE * before.len() == SLOPE * out.len()))]
    pub fn unpack<const ACTIVATION: usize, const SLOPE: usize, const TM: usize, const TN: usize>(
        rows: usize,
        columns: usize,
        m_tiles: usize,
        n_tiles: usize,
        packed: &[f32],
        before: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = out.get_mut(index) {
            let address = dense_tiled_address(item / columns, item % columns, TM, TN, n_tiles);
            *value = packed[address]
                * if SLOPE == 1 {
                    dense_tiled_slope(before[item], ACTIVATION)
                } else {
                    1.0
                };
        }
        let _ = (rows, m_tiles);
    }

    /// E(a) + J E(b) from the blocked products `X_a E` and `X_b E`, where J
    /// rotates each adjacent channel pair by +90 degrees.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, pairs >= 1, width == 2 * pairs,
        TM >= 1, TN >= 1, m_tiles >= 1, n_tiles >= 1,
        16 * TM * m_tiles >= rows, 16 * TN * n_tiles >= width,
        left.len() == m_tiles * n_tiles * 256 * TM * TN, right.len() == left.len(),
        out.len() == rows * width))]
    pub fn unpack_rotary<const TM: usize, const TN: usize>(
        rows: usize,
        width: usize,
        pairs: usize,
        m_tiles: usize,
        n_tiles: usize,
        left: &[f32],
        right: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        if let Some(value) = out.get_mut(index) {
            let (row, channel) = (item / width, item % width);
            let paired = right[dense_tiled_address(row, channel ^ 1, TM, TN, n_tiles)];
            *value = left[dense_tiled_address(row, channel, TM, TN, n_tiles)]
                + if channel.is_multiple_of(2) {
                    -paired
                } else {
                    paired
                };
        }
        let _ = (rows, pairs, m_tiles);
    }

    /// The residue operand of the rotary embedding gradient. Panel row `k`
    /// holds three `width`-column groups: input a, input b and the logit
    /// gradient of minibatch row `k`. Reduction padding is -0.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, classes >= 1, input.len() == rows * 2 * classes,
        logit_gradient.len() == rows * classes,
        width >= 1, depth >= 1, panels >= 1, out.len() == panels * depth * 3 * width))]
    pub fn pack_rotary_residues<const S: usize>(
        rows: usize,
        classes: usize,
        width: usize,
        depth: usize,
        panels: usize,
        input: &[f32],
        logit_gradient: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        let group = item % (3 * width);
        let k = item / (3 * width) % depth;
        let residue = item / (3 * width * depth) * width + group % width;
        let part = group / width;
        if let Some(value) = out.get_mut(index) {
            *value = if k >= rows {
                -0.0
            } else if residue >= classes {
                0.0
            } else if part == 0 {
                input[k * 2 * classes + residue]
            } else if part == 1 {
                input[k * 2 * classes + classes + residue]
            } else {
                logit_gradient[k * classes + residue]
            };
        }
        let _ = panels;
    }

    /// The channel operand of the rotary embedding gradient: the embedding
    /// gradient, its J^T pair term, and the final normalized hidden state.
    /// Reduction padding is +0.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        rows >= 1, pairs >= 1, channels == 2 * pairs,
        gradient.len() == rows * channels, hidden.len() == rows * channels,
        width >= 1, depth >= 1, panels >= 1, out.len() == panels * depth * 3 * width))]
    pub fn pack_rotary_channels<const S: usize>(
        rows: usize,
        channels: usize,
        pairs: usize,
        width: usize,
        depth: usize,
        panels: usize,
        gradient: &[f32],
        hidden: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let _ = S;
        let index = thread::index_1d();
        let item = index.get();
        let group = item % (3 * width);
        let k = item / (3 * width) % depth;
        let channel = item / (3 * width * depth) * width + group % width;
        let part = group / width;
        if let Some(value) = out.get_mut(index) {
            *value = if k >= rows || channel >= channels {
                0.0
            } else if part == 0 {
                gradient[k * channels + channel]
            } else if part == 1 {
                let paired = gradient[k * channels + (channel ^ 1)];
                if channel.is_multiple_of(2) {
                    paired
                } else {
                    -paired
                }
            } else {
                hidden[k * channels + channel]
            };
        }
        let _ = (pairs, panels);
    }

    /// `out[r][c] = sum_k a[k][r] g[k][c] + b[k][r] Jg[k][c] + z[k][r] h[k][c]`,
    /// adding each minibatch row's three terms before the running sum, in
    /// the `dense_tiled_address` layout. Unrolling the reduction loop four
    /// times rather than fully keeps the largest block in registers.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (
        TM >= 1, TN >= 1, STRIPE == 32 * TM * TN,
        depth == 16 * depth_tiles, depth_tiles >= 1, m_tiles >= 1, n_tiles >= 1,
        left.len() == m_tiles * depth * 48 * TM, right.len() == n_tiles * depth * 48 * TN,
        out.len() == m_tiles * n_tiles * 256 * TM * TN))]
    #[block_pipeline(rows = 16, iterations = depth_tiles)]
    pub fn rotary_gradient_product<const TM: usize, const TN: usize, const STRIPE: usize>(
        depth: usize,
        depth_tiles: usize,
        m_tiles: usize,
        n_tiles: usize,
        #[tile_source(columns = 48 * TM, stride = 48 * TM, offset = (block / n_tiles) * depth * 48 * TM)]
        left: &[f32],
        #[tile_source(columns = 48 * TN, stride = 48 * TN, offset = (block % n_tiles) * depth * 48 * TN)]
        right: &[f32],
        mut out: DisjointSlice<f32, thread::StripedWarp<STRIPE>>,
    ) {
        let init = || {
            let _ = (depth, m_tiles);
            [[0.0_f32; TN]; TM]
        };
        let step = |sums: &mut [[f32; TN]; TM],
                    _iteration: usize,
                    a: BlockTile<'_, f32>,
                    b: BlockTile<'_, f32>| {
            let lane = (thread::threadIdx_x() % 256) as usize;
            let row = lane / 16 * TM;
            let column = lane % 16 * TN;
            let mut k = 0;
            #[unroll(4)]
            while k < 16 {
                let mut first = [0.0_f32; TM];
                let mut second = [0.0_f32; TM];
                let mut logit = [0.0_f32; TM];
                let mut gradient = [0.0_f32; TN];
                let mut paired = [0.0_f32; TN];
                let mut hidden = [0.0_f32; TN];
                let mut i = 0;
                #[unroll]
                while i < TM {
                    first[i] = a.load(k, row + i);
                    second[i] = a.load(k, 16 * TM + row + i);
                    logit[i] = a.load(k, 32 * TM + row + i);
                    i += 1;
                }
                let mut j = 0;
                #[unroll]
                while j < TN {
                    gradient[j] = b.load(k, column + j);
                    paired[j] = b.load(k, 16 * TN + column + j);
                    hidden[j] = b.load(k, 32 * TN + column + j);
                    j += 1;
                }
                let mut i = 0;
                #[unroll]
                while i < TM {
                    let mut j = 0;
                    #[unroll]
                    while j < TN {
                        sums[i][j] +=
                            first[i] * gradient[j] + second[i] * paired[j] + logit[i] * hidden[j];
                        j += 1;
                    }
                    i += 1;
                }
                k += 1;
            }
        };
        let mut finish = |sums: [[f32; TN]; TM]| {
            let mut cells = thread::striped_columns::<STRIPE>();
            let mut i = 0;
            #[unroll]
            while i < TM {
                let mut j = 0;
                #[unroll]
                while j < TN {
                    if let Some(index) = cells.next()
                        && let Some(value) = out.get_mut(index)
                    {
                        *value = sums[i][j];
                    }
                    j += 1;
                }
                i += 1;
            }
        };
    }
}

/// A product `[m × k] · [k × n]` with `TM × TN` outputs per thread.
#[derive(Clone, Copy)]
struct Plan {
    tiles: [usize; 2],
    m_tiles: usize,
    n_tiles: usize,
    depth_tiles: usize,
}

impl Plan {
    fn new([m, n, k]: [usize; 3], tiles: [usize; 2]) -> Self {
        Self {
            tiles,
            m_tiles: m.div_ceil(16 * tiles[0]),
            n_tiles: n.div_ceil(16 * tiles[1]),
            depth_tiles: k.div_ceil(16),
        }
    }

    fn depth(self) -> usize {
        16 * self.depth_tiles
    }

    /// Panel width, depth and count of the left operand.
    fn left(self) -> [usize; 3] {
        [16 * self.tiles[0], self.depth(), self.m_tiles]
    }

    fn right(self) -> [usize; 3] {
        [16 * self.tiles[1], self.depth(), self.n_tiles]
    }

    /// Lengths of both panels, with `values` per operand cell, and of the
    /// blocked product.
    fn lengths(self, values: usize) -> [usize; 3] {
        let [[left, depth, m_tiles], [right, _, n_tiles]] = [self.left(), self.right()];
        [
            values * m_tiles * depth * left,
            values * n_tiles * depth * right,
            m_tiles * n_tiles * left * right,
        ]
    }
}

pub(super) struct Kernels {
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    /// Products aim for at least this many thread blocks.
    blocks: usize,
    /// Verification forces each tile shape in turn.
    choice: Option<usize>,
    /// Panels and blocked products, cached by their lengths.
    buffers: Mutex<BTreeMap<Vec<usize>, Vec<DeviceBuffer<f32>>>>,
    launches: Launches,
}

fn config(items: usize, block: usize) -> Result<LaunchConfig1D> {
    Ok(LaunchConfig1D::new(
        items.div_ceil(block).try_into()?,
        block.try_into()?,
        0,
    ))
}

#[allow(clippy::too_many_arguments)]
impl Kernels {
    pub(super) fn new(context: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Result<Self> {
        Self::with_choice(context, stream, None)
    }

    fn with_choice(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        choice: Option<usize>,
    ) -> Result<Self> {
        Ok(Self {
            stream: stream.clone(),
            module: kernels::load(context)?,
            blocks: usize::try_from(context.multiprocessor_count()?)?,
            choice,
            buffers: Mutex::new(BTreeMap::new()),
            launches: Launches::default(),
        })
    }

    /// The largest register block that still gives every multiprocessor a
    /// thread block, skipping blocks that only pad more than the next smaller
    /// one without adding thread blocks; products too small for that use the
    /// smallest block.
    fn plan(&self, shape: [usize; 3], candidates: &[[usize; 2]]) -> Plan {
        let [m, n, _] = shape;
        let blocks = |[tm, tn]: [usize; 2]| m.div_ceil(16 * tm) * n.div_ceil(16 * tn);
        let tiles = match self.choice {
            Some(choice) => candidates[choice % candidates.len()],
            None => candidates
                .windows(2)
                .find(|pair| blocks(pair[0]) >= self.blocks && blocks(pair[0]) < blocks(pair[1]))
                .map_or(candidates[candidates.len() - 1], |pair| pair[0]),
        };
        Plan::new(shape, tiles)
    }

    fn product_plan(&self, shape: [usize; 3]) -> Plan {
        self.plan(shape, &TILES)
    }

    /// Runs `f` with cached buffers of the given lengths. Every user writes
    /// each cell before reading it.
    fn with_buffers<const N: usize>(
        &self,
        lengths: [usize; N],
        f: impl FnOnce(&mut [DeviceBuffer<f32>; N]) -> Result<()>,
    ) -> Result<()> {
        let mut cache = self
            .buffers
            .lock()
            .map_err(|_| "CUDA matrix workspace lock poisoned")?;
        let buffers = match cache.entry(lengths.to_vec()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(
                lengths
                    .iter()
                    .map(|&length| Ok(DeviceBuffer::zeroed(&self.stream, length)?))
                    .collect::<Result<_>>()?,
            ),
        };
        f(buffers
            .as_mut_slice()
            .try_into()
            .map_err(|_| "CUDA matrix workspace has the wrong buffer count")?)
    }

    /// `shape` holds the input rows, the packed features, the row stride and
    /// the first packed feature.
    fn pack_panels<const A: usize, const T: usize>(
        &self,
        panels: [usize; 3],
        shape: [usize; 4],
        padding: f32,
        input: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [width, depth, count] = panels;
        let [rows, columns, stride, offset] = shape;
        let launch = self.launches.get(config(out.len(), 128)?, |c| {
            self.module.prepare_pack_panels::<A, T>(c)
        })?;
        self.module.pack_panels(
            &self.stream,
            &launch,
            rows,
            columns,
            stride,
            offset,
            width,
            depth,
            count,
            padding,
            input,
            out,
        )?;
        Ok(())
    }

    fn pack_weight_panels<const T: usize>(
        &self,
        panels: [usize; 3],
        [rows, columns]: [usize; 2],
        input: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [width, depth, count] = panels;
        let launch = self.launches.get(config(out.len(), 128)?, |c| {
            self.module.prepare_pack_weight_panels::<T>(c)
        })?;
        self.module.pack_weight_panels(
            &self.stream,
            &launch,
            rows,
            columns,
            width,
            depth,
            count,
            input,
            out,
        )?;
        Ok(())
    }

    fn product(
        &self,
        plan: Plan,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        match plan.tiles {
            [1, 1] => self.product_with::<1, 1, 32>(plan, left, right, out),
            [2, 2] => self.product_with::<2, 2, 128>(plan, left, right, out),
            [4, 4] => self.product_with::<4, 4, 512>(plan, left, right, out),
            [8, 4] => self.product_with::<8, 4, 1024>(plan, left, right, out),
            [8, 8] => self.product_with::<8, 8, 2048>(plan, left, right, out),
            _ => Err("unsupported CUDA product tile".into()),
        }
    }

    fn product_with<const TM: usize, const TN: usize, const STRIPE: usize>(
        &self,
        plan: Plan,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let config = LaunchConfig1D::new(
            (plan.m_tiles * plan.n_tiles).try_into()?,
            256,
            (2 * 16 * 16 * (TM + TN) * 4).try_into()?,
        );
        let launch = self.launches.get(config, |c| {
            self.module.prepare_blocked_product::<TM, TN, STRIPE>(c)
        })?;
        self.module.blocked_product(
            &self.stream,
            &launch,
            plan.depth(),
            plan.depth_tiles,
            plan.m_tiles,
            plan.n_tiles,
            left,
            right,
            out,
        )?;
        Ok(())
    }

    fn rotary_product(
        &self,
        plan: Plan,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        match plan.tiles {
            [1, 1] => self.rotary_product_with::<1, 1, 32>(plan, left, right, out),
            [2, 2] => self.rotary_product_with::<2, 2, 128>(plan, left, right, out),
            [4, 4] => self.rotary_product_with::<4, 4, 512>(plan, left, right, out),
            _ => Err("unsupported CUDA rotary gradient tile".into()),
        }
    }

    fn rotary_product_with<const TM: usize, const TN: usize, const STRIPE: usize>(
        &self,
        plan: Plan,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let config = LaunchConfig1D::new(
            (plan.m_tiles * plan.n_tiles).try_into()?,
            256,
            (2 * 16 * 48 * (TM + TN) * 4).try_into()?,
        );
        let launch = self.launches.get(config, |c| {
            self.module
                .prepare_rotary_gradient_product::<TM, TN, STRIPE>(c)
        })?;
        self.module.rotary_gradient_product(
            &self.stream,
            &launch,
            plan.depth(),
            plan.depth_tiles,
            plan.m_tiles,
            plan.n_tiles,
            left,
            right,
            out,
        )?;
        Ok(())
    }

    /// `before` is read only with `S` = 1.
    fn unpack<const A: usize, const S: usize>(
        &self,
        plan: Plan,
        shape: [usize; 2],
        packed: &DeviceBuffer<f32>,
        before: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        match plan.tiles {
            [1, 1] => self.unpack_with::<A, S, 1, 1>(plan, shape, packed, before, out),
            [2, 2] => self.unpack_with::<A, S, 2, 2>(plan, shape, packed, before, out),
            [4, 4] => self.unpack_with::<A, S, 4, 4>(plan, shape, packed, before, out),
            [8, 4] => self.unpack_with::<A, S, 8, 4>(plan, shape, packed, before, out),
            [8, 8] => self.unpack_with::<A, S, 8, 8>(plan, shape, packed, before, out),
            _ => Err("unsupported CUDA unpack tile".into()),
        }
    }

    fn unpack_with<const A: usize, const S: usize, const TM: usize, const TN: usize>(
        &self,
        plan: Plan,
        [rows, columns]: [usize; 2],
        packed: &DeviceBuffer<f32>,
        before: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self.launches.get(config(out.len(), 128)?, |c| {
            self.module.prepare_unpack::<A, S, TM, TN>(c)
        })?;
        self.module.unpack(
            &self.stream,
            &launch,
            rows,
            columns,
            plan.m_tiles,
            plan.n_tiles,
            packed,
            before,
            out,
        )?;
        Ok(())
    }

    fn unpack_rotary(
        &self,
        plan: Plan,
        shape: [usize; 2],
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        match plan.tiles {
            [1, 1] => self.unpack_rotary_with::<1, 1>(plan, shape, left, right, out),
            [2, 2] => self.unpack_rotary_with::<2, 2>(plan, shape, left, right, out),
            [4, 4] => self.unpack_rotary_with::<4, 4>(plan, shape, left, right, out),
            [8, 4] => self.unpack_rotary_with::<8, 4>(plan, shape, left, right, out),
            [8, 8] => self.unpack_rotary_with::<8, 8>(plan, shape, left, right, out),
            _ => Err("unsupported CUDA rotary unpack tile".into()),
        }
    }

    fn unpack_rotary_with<const TM: usize, const TN: usize>(
        &self,
        plan: Plan,
        [rows, width]: [usize; 2],
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self.launches.get(config(out.len(), 128)?, |c| {
            self.module.prepare_unpack_rotary::<TM, TN>(c)
        })?;
        self.module.unpack_rotary(
            &self.stream,
            &launch,
            rows,
            width,
            width / 2,
            plan.m_tiles,
            plan.n_tiles,
            left,
            right,
            out,
        )?;
        Ok(())
    }

    /// `out[r][o] = sum_i f(input[r][offset + i]) W[i][o]`; `columns` holds the
    /// input row stride and offset.
    pub(super) fn forward<const A: usize>(
        &self,
        layer: Dense,
        rows: usize,
        input: &DeviceBuffer<f32>,
        columns: [usize; 2],
        weights: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [stride, offset] = columns;
        let (inputs, outputs) = (layer.inputs, layer.outputs);
        let plan = self.product_plan([rows, outputs, inputs]);
        self.with_buffers(plan.lengths(1), |[left, right, packed]| {
            let shape = [rows, inputs, stride, offset];
            self.pack_panels::<A, 1>(plan.left(), shape, -0.0, input, left)?;
            self.pack_weight_panels::<0>(plan.right(), [inputs, outputs], weights, right)?;
            self.product(plan, left, right, packed)?;
            self.unpack::<IDENTITY, 0>(plan, [rows, outputs], packed, input, out)
        })
    }

    /// `out[r][i] = f'(before[r][i]) sum_o gradient[r][o] W[i][o]`.
    pub(super) fn input_gradient<const A: usize>(
        &self,
        layer: Dense,
        rows: usize,
        gradient: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        before: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let (inputs, outputs) = (layer.inputs, layer.outputs);
        let plan = self.product_plan([rows, inputs, outputs]);
        self.with_buffers(plan.lengths(1), |[left, right, packed]| {
            let shape = [rows, outputs, outputs, 0];
            self.pack_panels::<IDENTITY, 1>(plan.left(), shape, -0.0, gradient, left)?;
            self.pack_weight_panels::<1>(plan.right(), [inputs, outputs], weights, right)?;
            self.product(plan, left, right, packed)?;
            self.unpack::<A, 1>(plan, [rows, inputs], packed, before, out)
        })
    }

    /// `out[i][o] = sum_r f(input[r][offset + i]) gradient[r][o]`.
    pub(super) fn weight_gradient<const A: usize>(
        &self,
        layer: Dense,
        rows: usize,
        input: &DeviceBuffer<f32>,
        columns: [usize; 2],
        gradient: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [stride, offset] = columns;
        let (inputs, outputs) = (layer.inputs, layer.outputs);
        let plan = self.product_plan([inputs, outputs, rows]);
        self.with_buffers(plan.lengths(1), |[left, right, packed]| {
            let shape = [rows, inputs, stride, offset];
            self.pack_panels::<A, 0>(plan.left(), shape, -0.0, input, left)?;
            let shape = [rows, outputs, outputs, 0];
            self.pack_panels::<IDENTITY, 0>(plan.right(), shape, 0.0, gradient, right)?;
            self.product(plan, left, right, packed)?;
            self.unpack::<IDENTITY, 0>(plan, [inputs, outputs], packed, gradient, out)
        })
    }

    /// `z[r][c] = sum_k f(input[r][k]) E[c][k]` for `shape = [rows, classes, width]`.
    pub(super) fn tied_readout<const A: usize>(
        &self,
        shape: [usize; 3],
        input: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [rows, classes, width] = shape;
        let plan = self.product_plan([rows, classes, width]);
        self.with_buffers(plan.lengths(1), |[left, right, packed]| {
            let shape = [rows, width, width, 0];
            self.pack_panels::<A, 1>(plan.left(), shape, -0.0, input, left)?;
            self.pack_weight_panels::<1>(plan.right(), [classes, width], weights, right)?;
            self.product(plan, left, right, packed)?;
            self.unpack::<IDENTITY, 0>(plan, [rows, classes], packed, input, out)
        })
    }

    /// `out[r][k] = f'(before[r][k]) sum_c gradient[r][c] E[c][k]`.
    pub(super) fn tied_readout_input_gradient<const A: usize>(
        &self,
        shape: [usize; 3],
        gradient: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        before: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [rows, classes, width] = shape;
        let plan = self.product_plan([rows, width, classes]);
        self.with_buffers(plan.lengths(1), |[left, right, packed]| {
            let shape = [rows, classes, classes, 0];
            self.pack_panels::<IDENTITY, 1>(plan.left(), shape, -0.0, gradient, left)?;
            self.pack_weight_panels::<0>(plan.right(), [classes, width], weights, right)?;
            self.product(plan, left, right, packed)?;
            self.unpack::<A, 1>(plan, [rows, width], packed, before, out)
        })
    }

    /// E(a) + J E(b) for rows of two `classes`-wide operand encodings.
    pub(super) fn rotary_embedding(
        &self,
        shape: [usize; 3],
        input: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [rows, classes, width] = shape;
        let plan = self.product_plan([rows, width, classes]);
        let [left, right, packed] = plan.lengths(1);
        self.with_buffers(
            [left, right, packed, packed],
            |[left, right, first, second]| {
                self.pack_weight_panels::<0>(plan.right(), [classes, width], weights, right)?;
                for (offset, product) in [(0, &mut *first), (classes, &mut *second)] {
                    let shape = [rows, classes, 2 * classes, offset];
                    self.pack_panels::<IDENTITY, 1>(plan.left(), shape, -0.0, input, left)?;
                    self.product(plan, left, right, product)?;
                }
                self.unpack_rotary(plan, [rows, width], first, second, out)
            },
        )
    }

    /// The embedding gradient from its input-a, J^T input-b and transposed
    /// readout uses: `out[c][k] = sum_r a[r][c] g[r][k] + b[r][c] (J^T g)[r][k]
    /// + z[r][c] h[r][k]`.
    pub(super) fn rotary_embedding_gradient(
        &self,
        shape: [usize; 3],
        input: &DeviceBuffer<f32>,
        input_gradient: &DeviceBuffer<f32>,
        logit_gradient: &DeviceBuffer<f32>,
        hidden: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [rows, classes, width] = shape;
        let plan = self.plan([classes, width, rows], &ROTARY_TILES);
        self.with_buffers(plan.lengths(3), |[left, right, packed]| {
            let [panel, depth, panels] = plan.left();
            let launch = self.launches.get(config(left.len(), 128)?, |c| {
                self.module.prepare_pack_rotary_residues::<0>(c)
            })?;
            self.module.pack_rotary_residues(
                &self.stream,
                &launch,
                rows,
                classes,
                panel,
                depth,
                panels,
                input,
                logit_gradient,
                left,
            )?;
            let [panel, depth, panels] = plan.right();
            let launch = self.launches.get(config(right.len(), 128)?, |c| {
                self.module.prepare_pack_rotary_channels::<0>(c)
            })?;
            self.module.pack_rotary_channels(
                &self.stream,
                &launch,
                rows,
                width,
                width / 2,
                panel,
                depth,
                panels,
                input_gradient,
                hidden,
                right,
            )?;
            self.rotary_product(plan, left, right, packed)?;
            self.unpack::<IDENTITY, 0>(plan, [classes, width], packed, hidden, out)
        })
    }
}

fn compare(actual: &[f32], expected: &[f64], label: &str) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: length mismatch").into());
    }
    let scale = expected.iter().fold(1.0_f64, |s, v| s.max(v.abs()));
    let mut error = 0.0_f64;
    for (&a, &b) in actual.iter().zip(expected) {
        if !a.is_finite() || !b.is_finite() {
            return Err(format!("{label}: nonfinite value").into());
        }
        error = error.max((f64::from(a) - b).abs());
    }
    if error > 1e-5 * scale {
        return Err(format!("{label}: error {error}, scale {scale}").into());
    }
    Ok(())
}

/// Checks the automatic tile choice against f64 and every forced tile shape
/// bit for bit against the automatic choice.
fn compare_choices(results: &[Vec<f32>], expected: &[f64], label: &str) -> Result<()> {
    compare(&results[0], expected, label)?;
    for (choice, result) in results.iter().enumerate().skip(1) {
        let same = result.len() == results[0].len()
            && result
                .iter()
                .zip(&results[0])
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !same {
            return Err(format!("{label}: forced tile choice {} changed bits", choice - 1).into());
        }
    }
    Ok(())
}

fn wave(count: usize, frequency: f64, amplitude: f64) -> Vec<f32> {
    (0..count)
        .map(|i| ((i as f64 * frequency).sin() * amplitude) as f32)
        .collect()
}

fn activation<const A: usize>(v: f64) -> f64 {
    if A == RELU {
        v.max(0.0)
    } else if A == SQUARE {
        v * v
    } else {
        v
    }
}

fn slope<const A: usize>(v: f64) -> f64 {
    if A == RELU {
        if v > 0.0 { 1.0 } else { 0.0 }
    } else if A == SQUARE {
        2.0 * v
    } else {
        1.0
    }
}

fn verify_dense<const A: usize>(choices: &[Kernels], shape: [usize; 5]) -> Result<()> {
    let [rows, inputs, outputs, stride, offset] = shape;
    let stream = &choices[0].stream;
    let input = wave(rows * stride, 0.31, 0.4);
    let weights: Vec<_> = wave(inputs * outputs, 0.17, 0.2)
        .into_iter()
        .map(|w| [w, 0.0, 0.0])
        .collect();
    let gradient = wave(rows * outputs, 0.43, 0.3);
    let before = wave(rows * inputs, 0.19, 0.4);
    let input_d = DeviceBuffer::from_host(stream, &input)?;
    let weights_d = DeviceBuffer::from_host(stream, &weights)?;
    let gradient_d = DeviceBuffer::from_host(stream, &gradient)?;
    let before_d = DeviceBuffer::from_host(stream, &before)?;
    let layer = Dense { inputs, outputs };
    let mut results = [Vec::new(), Vec::new(), Vec::new()];
    for kernels in choices {
        let mut forward = DeviceBuffer::zeroed(stream, rows * outputs)?;
        let mut dx = DeviceBuffer::zeroed(stream, rows * inputs)?;
        let mut dw = DeviceBuffer::zeroed(stream, inputs * outputs)?;
        let columns = [stride, offset];
        kernels.forward::<A>(layer, rows, &input_d, columns, &weights_d, &mut forward)?;
        kernels.input_gradient::<A>(layer, rows, &gradient_d, &weights_d, &before_d, &mut dx)?;
        kernels.weight_gradient::<A>(layer, rows, &input_d, columns, &gradient_d, &mut dw)?;
        for (result, buffer) in results.iter_mut().zip([forward, dx, dw]) {
            result.push(buffer.to_host_vec(stream)?);
        }
    }
    let mut expected_forward = vec![0.0; rows * outputs];
    let mut expected_dx = vec![0.0; rows * inputs];
    let mut expected_dw = vec![0.0; inputs * outputs];
    for r in 0..rows {
        for i in 0..inputs {
            let x = activation::<A>(f64::from(input[r * stride + offset + i]));
            for o in 0..outputs {
                let w = f64::from(weights[i * outputs + o][0]);
                let g = f64::from(gradient[r * outputs + o]);
                expected_forward[r * outputs + o] += x * w;
                expected_dx[r * inputs + i] += g * w;
                expected_dw[i * outputs + o] += x * g;
            }
        }
    }
    for (x, &b) in expected_dx.iter_mut().zip(&before) {
        *x *= slope::<A>(f64::from(b));
    }
    let [forward, dx, dw] = &results;
    compare_choices(forward, &expected_forward, "tiled CUDA forward")?;
    compare_choices(dx, &expected_dx, "tiled CUDA input gradients")?;
    compare_choices(dw, &expected_dw, "tiled CUDA weight gradients")?;
    println!(
        "PASS tiled CUDA A={A}, {rows}x{inputs}x{outputs}, stride={stride}, offset={offset}: all outputs and both gradients match f64 and every tile shape"
    );
    Ok(())
}

/// The tied readout and its input gradient, and with `A` = identity the
/// rotary embedding and its gradient, on dense rather than one-hot inputs.
fn verify_embedding<const A: usize>(choices: &[Kernels], shape: [usize; 3]) -> Result<()> {
    let [rows, classes, width] = shape;
    let stream = &choices[0].stream;
    let operands = wave(rows * 2 * classes, 0.29, 0.5);
    let weights: Vec<_> = wave(classes * width, 0.13, 0.3)
        .into_iter()
        .map(|w| [w, 0.0, 0.0])
        .collect();
    let hidden = wave(rows * width, 0.37, 0.6);
    let logit_gradient = wave(rows * classes, 0.41, 0.2);
    let gradient = wave(rows * width, 0.23, 0.3);
    let operands_d = DeviceBuffer::from_host(stream, &operands)?;
    let weights_d = DeviceBuffer::from_host(stream, &weights)?;
    let hidden_d = DeviceBuffer::from_host(stream, &hidden)?;
    let logit_gradient_d = DeviceBuffer::from_host(stream, &logit_gradient)?;
    let gradient_d = DeviceBuffer::from_host(stream, &gradient)?;
    let rotary = A == IDENTITY;
    let mut results = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for kernels in choices {
        let mut logits = DeviceBuffer::zeroed(stream, rows * classes)?;
        let mut dn = DeviceBuffer::zeroed(stream, rows * width)?;
        let mut embedded = DeviceBuffer::zeroed(stream, rows * width)?;
        let mut de = DeviceBuffer::zeroed(stream, classes * width)?;
        kernels.tied_readout::<A>(shape, &hidden_d, &weights_d, &mut logits)?;
        kernels.tied_readout_input_gradient::<A>(
            shape,
            &logit_gradient_d,
            &weights_d,
            &hidden_d,
            &mut dn,
        )?;
        if rotary {
            kernels.rotary_embedding(shape, &operands_d, &weights_d, &mut embedded)?;
            kernels.rotary_embedding_gradient(
                shape,
                &operands_d,
                &gradient_d,
                &logit_gradient_d,
                &hidden_d,
                &mut de,
            )?;
        }
        for (result, buffer) in results.iter_mut().zip([logits, dn, embedded, de]) {
            result.push(buffer.to_host_vec(stream)?);
        }
    }
    let e = |c: usize, k: usize| f64::from(weights[c * width + k][0]);
    let mut expected_logits = vec![0.0; rows * classes];
    let mut expected_dn = vec![0.0; rows * width];
    let mut expected_embedded = vec![0.0; rows * width];
    let mut expected_de = vec![0.0; classes * width];
    for r in 0..rows {
        for c in 0..classes {
            let a = f64::from(operands[r * 2 * classes + c]);
            let b = f64::from(operands[r * 2 * classes + classes + c]);
            let z = f64::from(logit_gradient[r * classes + c]);
            for k in 0..width {
                let h = f64::from(hidden[r * width + k]);
                let g = f64::from(gradient[r * width + k]);
                let paired = f64::from(gradient[r * width + (k ^ 1)]);
                let sign = if k.is_multiple_of(2) { -1.0 } else { 1.0 };
                expected_logits[r * classes + c] += activation::<A>(h) * e(c, k);
                expected_dn[r * width + k] += z * e(c, k);
                if rotary {
                    expected_embedded[r * width + k] += a * e(c, k) + sign * b * e(c, k ^ 1);
                    expected_de[c * width + k] += a * g - sign * b * paired + z * h;
                }
            }
        }
    }
    for (x, &h) in expected_dn.iter_mut().zip(&hidden) {
        *x *= slope::<A>(f64::from(h));
    }
    let [logits, dn, embedded, de] = &results;
    compare_choices(logits, &expected_logits, "tiled CUDA tied readout")?;
    compare_choices(dn, &expected_dn, "tiled CUDA tied readout input gradients")?;
    if rotary {
        compare_choices(embedded, &expected_embedded, "tiled CUDA rotary embedding")?;
        compare_choices(de, &expected_de, "tiled CUDA rotary embedding gradients")?;
    }
    println!(
        "PASS tiled CUDA A={A}, rows={rows}, classes={classes}, width={width}: tied readout{} match f64 and every tile shape",
        if rotary {
            ", rotary embedding and both gradients"
        } else {
            " and its input gradient"
        }
    );
    Ok(())
}

pub(super) fn verify(device: usize) -> Result<()> {
    let context = CudaContext::new(device)?;
    let stream = context.default_stream();
    // The automatic choice first, then every forced tile shape.
    let choices = (0..=TILES.len())
        .map(|choice| Kernels::with_choice(&context, &stream, choice.checked_sub(1)))
        .collect::<Result<Vec<_>>>()?;
    for shape in [
        [1, 97, 1024, 103, 3],
        [7, 513, 97, 519, 4],
        [48, 1024, 97, 1024, 0],
        [37, 1024, 1024, 1024, 0],
        [512, 128, 129, 134, 5],
        [300, 200, 700, 200, 0],
    ] {
        verify_dense::<IDENTITY>(&choices, shape)?;
        verify_dense::<RELU>(&choices, shape)?;
        verify_dense::<SQUARE>(&choices, shape)?;
    }
    for shape in [[1, 7, 6], [48, 97, 16], [37, 13, 130], [512, 97, 1024]] {
        verify_embedding::<IDENTITY>(&choices, shape)?;
        verify_embedding::<RELU>(&choices, shape)?;
        verify_embedding::<SQUARE>(&choices, shape)?;
    }
    Ok(())
}
