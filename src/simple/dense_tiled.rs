//! Shared-memory CUDA matrix products for the large Dense layers.
//! The compiler stages read-only tiles and supplies both block barriers.

use super::{Dense, IDENTITY, RELU, SQUARE};
use crate::Result;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::shared::BlockTile;
use cuda_device::{DisjointSlice, device, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[cuda_module]
#[allow(clippy::too_many_arguments)]
mod kernels {
    use super::*;
    #[device]
    fn dense_tiled_activate(value: f32, code: usize) -> f32 {
        if code == RELU {
            value.max(0.0)
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

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        TRANSPOSE <= 1, rows >= 1, columns >= 1, stride >= 1,
        offset + columns <= stride, input.len() == rows * stride,
        tiles >= 1, padded == 16 * tiles,
        padded >= TRANSPOSE * rows + (1 - TRANSPOSE) * columns,
        padded < TRANSPOSE * rows + (1 - TRANSPOSE) * columns + 16,
        out.len() == (TRANSPOSE * columns + (1 - TRANSPOSE) * rows) * padded))]
    pub fn pack_input<const ACTIVATION: usize, const TRANSPOSE: usize>(
        rows: usize,
        columns: usize,
        stride: usize,
        offset: usize,
        padded: usize,
        tiles: usize,
        input: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let _ = (tiles, padded);
        let reduction = TRANSPOSE * columns + (1 - TRANSPOSE) * rows;
        let tile = index.get() / (reduction * 16);
        let within = index.get() % (reduction * 16);
        let packed_column = tile * 16 + within % 16;
        let packed_row = within / 16;
        let row = if TRANSPOSE == 1 {
            packed_column
        } else {
            packed_row
        };
        let column = if TRANSPOSE == 1 {
            packed_row
        } else {
            packed_column
        };
        if let Some(value) = out.get_mut(index) {
            *value = if row < rows && column < columns {
                dense_tiled_activate(input[row * stride + offset + column], ACTIVATION)
            } else {
                0.0
            };
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        TRANSPOSE <= 1, rows >= 1, columns >= 1, input.len() == rows * columns,
        tiles >= 1, padded == 16 * tiles,
        padded >= TRANSPOSE * rows + (1 - TRANSPOSE) * columns,
        padded < TRANSPOSE * rows + (1 - TRANSPOSE) * columns + 16,
        out.len() == (TRANSPOSE * columns + (1 - TRANSPOSE) * rows) * padded))]
    pub fn pack_weights<const TRANSPOSE: usize>(
        rows: usize,
        columns: usize,
        padded: usize,
        tiles: usize,
        input: &[[f32; 3]],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let _ = (tiles, padded);
        let reduction = TRANSPOSE * columns + (1 - TRANSPOSE) * rows;
        let tile = index.get() / (reduction * 16);
        let within = index.get() % (reduction * 16);
        let packed_column = tile * 16 + within % 16;
        let packed_row = within / 16;
        let row = if TRANSPOSE == 1 {
            packed_column
        } else {
            packed_row
        };
        let column = if TRANSPOSE == 1 {
            packed_row
        } else {
            packed_column
        };
        if let Some(value) = out.get_mut(index) {
            *value = if row < rows && column < columns {
                input[row * columns + column][0]
            } else {
                0.0
            };
        }
    }

    // Both sources are K-major. Padding removes cross-row tail ambiguity, and
    // packed output tiles give each ordinary 1D index exactly one owned cell.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (
        reduction >= 1, m_tiles >= 1, n_tiles >= 1, m == 16 * m_tiles, n == 16 * n_tiles,
        left.len() == reduction * m, right.len() == reduction * n,
        out.len() == m * n))]
    #[block_pipeline(rows = 16, iterations = (reduction + 15) / 16)]
    pub fn tiled_product<const S: usize>(
        reduction: usize,
        m: usize,
        n: usize,
        m_tiles: usize,
        n_tiles: usize,
        #[tile_source(columns = 16, stride = 16, offset = (block / n_tiles) * reduction * 16)] left: &[f32],
        #[tile_source(columns = 16, stride = 16, offset = (block % n_tiles) * reduction * 16)]
        right: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let init = || {
            let _ = (S, m, n, m_tiles);
            0.0_f32
        };
        let step =
            |sum: &mut f32, iteration: usize, a: BlockTile<'_, f32>, b: BlockTile<'_, f32>| {
                let lane = thread::threadIdx_x() as usize;
                let row = lane / 16;
                let column = lane % 16;
                let mut k = 0;
                #[unroll]
                while k < 16 {
                    if iteration * 16 + k < reduction {
                        *sum += a.load(k, row) * b.load(k, column);
                    }
                    k += 1;
                }
            };
        let mut finish = |sum: f32| {
            if let Some(value) = out.get_mut(thread::index_1d()) {
                *value = sum;
            }
        };
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1), requires = (
        SLOPE <= 1, rows >= 1, columns >= 1, m_tiles >= 1, n_tiles >= 1,
        m == 16 * m_tiles, n == 16 * n_tiles,
        m >= rows, m < rows + 16, n >= columns, n < columns + 16,
        packed.len() == m * n, out.len() == rows * columns,
        SLOPE * before.len() == SLOPE * out.len()))]
    pub fn unpack<const ACTIVATION: usize, const SLOPE: usize>(
        rows: usize,
        columns: usize,
        m: usize,
        n: usize,
        m_tiles: usize,
        n_tiles: usize,
        packed: &[f32],
        before: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let item = index.get();
        let row = item / columns;
        let column = item % columns;
        if let Some(value) = out.get_mut(index) {
            let address = ((row / 16) * (n / 16) + column / 16) * 256 + row % 16 * 16 + column % 16;
            *value = packed[address]
                * if SLOPE == 1 {
                    dense_tiled_slope(before[item], ACTIVATION)
                } else {
                    1.0
                };
        }
        let _ = (rows, m, m_tiles, n_tiles);
    }
}

struct Workspace {
    left: DeviceBuffer<f32>,
    right: DeviceBuffer<f32>,
    packed: DeviceBuffer<f32>,
}
impl Workspace {
    fn new(stream: &Arc<CudaStream>, m: usize, n: usize, k: usize) -> Result<Self> {
        let m = m.div_ceil(16) * 16;
        let n = n.div_ceil(16) * 16;
        Ok(Self {
            left: DeviceBuffer::zeroed(stream, k * m)?,
            right: DeviceBuffer::zeroed(stream, k * n)?,
            packed: DeviceBuffer::zeroed(stream, m * n)?,
        })
    }
}
pub(super) struct Kernels {
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    workspaces: Mutex<BTreeMap<[usize; 3], Workspace>>,
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
        Ok(Self {
            stream: stream.clone(),
            module: kernels::load(context)?,
            workspaces: Mutex::new(BTreeMap::new()),
        })
    }
    fn pack_input<const A: usize, const T: usize>(
        &self,
        rows: usize,
        columns: usize,
        stride: usize,
        offset: usize,
        padded: usize,
        input: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self
            .module
            .prepare_pack_input::<A, T>(config(out.len(), 128)?)?;
        self.module.pack_input(
            &self.stream,
            &launch,
            rows,
            columns,
            stride,
            offset,
            padded,
            padded / 16,
            input,
            out,
        )?;
        Ok(())
    }
    fn pack_weights<const T: usize>(
        &self,
        rows: usize,
        columns: usize,
        padded: usize,
        input: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let launch = self
            .module
            .prepare_pack_weights::<T>(config(out.len(), 128)?)?;
        self.module.pack_weights(
            &self.stream,
            &launch,
            rows,
            columns,
            padded,
            padded / 16,
            input,
            out,
        )?;
        Ok(())
    }
    fn product<const A: usize, const S: usize>(
        &self,
        shape: [usize; 3],
        workspace: &mut Workspace,
        before: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let [rows, columns, reduction] = shape;
        let m = rows.div_ceil(16) * 16;
        let n = columns.div_ceil(16) * 16;
        let launch = self.module.prepare_tiled_product::<0>(LaunchConfig1D::new(
            workspace.packed.len().div_ceil(256).try_into()?,
            256,
            2 * 2 * 16 * 16 * 4,
        ))?;
        self.module.tiled_product(
            &self.stream,
            &launch,
            reduction,
            m,
            n,
            m / 16,
            n / 16,
            &workspace.left,
            &workspace.right,
            &mut workspace.packed,
        )?;
        let launch = self
            .module
            .prepare_unpack::<A, S>(config(out.len(), 128)?)?;
        self.module.unpack(
            &self.stream,
            &launch,
            rows,
            columns,
            m,
            n,
            m / 16,
            n / 16,
            &workspace.packed,
            before,
            out,
        )?;
        Ok(())
    }
    pub(super) fn forward<const A: usize>(
        &self,
        layer: Dense,
        rows: usize,
        input: &DeviceBuffer<f32>,
        columns: [usize; 2],
        weights: &DeviceBuffer<[f32; 3]>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let shape = [rows, layer.outputs, layer.inputs];
        let [stride, offset] = columns;
        let mut cache = self
            .workspaces
            .lock()
            .map_err(|_| "CUDA matrix workspace lock poisoned")?;
        if !cache.contains_key(&shape) {
            cache.insert(
                shape,
                Workspace::new(&self.stream, shape[0], shape[1], shape[2])?,
            );
        }
        let workspace = cache.get_mut(&shape).unwrap();
        self.pack_input::<A, 1>(
            rows,
            layer.inputs,
            stride,
            offset,
            rows.div_ceil(16) * 16,
            input,
            &mut workspace.left,
        )?;
        self.pack_weights::<0>(
            layer.inputs,
            layer.outputs,
            layer.outputs.div_ceil(16) * 16,
            weights,
            &mut workspace.right,
        )?;
        self.product::<IDENTITY, 0>(shape, workspace, input, out)
    }
    pub(super) fn input_gradient<const A: usize>(
        &self,
        layer: Dense,
        rows: usize,
        gradient: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<[f32; 3]>,
        before: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let shape = [rows, layer.inputs, layer.outputs];
        let mut cache = self
            .workspaces
            .lock()
            .map_err(|_| "CUDA matrix workspace lock poisoned")?;
        if !cache.contains_key(&shape) {
            cache.insert(
                shape,
                Workspace::new(&self.stream, shape[0], shape[1], shape[2])?,
            );
        }
        let workspace = cache.get_mut(&shape).unwrap();
        self.pack_input::<IDENTITY, 1>(
            rows,
            layer.outputs,
            layer.outputs,
            0,
            rows.div_ceil(16) * 16,
            gradient,
            &mut workspace.left,
        )?;
        self.pack_weights::<1>(
            layer.inputs,
            layer.outputs,
            layer.inputs.div_ceil(16) * 16,
            weights,
            &mut workspace.right,
        )?;
        self.product::<A, 1>(shape, workspace, before, out)
    }
    pub(super) fn weight_gradient<const A: usize>(
        &self,
        layer: Dense,
        rows: usize,
        input: &DeviceBuffer<f32>,
        columns: [usize; 2],
        gradient: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let shape = [layer.inputs, layer.outputs, rows];
        let [stride, offset] = columns;
        let mut cache = self
            .workspaces
            .lock()
            .map_err(|_| "CUDA matrix workspace lock poisoned")?;
        if !cache.contains_key(&shape) {
            cache.insert(
                shape,
                Workspace::new(&self.stream, shape[0], shape[1], shape[2])?,
            );
        }
        let workspace = cache.get_mut(&shape).unwrap();
        self.pack_input::<A, 0>(
            rows,
            layer.inputs,
            stride,
            offset,
            layer.inputs.div_ceil(16) * 16,
            input,
            &mut workspace.left,
        )?;
        self.pack_input::<IDENTITY, 0>(
            rows,
            layer.outputs,
            layer.outputs,
            0,
            layer.outputs.div_ceil(16) * 16,
            gradient,
            &mut workspace.right,
        )?;
        self.product::<IDENTITY, 0>(shape, workspace, input, out)
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

fn verify_case<const A: usize>(kernels: &Kernels, shape: [usize; 5]) -> Result<()> {
    let [rows, inputs, outputs, stride, offset] = shape;
    let input: Vec<_> = (0..rows * stride)
        .map(|i| ((i as f64 * 0.31).sin() * 0.4) as f32)
        .collect();
    let weights: Vec<_> = (0..inputs * outputs)
        .map(|i| [((i as f64 * 0.17).cos() * 0.2) as f32, 0.0, 0.0])
        .collect();
    let gradient: Vec<_> = (0..rows * outputs)
        .map(|i| ((i as f64 * 0.43).sin() * 0.3) as f32)
        .collect();
    let before: Vec<_> = (0..rows * inputs)
        .map(|i| ((i as f64 * 0.19).cos() * 0.4) as f32)
        .collect();
    let input_d = DeviceBuffer::from_host(&kernels.stream, &input)?;
    let weights_d = DeviceBuffer::from_host(&kernels.stream, &weights)?;
    let gradient_d = DeviceBuffer::from_host(&kernels.stream, &gradient)?;
    let before_d = DeviceBuffer::from_host(&kernels.stream, &before)?;
    let mut forward = DeviceBuffer::zeroed(&kernels.stream, rows * outputs)?;
    let mut dx = DeviceBuffer::zeroed(&kernels.stream, rows * inputs)?;
    let mut dw = DeviceBuffer::zeroed(&kernels.stream, inputs * outputs)?;
    let layer = Dense { inputs, outputs };
    kernels.forward::<A>(
        layer,
        rows,
        &input_d,
        [stride, offset],
        &weights_d,
        &mut forward,
    )?;
    kernels.input_gradient::<A>(layer, rows, &gradient_d, &weights_d, &before_d, &mut dx)?;
    kernels.weight_gradient::<A>(
        layer,
        rows,
        &input_d,
        [stride, offset],
        &gradient_d,
        &mut dw,
    )?;
    let activation = |v: f64| {
        if A == RELU {
            v.max(0.0)
        } else if A == SQUARE {
            v * v
        } else {
            v
        }
    };
    let slope = |v: f64| {
        if A == RELU {
            if v > 0.0 { 1.0 } else { 0.0 }
        } else if A == SQUARE {
            2.0 * v
        } else {
            1.0
        }
    };
    let mut expected_forward = vec![0.0; rows * outputs];
    let mut expected_dx = vec![0.0; rows * inputs];
    let mut expected_dw = vec![0.0; inputs * outputs];
    for r in 0..rows {
        for i in 0..inputs {
            for o in 0..outputs {
                let x = activation(f64::from(input[r * stride + offset + i]));
                let w = f64::from(weights[i * outputs + o][0]);
                let g = f64::from(gradient[r * outputs + o]);
                expected_forward[r * outputs + o] += x * w;
                expected_dx[r * inputs + i] += g * w;
                expected_dw[i * outputs + o] += x * g;
            }
        }
    }
    for (x, &b) in expected_dx.iter_mut().zip(&before) {
        *x *= slope(f64::from(b));
    }
    compare(
        &forward.to_host_vec(&kernels.stream)?,
        &expected_forward,
        "tiled CUDA forward",
    )?;
    compare(
        &dx.to_host_vec(&kernels.stream)?,
        &expected_dx,
        "tiled CUDA input gradients",
    )?;
    compare(
        &dw.to_host_vec(&kernels.stream)?,
        &expected_dw,
        "tiled CUDA weight gradients",
    )?;
    println!(
        "PASS tiled CUDA A={A}, {rows}x{inputs}x{outputs}, stride={stride}, offset={offset}: all outputs and both gradients match f64"
    );
    Ok(())
}

pub(super) fn verify(device: usize) -> Result<()> {
    let context = CudaContext::new(device)?;
    let stream = context.default_stream();
    let kernels = Kernels::new(&context, &stream)?;
    for shape in [
        [1, 97, 1024, 103, 3],
        [7, 513, 97, 519, 4],
        [48, 1024, 97, 1024, 0],
        [37, 1024, 1024, 1024, 0],
        [512, 128, 129, 134, 5],
    ] {
        verify_case::<IDENTITY>(&kernels, shape)?;
        verify_case::<RELU>(&kernels, shape)?;
        verify_case::<SQUARE>(&kernels, shape)?;
    }
    Ok(())
}
