use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn vecadd(a: &[f32], b: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = output.get_mut(idx) {
            *value = a[i] + b[i];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 1024;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
    let a_device = DeviceBuffer::from_host(&stream, &a)?;
    let b_device = DeviceBuffer::from_host(&stream, &b)?;
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    // SAFETY: this package owns the device bundle compiled from kernels above.
    let module = unsafe { kernels::load(&ctx)? };
    let launch = module.prepare_vecadd(LaunchConfig1D::new((N as u32).div_ceil(256), 256, 0))?;
    module.vecadd(&stream, &launch, &a_device, &b_device, &mut output)?;

    let result = output.to_host_vec(&stream)?;
    for (i, value) in result.iter().enumerate() {
        let expected = a[i] + b[i];
        if !value.is_finite() || (*value - expected).abs() > 1e-5 {
            return Err(format!("vecadd[{i}]: expected {expected}, got {value}").into());
        }
    }

    println!("CUDA vecadd: all {N} elements correct");
    Ok(())
}
