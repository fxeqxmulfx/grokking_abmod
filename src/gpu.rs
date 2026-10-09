use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, device, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

use crate::mesh::{
    GRADIENT_SIZE, MAX_NODES, MAX_PARAMETERS, MISSING, validate_finite, validate_training_data,
};
use crate::{MeshTopology, ReferenceMesh, Result};

const BLOCK: u32 = 128;
const MAX_BATCH_SIZE: usize = 65_536;

#[cuda_module]
mod kernels {
    use super::*;

    #[device]
    fn evaluate(x: f32, nodes: &[[u32; 5]], state: &[[f32; 3]], values: &mut [f32; MAX_NODES + 1]) {
        values[0] = x;
        for (offset, node) in nodes.iter().enumerate() {
            let mut value = state[node[4] as usize][0];
            if node[0] != MISSING {
                value += state[node[2] as usize][0] * values[node[0] as usize];
            }
            if node[1] != MISSING {
                value += state[node[3] as usize][0] * values[node[1] as usize];
            }
            values[offset + 1] = if value >= 0.0 {
                value
            } else {
                value.exp() - 1.0
            };
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1),
        requires = (nodes.len() <= 288, state.len() <= 832, out.len() == inputs.len()))]
    pub fn forward(
        nodes: &[[u32; 5]],
        state: &[[f32; 3]],
        inputs: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let index = thread::index_1d();
        let sample = index.get();
        if let Some(slot) = out.get_mut(index) {
            let mut values = [0.0; MAX_NODES + 1];
            evaluate(inputs[sample], nodes, state, &mut values);
            *slot = values[nodes.len()];
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1),
        requires = (nodes.len() <= 288, state.len() <= 832,
            inputs.len() == targets.len(), out.len() == inputs.len()))]
    pub fn backward(
        nodes: &[[u32; 5]],
        state: &[[f32; 3]],
        inputs: &[f32],
        targets: &[f32],
        mut out: DisjointSlice<[f32; GRADIENT_SIZE]>,
    ) {
        let index = thread::index_1d();
        let sample = index.get();
        if let Some(gradient) = out.get_mut(index) {
            let mut values = [0.0; MAX_NODES + 1];
            let mut adjoints = [0.0; MAX_NODES + 1];
            evaluate(inputs[sample], nodes, state, &mut values);
            let error = values[nodes.len()] - targets[sample];
            gradient[MAX_PARAMETERS] = error * error;
            for slot in gradient.iter_mut().take(state.len()) {
                *slot = 0.0;
            }
            adjoints[nodes.len()] = 2.0 * error;
            for offset in (0..nodes.len()).rev() {
                let node = nodes[offset];
                let value = values[offset + 1];
                let derivative = if value >= 0.0 { 1.0 } else { value + 1.0 };
                let delta = adjoints[offset + 1] * derivative;
                gradient[node[4] as usize] = delta;
                if node[0] != MISSING {
                    gradient[node[2] as usize] = delta * values[node[0] as usize];
                    adjoints[node[0] as usize] += delta * state[node[2] as usize][0];
                }
                if node[1] != MISSING {
                    gradient[node[3] as usize] = delta * values[node[1] as usize];
                    adjoints[node[1] as usize] += delta * state[node[3] as usize][0];
                }
            }
        }
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1),
        requires = (gradients.len() >= 1, state.len() <= 832))]
    pub fn adam(
        learning_rate: f32,
        correction1: f32,
        correction2: f32,
        gradients: &[[f32; GRADIENT_SIZE]],
        mut state: DisjointSlice<[f32; 3]>,
    ) {
        let index = thread::index_1d();
        let parameter = index.get();
        if let Some(slot) = state.get_mut(index) {
            let mut sum = 0.0;
            for gradient in gradients {
                sum += gradient[parameter];
            }
            let gradient = sum / gradients.len() as f32;
            slot[1] = 0.9 * slot[1] + 0.1 * gradient;
            slot[2] = 0.999 * slot[2] + 0.001 * gradient * gradient;
            let m = slot[1] / correction1;
            let v = slot[2] / correction2;
            slot[0] -= learning_rate * m / (v.sqrt() + 1e-8);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrainOptions {
    pub epochs: usize,
    pub learning_rate: f32,
    pub report_every: usize,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            epochs: 3_000,
            learning_rate: 0.01,
            report_every: 250,
        }
    }
}

impl TrainOptions {
    pub fn validate(&self) -> Result<()> {
        if self.epochs == 0 || self.report_every == 0 {
            return Err("epochs and report_every must be positive".into());
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive and finite".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LossPoint {
    pub epoch: usize,
    pub mse: f64,
}

#[derive(Debug)]
pub struct TrainReport {
    pub history: Vec<LossPoint>,
    pub elapsed: Duration,
}

/// A scalar-input, scalar-output ELU diamond mesh.
///
/// Parameters, activations, gradients, and Adam updates are evaluated on the
/// GPU. Safe typed launch contracts and disjoint sample/parameter ownership
/// handle all kernel launches; no raw pointers or atomics are used.
pub struct GpuMesh {
    topology: MeshTopology,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    nodes: DeviceBuffer<[u32; 5]>,
    state: DeviceBuffer<[f32; 3]>,
    beta1_power: f32,
    beta2_power: f32,
}

impl GpuMesh {
    pub fn new(device: usize, depth: usize, seed: u64) -> Result<Self> {
        let reference = ReferenceMesh::new(depth, seed)?;
        let topology = reference.topology().clone();
        let context = CudaContext::new(device)?;
        let stream = context.default_stream();
        // SAFETY: this crate owns the embedded bundle compiled from kernels
        // above; no caller-supplied device artifact is accepted.
        let module = unsafe { kernels::load(&context)? };
        let nodes = DeviceBuffer::from_host(&stream, &topology.descriptors())?;
        let initial: Vec<_> = reference
            .parameters()
            .iter()
            .map(|&weight| [weight, 0.0, 0.0])
            .collect();
        let state = DeviceBuffer::from_host(&stream, &initial)?;
        Ok(Self {
            topology,
            stream,
            module,
            nodes,
            state,
            beta1_power: 1.0,
            beta2_power: 1.0,
        })
    }

    pub fn topology(&self) -> &MeshTopology {
        &self.topology
    }

    pub fn parameters(&self) -> Result<Vec<f32>> {
        Ok(self
            .state
            .to_host_vec(&self.stream)?
            .iter()
            .map(|slot| slot[0])
            .collect())
    }

    /// Replaces model parameters and resets Adam's moments and step history.
    pub fn set_parameters(&mut self, parameters: &[f32]) -> Result<()> {
        if parameters.len() != self.topology.parameter_count() {
            return Err("parameter count does not match the mesh topology".into());
        }
        validate_finite(parameters, "parameters")?;
        let state: Vec<_> = parameters.iter().map(|&p| [p, 0.0, 0.0]).collect();
        self.state.copy_from_host(&self.stream, &state)?;
        self.beta1_power = 1.0;
        self.beta2_power = 1.0;
        Ok(())
    }

    pub fn predict(&self, inputs: &[f32]) -> Result<Vec<f32>> {
        validate_finite(inputs, "inputs")?;
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let config = launch_config(inputs.len())?;
        let input = DeviceBuffer::from_host(&self.stream, inputs)?;
        let mut output = DeviceBuffer::<f32>::zeroed(&self.stream, inputs.len())?;
        let launch = self.module.prepare_forward(config)?;
        self.module.forward(
            &self.stream,
            &launch,
            &self.nodes,
            &self.state,
            &input,
            &mut output,
        )?;
        let predictions = output.to_host_vec(&self.stream)?;
        validate_finite(&predictions, "predictions")?;
        Ok(predictions)
    }

    /// Returns GPU-computed MSE and mean gradients for diagnostics.
    pub fn loss_and_gradient(&self, inputs: &[f32], targets: &[f32]) -> Result<(f64, Vec<f64>)> {
        validate_batch(inputs, targets)?;
        let input = DeviceBuffer::from_host(&self.stream, inputs)?;
        let target = DeviceBuffer::from_host(&self.stream, targets)?;
        let mut gradients =
            DeviceBuffer::<[f32; GRADIENT_SIZE]>::zeroed(&self.stream, inputs.len())?;
        let launch = self.module.prepare_backward(launch_config(inputs.len())?)?;
        self.module.backward(
            &self.stream,
            &launch,
            &self.nodes,
            &self.state,
            &input,
            &target,
            &mut gradients,
        )?;
        let rows = gradients.to_host_vec(&self.stream)?;
        let mut mean = vec![0.0; self.topology.parameter_count()];
        let mut loss = 0.0;
        for row in rows {
            loss += f64::from(row[MAX_PARAMETERS]);
            for (parameter, total) in mean.iter_mut().enumerate() {
                *total += f64::from(row[parameter]);
            }
        }
        for value in &mut mean {
            *value /= inputs.len() as f64;
        }
        if !loss.is_finite() || mean.iter().any(|value| !value.is_finite()) {
            return Err("loss or gradients overflowed; use smaller parameters or data".into());
        }
        Ok((loss / inputs.len() as f64, mean))
    }

    /// Full-batch Adam. Successive calls continue the existing optimizer state.
    pub fn train(
        &mut self,
        inputs: &[f32],
        targets: &[f32],
        options: TrainOptions,
        mut on_report: impl FnMut(LossPoint),
    ) -> Result<TrainReport> {
        options.validate()?;
        validate_batch(inputs, targets)?;
        let start = Instant::now();
        let initial = LossPoint {
            epoch: 0,
            mse: mse(&self.predict(inputs)?, targets),
        };
        let mut history = vec![initial];
        on_report(initial);
        let input = DeviceBuffer::from_host(&self.stream, inputs)?;
        let target = DeviceBuffer::from_host(&self.stream, targets)?;
        let mut gradients =
            DeviceBuffer::<[f32; GRADIENT_SIZE]>::zeroed(&self.stream, inputs.len())?;
        let backward = self.module.prepare_backward(launch_config(inputs.len())?)?;
        let adam = self.module.prepare_adam(launch_config(self.state.len())?)?;
        for epoch in 1..=options.epochs {
            self.module.backward(
                &self.stream,
                &backward,
                &self.nodes,
                &self.state,
                &input,
                &target,
                &mut gradients,
            )?;
            self.beta1_power *= 0.9;
            self.beta2_power *= 0.999;
            self.module.adam(
                &self.stream,
                &adam,
                options.learning_rate,
                1.0 - self.beta1_power,
                1.0 - self.beta2_power,
                &gradients,
                &mut self.state,
            )?;
            if epoch.is_multiple_of(options.report_every) || epoch == options.epochs {
                let point = LossPoint {
                    epoch,
                    mse: mse(&self.predict(inputs)?, targets),
                };
                history.push(point);
                on_report(point);
            }
        }
        self.stream.synchronize()?;
        Ok(TrainReport {
            history,
            elapsed: start.elapsed(),
        })
    }
}

fn launch_config(elements: usize) -> Result<LaunchConfig1D> {
    let count = u32::try_from(elements).map_err(|_| "input count exceeds the 1-D launch limit")?;
    Ok(LaunchConfig1D::new(count.div_ceil(BLOCK), BLOCK, 0))
}

fn validate_batch(inputs: &[f32], targets: &[f32]) -> Result<()> {
    validate_training_data(inputs, targets)?;
    if inputs.len() > MAX_BATCH_SIZE {
        return Err(
            format!("full-batch training supports at most {MAX_BATCH_SIZE} samples").into(),
        );
    }
    Ok(())
}

fn mse(predictions: &[f32], targets: &[f32]) -> f64 {
    predictions
        .iter()
        .zip(targets)
        .map(|(&p, &y)| {
            let error = f64::from(p) - f64::from(y);
            error * error
        })
        .sum::<f64>()
        / predictions.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_rejects_invalid_optimizer_options() {
        let defaults = TrainOptions::default();
        for options in [
            TrainOptions {
                epochs: 0,
                ..defaults
            },
            TrainOptions {
                report_every: 0,
                ..defaults
            },
            TrainOptions {
                learning_rate: 0.0,
                ..defaults
            },
            TrainOptions {
                learning_rate: -0.1,
                ..defaults
            },
            TrainOptions {
                learning_rate: f32::NAN,
                ..defaults
            },
        ] {
            assert!(options.validate().is_err());
        }
        assert!(defaults.validate().is_ok());
        assert!(
            validate_batch(
                &vec![0.0; MAX_BATCH_SIZE + 1],
                &vec![0.0; MAX_BATCH_SIZE + 1]
            )
            .is_err()
        );
    }
}
