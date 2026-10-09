use crate::Result;

pub const MAX_DEPTH: usize = 32;
pub(crate) const MAX_NODES: usize = 288;
pub(crate) const MAX_PARAMETERS: usize = 832;
pub(crate) const GRADIENT_SIZE: usize = MAX_PARAMETERS + 1;
pub(crate) const MISSING: u32 = u32::MAX;

/// One neuron. Parent 0 denotes the scalar input; neuron IDs start at 1.
#[derive(Clone, Debug)]
pub struct Node {
    pub layer: usize,
    pub position: usize,
    pub parents: [Option<usize>; 2],
    pub weights: [Option<usize>; 2],
    pub bias: usize,
}

/// The ancestors of the central output in Wolfram's zero-padded mesh.
///
/// Removing neurons outside the output's dependency cone turns the expanding
/// mesh into a diamond without changing its scalar function.
#[derive(Clone, Debug)]
pub struct MeshTopology {
    depth: usize,
    nodes: Vec<Node>,
    widths: Vec<usize>,
    parameter_count: usize,
}

impl MeshTopology {
    pub fn new(depth: usize) -> Result<Self> {
        if depth == 0 || !depth.is_multiple_of(2) || depth > MAX_DEPTH {
            return Err(format!("depth must be even and between 2 and {MAX_DEPTH}").into());
        }
        let half = depth / 2;
        let mut nodes = Vec::new();
        let mut widths = vec![1];
        let mut previous = vec![0];
        let mut parameter_count = 0;
        for layer in 1..=depth {
            let growing = layer <= half;
            let width = if growing {
                previous.len() + 1
            } else {
                previous.len() - 1
            };
            let mut current = Vec::with_capacity(width);
            for position in 0..width {
                let parents = if growing {
                    [
                        position.checked_sub(1).map(|i| previous[i]),
                        previous.get(position).copied(),
                    ]
                } else {
                    [Some(previous[position]), Some(previous[position + 1])]
                };
                let mut weights = [None; 2];
                for (edge, parent) in parents.iter().enumerate() {
                    if parent.is_some() {
                        weights[edge] = Some(parameter_count);
                        parameter_count += 1;
                    }
                }
                let bias = parameter_count;
                parameter_count += 1;
                current.push(nodes.len() + 1);
                nodes.push(Node {
                    layer,
                    position,
                    parents,
                    weights,
                    bias,
                });
            }
            widths.push(width);
            previous = current;
        }
        debug_assert!(nodes.len() <= MAX_NODES && parameter_count <= MAX_PARAMETERS);
        Ok(Self {
            depth,
            nodes,
            widths,
            parameter_count,
        })
    }

    pub fn depth(&self) -> usize {
        self.depth
    }
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn widths(&self) -> &[usize] {
        &self.widths
    }
    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }
    pub fn weight_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|node| node.weights.iter().flatten().count())
            .sum()
    }

    pub(crate) fn descriptors(&self) -> Vec<[u32; 5]> {
        self.nodes
            .iter()
            .map(|node| {
                [
                    node.parents[0].map_or(MISSING, |i| i as u32),
                    node.parents[1].map_or(MISSING, |i| i as u32),
                    node.weights[0].map_or(MISSING, |i| i as u32),
                    node.weights[1].map_or(MISSING, |i| i as u32),
                    node.bias as u32,
                ]
            })
            .collect()
    }
}

/// Independent f64 reference implementation for checking the GPU kernels.
#[derive(Clone, Debug)]
pub struct ReferenceMesh {
    topology: MeshTopology,
    parameters: Vec<f32>,
}

impl ReferenceMesh {
    pub fn new(depth: usize, seed: u64) -> Result<Self> {
        let topology = MeshTopology::new(depth)?;
        let mut random = Random::new(seed);
        let mut parameters = vec![0.0; topology.parameter_count()];
        for node in topology.nodes() {
            let fan_in = node.parents.iter().flatten().count() as f32;
            let scale = (6.0 / (fan_in + 1.0)).sqrt();
            for index in node.weights.iter().flatten() {
                parameters[*index] = (2.0 * random.unit() - 1.0) * scale;
            }
            parameters[node.bias] = 0.1 * (random.unit() - 0.5);
        }
        Ok(Self {
            topology,
            parameters,
        })
    }

    pub fn from_parameters(depth: usize, parameters: Vec<f32>) -> Result<Self> {
        let topology = MeshTopology::new(depth)?;
        if parameters.len() != topology.parameter_count() {
            return Err("parameter count does not match the mesh topology".into());
        }
        validate_finite(&parameters, "parameters")?;
        Ok(Self {
            topology,
            parameters,
        })
    }

    pub fn topology(&self) -> &MeshTopology {
        &self.topology
    }
    pub fn parameters(&self) -> &[f32] {
        &self.parameters
    }

    fn activations(&self, input: f32) -> Vec<f64> {
        let mut values = vec![f64::from(input)];
        for node in self.topology.nodes() {
            let mut value = f64::from(self.parameters[node.bias]);
            for edge in 0..2 {
                if let (Some(parent), Some(weight)) = (node.parents[edge], node.weights[edge]) {
                    value += values[parent] * f64::from(self.parameters[weight]);
                }
            }
            values.push(if value >= 0.0 { value } else { value.exp_m1() });
        }
        values
    }

    pub fn predict(&self, inputs: &[f32]) -> Result<Vec<f32>> {
        validate_finite(inputs, "inputs")?;
        let values: Vec<_> = inputs
            .iter()
            .map(|&x| *self.activations(x).last().unwrap() as f32)
            .collect();
        validate_finite(&values, "predictions")?;
        Ok(values)
    }

    pub fn loss_and_gradient(&self, inputs: &[f32], targets: &[f32]) -> Result<(f64, Vec<f64>)> {
        validate_training_data(inputs, targets)?;
        let mut gradient = vec![0.0; self.parameters.len()];
        let mut loss = 0.0;
        for (&input, &target) in inputs.iter().zip(targets) {
            let values = self.activations(input);
            let output = values.len() - 1;
            let error = values[output] - f64::from(target);
            loss += error * error;
            let mut adjoints = vec![0.0; values.len()];
            adjoints[output] = 2.0 * error;
            for (offset, node) in self.topology.nodes().iter().enumerate().rev() {
                let id = offset + 1;
                let derivative = if values[id] >= 0.0 {
                    1.0
                } else {
                    values[id] + 1.0
                };
                let delta = adjoints[id] * derivative;
                gradient[node.bias] += delta;
                for edge in 0..2 {
                    if let (Some(parent), Some(weight)) = (node.parents[edge], node.weights[edge]) {
                        gradient[weight] += delta * values[parent];
                        adjoints[parent] += delta * f64::from(self.parameters[weight]);
                    }
                }
            }
        }
        for value in &mut gradient {
            *value /= inputs.len() as f64;
        }
        if !loss.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            return Err("loss or gradients overflowed; use smaller parameters or data".into());
        }
        Ok((loss / inputs.len() as f64, gradient))
    }
}

/// The target used in the article: 1 on [-2, -1) and [0, 2), otherwise 0.
pub fn piecewise_target(x: f32) -> f32 {
    if (-2.0..-1.0).contains(&x) || (0.0..2.0).contains(&x) {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn validate_finite(values: &[f32], name: &str) -> Result<()> {
    if values.iter().any(|x| !x.is_finite()) {
        return Err(format!("{name} must contain only finite values").into());
    }
    Ok(())
}

pub(crate) fn validate_training_data(inputs: &[f32], targets: &[f32]) -> Result<()> {
    if inputs.is_empty() || inputs.len() != targets.len() {
        return Err("training inputs and targets must have the same nonzero length".into());
    }
    validate_finite(inputs, "inputs")?;
    validate_finite(targets, "targets")
}

pub(crate) struct Random {
    state: u64,
}
impl Random {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub(crate) fn unit(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut bits = self.state;
        bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
        bits ^= bits >> 31;
        ((bits >> 40) as u32) as f32 / 16_777_216.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn article_diamond_has_the_expected_connections() {
        let topology = MeshTopology::new(14).unwrap();
        assert_eq!(
            topology.widths(),
            &[1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
        );
        assert_eq!(topology.nodes().len(), 63);
        assert_eq!(topology.weight_count(), 112);
        assert_eq!(topology.parameter_count(), 175);
        for (offset, node) in topology.nodes().iter().enumerate() {
            assert!(
                node.parents
                    .iter()
                    .flatten()
                    .all(|&parent| parent <= offset)
            );
        }
    }

    #[test]
    fn pruned_diamond_matches_the_full_zero_padded_mesh() {
        for depth in [2, 6, 14, MAX_DEPTH] {
            let model = ReferenceMesh::new(depth, 123).unwrap();
            for x in [-2.5, -0.7, 0.0, 1.1, 2.8] {
                let mut values = vec![f64::from(x)];
                for layer in 1..=depth {
                    let mut next = vec![0.0; layer + 1];
                    let first = layer.saturating_sub(depth / 2);
                    for (position, slot) in next.iter_mut().enumerate() {
                        let active = model
                            .topology
                            .nodes()
                            .iter()
                            .find(|n| n.layer == layer && n.position + first == position);
                        let left = position.checked_sub(1).map_or(0.0, |i| values[i]);
                        let right = values.get(position).copied().unwrap_or(0.0);
                        let value = if let Some(node) = active {
                            f64::from(model.parameters[node.bias])
                                + node.weights[0]
                                    .map_or(0.0, |w| f64::from(model.parameters[w]) * left)
                                + node.weights[1]
                                    .map_or(0.0, |w| f64::from(model.parameters[w]) * right)
                        } else {
                            // Arbitrary discarded-neuron values must not affect the output.
                            0.37 * left - 0.81 * right + 0.23
                        };
                        *slot = if value >= 0.0 { value } else { value.exp_m1() };
                    }
                    values = next;
                }
                let expected = values[depth / 2];
                let actual = *model.activations(x).last().unwrap();
                assert!((actual - expected).abs() < 1e-10, "depth={depth}, x={x}");
            }
        }
    }

    #[test]
    fn backprop_matches_finite_differences_for_every_parameter() {
        let inputs = [-2.4, -0.3, 0.7, 1.8];
        let targets: Vec<_> = inputs.iter().map(|&x| piecewise_target(x)).collect();
        for depth in [2, 6, 14] {
            let model = ReferenceMesh::new(depth, 7).unwrap();
            let (_, gradient) = model.loss_and_gradient(&inputs, &targets).unwrap();
            for (index, analytic) in gradient.iter().enumerate() {
                let mut plus = model.clone();
                let mut minus = model.clone();
                plus.parameters[index] += 0.001;
                minus.parameters[index] -= 0.001;
                let step = f64::from(plus.parameters[index]) - f64::from(minus.parameters[index]);
                let numerical = (plus.loss_and_gradient(&inputs, &targets).unwrap().0
                    - minus.loss_and_gradient(&inputs, &targets).unwrap().0)
                    / step;
                assert!(
                    (analytic - numerical).abs() < 2e-4 * (1.0 + analytic.abs()),
                    "depth={depth}, parameter={index}, analytic={analytic}, numerical={numerical}"
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_shapes_and_nonfinite_data() {
        for depth in [0, 1, 3, MAX_DEPTH + 2, usize::MAX] {
            assert!(MeshTopology::new(depth).is_err());
        }
        assert!(ReferenceMesh::from_parameters(2, vec![0.0; 6]).is_err());
        assert!(ReferenceMesh::from_parameters(2, vec![f32::NAN; 7]).is_err());
        let model = ReferenceMesh::new(2, 1).unwrap();
        assert!(model.loss_and_gradient(&[], &[]).is_err());
        assert!(model.loss_and_gradient(&[0.0], &[]).is_err());
        assert!(model.loss_and_gradient(&[f32::INFINITY], &[0.0]).is_err());
        assert!(model.loss_and_gradient(&[0.0], &[f32::NAN]).is_err());
        let overflowing =
            ReferenceMesh::from_parameters(MAX_DEPTH, vec![f32::MAX; MAX_PARAMETERS]).unwrap();
        assert!(overflowing.loss_and_gradient(&[1.0], &[0.0]).is_err());
    }

    #[test]
    fn target_has_the_article_boundary_values() {
        for (x, expected) in [
            (-3.0, 0.0),
            (-2.0, 1.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (2.0, 0.0),
            (3.0, 0.0),
        ] {
            assert_eq!(piecewise_target(x), expected);
        }
    }
}
