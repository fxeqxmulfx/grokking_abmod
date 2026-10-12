use crate::{
    Result,
    gpt::reference::{Tape, rmsnorm},
    mesh::{Random, validate_finite},
};

use super::{Activation, Architecture, SimpleConfig};

/// Independent f64 scalar automatic differentiation of the attention-free
/// models, used only for verification. The GPU kernels use hand-written
/// derivatives instead.
#[derive(Clone, Debug)]
pub struct ReferenceSimpleModel {
    config: SimpleConfig,
    parameters: Vec<f32>,
}

impl ReferenceSimpleModel {
    pub fn new(config: SimpleConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            parameters: config.initial_parameters(seed),
        })
    }

    pub fn from_parameters(config: SimpleConfig, parameters: Vec<f32>) -> Result<Self> {
        config.validate()?;
        if parameters.len() != config.parameter_count() {
            return Err("parameter count does not match the model".into());
        }
        validate_finite(&parameters, "parameters")?;
        Ok(Self { config, parameters })
    }

    pub fn parameters(&self) -> &[f32] {
        &self.parameters
    }

    /// Logits for rows of `config.input_features()` features.
    pub fn logits(&self, features: &[f32]) -> Result<Vec<Vec<f64>>> {
        self.validate_features(features)?;
        let (tape, logits) = forward(self.config, self.f64_parameters(), features);
        Ok(logits
            .iter()
            .map(|row| row.iter().map(|&id| tape.values[id]).collect())
            .collect())
    }

    /// Mean cross-entropy of the answers and its gradient for every parameter.
    pub fn loss_and_gradient(&self, features: &[f32], targets: &[u32]) -> Result<(f64, Vec<f64>)> {
        let rows = self.validate_features(features)?;
        if targets.len() != rows || targets.iter().any(|&t| t as usize >= self.config.modulus) {
            return Err("each row needs one answer class below the modulus".into());
        }
        let (loss, gradients) = loss_and_gradient(
            self.config,
            self.f64_parameters(),
            features,
            targets,
            self.parameters.len(),
        );
        if !loss.is_finite() || gradients.iter().any(|g| !g.is_finite()) {
            return Err("nonfinite reference loss or gradients".into());
        }
        Ok((loss, gradients))
    }

    fn f64_parameters(&self) -> impl Iterator<Item = f64> + '_ {
        self.parameters.iter().map(|&p| f64::from(p))
    }

    fn validate_features(&self, features: &[f32]) -> Result<usize> {
        let columns = self.config.input_features();
        if features.is_empty() || !features.len().is_multiple_of(columns) {
            return Err("features must hold one or more complete rows".into());
        }
        validate_finite(features, "features")?;
        Ok(features.len() / columns)
    }
}

/// Weights for derivative checks: twice the initial dense weights, so that
/// every nonlinearity matters, and uneven mesh weights in place of the
/// identity, so that every mesh link carries a signal.
pub(crate) fn verification_parameters(config: SimpleConfig, seed: u64) -> Vec<f32> {
    let mut parameters: Vec<_> = config
        .initial_parameters(seed)
        .iter()
        .map(|&p| 2.0 * p)
        .collect();
    let parents = config.connectivity.parents();
    let mesh = parameters.len() - parents * config.width * config.stored_mesh_layers();
    let mut random = Random::new(seed);
    for unit in parameters[mesh..].chunks_exact_mut(parents) {
        unit[0] = 0.75 + 0.5 * random.unit();
        if let Some(other) = unit.get_mut(1) {
            *other = random.unit() - 0.5;
        }
    }
    parameters
}

fn loss_and_gradient(
    config: SimpleConfig,
    parameters: impl IntoIterator<Item = f64>,
    features: &[f32],
    targets: &[u32],
    count: usize,
) -> (f64, Vec<f64>) {
    let (mut tape, logits) = forward(config, parameters, features);
    let losses: Vec<_> = logits
        .iter()
        .zip(targets)
        .map(|(row, &target)| {
            let (maximum, sum, _) = tape.shifted_exponentials(row);
            let log = tape.log(sum);
            let normalized = tape.shift(log, maximum);
            tape.subtract(normalized, row[target as usize])
        })
        .collect();
    let sum = tape.sum(&losses);
    let loss = tape.scale(sum, 1.0 / targets.len() as f64);
    (tape.values[loss], tape.backward(loss, count))
}

/// Builds every row's logits on one tape whose first nodes are the parameters.
fn forward(
    config: SimpleConfig,
    parameters: impl IntoIterator<Item = f64>,
    features: &[f32],
) -> (Tape, Vec<Vec<usize>>) {
    let mut tape = Tape::new(parameters);
    let layers = config.layers();
    let offsets: Vec<_> = layers
        .iter()
        .scan(0, |offset, layer| {
            let start = *offset;
            *offset += layer.parameters();
            Some(start)
        })
        .collect();
    let mesh_offset: usize = layers.iter().map(|layer| layer.parameters()).sum();
    let dense = |tape: &mut Tape, input: &[usize], layer: usize| -> Vec<usize> {
        let outputs = layers[layer].outputs;
        (0..outputs)
            .map(|column| {
                let weights: Vec<_> = (0..input.len())
                    .map(|row| offsets[layer] + row * outputs + column)
                    .collect();
                tape.dot(input, &weights)
            })
            .collect()
    };
    let activate = |tape: &mut Tape, input: &[usize]| -> Vec<usize> {
        input
            .iter()
            .map(|&id| match config.activation {
                Activation::Relu => tape.relu(id),
                Activation::Square => tape.multiply(id, id),
            })
            .collect()
    };
    let mut logits = Vec::new();
    for row in features.chunks_exact(config.input_features()) {
        let x: Vec<_> = row.iter().map(|&v| tape.constant(f64::from(v))).collect();
        logits.push(match config.architecture {
            Architecture::Mlp => {
                let mut hidden = dense(&mut tape, &x, 0);
                for layer in 1..=config.depth {
                    let activated = activate(&mut tape, &hidden);
                    hidden = dense(&mut tape, &activated, layer);
                }
                hidden
            }
            Architecture::MeshMlp | Architecture::TiedConcatMeshMlp => {
                let (width, parents) = (config.width, config.connectivity.parents());
                let tied = config.architecture == Architecture::TiedConcatMeshMlp;
                let mut hidden = if tied {
                    let (a, b) = x.split_at(config.operand_features());
                    let mut joined = dense(&mut tape, a, 0);
                    joined.extend(dense(&mut tape, b, 0));
                    dense(&mut tape, &joined, 1)
                } else {
                    dense(&mut tape, &x, 0)
                };
                for layer in 0..config.mesh_layers() {
                    let activated = activate(&mut tape, &hidden);
                    let stride = config.mesh_stride(layer);
                    let offset = mesh_offset + parents * width * config.stored_mesh_layer(layer);
                    hidden = (0..width)
                        .map(|channel| {
                            let inputs =
                                [activated[channel], activated[(channel + stride) % width]];
                            let unit = offset + parents * channel;
                            tape.dot(&inputs[..parents], &[unit, unit + 1][..parents])
                        })
                        .collect();
                }
                let activated = activate(&mut tape, &hidden);
                if tied {
                    (0..config.modulus)
                        .map(|class| {
                            let weights: Vec<_> =
                                (0..width).map(|channel| class * width + channel).collect();
                            tape.dot(&activated, &weights)
                        })
                        .collect()
                } else {
                    dense(&mut tape, &activated, 1)
                }
            }
            Architecture::ResidualMeshMlp
            | Architecture::TiedConcatResidualMeshMlp
            | Architecture::TiedRotaryResidualMeshMlp => {
                let width = config.width;
                let parents = config.connectivity.parents();
                let tied = config.architecture.tied_embeddings();
                let mut hidden = if config.architecture.rotary_embeddings() {
                    let (a, b) = x.split_at(config.operand_features());
                    let left = dense(&mut tape, a, 0);
                    let right = dense(&mut tape, b, 0);
                    (0..width)
                        .map(|channel| {
                            let rotated = tape.scale(
                                right[channel ^ 1],
                                if channel % 2 == 0 { -1.0 } else { 1.0 },
                            );
                            tape.add(left[channel], rotated)
                        })
                        .collect()
                } else if tied {
                    let (a, b) = x.split_at(config.operand_features());
                    let mut joined = dense(&mut tape, a, 0);
                    joined.extend(dense(&mut tape, b, 0));
                    dense(&mut tape, &joined, 1)
                } else {
                    dense(&mut tape, &x, 0)
                };
                for block in 0..config.depth {
                    let normalized = rmsnorm(&mut tape, std::slice::from_ref(&hidden), 1).remove(0);
                    let mut branch = normalized;
                    for layer in 0..config.mesh_depth {
                        if layer > 0 {
                            branch = activate(&mut tape, &branch);
                        }
                        let mesh = block * config.mesh_depth + layer;
                        let stride = config.mesh_stride(mesh);
                        branch = (0..width)
                            .map(|channel| {
                                let mut inputs = vec![branch[channel]];
                                if parents == 2 {
                                    inputs.push(branch[(channel + stride) % width]);
                                }
                                let start = mesh_offset + (mesh * width + channel) * parents;
                                tape.dot(&inputs, &(start..start + parents).collect::<Vec<_>>())
                            })
                            .collect();
                    }
                    hidden = hidden
                        .iter()
                        .zip(&branch)
                        .map(|(&a, &b)| tape.add(a, b))
                        .collect();
                }
                let normalized = rmsnorm(&mut tape, &[hidden], 1).remove(0);
                if tied {
                    (0..config.modulus)
                        .map(|class| {
                            let weights: Vec<_> =
                                (0..width).map(|channel| class * width + channel).collect();
                            tape.dot(&normalized, &weights)
                        })
                        .collect()
                } else {
                    dense(&mut tape, &normalized, 1)
                }
            }
            Architecture::ResidualMlp => {
                let mut hidden = dense(&mut tape, &x, 0);
                for block in 0..config.depth {
                    let normalized = rmsnorm(&mut tape, std::slice::from_ref(&hidden), 1).remove(0);
                    let expanded = dense(&mut tape, &normalized, 1 + 2 * block);
                    let activated = activate(&mut tape, &expanded);
                    let branch = dense(&mut tape, &activated, 2 + 2 * block);
                    hidden = hidden
                        .iter()
                        .zip(&branch)
                        .map(|(&a, &b)| tape.add(a, b))
                        .collect();
                }
                let normalized = rmsnorm(&mut tape, &[hidden], 1).remove(0);
                dense(&mut tape, &normalized, 1 + 2 * config.depth)
            }
            Architecture::Bilinear => {
                let (a, b) = x.split_at(config.operand_features());
                let left = dense(&mut tape, a, 0);
                let right = dense(&mut tape, b, 1);
                let product: Vec<_> = left
                    .iter()
                    .zip(&right)
                    .map(|(&l, &r)| tape.multiply(l, r))
                    .collect();
                dense(&mut tape, &product, 2)
            }
        });
    }
    (tape, logits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gpt::{EquationDataset, ModularOperation},
        simple::{Connectivity, Encoding},
    };

    fn variants() -> Vec<SimpleConfig> {
        let base = SimpleConfig {
            modulus: 7,
            width: 5,
            ..SimpleConfig::default()
        };
        let mut configs = Vec::new();
        for activation in [Activation::Relu, Activation::Square] {
            for encoding in [
                Encoding::OneHot,
                Encoding::Binary,
                Encoding::Scalar,
                Encoding::DiscreteLog,
            ] {
                configs.push(SimpleConfig {
                    activation,
                    encoding,
                    depth: 2,
                    ..base
                });
                configs.push(SimpleConfig {
                    architecture: Architecture::ResidualMlp,
                    activation,
                    encoding,
                    width: 4,
                    depth: 2,
                    expansion: 2,
                    ..base
                });
                for connectivity in [Connectivity::Neighbor, Connectivity::Own] {
                    configs.push(SimpleConfig {
                        architecture: Architecture::MeshMlp,
                        activation,
                        encoding,
                        depth: 3,
                        connectivity,
                        ..base
                    });
                }
            }
        }
        // A width of 6 has butterfly strides 1, 2, 1.
        configs.push(SimpleConfig {
            architecture: Architecture::MeshMlp,
            connectivity: Connectivity::Butterfly,
            width: 6,
            depth: 4,
            ..base
        });
        // Tied layers add the gradients of every use of their weights.
        for connectivity in [Connectivity::Own, Connectivity::Butterfly] {
            configs.push(SimpleConfig {
                architecture: Architecture::MeshMlp,
                connectivity,
                width: 6,
                depth: 4,
                tied_mesh: true,
                ..base
            });
        }
        for encoding in [Encoding::OneHot, Encoding::DiscreteLog] {
            configs.push(SimpleConfig {
                architecture: Architecture::Bilinear,
                encoding,
                ..base
            });
        }
        for activation in [Activation::Relu, Activation::Square] {
            for connectivity in [Connectivity::Own, Connectivity::Butterfly] {
                configs.push(SimpleConfig {
                    architecture: Architecture::TiedConcatMeshMlp,
                    activation,
                    connectivity,
                    depth: 3,
                    ..base
                });
            }
        }
        configs.push(SimpleConfig {
            architecture: Architecture::TiedConcatMeshMlp,
            depth: 4,
            tied_mesh: true,
            ..base
        });
        for activation in [Activation::Relu, Activation::Square] {
            for connectivity in [
                Connectivity::Own,
                Connectivity::Neighbor,
                Connectivity::Butterfly,
            ] {
                configs.push(SimpleConfig {
                    architecture: Architecture::ResidualMeshMlp,
                    activation,
                    connectivity,
                    depth: 2,
                    mesh_depth: 3,
                    ..base
                });
                configs.push(SimpleConfig {
                    architecture: Architecture::TiedConcatResidualMeshMlp,
                    activation,
                    connectivity,
                    depth: 2,
                    mesh_depth: 3,
                    ..base
                });
                configs.push(SimpleConfig {
                    architecture: Architecture::TiedRotaryResidualMeshMlp,
                    activation,
                    connectivity,
                    width: 6,
                    depth: 2,
                    mesh_depth: 3,
                    ..base
                });
            }
        }
        configs
    }

    fn batch(config: SimpleConfig) -> (Vec<f32>, Vec<u32>) {
        let dataset =
            EquationDataset::new(config.modulus, ModularOperation::Divide, 0.5, 7).unwrap();
        config.encode(&dataset.split(false)[..6]).unwrap()
    }

    #[test]
    fn reference_gradients_match_central_differences() {
        for config in variants() {
            let parameters: Vec<_> = verification_parameters(config, 5)
                .iter()
                .map(|&p| f64::from(p))
                .collect();
            let (features, targets) = batch(config);
            let count = parameters.len();
            let (_, gradient) = loss_and_gradient(
                config,
                parameters.iter().copied(),
                &features,
                &targets,
                count,
            );
            let scale = gradient.iter().fold(0.0_f64, |a, g| a.max(g.abs()));
            assert!(scale > 1e-4, "{config:?}");
            for index in (0..count).step_by(3) {
                let mut shifted = parameters.clone();
                let step = 1e-6;
                shifted[index] += step;
                let (plus, _) = loss_and_gradient(config, shifted.clone(), &features, &targets, 0);
                shifted[index] -= 2.0 * step;
                let (minus, _) = loss_and_gradient(config, shifted, &features, &targets, 0);
                let numeric = (plus - minus) / (2.0 * step);
                assert!(
                    (numeric - gradient[index]).abs() <= 1e-6 * scale + 1e-5 * numeric.abs(),
                    "{config:?} parameter {index}: numeric {numeric}, tape {}",
                    gradient[index]
                );
            }
        }
    }

    #[test]
    fn models_compute_the_documented_functions() {
        let config = SimpleConfig {
            architecture: Architecture::Bilinear,
            encoding: Encoding::Scalar,
            modulus: 3,
            width: 1,
            ..SimpleConfig::default()
        };
        // left = 2a, right = -b, logits = (left * right) * [1, 0, -1].
        let model =
            ReferenceSimpleModel::from_parameters(config, vec![2.0, -1.0, 1.0, 0.0, -1.0]).unwrap();
        let logits = model.logits(&[0.5, 1.0]).unwrap();
        assert_eq!(logits, [[-1.0, 0.0, 1.0]]);
        let config = SimpleConfig {
            architecture: Architecture::Mlp,
            activation: Activation::Square,
            ..config
        };
        // hidden = (a - 3b)^2, logits = hidden * [1, 2, 3].
        let model =
            ReferenceSimpleModel::from_parameters(config, vec![1.0, -3.0, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(model.logits(&[1.0, 1.0]).unwrap(), [[4.0, 8.0, 12.0]]);
        let relu = SimpleConfig {
            activation: Activation::Relu,
            ..config
        };
        let model =
            ReferenceSimpleModel::from_parameters(relu, vec![1.0, -3.0, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(model.logits(&[1.0, 1.0]).unwrap(), [[0.0, 0.0, 0.0]]);
        let residual = SimpleConfig {
            architecture: Architecture::ResidualMlp,
            activation: Activation::Relu,
            depth: 1,
            expansion: 1,
            ..config
        };
        // hidden = a + b = 2 normalizes to about 1. The block adds 0.5 * relu(1),
        // and the final RMSNorm of about 2.5 is again about 1.
        let model = ReferenceSimpleModel::from_parameters(
            residual,
            vec![1.0, 1.0, 1.0, 0.5, 1.0, -1.0, 0.0],
        )
        .unwrap();
        let logits = model.logits(&[1.0, 1.0]).unwrap();
        for (actual, expected) in logits[0].iter().zip([1.0, -1.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-6, "{actual} {expected}");
        }
        let mesh = SimpleConfig {
            architecture: Architecture::MeshMlp,
            activation: Activation::Relu,
            width: 2,
            depth: 2,
            ..config
        };
        // hidden = relu([a + b, 2a - b]) = [2, 1]. The mesh layer gives
        // [2 + 0.5 * 1, -1 * 1 + 3 * 2] = [2.5, 5], and the readout
        // [[1, 0, -1], [0, 1, 1]] gives [2.5, 5, 2.5].
        let parameters = vec![
            1.0, 2.0, 1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.5, -1.0, 3.0,
        ];
        let model = ReferenceSimpleModel::from_parameters(mesh, parameters).unwrap();
        assert_eq!(model.logits(&[1.0, 1.0]).unwrap(), [[2.5, 5.0, 2.5]]);
        let own = SimpleConfig {
            connectivity: Connectivity::Own,
            ..mesh
        };
        // Gains [0.5, -2] turn relu([2, 1]) into [1, -2], whose ReLU [1, 0]
        // reads out as [1, 0, -1].
        let parameters = vec![
            1.0, 2.0, 1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.5, -2.0,
        ];
        let model = ReferenceSimpleModel::from_parameters(own, parameters).unwrap();
        assert_eq!(model.logits(&[1.0, 1.0]).unwrap(), [[1.0, 0.0, -1.0]]);
        let tied = SimpleConfig {
            depth: 3,
            tied_mesh: true,
            ..own
        };
        // Gains [0.5, 3] applied twice turn relu([2, 1]) into [0.5, 9], which
        // reads out as [0.5, 9, 8.5].
        let parameters = vec![1.0, 2.0, 1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.5, 3.0];
        let model = ReferenceSimpleModel::from_parameters(tied, parameters).unwrap();
        assert_eq!(model.logits(&[1.0, 1.0]).unwrap(), [[0.5, 9.0, 8.5]]);
    }

    #[test]
    fn tied_meshes_repeat_their_one_stored_layer() {
        let base = SimpleConfig {
            architecture: Architecture::MeshMlp,
            modulus: 7,
            width: 5,
            depth: 5,
            ..SimpleConfig::default()
        };
        let (features, _) = batch(base);
        for connectivity in [Connectivity::Own, Connectivity::Butterfly] {
            let untied = SimpleConfig {
                connectivity,
                ..base
            };
            let tied = SimpleConfig {
                tied_mesh: true,
                ..untied
            };
            let shared = verification_parameters(tied, 5);
            let (dense, mesh) = shared.split_at(shared.len() - connectivity.parents() * 5);
            let mut repeated = dense.to_vec();
            for _ in 0..untied.mesh_layers() {
                repeated.extend_from_slice(mesh);
            }
            let expected = ReferenceSimpleModel::from_parameters(untied, repeated)
                .unwrap()
                .logits(&features)
                .unwrap();
            let model = ReferenceSimpleModel::from_parameters(tied, shared).unwrap();
            assert_eq!(model.logits(&features).unwrap(), expected);
        }
    }

    #[test]
    fn identity_meshes_preserve_the_one_layer_relu_mlp() {
        let mlp = SimpleConfig {
            modulus: 7,
            width: 5,
            ..SimpleConfig::default()
        };
        let (features, _) = batch(mlp);
        let expected = ReferenceSimpleModel::new(mlp, 5)
            .unwrap()
            .logits(&features)
            .unwrap();
        for connectivity in [
            Connectivity::Own,
            Connectivity::Neighbor,
            Connectivity::Butterfly,
        ] {
            let mesh = SimpleConfig {
                architecture: Architecture::MeshMlp,
                depth: 4,
                connectivity,
                ..mlp
            };
            let model = ReferenceSimpleModel::new(mesh, 5).unwrap();
            assert_eq!(model.logits(&features).unwrap(), expected);
        }
    }

    #[test]
    fn reference_rejects_malformed_rows_targets_and_parameters() {
        let config = SimpleConfig {
            modulus: 7,
            width: 3,
            ..SimpleConfig::default()
        };
        let model = ReferenceSimpleModel::new(config, 1).unwrap();
        let (features, targets) = batch(config);
        model.loss_and_gradient(&features, &targets).unwrap();
        assert!(model.logits(&features[1..]).is_err());
        assert!(model.logits(&[]).is_err());
        assert!(model.loss_and_gradient(&features, &targets[1..]).is_err());
        let mut invalid = targets.clone();
        invalid[0] = 7;
        assert!(model.loss_and_gradient(&features, &invalid).is_err());
        let mut nan = features.clone();
        nan[0] = f32::NAN;
        assert!(model.logits(&nan).is_err());
        assert!(ReferenceSimpleModel::from_parameters(config, vec![0.0; 3]).is_err());
    }
}
