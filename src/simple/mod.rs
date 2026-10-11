//! Attention-free baselines for the grokking experiments. Each model reads the
//! two encoded operands of `a op b` and predicts the answer through a bias-free
//! linear readout: a plain MLP, an MLP whose deeper hidden layers are Mesh GPT's
//! two-parent mesh layers, a pre-norm residual MLP, or the elementwise product
//! of two operand embeddings.

mod gpu;
mod reference;

use std::{fmt, str::FromStr};

use crate::{
    Result,
    gpt::{Equation, MeshConnectivity},
    mesh::Random,
};

pub use gpu::{DeviceBatches, DeviceTable, GpuSimpleModel};
pub use reference::ReferenceSimpleModel;

/// Largest supported layer width, including a residual block's expansion.
pub const MAX_SIMPLE_WIDTH: usize = 16_384;
/// Largest number of hidden layers or residual blocks.
pub const MAX_SIMPLE_DEPTH: usize = 64;
/// Largest number of classes, and therefore of residues.
pub const MAX_SIMPLE_CLASSES: usize = 1024;
/// Largest number of rows in one table, batch, or evaluation.
pub const MAX_SIMPLE_ROWS: usize = 131_072;
const MAX_SIMPLE_PARAMETERS: usize = 1 << 28;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    /// Concatenated operands, `depth` activated hidden layers, and a readout.
    Mlp,
    /// An MLP whose hidden layers after the first are mesh layers: unit `c`
    /// weights its own activated channel and one other, `c + stride`, as in
    /// Mesh GPT. Depth 1 is the one-layer MLP.
    MeshMlp,
    /// An operand embedding, `depth` pre-RMSNorm feed-forward blocks with
    /// residual connections, a final RMSNorm, and a readout.
    ResidualMlp,
    /// The elementwise product of separate `a` and `b` embeddings, read out.
    Bilinear,
}

impl fmt::Display for Architecture {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Mlp => "mlp",
            Self::MeshMlp => "mesh-mlp",
            Self::ResidualMlp => "residual-mlp",
            Self::Bilinear => "bilinear",
        })
    }
}

impl FromStr for Architecture {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "mlp" => Ok(Self::Mlp),
            "mesh-mlp" => Ok(Self::MeshMlp),
            "residual-mlp" => Ok(Self::ResidualMlp),
            "bilinear" => Ok(Self::Bilinear),
            _ => Err(format!("unknown model: {value}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Square,
}

impl fmt::Display for Activation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Relu => "relu",
            Self::Square => "square",
        })
    }
}

impl FromStr for Activation {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "relu" => Ok(Self::Relu),
            "square" => Ok(Self::Square),
            _ => Err(format!("unknown activation: {value}")),
        }
    }
}

/// How one operand is presented to the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// An abstract identity: one feature per residue.
    OneHot,
    /// The residue's bits, least significant first, as -1 or +1.
    Binary,
    /// The residue's value, mapped linearly onto [-1, 1].
    Scalar,
    /// A zero flag and the discrete logarithm to the smallest primitive root,
    /// mapped linearly onto [-1, 1). This supplies the order in which division
    /// is subtraction.
    DiscreteLog,
}

impl Encoding {
    /// Features per operand.
    pub fn features(self, modulus: usize) -> usize {
        match self {
            Self::OneHot => modulus,
            Self::Binary => (usize::BITS - (modulus - 1).leading_zeros()).max(1) as usize,
            Self::Scalar => 1,
            Self::DiscreteLog => 2,
        }
    }

    /// Feature rows for every residue in `0..modulus`.
    pub fn table(self, modulus: usize) -> Result<Vec<Vec<f32>>> {
        if !(2..=MAX_SIMPLE_CLASSES).contains(&modulus) {
            return Err(format!("modulus must be in 2..={MAX_SIMPLE_CLASSES}").into());
        }
        let features = self.features(modulus);
        Ok(match self {
            Self::OneHot => (0..modulus)
                .map(|value| {
                    let mut row = vec![0.0; features];
                    row[value] = 1.0;
                    row
                })
                .collect(),
            Self::Binary => (0..modulus)
                .map(|value| {
                    (0..features)
                        .map(|bit| if value >> bit & 1 == 1 { 1.0 } else { -1.0 })
                        .collect()
                })
                .collect(),
            Self::Scalar => {
                let middle = (modulus - 1) as f64 / 2.0;
                (0..modulus)
                    .map(|value| vec![((value as f64 - middle) / middle) as f32])
                    .collect()
            }
            Self::DiscreteLog => {
                let logarithms = discrete_logarithms(modulus)?;
                let middle = (modulus - 2) as f64 / 2.0;
                (0..modulus)
                    .map(|value| {
                        if value == 0 {
                            vec![1.0, 0.0]
                        } else if middle == 0.0 {
                            vec![0.0, 0.0]
                        } else {
                            vec![0.0, ((logarithms[value] as f64 - middle) / middle) as f32]
                        }
                    })
                    .collect()
            }
        })
    }
}

impl fmt::Display for Encoding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OneHot => "one-hot",
            Self::Binary => "binary",
            Self::Scalar => "scalar",
            Self::DiscreteLog => "discrete-log",
        })
    }
}

impl FromStr for Encoding {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "one-hot" => Ok(Self::OneHot),
            "binary" => Ok(Self::Binary),
            "scalar" => Ok(Self::Scalar),
            "discrete-log" => Ok(Self::DiscreteLog),
            _ => Err(format!("unknown encoding: {value}")),
        }
    }
}

/// The smallest primitive root of a prime modulus.
pub fn primitive_root(modulus: usize) -> Result<usize> {
    (1..modulus)
        .find(|&root| {
            let mut value = 1;
            let mut seen = vec![false; modulus];
            for _ in 0..modulus - 1 {
                seen[value] = true;
                value = value * root % modulus;
            }
            seen[1..].iter().all(|&x| x)
        })
        .ok_or_else(|| "discrete logarithms require a prime modulus".into())
}

/// `log[g^k mod p] = k` for the smallest primitive root `g`; `log[0]` is unused.
fn discrete_logarithms(modulus: usize) -> Result<Vec<usize>> {
    let root = primitive_root(modulus)?;
    let mut logarithms = vec![0; modulus];
    let mut value = 1;
    for exponent in 0..modulus - 1 {
        logarithms[value] = exponent;
        value = value * root % modulus;
    }
    Ok(logarithms)
}

/// One bias-free layer, `outputs = inputs W` with `W` stored input-major.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dense {
    pub inputs: usize,
    pub outputs: usize,
}

impl Dense {
    pub fn parameters(self) -> usize {
        self.inputs * self.outputs
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimpleConfig {
    pub architecture: Architecture,
    pub activation: Activation,
    pub encoding: Encoding,
    /// Residues, and therefore answer classes.
    pub modulus: usize,
    /// Hidden width of an MLP, or the residual and embedding width.
    pub width: usize,
    /// MLP and mesh-MLP hidden layers or residual-MLP blocks. Bilinear models ignore it.
    pub depth: usize,
    /// Feed-forward width of a residual block, as a multiple of `width`.
    pub expansion: usize,
    /// The second parent of each mesh-layer unit. Only mesh MLPs use it.
    pub connectivity: MeshConnectivity,
}

impl Default for SimpleConfig {
    fn default() -> Self {
        Self {
            architecture: Architecture::Mlp,
            activation: Activation::Relu,
            encoding: Encoding::OneHot,
            modulus: 97,
            width: 128,
            depth: 1,
            expansion: 4,
            connectivity: MeshConnectivity::Neighbor,
        }
    }
}

impl SimpleConfig {
    pub fn validate(&self) -> Result<()> {
        if !(2..=MAX_SIMPLE_CLASSES).contains(&self.modulus) {
            return Err(format!("modulus must be in 2..={MAX_SIMPLE_CLASSES}").into());
        }
        if !(1..=MAX_SIMPLE_WIDTH).contains(&self.width) {
            return Err(format!("width must be in 1..={MAX_SIMPLE_WIDTH}").into());
        }
        if self.architecture != Architecture::Bilinear
            && !(1..=MAX_SIMPLE_DEPTH).contains(&self.depth)
        {
            return Err(format!("depth must be in 1..={MAX_SIMPLE_DEPTH}").into());
        }
        if self.architecture == Architecture::ResidualMlp
            && (self.expansion == 0
                || self
                    .width
                    .checked_mul(self.expansion)
                    .is_none_or(|hidden| hidden > MAX_SIMPLE_WIDTH))
        {
            return Err(format!("residual hidden width must be in 1..={MAX_SIMPLE_WIDTH}").into());
        }
        // A unit's second parent must be another channel.
        if self.mesh_layers() > 0 && self.width < 2 {
            return Err("mesh layers need a width of at least 2".into());
        }
        if self.encoding == Encoding::DiscreteLog {
            primitive_root(self.modulus)?;
        }
        if self.parameter_count() > MAX_SIMPLE_PARAMETERS {
            return Err(format!("models are limited to {MAX_SIMPLE_PARAMETERS} parameters").into());
        }
        Ok(())
    }

    /// Features per operand.
    pub fn operand_features(&self) -> usize {
        self.encoding.features(self.modulus)
    }

    /// Features of one equation row: `a` first, then `b`.
    pub fn input_features(&self) -> usize {
        2 * self.operand_features()
    }

    /// Dense layers in parameter order. A bilinear model's first two layers
    /// read `a` and `b` respectively; every other first layer reads both. A
    /// mesh MLP stores its mesh layers after its input layer and readout.
    pub fn layers(&self) -> Vec<Dense> {
        let (classes, width) = (self.modulus, self.width);
        let dense = |inputs, outputs| Dense { inputs, outputs };
        match self.architecture {
            Architecture::Mlp => {
                let mut layers = vec![dense(self.input_features(), width)];
                layers.extend((1..self.depth).map(|_| dense(width, width)));
                layers.push(dense(width, classes));
                layers
            }
            Architecture::MeshMlp => {
                vec![dense(self.input_features(), width), dense(width, classes)]
            }
            Architecture::ResidualMlp => {
                let hidden = width * self.expansion;
                let mut layers = vec![dense(self.input_features(), width)];
                for _ in 0..self.depth {
                    layers.push(dense(width, hidden));
                    layers.push(dense(hidden, width));
                }
                layers.push(dense(width, classes));
                layers
            }
            Architecture::Bilinear => vec![
                dense(self.operand_features(), width),
                dense(self.operand_features(), width),
                dense(width, classes),
            ],
        }
    }

    /// Mesh layers of a mesh MLP, each with a self and a neighbor weight per unit.
    pub fn mesh_layers(&self) -> usize {
        match self.architecture {
            Architecture::MeshMlp => self.depth.saturating_sub(1),
            _ => 0,
        }
    }

    /// Distance from each unit of mesh layer `layer` to its second parent.
    pub fn mesh_stride(&self, layer: usize) -> usize {
        self.connectivity.stride(self.width, layer)
    }

    /// Parameter group sizes in storage order: the dense layers, then the mesh layers.
    pub(crate) fn groups(&self) -> Vec<usize> {
        let mut groups: Vec<_> = self.layers().into_iter().map(Dense::parameters).collect();
        groups.extend(std::iter::repeat_n(2 * self.width, self.mesh_layers()));
        groups
    }

    pub fn parameter_count(&self) -> usize {
        self.groups().into_iter().sum()
    }

    /// PyTorch's default `nn.Linear` initialization, U(-1/sqrt(fan_in), 1/sqrt(fan_in)),
    /// drawn layer by layer from a seeded SplitMix64 stream. Mesh layers start
    /// as the identity and draw nothing, so a ReLU mesh MLP of any depth first
    /// computes the one-layer MLP of the same seed.
    pub fn initial_parameters(&self, seed: u64) -> Vec<f32> {
        let mut random = Random::new(seed ^ 0x73696d706c650004);
        let mut parameters = Vec::with_capacity(self.parameter_count());
        for layer in self.layers() {
            let bound = 1.0 / (layer.inputs as f32).sqrt();
            parameters.extend((0..layer.parameters()).map(|_| (2.0 * random.unit() - 1.0) * bound));
        }
        for _ in 0..self.mesh_layers() * self.width {
            parameters.extend([1.0, 0.0]);
        }
        parameters
    }

    /// Feature rows `[a | b]` and answer classes for a list of equations.
    pub fn encode(&self, equations: &[Equation]) -> Result<(Vec<f32>, Vec<u32>)> {
        let table = self.encoding.table(self.modulus)?;
        let mut features = Vec::with_capacity(equations.len() * self.input_features());
        let mut targets = Vec::with_capacity(equations.len());
        for equation in equations {
            if equation.a >= self.modulus
                || equation.b >= self.modulus
                || equation.answer >= self.modulus
            {
                return Err("equation residue is outside the model's modulus".into());
            }
            features.extend_from_slice(&table[equation.a]);
            features.extend_from_slice(&table[equation.b]);
            targets.push(equation.answer as u32);
        }
        Ok((features, targets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings_present_identities_bits_values_and_logarithms() {
        let one_hot = Encoding::OneHot.table(97).unwrap();
        for (value, row) in one_hot.iter().enumerate() {
            assert_eq!(row.len(), 97);
            assert_eq!(row.iter().sum::<f32>(), 1.0);
            assert_eq!(row[value], 1.0);
        }
        let binary = Encoding::Binary.table(97).unwrap();
        assert_eq!(binary[5], [1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
        assert_eq!(binary[96], [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]);
        let scalar = Encoding::Scalar.table(97).unwrap();
        assert_eq!(
            [scalar[0][0], scalar[48][0], scalar[96][0]],
            [-1.0, 0.0, 1.0]
        );
        assert_eq!(primitive_root(97).unwrap(), 5);
        // 5 generates the nonzero residues mod 97: log 1 = 0, log 5 = 1.
        let logarithms = Encoding::DiscreteLog.table(97).unwrap();
        assert_eq!(logarithms[0], [1.0, 0.0]);
        assert_eq!(logarithms[1], [0.0, -1.0]);
        assert_eq!(logarithms[5], [0.0, (-46.5_f64 / 47.5) as f32]);
        let table = discrete_logarithms(97).unwrap();
        for a in 1..97 {
            for b in 1..97 {
                // Division subtracts logarithms modulo the group order.
                let quotient = a * inverse(b, 97) % 97;
                assert_eq!(table[quotient], (table[a] + 96 - table[b]) % 96);
            }
        }
        assert_eq!(Encoding::Binary.features(2), 1);
        assert_eq!(
            Encoding::DiscreteLog.table(2).unwrap(),
            [[1.0, 0.0], [0.0, 0.0]]
        );
        assert!(primitive_root(8).is_err());
        assert!(Encoding::DiscreteLog.table(15).is_err());
    }

    fn inverse(value: usize, modulus: usize) -> usize {
        (1..modulus).find(|x| x * value % modulus == 1).unwrap()
    }

    #[test]
    fn parameter_counts_follow_the_layer_shapes() {
        let mlp = SimpleConfig::default();
        assert_eq!(mlp.parameter_count(), 291 * 128);
        let deep = SimpleConfig {
            depth: 3,
            width: 16,
            ..mlp
        };
        assert_eq!(deep.parameter_count(), 194 * 16 + 2 * 16 * 16 + 16 * 97);
        let mesh = SimpleConfig {
            architecture: Architecture::MeshMlp,
            ..deep
        };
        assert_eq!(mesh.parameter_count(), 291 * 16 + 2 * 2 * 16);
        assert_eq!(
            mesh.layers(),
            [
                Dense {
                    inputs: 194,
                    outputs: 16
                },
                Dense {
                    inputs: 16,
                    outputs: 97
                }
            ]
        );
        let residual = SimpleConfig {
            architecture: Architecture::ResidualMlp,
            width: 32,
            depth: 2,
            ..mlp
        };
        assert_eq!(residual.parameter_count(), 291 * 32 + 2 * 8 * 32 * 32);
        let bilinear = SimpleConfig {
            architecture: Architecture::Bilinear,
            width: 8,
            ..mlp
        };
        assert_eq!(bilinear.parameter_count(), 291 * 8);
        let numbers = SimpleConfig {
            encoding: Encoding::Binary,
            ..bilinear
        };
        assert_eq!(
            numbers.layers()[0],
            Dense {
                inputs: 7,
                outputs: 8
            }
        );
        assert_eq!(numbers.input_features(), 14);
    }

    #[test]
    fn configurations_reject_empty_oversized_and_composite_cases() {
        let valid = SimpleConfig::default();
        valid.validate().unwrap();
        for config in [
            SimpleConfig { width: 0, ..valid },
            SimpleConfig { depth: 0, ..valid },
            SimpleConfig {
                modulus: 1,
                ..valid
            },
            SimpleConfig {
                modulus: MAX_SIMPLE_CLASSES + 1,
                ..valid
            },
            SimpleConfig {
                width: MAX_SIMPLE_WIDTH + 1,
                ..valid
            },
            SimpleConfig {
                architecture: Architecture::ResidualMlp,
                expansion: 0,
                ..valid
            },
            SimpleConfig {
                architecture: Architecture::ResidualMlp,
                expansion: usize::MAX,
                ..valid
            },
            SimpleConfig {
                encoding: Encoding::DiscreteLog,
                modulus: 96,
                ..valid
            },
            SimpleConfig {
                architecture: Architecture::MeshMlp,
                depth: 0,
                ..valid
            },
            SimpleConfig {
                architecture: Architecture::MeshMlp,
                width: 1,
                depth: 2,
                ..valid
            },
        ] {
            assert!(config.validate().is_err(), "{config:?}");
        }
        for config in [
            SimpleConfig {
                architecture: Architecture::Bilinear,
                depth: 0,
                ..valid
            },
            // Without mesh layers, a single channel is a valid one-layer MLP.
            SimpleConfig {
                architecture: Architecture::MeshMlp,
                width: 1,
                ..valid
            },
        ] {
            config.validate().unwrap();
        }
    }

    #[test]
    fn mesh_mlps_extend_the_one_layer_mlp_with_identity_meshes() {
        let mlp = SimpleConfig {
            width: 8,
            ..SimpleConfig::default()
        };
        let mesh = SimpleConfig {
            architecture: Architecture::MeshMlp,
            depth: 4,
            connectivity: MeshConnectivity::Butterfly,
            ..mlp
        };
        let dense = mlp.initial_parameters(3);
        let parameters = mesh.initial_parameters(3);
        assert_eq!(parameters.len(), mesh.parameter_count());
        assert_eq!(parameters[..dense.len()], dense);
        assert_eq!(parameters.len() - dense.len(), 3 * 2 * 8);
        let (meshes, _) = parameters[dense.len()..].as_chunks::<2>();
        assert!(meshes.iter().all(|&pair| pair == [1.0, 0.0]));
        assert_eq!(
            SimpleConfig { depth: 1, ..mesh }.initial_parameters(3),
            dense
        );
        // Butterfly strides double up to half the largest power of two in the width.
        let strides: Vec<_> = (0..4).map(|layer| mesh.mesh_stride(layer)).collect();
        assert_eq!(strides, [1, 2, 4, 1]);
        let neighbor = SimpleConfig {
            connectivity: MeshConnectivity::Neighbor,
            ..mesh
        };
        assert_eq!(neighbor.mesh_stride(2), 1);
        assert_eq!(mlp.mesh_layers(), 0);
        assert_eq!("mesh-mlp".parse(), Ok(Architecture::MeshMlp));
        assert_eq!(Architecture::MeshMlp.to_string(), "mesh-mlp");
    }

    #[test]
    fn initialization_is_seeded_and_bounded_by_fan_in() {
        let config = SimpleConfig {
            architecture: Architecture::ResidualMlp,
            width: 16,
            depth: 2,
            ..SimpleConfig::default()
        };
        let parameters = config.initial_parameters(3);
        assert_eq!(parameters, config.initial_parameters(3));
        assert_ne!(parameters, config.initial_parameters(4));
        let mut offset = 0;
        for layer in config.layers() {
            let bound = 1.0 / (layer.inputs as f32).sqrt();
            let weights = &parameters[offset..offset + layer.parameters()];
            assert!(weights.iter().all(|w| w.abs() <= bound));
            // A uniform draw fills most of its interval.
            assert!(weights.iter().any(|&w| w > 0.9 * bound));
            assert!(weights.iter().any(|&w| w < -0.9 * bound));
            offset += layer.parameters();
        }
        assert_eq!(offset, parameters.len());
    }

    #[test]
    fn equations_encode_both_operands_and_the_answer() {
        let config = SimpleConfig {
            encoding: Encoding::Scalar,
            modulus: 5,
            ..SimpleConfig::default()
        };
        let equations = [
            Equation {
                a: 4,
                b: 2,
                answer: 2,
            },
            Equation {
                a: 0,
                b: 1,
                answer: 0,
            },
        ];
        let (features, targets) = config.encode(&equations).unwrap();
        assert_eq!(features, [1.0, 0.0, -1.0, -0.5]);
        assert_eq!(targets, [2, 0]);
        assert!(
            config
                .encode(&[Equation {
                    a: 5,
                    b: 1,
                    answer: 0
                }])
                .is_err()
        );
    }
}
