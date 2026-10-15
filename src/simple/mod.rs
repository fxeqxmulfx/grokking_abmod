//! Attention-free baselines for the grokking experiments. Each model reads the
//! two encoded operands of `a op b` and predicts the answer: dense/tied readout
//! baselines, residual models, operand products, and a fully two-parent network
//! with a sparse input and sparse readout.

mod gpu;
mod reference;

use std::{fmt, str::FromStr};

use crate::{
    Result,
    gpt::{Equation, MeshConnectivity},
    mesh::Random,
};

pub use gpu::{DeviceBatches, DeviceTable, Evaluation, GpuSimpleModel};
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
    /// Mesh GPT, or only its own channel with `Connectivity::Own`. Depth 1 is
    /// the one-layer MLP.
    MeshMlp,
    /// An entirely two-parent network: sparse input, hidden and output layers,
    /// local activations and per-node biases. No dense embedding or readout,
    /// attention, residual branch or normalization mixes additional parents.
    FullMeshMlp,
    /// Shared linear Neighbor embedding chains for a/b and their transposed
    /// readout, Rotate90 roles, pre-RMSNorm residual mesh blocks and final RMSNorm.
    SharedNeighborMeshMlp,
    /// Shared operand embeddings, their ordered concatenation projected to
    /// `width`, `depth - 1` mesh layers, and the shared embedding's transpose
    /// as readout. The embedding has one optimizer state for all three uses.
    TiedConcatMeshMlp,
    /// An operand embedding, `depth` pre-RMSNorm feed-forward blocks with
    /// residual connections, a final RMSNorm, and a readout.
    ResidualMlp,
    /// Pre-RMSNorm residual blocks whose branches are fixed-width mesh
    /// projections, with `mesh_depth` layers per branch and no dense FFN.
    ResidualMeshMlp,
    /// Shared operand embeddings, an ordered concatenation projection,
    /// pre-RMSNorm residual mesh blocks, and the embedding's transposed readout.
    TiedConcatResidualMeshMlp,
    /// Shared operand/readout embedding. Rotate adjacent coordinate pairs of
    /// operand b by 90 degrees, add operand a, then use residual mesh blocks.
    TiedRotaryResidualMeshMlp,
    /// Shared rotated operand embeddings and tied readout, with one activated
    /// width-by-width dense projection in each pre-RMSNorm residual branch.
    TiedRotaryResidualDenseMlp,
    /// The elementwise product of separate `a` and `b` embeddings, read out.
    Bilinear,
}

impl fmt::Display for Architecture {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Mlp => "mlp",
            Self::MeshMlp => "mesh-mlp",
            Self::FullMeshMlp => "full-mesh-mlp",
            Self::SharedNeighborMeshMlp => "shared-neighbor-mesh-mlp",
            Self::TiedConcatMeshMlp => "tied-concat-mesh-mlp",
            Self::ResidualMlp => "residual-mlp",
            Self::ResidualMeshMlp => "residual-mesh-mlp",
            Self::TiedConcatResidualMeshMlp => "tied-concat-residual-mesh-mlp",
            Self::TiedRotaryResidualMeshMlp => "tied-rotary-residual-mesh-mlp",
            Self::TiedRotaryResidualDenseMlp => "tied-rotary-residual-dense-mlp",
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
            "full-mesh-mlp" => Ok(Self::FullMeshMlp),
            "shared-neighbor-mesh-mlp" => Ok(Self::SharedNeighborMeshMlp),
            "tied-concat-mesh-mlp" => Ok(Self::TiedConcatMeshMlp),
            "residual-mlp" => Ok(Self::ResidualMlp),
            "residual-mesh-mlp" => Ok(Self::ResidualMeshMlp),
            "tied-concat-residual-mesh-mlp" => Ok(Self::TiedConcatResidualMeshMlp),
            "tied-rotary-residual-mesh-mlp" => Ok(Self::TiedRotaryResidualMeshMlp),
            "tied-rotary-residual-dense-mlp" => Ok(Self::TiedRotaryResidualDenseMlp),
            "bilinear" => Ok(Self::Bilinear),
            _ => Err(format!("unknown model: {value}")),
        }
    }
}

impl Architecture {
    pub fn tied_embeddings(self) -> bool {
        matches!(
            self,
            Self::TiedConcatMeshMlp
                | Self::SharedNeighborMeshMlp
                | Self::TiedConcatResidualMeshMlp
                | Self::TiedRotaryResidualMeshMlp
                | Self::TiedRotaryResidualDenseMlp
        )
    }

    pub fn residual_mesh(self) -> bool {
        matches!(
            self,
            Self::ResidualMeshMlp
                | Self::TiedConcatResidualMeshMlp
                | Self::TiedRotaryResidualMeshMlp
        )
    }

    pub fn rotary_embeddings(self) -> bool {
        matches!(
            self,
            Self::TiedRotaryResidualMeshMlp
                | Self::TiedRotaryResidualDenseMlp
                | Self::SharedNeighborMeshMlp
        )
    }

    pub fn residual_dense(self) -> bool {
        self == Self::TiedRotaryResidualDenseMlp
    }
}

/// The parents of each mesh-layer unit, its own channel first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Connectivity {
    /// Ablation: no second parent, so a mesh layer scales each activated
    /// channel by a learned gain.
    Own,
    /// Mesh GPT's neighbor connectivity: the second parent is channel `c + 1`.
    #[default]
    Neighbor,
    /// Mesh GPT's butterfly connectivity: in mesh layer `l` the second parent
    /// is channel `c + 2^(l mod floor(log2 width))`.
    Butterfly,
}

impl Connectivity {
    pub fn name(self) -> &'static str {
        match self {
            Self::Own => "own",
            Self::Neighbor => MeshConnectivity::Neighbor.name(),
            Self::Butterfly => MeshConnectivity::Butterfly.name(),
        }
    }

    /// Weights per mesh-layer unit.
    pub fn parents(self) -> usize {
        match self {
            Self::Own => 1,
            Self::Neighbor | Self::Butterfly => 2,
        }
    }

    /// Distance from a unit to its second parent, in `1..width`, or 0 without one.
    fn stride(self, width: usize, layer: usize) -> usize {
        match self {
            Self::Own => 0,
            Self::Neighbor => MeshConnectivity::Neighbor.stride(width, layer),
            Self::Butterfly => MeshConnectivity::Butterfly.stride(width, layer),
        }
    }
}

impl fmt::Display for Connectivity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

impl FromStr for Connectivity {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "own" => Ok(Self::Own),
            "neighbor" => Ok(Self::Neighbor),
            "butterfly" => Ok(Self::Butterfly),
            _ => Err(format!("unknown mesh connectivity: {value}")),
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

/// Initialization of the separate, fully sparse two-parent network.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FullMeshInitialization {
    /// Independent Kaiming-normal hidden weights, linear-gain output weights.
    #[default]
    Kaiming,
    /// Random input/output layers, identity hidden layers that learn mixing.
    IdentityHidden,
}

impl fmt::Display for FullMeshInitialization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Kaiming => "kaiming",
            Self::IdentityHidden => "identity-hidden",
        })
    }
}

impl FromStr for FullMeshInitialization {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "kaiming" => Ok(Self::Kaiming),
            "identity-hidden" => Ok(Self::IdentityHidden),
            _ => Err(format!("unknown full-mesh initialization: {value}")),
        }
    }
}

/// Initialization controls for the tied dense embedding/readout matrix.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DenseEmbeddingInitialization {
    /// Historical U(-1/sqrt(classes), 1/sqrt(classes)) embedding.
    #[default]
    UniformFanIn,
    /// Normal(0, 1/width) variance for the shared readout's width inputs.
    NormalWidth,
}

impl fmt::Display for DenseEmbeddingInitialization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UniformFanIn => "uniform-fan-in",
            Self::NormalWidth => "normal-width",
        })
    }
}

impl FromStr for DenseEmbeddingInitialization {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "uniform-fan-in" => Ok(Self::UniformFanIn),
            "normal-width" => Ok(Self::NormalWidth),
            _ => Err(format!("unknown dense embedding initialization: {value}")),
        }
    }
}

/// Matched initialization controls for the shared linear Neighbor chain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SharedEmbeddingInitialization {
    /// Historical identity plus small Gaussian neighbor weights.
    #[default]
    NearIdentity,
    /// Small orthogonal paired rotations, with unit implicit class columns.
    UnitNormWeak,
    /// Signed 45-degree paired rotations, with the same column normalization.
    UnitNormStrong,
}

impl SharedEmbeddingInitialization {
    /// Gaussian neighbor deviation or the paired rotation's tangent.
    pub fn ratio(self, depth: usize) -> f64 {
        match self {
            Self::NearIdentity | Self::UnitNormWeak => 0.1 / (depth as f64).sqrt(),
            Self::UnitNormStrong => 1.0,
        }
    }

    pub fn normalizes_columns(self) -> bool {
        self != Self::NearIdentity
    }
}

impl fmt::Display for SharedEmbeddingInitialization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NearIdentity => "near-identity",
            Self::UnitNormWeak => "unit-norm-weak",
            Self::UnitNormStrong => "unit-norm-strong",
        })
    }
}

impl FromStr for SharedEmbeddingInitialization {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "near-identity" => Ok(Self::NearIdentity),
            "unit-norm-weak" => Ok(Self::UnitNormWeak),
            "unit-norm-strong" => Ok(Self::UnitNormStrong),
            _ => Err(format!("unknown shared embedding initialization: {value}")),
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

/// A two-parent affine layer. Each output has two weights and one bias. Biases
/// are parameters, not connections from other activation nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoParentLayer {
    pub inputs: usize,
    pub outputs: usize,
    pub stride: usize,
}

impl TwoParentLayer {
    pub fn parents(self, output: usize) -> [usize; 2] {
        let first = output * self.inputs / self.outputs;
        [first, (first + self.stride) % self.inputs]
    }

    pub fn parameters(self) -> usize {
        3 * self.outputs
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
    /// MLP and full-mesh hidden layers or residual blocks. Bilinear models ignore it.
    pub depth: usize,
    /// Feed-forward width of a residual block, as a multiple of `width`.
    pub expansion: usize,
    /// Fixed-width mesh layers per residual branch. Only residual mesh MLPs use it.
    pub mesh_depth: usize,
    /// Apply the configured activation after the last mesh layer, before the skip sum.
    /// Residual mesh branches otherwise activate only between their mesh layers.
    pub mesh_output_activation: bool,
    /// The parents of each mesh-layer unit. Only mesh MLPs use it.
    pub connectivity: Connectivity,
    /// Every mesh layer uses the first mesh layer's weights, so a mesh MLP
    /// of any depth stores one mesh layer. Only mesh MLPs use it.
    pub tied_mesh: bool,
    /// Core initialization of the full mesh and shared Neighbor architectures.
    pub full_mesh_initialization: FullMeshInitialization,
    /// Linear Neighbor layers shared by a/b and transposed readout. Only the
    /// fully sparse SharedNeighborMeshMlp uses this depth.
    pub embedding_mesh_depth: usize,
    /// Initialization-only mixing and column scale of the shared linear chain.
    pub shared_embedding_initialization: SharedEmbeddingInitialization,
    /// Initialization-only distribution of E in the rotary dense residual model.
    pub dense_embedding_initialization: DenseEmbeddingInitialization,
    /// Factors per hidden channel whose balanced L2 penalty the weight decay
    /// imitates. For `n` other than 2, AdamW decays the input column and
    /// readout row of channel `c` at `λ s_c^(2/n - 1)`, where
    /// `s_c = (|E_c|² + |R_c|²) / 2`: the gradient of `(nλ/2) Σ_c s_c^(2/n)`,
    /// which at `|E_c| = |R_c|` is the least L2 penalty of `n` factors with
    /// the product `|E_c| |R_c|`. Only one-layer MLPs use other values.
    pub decay_factors: usize,
    /// Mesh-layer weights train at this multiple of the learning rate, their
    /// decay included. Only mesh MLPs with mesh layers use other values.
    pub mesh_learning_rate_multiplier: usize,
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
            mesh_depth: 4,
            mesh_output_activation: false,
            connectivity: Connectivity::Neighbor,
            tied_mesh: false,
            full_mesh_initialization: FullMeshInitialization::Kaiming,
            embedding_mesh_depth: 4,
            shared_embedding_initialization: SharedEmbeddingInitialization::NearIdentity,
            dense_embedding_initialization: DenseEmbeddingInitialization::UniformFanIn,
            decay_factors: 2,
            mesh_learning_rate_multiplier: 1,
        }
    }
}

impl SimpleConfig {
    pub fn validate(&self) -> Result<()> {
        if self.dense_embedding_initialization != DenseEmbeddingInitialization::UniformFanIn
            && self.architecture != Architecture::TiedRotaryResidualDenseMlp
        {
            return Err(
                "width-normal embedding initialization needs the rotary dense residual model"
                    .into(),
            );
        }
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
        if self.architecture.residual_mesh() && !(1..=MAX_SIMPLE_DEPTH).contains(&self.mesh_depth) {
            return Err(format!("mesh depth must be in 1..={MAX_SIMPLE_DEPTH}").into());
        }
        if self.architecture == Architecture::SharedNeighborMeshMlp
            && !(1..=MAX_SIMPLE_DEPTH).contains(&self.embedding_mesh_depth)
        {
            return Err("shared Neighbor embedding depth must be in 1..=64".into());
        }
        if self.shared_embedding_initialization.normalizes_columns()
            && (self.architecture != Architecture::SharedNeighborMeshMlp
                || !self.embedding_mesh_depth.is_multiple_of(2))
        {
            return Err("unit-norm embedding initialization needs a shared Neighbor model with even embedding depth".into());
        }
        if self.mesh_output_activation && !self.architecture.residual_mesh() {
            return Err("only residual mesh branches activate their final mesh output".into());
        }
        if self.mesh_layers() > 0 && self.connectivity.parents() == 2 && self.width < 2 {
            return Err("a second mesh parent needs a width of at least 2".into());
        }
        if self.tied_mesh
            && !matches!(
                self.architecture,
                Architecture::MeshMlp | Architecture::TiedConcatMeshMlp
            )
        {
            return Err("only mesh MLPs tie mesh layers".into());
        }
        if matches!(
            self.architecture,
            Architecture::TiedConcatMeshMlp | Architecture::TiedConcatResidualMeshMlp
        ) && (self.encoding != Encoding::OneHot || 2 * self.width > MAX_SIMPLE_WIDTH)
        {
            return Err(format!(
                "tied concatenation needs one-hot encoding and width <= {}",
                MAX_SIMPLE_WIDTH / 2
            )
            .into());
        }
        if self.architecture.rotary_embeddings()
            && (self.encoding != Encoding::OneHot
                || self.width < 2
                || !self.width.is_multiple_of(2))
        {
            return Err("rotary roles need one-hot encoding and a positive even width".into());
        }
        if self.architecture == Architecture::FullMeshMlp
            && (self.encoding != Encoding::Binary || self.connectivity == Connectivity::Own)
        {
            return Err(
                "a full mesh needs ordinary binary inputs and two-parent connectivity".into(),
            );
        }
        if self.full_mesh_initialization != FullMeshInitialization::Kaiming
            && !matches!(
                self.architecture,
                Architecture::FullMeshMlp | Architecture::SharedNeighborMeshMlp
            )
        {
            return Err("only a full mesh configures its mesh initialization".into());
        }
        if self.architecture == Architecture::SharedNeighborMeshMlp
            && (self.encoding != Encoding::OneHot
                || self.width < self.modulus
                || !self.width.is_multiple_of(2)
                || !(1..=MAX_SIMPLE_DEPTH).contains(&self.embedding_mesh_depth)
                || self.connectivity == Connectivity::Own)
        {
            return Err("shared Neighbor meshes need one-hot inputs, even width >= classes, two core parents and embedding depth 1..=64".into());
        }
        if self.decay_factors < 2 {
            return Err("weight decay imitates at least two factors per channel".into());
        }
        if self.decay_factors != 2 && (self.architecture != Architecture::Mlp || self.depth != 1) {
            return Err("only one-layer MLPs decay as more factors per channel".into());
        }
        if self.mesh_learning_rate_multiplier == 0 {
            return Err("the mesh learning rate multiplier must be positive".into());
        }
        if self.mesh_learning_rate_multiplier != 1 && self.mesh_layers() == 0 {
            return Err("only mesh layers multiply their learning rate".into());
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
            Architecture::MeshMlp | Architecture::ResidualMeshMlp => {
                vec![dense(self.input_features(), width), dense(width, classes)]
            }
            Architecture::FullMeshMlp | Architecture::SharedNeighborMeshMlp => Vec::new(),
            Architecture::TiedConcatMeshMlp | Architecture::TiedConcatResidualMeshMlp => {
                vec![dense(classes, width), dense(2 * width, width)]
            }
            Architecture::TiedRotaryResidualMeshMlp => vec![dense(classes, width)],
            Architecture::TiedRotaryResidualDenseMlp => {
                let mut layers = vec![dense(classes, width)];
                layers.extend(std::iter::repeat_n(dense(width, width), self.depth));
                layers
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

    /// Mesh layers of a mesh MLP, each with one weight per unit and parent.
    pub fn mesh_layers(&self) -> usize {
        match self.architecture {
            Architecture::FullMeshMlp => self.depth + 1,
            Architecture::SharedNeighborMeshMlp => self.embedding_mesh_depth + self.depth,
            Architecture::MeshMlp | Architecture::TiedConcatMeshMlp => self.depth.saturating_sub(1),
            Architecture::ResidualMeshMlp
            | Architecture::TiedConcatResidualMeshMlp
            | Architecture::TiedRotaryResidualMeshMlp => self.depth * self.mesh_depth,
            _ => 0,
        }
    }

    /// All affine layers of FullMeshMlp, including the sparse readout. The input
    /// layer pairs corresponding a/b bits. Width changes distribute the first
    /// parents evenly; same-width layers use the requested mesh connectivity.
    pub fn two_parent_layers(&self) -> Vec<TwoParentLayer> {
        if self.architecture == Architecture::SharedNeighborMeshMlp {
            return (0..self.depth)
                .map(|layer| TwoParentLayer {
                    inputs: self.width,
                    outputs: self.width,
                    stride: self.mesh_stride(layer),
                })
                .collect();
        }
        if self.architecture != Architecture::FullMeshMlp {
            return Vec::new();
        }
        let mut layers = vec![TwoParentLayer {
            inputs: self.input_features(),
            outputs: self.width,
            stride: self.operand_features(),
        }];
        layers.extend((1..self.depth).map(|layer| TwoParentLayer {
            inputs: self.width,
            outputs: self.width,
            stride: self.mesh_stride(layer - 1),
        }));
        layers.push(TwoParentLayer {
            inputs: self.width,
            outputs: self.modulus,
            stride: self.mesh_stride(self.depth - 1),
        });
        layers
    }

    /// Potential input dependencies of each full-mesh logit, independent of
    /// current weights and activations. Binary inputs use at most 20 bits.
    pub fn full_mesh_receptive_fields(&self) -> Vec<u64> {
        if self.architecture != Architecture::FullMeshMlp {
            return Vec::new();
        }
        let mut fields: Vec<_> = (0..self.input_features()).map(|bit| 1_u64 << bit).collect();
        for layer in self.two_parent_layers() {
            fields = (0..layer.outputs)
                .map(|output| {
                    let [a, b] = layer.parents(output);
                    fields[a] | fields[b]
                })
                .collect();
        }
        fields
    }

    /// Input identity routes through the learned meshes, excluding global RMS
    /// statistics. Padding contributes no feature. This is a lower bound on
    /// full dependencies, and avoids using a norm scalar as a shortcut in the
    /// depth/coverage check.
    pub fn shared_neighbor_input_coverage(&self) -> Vec<usize> {
        if self.architecture != Architecture::SharedNeighborMeshMlp {
            return Vec::new();
        }
        let words = (2 * self.modulus).div_ceil(64);
        let combine = |fields: &[Vec<u64>], stride: usize| -> Vec<Vec<u64>> {
            (0..self.width)
                .map(|c| {
                    fields[c]
                        .iter()
                        .zip(&fields[(c + stride) % self.width])
                        .map(|(a, b)| a | b)
                        .collect()
                })
                .collect()
        };
        let mut a = vec![vec![0_u64; words]; self.width];
        let mut b = a.clone();
        for class in 0..self.modulus {
            a[class][class / 64] |= 1 << (class % 64);
            let bit = self.modulus + class;
            b[class][bit / 64] |= 1 << (bit % 64);
        }
        for _ in 0..self.embedding_mesh_depth {
            a = combine(&a, 1);
            b = combine(&b, 1);
        }
        let mut fields: Vec<Vec<u64>> = (0..self.width)
            .map(|c| {
                let rotated = if c.is_multiple_of(2) { c + 1 } else { c - 1 };
                a[c].iter().zip(&b[rotated]).map(|(a, b)| a | b).collect()
            })
            .collect();
        for layer in 0..self.depth {
            fields = combine(&fields, self.mesh_stride(layer));
        }
        for _ in 0..self.embedding_mesh_depth {
            fields = combine(&fields, self.width - 1);
        }
        fields
            .iter()
            .take(self.modulus)
            .map(|field| field.iter().map(|word| word.count_ones() as usize).sum())
            .collect()
    }

    pub(crate) fn residual_branch_depth(&self) -> usize {
        if self.architecture.residual_dense() {
            1
        } else {
            self.mesh_depth
        }
    }

    pub(crate) fn residual_branch_activation(&self) -> bool {
        self.architecture.residual_dense() || self.mesh_output_activation
    }

    /// Mesh layers with their own weights: at most one when they are tied.
    pub fn stored_mesh_layers(&self) -> usize {
        if self.tied_mesh {
            self.mesh_layers().min(1)
        } else {
            self.mesh_layers()
        }
    }

    /// The stored mesh layer whose weights mesh layer `layer` uses.
    pub(crate) fn stored_mesh_layer(&self, layer: usize) -> usize {
        if self.tied_mesh { 0 } else { layer }
    }

    /// The exponent `2/n - 1` of the hidden-channel decay multipliers.
    pub(crate) fn decay_exponent(&self) -> f32 {
        (2.0 / self.decay_factors as f64 - 1.0) as f32
    }

    /// Distance from each unit of mesh layer `layer` to its second parent, or
    /// 0 without one.
    pub fn mesh_stride(&self, layer: usize) -> usize {
        self.connectivity.stride(self.width, layer)
    }

    /// Parameter group sizes in storage order: the dense layers, then the
    /// stored mesh layers.
    pub(crate) fn groups(&self) -> Vec<usize> {
        if self.architecture == Architecture::SharedNeighborMeshMlp {
            let mut groups = vec![2 * self.width; self.embedding_mesh_depth];
            groups.extend(std::iter::repeat_n(3 * self.width, self.depth));
            return groups;
        }
        if self.architecture == Architecture::FullMeshMlp {
            return self
                .two_parent_layers()
                .into_iter()
                .map(TwoParentLayer::parameters)
                .collect();
        }
        let mut groups: Vec<_> = self.layers().into_iter().map(Dense::parameters).collect();
        let mesh = self.connectivity.parents() * self.width;
        groups.extend(std::iter::repeat_n(mesh, self.stored_mesh_layers()));
        groups
    }

    pub fn parameter_count(&self) -> usize {
        self.groups().into_iter().sum()
    }

    /// PyTorch's default `nn.Linear` initialization, U(-1/sqrt(fan_in), 1/sqrt(fan_in)),
    /// drawn layer by layer from a seeded SplitMix64 stream. Mesh layers start
    /// as the identity and draw nothing, so a ReLU mesh MLP of any depth first
    /// computes the one-layer MLP of the same seed. Rotary dense residual
    /// projections instead use Kaiming normal, fan-in, with ReLU gain sqrt(2).
    /// Its optional width-normal E consumes the same random draws as the
    /// historical uniform E, preserving every initial dense branch weight.
    /// Full meshes initialize their two-parent layers separately and zero biases.
    pub fn initial_parameters(&self, seed: u64) -> Vec<f32> {
        let mut random = Random::new(seed ^ 0x73696d706c650004);
        let mut parameters = Vec::with_capacity(self.parameter_count());
        if self.architecture == Architecture::SharedNeighborMeshMlp {
            // Both controls draw exactly the same Gaussian stream. Their core
            // weights therefore match, despite different embedding mixing.
            let mode = self.shared_embedding_initialization;
            let ratio = mode.ratio(self.embedding_mesh_depth);
            let own = if mode.normalizes_columns() {
                (1.0 + ratio * ratio).sqrt().recip()
            } else {
                1.0
            };
            let deviation = ratio * own;
            for _ in 0..self.embedding_mesh_depth * self.width {
                let radius = (-2.0 * (1.0 - f64::from(random.unit())).ln()).sqrt();
                let angle = std::f64::consts::TAU * f64::from(random.unit());
                parameters
                    .extend_from_slice(&[own as f32, (deviation * radius * angle.cos()) as f32]);
            }
            if mode.normalizes_columns() {
                self.initialize_shared_rotation_factors(&mut parameters, ratio);
                self.normalize_shared_embedding_columns(&mut parameters);
            }
            for layer in 0..self.depth {
                let identity = self.full_mesh_initialization
                    == FullMeshInitialization::IdentityHidden
                    && layer > 0;
                for _ in 0..self.width {
                    let weights = if identity {
                        [1.0, 0.0]
                    } else {
                        let radius = (-2.0 * (1.0 - f64::from(random.unit())).ln()).sqrt();
                        let angle = std::f64::consts::TAU * f64::from(random.unit());
                        [(radius * angle.cos()) as f32, (radius * angle.sin()) as f32]
                    };
                    parameters.extend_from_slice(&[weights[0], weights[1], 0.0]);
                }
            }
            return parameters;
        }
        if self.architecture == Architecture::FullMeshMlp {
            for (index, layer) in self.two_parent_layers().into_iter().enumerate() {
                let identity = self.full_mesh_initialization
                    == FullMeshInitialization::IdentityHidden
                    && index > 0
                    && index < self.depth;
                let deviation = if index == self.depth {
                    0.5_f64.sqrt()
                } else {
                    1.0
                };
                for _ in 0..layer.outputs {
                    let weights = if identity {
                        [1.0, 0.0]
                    } else {
                        let radius = (-2.0 * (1.0 - f64::from(random.unit())).ln()).sqrt();
                        let angle = std::f64::consts::TAU * f64::from(random.unit());
                        [
                            (deviation * radius * angle.cos()) as f32,
                            (deviation * radius * angle.sin()) as f32,
                        ]
                    };
                    parameters.extend_from_slice(&[weights[0], weights[1], 0.0]);
                }
            }
            return parameters;
        }
        for (index, layer) in self.layers().into_iter().enumerate() {
            let normal_embedding = index == 0
                && self.dense_embedding_initialization == DenseEmbeddingInitialization::NormalWidth;
            if normal_embedding || self.architecture.residual_dense() && index > 0 {
                // Box-Muller from the same SplitMix64 stream. Use f64 math
                // before the final FP32 cast, matching the browser initializer.
                let deviation = if normal_embedding {
                    (1.0 / layer.outputs as f64).sqrt()
                } else {
                    (2.0 / layer.inputs as f64).sqrt()
                };
                for offset in (0..layer.parameters()).step_by(2) {
                    let radius = (-2.0 * (1.0 - f64::from(random.unit())).ln()).sqrt();
                    let angle = std::f64::consts::TAU * f64::from(random.unit());
                    parameters.push((deviation * radius * angle.cos()) as f32);
                    if offset + 1 < layer.parameters() {
                        parameters.push((deviation * radius * angle.sin()) as f32);
                    }
                }
                continue;
            }
            let bound = 1.0 / (layer.inputs as f32).sqrt();
            parameters.extend((0..layer.parameters()).map(|_| (2.0 * random.unit() - 1.0) * bound));
        }
        // A unit weights its own channel by 1 and any second parent by 0.
        let unit = &[1.0, 0.0][..self.connectivity.parents()];
        for _ in 0..self.stored_mesh_layers() * self.width {
            parameters.extend_from_slice(unit);
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

    /// Two ordinary Neighbor layers implement an orthogonal pair rotation and
    /// a cyclic shift. On input (x[c+1], x[c+2]), the two output rows are
    /// (cos*x + sin*y, -sin*x + cos*y). Both controls use the same random signs.
    fn initialize_shared_rotation_factors(&self, parameters: &mut [f32], ratio: f64) {
        let angle = ratio.atan();
        let cosine = angle.cos();
        for layer in (0..self.embedding_mesh_depth).step_by(2) {
            for c in (0..self.width).step_by(2) {
                let first = 2 * (layer * self.width + c);
                let second = first + 2 * self.width;
                let sign = if parameters[first + 1] >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let sine = sign * angle.sin();
                parameters[first..first + 4].copy_from_slice(&[
                    0.0,
                    1.0,
                    -sine as f32,
                    cosine as f32,
                ]);
                parameters[second..second + 4].copy_from_slice(&[
                    cosine.recip() as f32,
                    (sine / cosine) as f32,
                    1.0,
                    0.0,
                ]);
            }
        }
    }

    /// Applies E <- E D by scaling the first layer's incoming columns. This
    /// preserves the sparse topology, linear chain and exactly tied transpose.
    /// Only the 97 (or configured modulus) used class columns are normalized.
    fn normalize_shared_embedding_columns(&self, parameters: &mut [f32]) {
        let mut scales = vec![1.0; self.width];
        for (class, scale) in scales.iter_mut().take(self.modulus).enumerate() {
            let mut row = vec![0.0_f64; self.width];
            row[class] = 1.0;
            for layer in 0..self.embedding_mesh_depth {
                row = (0..self.width)
                    .map(|c| {
                        let unit = 2 * (layer * self.width + c);
                        f64::from(parameters[unit]) * row[c]
                            + f64::from(parameters[unit + 1]) * row[(c + 1) % self.width]
                    })
                    .collect();
            }
            // The paired rotation factors are nonsingular with bounded scales.
            let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(norm.is_finite() && norm > 0.0);
            *scale = norm.recip() as f32;
        }
        for c in 0..self.width {
            parameters[2 * c] *= scales[c];
            parameters[2 * c + 1] *= scales[(c + 1) % self.width];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_neighbor_stores_embedding_once_and_has_no_dense_layers() {
        let config = SimpleConfig {
            architecture: Architecture::SharedNeighborMeshMlp,
            width: 128,
            depth: 8,
            embedding_mesh_depth: 4,
            ..SimpleConfig::default()
        };
        config.validate().unwrap();
        assert!(config.layers().is_empty());
        assert!(config.architecture.tied_embeddings());
        assert!(config.architecture.rotary_embeddings());
        assert_eq!(config.parameter_count(), 4096);
        assert_eq!(config.groups(), [vec![256; 4], vec![384; 8]].concat());
        assert!(
            config
                .shared_neighbor_input_coverage()
                .iter()
                .all(|&count| count < 194)
        );
        let global = SimpleConfig {
            depth: 64,
            embedding_mesh_depth: 32,
            ..config
        };
        assert!(
            global
                .shared_neighbor_input_coverage()
                .iter()
                .all(|&count| count == 194)
        );
        let initial = config.initial_parameters(11);
        assert_eq!(initial.len(), 4096);
        assert_eq!(initial, config.initial_parameters(11));
        assert_ne!(initial, config.initial_parameters(7));
        assert!(
            initial[..1024]
                .as_chunks::<2>()
                .0
                .iter()
                .all(|unit| unit[0] == 1.0 && unit[1] != 0.0)
        );
        assert!(
            initial[1024..]
                .as_chunks::<3>()
                .0
                .iter()
                .all(|unit| unit[2] == 0.0)
        );
        for layer in config.two_parent_layers() {
            assert_eq!((layer.inputs, layer.outputs, layer.stride), (128, 128, 1));
        }
        for invalid in [
            SimpleConfig {
                width: 96,
                ..config
            },
            SimpleConfig {
                width: 129,
                ..config
            },
            SimpleConfig {
                encoding: Encoding::Binary,
                ..config
            },
            SimpleConfig {
                embedding_mesh_depth: 0,
                ..config
            },
            SimpleConfig {
                embedding_mesh_depth: usize::MAX,
                ..config
            },
            SimpleConfig {
                tied_mesh: true,
                ..config
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn balanced_shared_initialization_matches_core_weights_and_column_norms() {
        let base = SimpleConfig {
            architecture: Architecture::SharedNeighborMeshMlp,
            modulus: 97,
            width: 128,
            depth: 64,
            embedding_mesh_depth: 32,
            full_mesh_initialization: FullMeshInitialization::IdentityHidden,
            ..SimpleConfig::default()
        };
        let offset = 2 * base.width * base.embedding_mesh_depth;
        let expected = (1.0 / base.width as f64 + f64::from(1e-5_f32))
            .sqrt()
            .recip();
        for seed in [11, 17, 23] {
            let legacy = base.initial_parameters(seed);
            for mode in [
                SharedEmbeddingInitialization::UnitNormWeak,
                SharedEmbeddingInitialization::UnitNormStrong,
            ] {
                let config = SimpleConfig {
                    shared_embedding_initialization: mode,
                    ..base
                };
                config.validate().unwrap();
                let weights = config.initial_parameters(seed);
                assert_eq!(weights.len(), 32768);
                assert_eq!(&weights[offset..], &legacy[offset..]);
                assert!(weights.iter().all(|v| v.is_finite()));
                assert_eq!(weights, config.initial_parameters(seed));
                // With a zero residual branch and b absent, the diagonal tied
                // readout is |E_a|^2 / sqrt(|E_a|^2/W + epsilon). This checks
                // unit norms through the independent forward and transpose.
                let probe = SimpleConfig { depth: 1, ..config };
                let mut parameters = weights[..offset + 3 * config.width].to_vec();
                parameters[offset..].fill(0.0);
                let model = ReferenceSimpleModel::from_parameters(probe, parameters).unwrap();
                for start in (0..config.modulus).step_by(8) {
                    let end = (start + 8).min(config.modulus);
                    let mut features = vec![0.0; (end - start) * probe.input_features()];
                    for class in start..end {
                        features[(class - start) * probe.input_features() + class] = 1.0;
                    }
                    for (i, row) in model.logits(&features).unwrap().iter().enumerate() {
                        for (q, &value) in row.iter().enumerate() {
                            let target = if q == start + i { expected } else { 0.0 };
                            assert!(
                                (value - target).abs() < 1e-5,
                                "{mode}, seed {seed}, input {}, output {q}",
                                start + i
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn balanced_shared_initialization_rejects_other_models_and_odd_depths() {
        for mode in [
            SharedEmbeddingInitialization::UnitNormWeak,
            SharedEmbeddingInitialization::UnitNormStrong,
        ] {
            assert!(
                SimpleConfig {
                    shared_embedding_initialization: mode,
                    ..SimpleConfig::default()
                }
                .validate()
                .is_err()
            );
            assert!(
                SimpleConfig {
                    architecture: Architecture::SharedNeighborMeshMlp,
                    modulus: 7,
                    width: 8,
                    embedding_mesh_depth: 3,
                    shared_embedding_initialization: mode,
                    ..SimpleConfig::default()
                }
                .validate()
                .is_err()
            );
            assert_eq!(
                mode.to_string()
                    .parse::<SharedEmbeddingInitialization>()
                    .unwrap(),
                mode
            );
        }
    }

    #[test]
    fn shared_neighbor_reverse_chain_equals_the_implicit_embedding_transpose() {
        let config = SimpleConfig {
            architecture: Architecture::SharedNeighborMeshMlp,
            modulus: 7,
            width: 8,
            depth: 2,
            embedding_mesh_depth: 3,
            ..SimpleConfig::default()
        };
        let weights = config.initial_parameters(11);
        let h: Vec<_> = (0..8).map(|c| (c as f64 * 0.73).cos()).collect();
        let mut readout = h.clone();
        for layer in (0..3).rev() {
            readout = (0..8)
                .map(|c| {
                    let previous = (c + 7) % 8;
                    f64::from(weights[2 * (layer * 8 + c)]) * readout[c]
                        + f64::from(weights[2 * (layer * 8 + previous) + 1]) * readout[previous]
                })
                .collect();
        }
        for (class, &actual) in readout.iter().enumerate().take(7) {
            let mut embedding = vec![0.0; 8];
            embedding[class] = 1.0;
            for layer in 0..3 {
                embedding = (0..8)
                    .map(|c| {
                        f64::from(weights[2 * (layer * 8 + c)]) * embedding[c]
                            + f64::from(weights[2 * (layer * 8 + c) + 1]) * embedding[(c + 1) % 8]
                    })
                    .collect();
            }
            let expected: f64 = embedding.iter().zip(&h).map(|(a, b)| a * b).sum();
            assert!((expected - actual).abs() < 1e-12);
        }
    }

    #[test]
    fn full_mesh_has_two_distinct_parents_including_input_and_readout() {
        let config = SimpleConfig {
            architecture: Architecture::FullMeshMlp,
            encoding: Encoding::Binary,
            width: 128,
            depth: 8,
            connectivity: Connectivity::Butterfly,
            ..SimpleConfig::default()
        };
        config.validate().unwrap();
        assert!(config.layers().is_empty());
        let layers = config.two_parent_layers();
        assert_eq!(layers.len(), 9);
        assert_eq!((layers[0].inputs, layers[0].outputs), (14, 128));
        assert_eq!((layers[8].inputs, layers[8].outputs), (128, 97));
        assert_eq!(config.parameter_count(), 3 * (8 * 128 + 97));
        for layer in layers {
            for output in 0..layer.outputs {
                let [a, b] = layer.parents(output);
                assert!(a < layer.inputs && b < layer.inputs && a != b);
            }
        }
        let all_inputs = (1_u64 << 14) - 1;
        assert!(
            config
                .full_mesh_receptive_fields()
                .iter()
                .all(|&field| field == all_inputs)
        );
        let local = SimpleConfig {
            connectivity: Connectivity::Neighbor,
            ..config
        };
        assert!(
            local
                .full_mesh_receptive_fields()
                .iter()
                .all(|&field| field != all_inputs)
        );
        let deep = SimpleConfig { depth: 64, ..local };
        assert!(
            deep.full_mesh_receptive_fields()
                .iter()
                .all(|&field| field == all_inputs)
        );
        for invalid in [
            SimpleConfig {
                encoding: Encoding::OneHot,
                ..config
            },
            SimpleConfig {
                connectivity: Connectivity::Own,
                ..config
            },
            SimpleConfig {
                tied_mesh: true,
                ..config
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn full_mesh_initialization_is_seeded_and_identity_applies_only_to_hidden_layers() {
        let config = SimpleConfig {
            architecture: Architecture::FullMeshMlp,
            encoding: Encoding::Binary,
            modulus: 7,
            width: 8,
            depth: 3,
            ..SimpleConfig::default()
        };
        let weights = config.initial_parameters(11);
        assert_eq!(weights.len(), config.parameter_count());
        assert_eq!(weights, config.initial_parameters(11));
        assert_ne!(weights, config.initial_parameters(7));
        assert!(weights.as_chunks::<3>().0.iter().all(|unit| unit[2] == 0.0));
        assert!(weights.as_chunks::<3>().0.iter().any(|unit| unit[1] != 0.0));
        let identity = SimpleConfig {
            full_mesh_initialization: FullMeshInitialization::IdentityHidden,
            ..config
        };
        identity.validate().unwrap();
        let weights = identity.initial_parameters(11);
        assert!(
            weights[24..72]
                .as_chunks::<3>()
                .0
                .iter()
                .all(|unit| *unit == [1.0, 0.0, 0.0])
        );
        assert!(
            weights[..24]
                .as_chunks::<3>()
                .0
                .iter()
                .any(|unit| unit[1] != 0.0)
        );
        assert!(
            weights[72..]
                .as_chunks::<3>()
                .0
                .iter()
                .any(|unit| unit[1] != 0.0)
        );
    }

    #[test]
    fn residual_mesh_config_bounds_branch_depth_and_counts_only_edges() {
        let config = SimpleConfig {
            architecture: Architecture::ResidualMeshMlp,
            width: 64,
            depth: 2,
            mesh_depth: 4,
            ..SimpleConfig::default()
        };
        config.validate().unwrap();
        assert_eq!(config.parameter_count(), 19_648);
        assert_eq!(config.mesh_layers(), 8);
        for mesh_depth in [0, MAX_SIMPLE_DEPTH + 1] {
            assert!(
                SimpleConfig {
                    mesh_depth,
                    ..config
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            SimpleConfig {
                tied_mesh: true,
                ..config
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn rotary_residual_mesh_stores_one_embedding_and_requires_coordinate_pairs() {
        let config = SimpleConfig {
            architecture: Architecture::TiedRotaryResidualMeshMlp,
            activation: Activation::Square,
            width: 128,
            depth: 2,
            mesh_depth: 3,
            ..SimpleConfig::default()
        };
        config.validate().unwrap();
        assert_eq!(config.parameter_count(), 13_952);
        assert_eq!(config.groups(), [12_416, 256, 256, 256, 256, 256, 256]);
        assert_eq!(config.initial_parameters(11).len(), 13_952);
        for width in [1, 3, 127] {
            assert!(SimpleConfig { width, ..config }.validate().is_err());
        }
        for encoding in [Encoding::Binary, Encoding::Scalar, Encoding::DiscreteLog] {
            assert!(SimpleConfig { encoding, ..config }.validate().is_err());
        }
        for mesh_depth in [0, MAX_SIMPLE_DEPTH + 1] {
            assert!(
                SimpleConfig {
                    mesh_depth,
                    ..config
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            SimpleConfig {
                tied_mesh: true,
                ..config
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn normal_width_embedding_preserves_dense_weights_and_has_width_scaled_variance() {
        for width in [128, 1024] {
            let base = SimpleConfig {
                architecture: Architecture::TiedRotaryResidualDenseMlp,
                width,
                depth: 1,
                ..SimpleConfig::default()
            };
            let normal = SimpleConfig {
                dense_embedding_initialization: DenseEmbeddingInitialization::NormalWidth,
                ..base
            };
            normal.validate().unwrap();
            for seed in [11, 17, 23] {
                let original = base.initial_parameters(seed);
                let changed = normal.initial_parameters(seed);
                assert_eq!(changed, normal.initial_parameters(seed));
                assert_eq!(changed.len(), original.len());
                let embedding = normal.modulus * width;
                assert_eq!(changed[embedding..], original[embedding..]);
                assert_ne!(changed[..embedding], original[..embedding]);
                let mean = changed[..embedding]
                    .iter()
                    .map(|&x| f64::from(x))
                    .sum::<f64>()
                    / embedding as f64;
                let variance = changed[..embedding]
                    .iter()
                    .map(|&x| (f64::from(x) - mean).powi(2))
                    .sum::<f64>()
                    / embedding as f64;
                assert!(mean.abs() < 0.025 / (width as f64).sqrt());
                assert!((variance * width as f64 - 1.0).abs() < 0.025);
                // Expected squared class norm is one, without projecting rows
                // onto a sphere or constraining parameters during training.
                let mean_squared_norm = changed[..embedding]
                    .iter()
                    .map(|&x| f64::from(x).powi(2))
                    .sum::<f64>()
                    / normal.modulus as f64;
                assert!((mean_squared_norm - 1.0).abs() < 0.025);
            }
        }
    }

    #[test]
    fn normal_width_embedding_rejects_unrelated_architectures_and_roundtrips() {
        let mode = DenseEmbeddingInitialization::NormalWidth;
        assert_eq!(
            mode.to_string()
                .parse::<DenseEmbeddingInitialization>()
                .unwrap(),
            mode
        );
        assert!(
            "normal-typo"
                .parse::<DenseEmbeddingInitialization>()
                .is_err()
        );
        for architecture in [
            Architecture::Mlp,
            Architecture::TiedRotaryResidualMeshMlp,
            Architecture::SharedNeighborMeshMlp,
        ] {
            assert!(
                SimpleConfig {
                    architecture,
                    dense_embedding_initialization: mode,
                    ..SimpleConfig::default()
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn rotary_dense_branch_uses_seeded_kaiming_weights_and_trains_all_connections() {
        use super::ReferenceSimpleModel;
        use crate::gpt::{EquationDataset, ModularOperation};
        let dense = SimpleConfig {
            architecture: Architecture::TiedRotaryResidualDenseMlp,
            activation: Activation::Relu,
            modulus: 7,
            width: 6,
            depth: 1,
            ..SimpleConfig::default()
        };
        dense.validate().unwrap();
        assert_eq!(dense.groups(), [42, 36]);
        let mesh = SimpleConfig {
            architecture: Architecture::TiedRotaryResidualMeshMlp,
            mesh_depth: 1,
            mesh_output_activation: true,
            ..dense
        };
        let dataset = EquationDataset::new(7, ModularOperation::Divide, 0.5, 11).unwrap();
        let (features, targets) = dense.encode(&dataset.split(false)[..3]).unwrap();
        let model = ReferenceSimpleModel::new(dense, 11).unwrap();
        let sparse = ReferenceSimpleModel::new(mesh, 11).unwrap();
        assert_eq!(&model.parameters()[..42], &sparse.parameters()[..42]);
        assert_eq!(dense.initial_parameters(11), dense.initial_parameters(11));
        assert_ne!(dense.initial_parameters(11), dense.initial_parameters(7));
        let (_, gradient) = model.loss_and_gradient(&features, &targets).unwrap();
        assert!(
            gradient[42..]
                .iter()
                .enumerate()
                .any(|(i, value)| i / 6 != i % 6 && value.abs() > 1e-8)
        );
        for (width, parameters) in [(128, 28_800), (1024, 1_147_904)] {
            assert_eq!(
                SimpleConfig {
                    width,
                    modulus: 97,
                    ..dense
                }
                .parameter_count(),
                parameters
            );
        }
        let wide = SimpleConfig {
            width: 128,
            modulus: 97,
            ..dense
        };
        let initialized = wide.initial_parameters(11);
        let matrix = &initialized[97 * 128..];
        let mean = matrix.iter().map(|&v| f64::from(v)).sum::<f64>() / matrix.len() as f64;
        let variance = matrix
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / matrix.len() as f64;
        assert!(mean.abs() < 0.004);
        assert!((variance / (2.0 / 128.0) - 1.0).abs() < 0.05);
    }

    #[test]
    fn final_mesh_activation_preserves_parameters_and_needs_a_residual_mesh() {
        let config = SimpleConfig {
            architecture: Architecture::TiedRotaryResidualMeshMlp,
            activation: Activation::Square,
            width: 128,
            depth: 1,
            mesh_depth: 1,
            ..SimpleConfig::default()
        };
        let activated = SimpleConfig {
            mesh_output_activation: true,
            ..config
        };
        activated.validate().unwrap();
        assert_eq!(activated.parameter_count(), 12_672);
        assert_eq!(activated.groups(), config.groups());
        assert_eq!(
            activated.initial_parameters(11),
            config.initial_parameters(11)
        );
        for architecture in [
            Architecture::Mlp,
            Architecture::MeshMlp,
            Architecture::TiedConcatMeshMlp,
            Architecture::ResidualMlp,
            Architecture::Bilinear,
        ] {
            assert!(
                SimpleConfig {
                    architecture,
                    ..activated
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn tied_residual_mesh_shares_embedding_across_inputs_and_readout() {
        let config = SimpleConfig {
            architecture: Architecture::TiedConcatResidualMeshMlp,
            activation: Activation::Square,
            width: 128,
            depth: 2,
            mesh_depth: 3,
            connectivity: Connectivity::Neighbor,
            ..SimpleConfig::default()
        };
        config.validate().unwrap();
        assert_eq!(config.parameter_count(), 46_720);
        assert_eq!(
            config.groups(),
            [12_416, 32_768, 256, 256, 256, 256, 256, 256]
        );
        assert_eq!(config.initial_parameters(11).len(), 46_720);
        assert_eq!(config.mesh_layers(), 6);
        for encoding in [Encoding::Binary, Encoding::Scalar, Encoding::DiscreteLog] {
            assert!(SimpleConfig { encoding, ..config }.validate().is_err());
        }
        for mesh_depth in [0, MAX_SIMPLE_DEPTH + 1] {
            assert!(
                SimpleConfig {
                    mesh_depth,
                    ..config
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            SimpleConfig {
                tied_mesh: true,
                ..config
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn tied_concatenation_stores_shared_embedding_projection_and_mesh_only() {
        let config = SimpleConfig {
            architecture: Architecture::TiedConcatMeshMlp,
            connectivity: Connectivity::Own,
            width: 64,
            depth: 5,
            ..SimpleConfig::default()
        };
        config.validate().unwrap();
        assert_eq!(config.parameter_count(), 14_656);
        assert_eq!(config.groups(), [97 * 64, 128 * 64, 64, 64, 64, 64]);
        assert_eq!(config.initial_parameters(7).len(), 14_656);
        for encoding in [Encoding::Binary, Encoding::Scalar, Encoding::DiscreteLog] {
            assert!(SimpleConfig { encoding, ..config }.validate().is_err());
        }
        assert!(
            SimpleConfig {
                width: MAX_SIMPLE_WIDTH,
                ..config
            }
            .validate()
            .is_err()
        );
    }

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
        let own = SimpleConfig {
            connectivity: Connectivity::Own,
            ..mesh
        };
        assert_eq!(own.parameter_count(), 291 * 16 + 2 * 16);
        // Tied mesh layers store one layer at any depth.
        for depth in [2, 3, 9] {
            let tied = SimpleConfig {
                depth,
                tied_mesh: true,
                ..own
            };
            assert_eq!(tied.parameter_count(), 291 * 16 + 16);
            assert_eq!(tied.stored_mesh_layers(), 1);
            assert_eq!(tied.stored_mesh_layer(depth - 2), 0);
        }
        let tied = SimpleConfig {
            tied_mesh: true,
            ..mesh
        };
        assert_eq!(tied.parameter_count(), 291 * 16 + 2 * 16);
        assert_eq!(SimpleConfig { depth: 1, ..tied }.stored_mesh_layers(), 0);
        assert_eq!(own.stored_mesh_layer(1), 1);
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
            SimpleConfig {
                tied_mesh: true,
                ..valid
            },
            SimpleConfig {
                decay_factors: 1,
                ..valid
            },
            SimpleConfig {
                depth: 2,
                decay_factors: 3,
                ..valid
            },
            SimpleConfig {
                architecture: Architecture::MeshMlp,
                decay_factors: 3,
                ..valid
            },
            SimpleConfig {
                architecture: Architecture::MeshMlp,
                depth: 2,
                mesh_learning_rate_multiplier: 0,
                ..valid
            },
            SimpleConfig {
                mesh_learning_rate_multiplier: 2,
                ..valid
            },
            // A depth-1 mesh MLP has no mesh layer to train faster.
            SimpleConfig {
                architecture: Architecture::MeshMlp,
                mesh_learning_rate_multiplier: 2,
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
            // Without second parents, mesh layers need only one channel.
            SimpleConfig {
                architecture: Architecture::MeshMlp,
                width: 1,
                depth: 2,
                connectivity: Connectivity::Own,
                ..valid
            },
            SimpleConfig {
                architecture: Architecture::MeshMlp,
                depth: 9,
                tied_mesh: true,
                ..valid
            },
            SimpleConfig {
                decay_factors: 10,
                ..valid
            },
            SimpleConfig {
                architecture: Architecture::MeshMlp,
                depth: 2,
                connectivity: Connectivity::Own,
                mesh_learning_rate_multiplier: 8,
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
            connectivity: Connectivity::Butterfly,
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
            connectivity: Connectivity::Neighbor,
            ..mesh
        };
        assert_eq!(neighbor.mesh_stride(2), 1);
        // Own-channel meshes start with unit gains and have no second parent.
        let own = SimpleConfig {
            connectivity: Connectivity::Own,
            ..mesh
        };
        let gains = own.initial_parameters(3);
        assert_eq!(gains.len(), own.parameter_count());
        assert_eq!(gains[..dense.len()], dense);
        assert_eq!(gains[dense.len()..], [1.0; 3 * 8]);
        let tied = SimpleConfig {
            tied_mesh: true,
            ..own
        };
        assert_eq!(tied.initial_parameters(3), gains[..dense.len() + 8]);
        assert_eq!(own.mesh_stride(2), 0);
        assert_eq!(mlp.mesh_layers(), 0);
        assert_eq!("mesh-mlp".parse(), Ok(Architecture::MeshMlp));
        assert_eq!(Architecture::MeshMlp.to_string(), "mesh-mlp");
        for connectivity in [
            Connectivity::Own,
            Connectivity::Neighbor,
            Connectivity::Butterfly,
        ] {
            assert_eq!(connectivity.to_string().parse(), Ok(connectivity));
        }
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
