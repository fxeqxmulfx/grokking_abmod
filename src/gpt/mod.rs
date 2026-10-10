//! A Transformer whose learned parameters are mesh edges, including tied embeddings.
//! Bias-free, with parameter-free RMSNorm, QK-Norm, slice RoPE, and XSA.

mod checkpoint;
mod config;
mod gpu;
mod grokking;
mod reference;
mod text;

pub use checkpoint::Checkpoint;
pub use config::{AdamWOptions, GptConfig, GptTrainOptions};
pub use gpu::{GpuMeshGpt, SpecializedMeshGpt};
pub use grokking::{
    EQUATION_CONTEXT, Equation, EquationDataset, EquationSampler, ModularOperation,
};
pub use reference::ReferenceGpt;
pub use text::{CharacterTokenizer, TextDataset};

/// Target value that excludes a position from loss and accuracy.
pub const IGNORE_TARGET: u32 = u32::MAX;

pub(crate) const MAX_WIDTH: usize = 128;
pub(crate) const MAX_CONTEXT: usize = 128;
pub(crate) const MAX_HEADS: usize = 8;
pub(crate) const MAX_VOCAB: usize = 256;
/// Maximum number of token positions in a validated training or inference batch.
pub const MAX_ROWS: usize = 131_072;
// Resource limits for validated configurations; kernel capacities are const generics.
pub(crate) const MODEL_MAX_WIDTH: usize = 4_096;
pub(crate) const MODEL_MAX_CONTEXT: usize = 4_096;
pub(crate) const MODEL_MAX_HEADS: usize = 64;
pub(crate) const MODEL_MAX_VOCAB: usize = 65_536;
pub(crate) const ROPE_BASE: f32 = 10_000.0;
pub(crate) const RMSNORM_EPSILON: f32 = 1e-5;
// Matches F.normalize's default norm floor in the XSA paper's pseudocode.
pub(crate) const XSA_EPSILON: f32 = 1e-12;
pub(crate) type Features<const W: usize = MAX_WIDTH> = [f32; W];
pub(crate) type Attention<const C: usize = MAX_CONTEXT, const H: usize = MAX_HEADS> = [[f32; C]; H];
pub(crate) type Vocabulary<const V: usize = MAX_VOCAB> = [f32; V];
// Shared scratch for inverse RMS and the backward gradient projection.
pub(crate) type NormStats = [f32; 2];
