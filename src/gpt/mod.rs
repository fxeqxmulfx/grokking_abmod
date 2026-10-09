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
pub use gpu::GpuMeshGpt;
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
pub(crate) const MAX_ROWS: usize = 4_096;
pub(crate) const ROPE_BASE: f32 = 10_000.0;
pub(crate) const RMSNORM_EPSILON: f32 = 1e-5;
// Matches F.normalize's default norm floor in the XSA paper's pseudocode.
pub(crate) const XSA_EPSILON: f32 = 1e-12;
pub(crate) type Features = [f32; MAX_WIDTH];
pub(crate) type Attention = [f32; MAX_CONTEXT * MAX_HEADS];
pub(crate) type Vocabulary = [f32; MAX_VOCAB];
// RMS-normalized channels followed by up to MAX_HEADS inverse RMS values.
pub(crate) type NormCache = [f32; MAX_WIDTH + MAX_HEADS];
