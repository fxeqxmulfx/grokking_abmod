//! A pre-RMSNorm Transformer with mesh mixers, slice RoPE, and exclusive attention.

mod checkpoint;
mod config;
mod gpu;
mod reference;
mod text;

pub use checkpoint::Checkpoint;
pub use config::{GptConfig, GptTrainOptions};
pub use gpu::GpuMeshGpt;
pub use reference::ReferenceGpt;
pub use text::{CharacterTokenizer, TextDataset};

pub(crate) const MAX_WIDTH: usize = 64;
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
// RMS-normalized channels followed by the inverse RMS in the last slot.
pub(crate) type NormCache = [f32; MAX_WIDTH + 1];
