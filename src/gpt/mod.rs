//! An autoregressive pre-norm Transformer with two-parent mesh channel mixers.

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
pub(crate) type Features = [f32; MAX_WIDTH];
pub(crate) type Attention = [f32; MAX_CONTEXT * MAX_HEADS];
pub(crate) type Vocabulary = [f32; MAX_VOCAB];
pub(crate) type NormCache = [f32; MAX_WIDTH + 1];
