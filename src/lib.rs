//! Diamond mesh neural networks with CUDA Oxide training and inference.

pub mod gpt;
mod gpu;
mod mesh;

pub use gpu::{GpuMesh, LossPoint, TrainOptions, TrainReport};
pub use mesh::{MAX_DEPTH, MeshTopology, Node, ReferenceMesh, piecewise_target};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
