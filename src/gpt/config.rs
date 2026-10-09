use crate::{Result, mesh::Random};

use super::{MAX_CONTEXT, MAX_HEADS, MAX_ROWS, MAX_VOCAB, MAX_WIDTH};

/// Every learned channel mixer is a stack of width-preserving mesh layers.
/// A neuron reads channels c and (c + 1) mod width from the previous layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub context: usize,
    pub width: usize,
    pub heads: usize,
    pub layers: usize,
    pub attention_mesh_depth: usize,
    pub ffn_mesh_depth: usize,
}

impl Default for GptConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32,
            context: 32,
            width: 32,
            heads: 4,
            layers: 2,
            attention_mesh_depth: 4,
            ffn_mesh_depth: 6,
        }
    }
}

impl GptConfig {
    pub fn validate(&self) -> Result<()> {
        if !(2..=MAX_VOCAB).contains(&self.vocab_size) {
            return Err("vocab_size must be in 2..256".into());
        }
        if !(1..=MAX_CONTEXT).contains(&self.context) {
            return Err("context must be in 1..128".into());
        }
        if !(2..=MAX_WIDTH).contains(&self.width) {
            return Err("width must be in 2..64".into());
        }
        if !(1..=MAX_HEADS).contains(&self.heads) || !self.width.is_multiple_of(self.heads) {
            return Err("heads must be in 1..8 and divide width".into());
        }
        if !(1..=4).contains(&self.layers) {
            return Err("layers must be in 1..4".into());
        }
        if !(1..=16).contains(&self.attention_mesh_depth)
            || !(1..=16).contains(&self.ffn_mesh_depth)
        {
            return Err("mesh depths must be in 1..16".into());
        }
        Ok(())
    }

    pub fn parameter_count(&self) -> Result<usize> {
        Ok(Layout::new(*self)?.count())
    }

    pub(crate) fn validate_tokens(&self, tokens: &[u32], sequence: usize) -> Result<()> {
        if sequence == 0
            || sequence > self.context
            || tokens.is_empty()
            || !tokens.len().is_multiple_of(sequence)
            || tokens.len() > MAX_ROWS
        {
            return Err(
                "tokens must form a nonempty batch of equal sequences within context and 4096 rows"
                    .into(),
            );
        }
        if tokens
            .iter()
            .any(|&token| token as usize >= self.vocab_size)
        {
            return Err("token ID is outside the vocabulary".into());
        }
        Ok(())
    }

    pub(crate) fn validate_batch(
        &self,
        inputs: &[u32],
        targets: &[u32],
        sequence: usize,
    ) -> Result<()> {
        self.validate_tokens(inputs, sequence)?;
        self.validate_tokens(targets, sequence)?;
        if inputs.len() != targets.len() {
            return Err("inputs and targets must have equal lengths".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GptTrainOptions {
    pub steps: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub report_every: usize,
}

impl Default for GptTrainOptions {
    fn default() -> Self {
        Self {
            steps: 1_000,
            learning_rate: 0.003,
            weight_decay: 0.01,
            report_every: 100,
        }
    }
}

impl GptTrainOptions {
    pub fn validate(&self) -> Result<()> {
        if self.steps == 0 || self.report_every == 0 {
            return Err("steps and report_every must be positive".into());
        }
        if !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.weight_decay.is_finite()
            || self.weight_decay < 0.0
        {
            return Err(
                "learning_rate must be positive and weight_decay nonnegative; both must be finite"
                    .into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Kind {
    Embedding,
    Norm,
    Mesh(usize),
    Bias,
}

#[derive(Clone, Debug)]
pub(crate) struct Group {
    pub offset: usize,
    pub size: usize,
    pub kind: Kind,
}

#[derive(Clone, Debug)]
pub(crate) struct Block {
    pub norm1: usize,
    pub query: usize,
    pub key: usize,
    pub value: usize,
    pub projection: usize,
    pub norm2: usize,
    pub ffn: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct Layout {
    pub groups: Vec<Group>,
    pub blocks: Vec<Block>,
    pub final_norm: usize,
    pub head_bias: usize,
}

impl Layout {
    pub(crate) fn new(config: GptConfig) -> Result<Self> {
        config.validate()?;
        let mut layout = Self {
            groups: Vec::new(),
            blocks: Vec::new(),
            final_norm: 0,
            head_bias: 0,
        };
        layout.push(
            (config.vocab_size + config.context) * config.width,
            Kind::Embedding,
        );
        for _ in 0..config.layers {
            let norm1 = layout.push(2 * config.width, Kind::Norm);
            let mut mesh = || {
                layout.push(
                    3 * config.width * config.attention_mesh_depth,
                    Kind::Mesh(config.attention_mesh_depth),
                )
            };
            let query = mesh();
            let key = mesh();
            let value = mesh();
            let projection = mesh();
            let norm2 = layout.push(2 * config.width, Kind::Norm);
            let ffn = layout.push(
                3 * config.width * config.ffn_mesh_depth,
                Kind::Mesh(config.ffn_mesh_depth),
            );
            layout.blocks.push(Block {
                norm1,
                query,
                key,
                value,
                projection,
                norm2,
                ffn,
            });
        }
        layout.final_norm = layout.push(2 * config.width, Kind::Norm);
        layout.head_bias = layout.push(config.vocab_size, Kind::Bias);
        Ok(layout)
    }

    fn push(&mut self, size: usize, kind: Kind) -> usize {
        let id = self.groups.len();
        self.groups.push(Group {
            offset: self.count(),
            size,
            kind,
        });
        id
    }

    pub(crate) fn count(&self) -> usize {
        self.groups
            .last()
            .map_or(0, |group| group.offset + group.size)
    }

    pub(crate) fn initialize(&self, config: GptConfig, seed: u64) -> Vec<f32> {
        let mut random = Random::new(seed);
        let mut parameters = vec![0.0; self.count()];
        for group in &self.groups {
            let values = &mut parameters[group.offset..group.offset + group.size];
            match group.kind {
                Kind::Embedding => {
                    for value in values {
                        *value = 0.2 * (random.unit() - 0.5);
                    }
                }
                Kind::Norm => values[..config.width].fill(1.0),
                Kind::Mesh(depth) => {
                    for layer in 0..depth {
                        for channel in 0..config.width {
                            let offset = layer * 3 * config.width + channel * 3;
                            values[offset] = 0.7 + 0.2 * (random.unit() - 0.5);
                            values[offset + 1] = 0.4 * (random.unit() - 0.5);
                            // Smaller output weights stabilize the residual branches.
                            if layer + 1 == depth {
                                values[offset] *= 0.3;
                                values[offset + 1] *= 0.3;
                            }
                        }
                    }
                }
                Kind::Bias => {}
            }
        }
        parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_rejects_invalid_and_overflowing_dimensions() {
        let valid = GptConfig::default();
        for config in [
            GptConfig {
                width: usize::MAX,
                ..valid
            },
            GptConfig {
                vocab_size: 1,
                ..valid
            },
            GptConfig {
                context: 0,
                ..valid
            },
            GptConfig { heads: 3, ..valid },
            GptConfig { layers: 5, ..valid },
            GptConfig {
                ffn_mesh_depth: 0,
                ..valid
            },
        ] {
            assert!(config.parameter_count().is_err());
        }
        assert!(
            valid
                .validate_tokens(&[valid.vocab_size as u32], 1)
                .is_err()
        );
        assert!(valid.validate_tokens(&[], 1).is_err());
        assert!(valid.validate_batch(&[0, 1], &[1], 2).is_err());
        let maximum = GptConfig {
            width: 64,
            heads: 8,
            context: 128,
            vocab_size: 256,
            layers: 4,
            attention_mesh_depth: 16,
            ffn_mesh_depth: 16,
        };
        assert!(maximum.parameter_count().unwrap() < 100_000);
    }
}
