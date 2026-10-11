use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use crate::{Result, mesh::validate_finite};

use super::{
    CharacterTokenizer, EmbeddingForm, EmbeddingNorm, EmbeddingOrder, GptConfig, MeshConnectivity,
    TokenEmbedding, config::Layout,
};

/// A versioned weights-only checkpoint. Loading resets the AdamW state.
/// The default architecture keeps the MSHGPT07 layout; MSHGPT08 adds the
/// token-embedding and mesh-connectivity ablation fields. MSHGPT09 also stores
/// geometric codes, constraints, and operator switches; older layouts still load.
pub struct Checkpoint {
    pub config: GptConfig,
    pub tokenizer: CharacterTokenizer,
    pub parameters: Vec<f32>,
}

impl Checkpoint {
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        if self.config.parameter_count()? != self.parameters.len()
            || self.config.vocab_size != self.tokenizer.vocab_size()
        {
            return Err("checkpoint model, parameters, and tokenizer do not match".into());
        }
        validate_finite(&self.parameters, "checkpoint parameters")?;
        Layout::new(self.config)?.validate_constraints(self.config, &self.parameters)?;
        let ablation = [
            usize::from(self.config.embedding == TokenEmbedding::Table),
            usize::from(self.config.connectivity == MeshConnectivity::Butterfly),
        ];
        let extended = self.config.geometry != super::EmbeddingGeometry::default()
            || !self.config.rope
            || !self.config.qk_norm
            || !self.config.xsa;
        let default = ablation == [0, 0] && !extended;
        let mut file = File::create(path)?;
        file.write_all(if extended {
            b"MSHGPT09"
        } else if default {
            b"MSHGPT07"
        } else {
            b"MSHGPT08"
        })?;
        let extra: &[usize] = if default { &[] } else { &ablation };
        for &field in [
            self.config.vocab_size,
            self.config.context,
            self.config.width,
            self.config.heads,
            self.config.layers,
            self.config.attention_mesh_depth,
            self.config.ffn_mesh_depth,
        ]
        .iter()
        .chain(extra)
        {
            file.write_all(&(field as u32).to_le_bytes())?;
        }
        if extended {
            let g = self.config.geometry;
            for field in [
                g.form as u32,
                g.order as u32,
                g.frequency as u32,
                g.harmonics as u32,
                g.overlap.to_bits(),
                g.normalization as u32,
                u32::from(g.frozen),
                g.preserve_features as u32,
                g.modulus as u32,
                g.seed as u32,
                (g.seed >> 32) as u32,
                u32::from(self.config.rope),
                u32::from(self.config.qk_norm),
                u32::from(self.config.xsa),
            ] {
                file.write_all(&field.to_le_bytes())?;
            }
        }
        for &character in &self.tokenizer.characters {
            file.write_all(&(character as u32).to_le_bytes())?;
        }
        for &value in &self.parameters {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::read(&mut File::open(path)?)
    }

    fn read(reader: &mut impl Read) -> Result<Self> {
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"MSHGPT07" && &magic != b"MSHGPT08" && &magic != b"MSHGPT09" {
            return Err(match &magic {
                b"MSHGPT01" | b"MSHGPT02" => "checkpoint uses learned position embeddings; retrain Mesh GPT with RoPE",
                b"MSHGPT03" => "checkpoint uses LayerNorm; retrain Mesh GPT with RMSNorm",
                b"MSHGPT04" => "checkpoint contains biases; retrain bias-free Mesh GPT",
                b"MSHGPT05" => "checkpoint uses learned RMSNorm without QK-Norm; retrain Mesh GPT with parameter-free QK-Norm",
                b"MSHGPT06" => "checkpoint uses a learned token table; retrain Mesh GPT with mesh token embeddings",
                _ => "unsupported Mesh GPT checkpoint format",
            }.into());
        }
        let mut word = || -> Result<[u8; 4]> {
            let mut bytes = [0; 4];
            reader.read_exact(&mut bytes)?;
            Ok(bytes)
        };
        let mut config = GptConfig {
            vocab_size: u32::from_le_bytes(word()?) as usize,
            context: u32::from_le_bytes(word()?) as usize,
            width: u32::from_le_bytes(word()?) as usize,
            heads: u32::from_le_bytes(word()?) as usize,
            layers: u32::from_le_bytes(word()?) as usize,
            attention_mesh_depth: u32::from_le_bytes(word()?) as usize,
            ffn_mesh_depth: u32::from_le_bytes(word()?) as usize,
            ..GptConfig::default()
        };
        if &magic == b"MSHGPT08" || &magic == b"MSHGPT09" {
            config.embedding = match u32::from_le_bytes(word()?) {
                0 => TokenEmbedding::Mesh,
                1 => TokenEmbedding::Table,
                _ => return Err("unknown checkpoint token embedding".into()),
            };
            config.connectivity = match u32::from_le_bytes(word()?) {
                0 => MeshConnectivity::Neighbor,
                1 => MeshConnectivity::Butterfly,
                _ => return Err("unknown checkpoint mesh connectivity".into()),
            };
            if &magic == b"MSHGPT08"
                && config.embedding == TokenEmbedding::Mesh
                && config.connectivity == MeshConnectivity::Neighbor
            {
                return Err("the default architecture must use the MSHGPT07 layout".into());
            }
        }
        if &magic == b"MSHGPT09" {
            config.geometry.form = EmbeddingForm::from_code(u32::from_le_bytes(word()?))?;
            config.geometry.order = EmbeddingOrder::from_code(u32::from_le_bytes(word()?))?;
            config.geometry.frequency = u32::from_le_bytes(word()?) as usize;
            config.geometry.harmonics = u32::from_le_bytes(word()?) as usize;
            config.geometry.overlap = f32::from_bits(u32::from_le_bytes(word()?));
            config.geometry.normalization = EmbeddingNorm::from_code(u32::from_le_bytes(word()?))?;
            config.geometry.frozen = read_switch(u32::from_le_bytes(word()?))?;
            config.geometry.preserve_features = u32::from_le_bytes(word()?) as usize;
            config.geometry.modulus = u32::from_le_bytes(word()?) as usize;
            let low = u64::from(u32::from_le_bytes(word()?));
            config.geometry.seed = low | (u64::from(u32::from_le_bytes(word()?)) << 32);
            config.rope = read_switch(u32::from_le_bytes(word()?))?;
            config.qk_norm = read_switch(u32::from_le_bytes(word()?))?;
            config.xsa = read_switch(u32::from_le_bytes(word()?))?;
            if config.geometry == super::EmbeddingGeometry::default()
                && config.rope
                && config.qk_norm
                && config.xsa
            {
                return Err("default geometry must use an older checkpoint layout".into());
            }
        }
        let count = config.parameter_count()?;
        let mut characters = Vec::with_capacity(config.vocab_size);
        for _ in 0..config.vocab_size {
            characters.push(
                char::from_u32(u32::from_le_bytes(word()?))
                    .ok_or("invalid checkpoint character")?,
            );
        }
        let tokenizer = CharacterTokenizer::from_characters(characters)?;
        let mut parameters = Vec::with_capacity(count);
        for _ in 0..count {
            parameters.push(f32::from_le_bytes(word()?));
        }
        validate_finite(&parameters, "checkpoint parameters")?;
        Layout::new(config)?.validate_constraints(config, &parameters)?;
        if reader.read(&mut [0])? != 0 {
            return Err("unexpected trailing checkpoint bytes".into());
        }
        Ok(Self {
            config,
            tokenizer,
            parameters,
        })
    }
}

fn read_switch(code: u32) -> Result<bool> {
    match code {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err("invalid checkpoint switch".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt::ReferenceGpt;

    #[test]
    fn geometric_checkpoints_preserve_codes_constraints_and_operator_switches() {
        let tokenizer = CharacterTokenizer::from_text("abcdefg").unwrap();
        for form in [
            EmbeddingForm::Circle,
            EmbeddingForm::Fourier,
            EmbeddingForm::Simplex,
            EmbeddingForm::Sphere,
            EmbeddingForm::Parity,
        ] {
            let config = GptConfig {
                vocab_size: 7,
                width: 8,
                heads: 2,
                embedding: TokenEmbedding::Table,
                rope: false,
                qk_norm: false,
                xsa: false,
                geometry: super::super::EmbeddingGeometry {
                    form,
                    seed: u64::MAX,
                    preserve_features: 2,
                    frozen: true,
                    ..super::super::EmbeddingGeometry::default()
                },
                ..GptConfig::default()
            };
            let checkpoint = Checkpoint {
                config,
                tokenizer: tokenizer.clone(),
                parameters: config.initial_parameters(7).unwrap(),
            };
            let path = std::env::temp_dir().join(format!(
                "geometric-checkpoint-{}-{form}.bin",
                std::process::id()
            ));
            checkpoint.save(&path).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            assert_eq!(&bytes[..8], b"MSHGPT09");
            let loaded = Checkpoint::read(&mut bytes.as_slice()).unwrap();
            assert_eq!(loaded.config, config);
            assert_eq!(loaded.parameters, checkpoint.parameters);
            assert_eq!(
                loaded.config.fixed_token_codes().unwrap(),
                config.fixed_token_codes().unwrap()
            );
            for (offset, code) in [(44, 99_u32), (48, 99), (64, 99), (68, 2), (92, 2)] {
                let mut broken = bytes.clone();
                broken[offset..offset + 4].copy_from_slice(&code.to_le_bytes());
                assert!(Checkpoint::read(&mut broken.as_slice()).is_err());
            }
            let mut broken = checkpoint.parameters.clone();
            let layout = Layout::new(config).unwrap();
            let id = layout.blocks[0].projection;
            let range = layout.constrained_range(config, id);
            broken[layout.groups[id].offset + range.start] = 1.0;
            assert!(
                Checkpoint {
                    parameters: broken,
                    ..checkpoint
                }
                .save(&path)
                .is_err()
            );
        }
    }

    #[test]
    fn checkpoint_rejects_corruption_and_round_trips_exact_weights() {
        let tokenizer = CharacterTokenizer::from_text("abcd ").unwrap();
        let config = GptConfig {
            vocab_size: tokenizer.vocab_size(),
            ..GptConfig::default()
        };
        let parameters = ReferenceGpt::new(config, 7).unwrap().parameters().to_vec();
        let path =
            std::env::temp_dir().join(format!("mesh-gpt-checkpoint-{}.bin", std::process::id()));
        let checkpoint = Checkpoint {
            config,
            tokenizer,
            parameters,
        };
        checkpoint.save(&path).unwrap();
        let loaded = Checkpoint::load(&path).unwrap();
        assert_eq!(loaded.config, checkpoint.config);
        assert_eq!(loaded.tokenizer, checkpoint.tokenizer);
        assert_eq!(loaded.parameters, checkpoint.parameters);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], b"MSHGPT07");
        std::fs::remove_file(path).unwrap();
        for corrupted in [
            vec![0; 60],
            bytes[..bytes.len() - 1].to_vec(),
            [bytes.clone(), vec![0]].concat(),
        ] {
            assert!(Checkpoint::read(&mut corrupted.as_slice()).is_err());
        }
        for (version, message) in [
            (b"MSHGPT01", "retrain Mesh GPT with RoPE"),
            (b"MSHGPT02", "retrain Mesh GPT with RoPE"),
            (b"MSHGPT03", "retrain Mesh GPT with RMSNorm"),
            (b"MSHGPT04", "retrain bias-free Mesh GPT"),
            (b"MSHGPT05", "parameter-free QK-Norm"),
            (b"MSHGPT06", "mesh token embeddings"),
        ] {
            let mut legacy = bytes.clone();
            legacy[..8].copy_from_slice(version);
            let error = Checkpoint::read(&mut legacy.as_slice()).err().unwrap();
            assert!(error.to_string().contains(message));
        }
        let mut nonfinite = bytes.clone();
        let last = nonfinite.len() - 4;
        nonfinite[last..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(Checkpoint::read(&mut nonfinite.as_slice()).is_err());
        let mut invalid = bytes;
        invalid[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Checkpoint::read(&mut invalid.as_slice()).is_err());
    }

    #[test]
    fn ablation_checkpoints_round_trip_their_architecture() {
        let tokenizer = CharacterTokenizer::from_text("abcde").unwrap();
        for (embedding, connectivity, codes) in [
            (TokenEmbedding::Table, MeshConnectivity::Neighbor, [1, 0]),
            (TokenEmbedding::Mesh, MeshConnectivity::Butterfly, [0, 1]),
            (TokenEmbedding::Table, MeshConnectivity::Butterfly, [1, 1]),
        ] {
            let config = GptConfig {
                vocab_size: tokenizer.vocab_size(),
                embedding,
                connectivity,
                ..GptConfig::default()
            };
            let parameters = ReferenceGpt::new(config, 3).unwrap().parameters().to_vec();
            let path = std::env::temp_dir().join(format!(
                "mesh-gpt-ablation-{}-{}{}.bin",
                std::process::id(),
                codes[0],
                codes[1]
            ));
            let checkpoint = Checkpoint {
                config,
                tokenizer: tokenizer.clone(),
                parameters,
            };
            checkpoint.save(&path).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            assert_eq!(&bytes[..8], b"MSHGPT08");
            for (field, code) in codes.into_iter().enumerate() {
                let start = 36 + 4 * field;
                assert_eq!(bytes[start..start + 4], (code as u32).to_le_bytes());
            }
            let loaded = Checkpoint::read(&mut bytes.as_slice()).unwrap();
            assert_eq!(loaded.config, config);
            assert_eq!(loaded.parameters, checkpoint.parameters);
            // Older readers would misplace these fields, and unknown codes must not load.
            let mut legacy = bytes.clone();
            legacy[..8].copy_from_slice(b"MSHGPT07");
            assert!(Checkpoint::read(&mut legacy.as_slice()).is_err());
            for field in [36, 40] {
                let mut unknown = bytes.clone();
                unknown[field..field + 4].copy_from_slice(&2_u32.to_le_bytes());
                assert!(Checkpoint::read(&mut unknown.as_slice()).is_err());
            }
            let mut default = bytes;
            default[36..44].fill(0);
            assert!(Checkpoint::read(&mut default.as_slice()).is_err());
        }
    }
}
