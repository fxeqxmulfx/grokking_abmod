use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use crate::{Result, mesh::validate_finite};

use super::{CharacterTokenizer, GptConfig};

/// A versioned weights-only checkpoint. Loading resets the AdamW state.
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
        let mut file = File::create(path)?;
        file.write_all(b"MSHGPT07")?;
        for field in [
            self.config.vocab_size,
            self.config.context,
            self.config.width,
            self.config.heads,
            self.config.layers,
            self.config.attention_mesh_depth,
            self.config.ffn_mesh_depth,
        ] {
            file.write_all(&(field as u32).to_le_bytes())?;
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
        if &magic != b"MSHGPT07" {
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
        let config = GptConfig {
            vocab_size: u32::from_le_bytes(word()?) as usize,
            context: u32::from_le_bytes(word()?) as usize,
            width: u32::from_le_bytes(word()?) as usize,
            heads: u32::from_le_bytes(word()?) as usize,
            layers: u32::from_le_bytes(word()?) as usize,
            attention_mesh_depth: u32::from_le_bytes(word()?) as usize,
            ffn_mesh_depth: u32::from_le_bytes(word()?) as usize,
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt::ReferenceGpt;

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
}
