use std::collections::BTreeSet;

use crate::{Result, mesh::Random};

use super::{GptConfig, MAX_ROWS, MODEL_MAX_VOCAB};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CharacterTokenizer {
    pub(crate) characters: Vec<char>,
}

impl CharacterTokenizer {
    pub fn from_text(text: &str) -> Result<Self> {
        Self::from_characters(text.chars().collect::<BTreeSet<_>>().into_iter().collect())
    }

    pub(crate) fn from_characters(characters: Vec<char>) -> Result<Self> {
        if !(2..=MODEL_MAX_VOCAB).contains(&characters.len())
            || characters.iter().copied().collect::<BTreeSet<_>>().len() != characters.len()
        {
            return Err("tokenizer must contain 2..65536 unique characters".into());
        }
        Ok(Self { characters })
    }

    pub fn vocab_size(&self) -> usize {
        self.characters.len()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        text.chars()
            .map(|character| {
                self.characters
                    .iter()
                    .position(|&c| c == character)
                    .map(|id| id as u32)
                    .ok_or_else(|| {
                        format!("character {character:?} is outside the tokenizer vocabulary")
                            .into()
                    })
            })
            .collect()
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        tokens
            .iter()
            .map(|&token| {
                self.characters
                    .get(token as usize)
                    .copied()
                    .ok_or_else(|| "token ID is outside the tokenizer vocabulary".into())
            })
            .collect()
    }
}

pub struct TextDataset {
    training: Vec<u32>,
    validation: Vec<u32>,
}

impl TextDataset {
    /// A contiguous 90/10 split; windows never cross the split boundary.
    pub fn new(tokens: Vec<u32>, config: GptConfig) -> Result<Self> {
        config.validate()?;
        if tokens.iter().any(|&id| id as usize >= config.vocab_size) {
            return Err("dataset has invalid token IDs".into());
        }
        let split = tokens.len() * 9 / 10;
        if split <= config.context || tokens.len() - split <= config.context {
            return Err("corpus is too short for a 90/10 split at this context length".into());
        }
        Ok(Self {
            training: tokens[..split].to_vec(),
            validation: tokens[split..].to_vec(),
        })
    }

    pub fn batch(
        &self,
        validation: bool,
        sequence: usize,
        batch_size: usize,
        seed: u64,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let corpus = if validation {
            &self.validation
        } else {
            &self.training
        };
        if sequence == 0
            || sequence >= corpus.len()
            || batch_size == 0
            || batch_size > MAX_ROWS / sequence
        {
            return Err("invalid dataset batch dimensions".into());
        }
        let mut random = Random::new(seed);
        let mut inputs = Vec::with_capacity(sequence * batch_size);
        let mut targets = Vec::with_capacity(sequence * batch_size);
        for _ in 0..batch_size {
            let start = (random.unit() * (corpus.len() - sequence) as f32) as usize;
            inputs.extend_from_slice(&corpus[start..start + sequence]);
            targets.extend_from_slice(&corpus[start + 1..start + sequence + 1]);
        }
        Ok((inputs, targets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_round_trip_and_split_boundaries() {
        let text = "abc def\n".repeat(80);
        let tokenizer = CharacterTokenizer::from_text(&text).unwrap();
        assert_eq!(
            tokenizer.decode(&tokenizer.encode(&text).unwrap()).unwrap(),
            text
        );
        assert!(tokenizer.encode("?").is_err());
        assert!(tokenizer.decode(&[999]).is_err());
        let tokens = tokenizer.encode(&text).unwrap();
        let dataset = TextDataset::new(
            tokens,
            GptConfig {
                vocab_size: tokenizer.vocab_size(),
                ..GptConfig::default()
            },
        )
        .unwrap();
        for validation in [false, true] {
            let (inputs, targets) = dataset.batch(validation, 8, 5, 42).unwrap();
            for (x, y) in inputs.chunks(8).zip(targets.chunks(8)) {
                assert_eq!(&x[1..], &y[..7]);
            }
        }
        assert!(dataset.batch(false, 0, 1, 0).is_err());
        assert!(dataset.batch(false, usize::MAX, 1, 0).is_err());
        let isolated = TextDataset::new(
            [vec![0; 450], vec![1; 50]].concat(),
            GptConfig {
                vocab_size: 2,
                ..GptConfig::default()
            },
        )
        .unwrap();
        for validation in [false, true] {
            let (x, y) = isolated.batch(validation, 8, 20, 42).unwrap();
            let expected = u32::from(validation);
            assert!(x.iter().chain(&y).all(|&id| id == expected));
        }
    }
}
