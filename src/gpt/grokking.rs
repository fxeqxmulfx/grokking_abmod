//! Abstract-symbol modular operation tables for the grokking experiments.

use std::{fmt, str::FromStr};

use crate::{Result, mesh::Random};

use super::{CharacterTokenizer, IGNORE_TARGET, MAX_ROWS, MAX_VOCAB};

pub const EQUATION_CONTEXT: usize = 4;

/// The modular arithmetic operations in Appendix A.1 of arXiv:2201.02177v1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModularOperation {
    Add,
    Subtract,
    Divide,
    Mixed,
    SquareSum,
    Quadratic,
    QuadraticLinear,
    CubicProduct,
    CubicMixed,
}

impl ModularOperation {
    pub fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "sub",
            Self::Divide => "div",
            Self::Mixed => "mixed",
            Self::SquareSum => "square-sum",
            Self::Quadratic => "quadratic",
            Self::QuadraticLinear => "quadratic-linear",
            Self::CubicProduct => "cubic-product",
            Self::CubicMixed => "cubic-mixed",
        }
    }

    pub fn symbol(self) -> char {
        match self {
            Self::Add => '+',
            Self::Subtract => '-',
            Self::Divide => '/',
            _ => '\u{2218}',
        }
    }

    fn apply(self, a: usize, b: usize, modulus: usize) -> usize {
        match self {
            Self::Add => (a + b) % modulus,
            Self::Subtract => (a + modulus - b) % modulus,
            Self::Divide => a * inverse(b, modulus) % modulus,
            Self::Mixed if b % 2 == 1 => a * inverse(b, modulus) % modulus,
            Self::Mixed => (a + modulus - b) % modulus,
            Self::SquareSum => (a * a + b * b) % modulus,
            Self::Quadratic => (a * a + a * b + b * b) % modulus,
            Self::QuadraticLinear => (a * a + a * b + b * b + a) % modulus,
            Self::CubicProduct => (a * a * a + a * b) % modulus,
            Self::CubicMixed => (a * a * a + a * b * b + b) % modulus,
        }
    }
}

impl fmt::Display for ModularOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

impl FromStr for ModularOperation {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "add" => Ok(Self::Add),
            "sub" => Ok(Self::Subtract),
            "div" => Ok(Self::Divide),
            "mixed" => Ok(Self::Mixed),
            "square-sum" => Ok(Self::SquareSum),
            "quadratic" => Ok(Self::Quadratic),
            "quadratic-linear" => Ok(Self::QuadraticLinear),
            "cubic-product" => Ok(Self::CubicProduct),
            "cubic-mixed" => Ok(Self::CubicMixed),
            _ => Err(format!("unknown operation: {value}")),
        }
    }
}

fn inverse(value: usize, modulus: usize) -> usize {
    let mut result = 1;
    let mut base = value;
    let mut exponent = modulus - 2;
    while exponent != 0 {
        if exponent % 2 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exponent /= 2;
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Equation {
    pub a: usize,
    pub b: usize,
    pub answer: usize,
}

/// A seeded split of an entire binary operation table, with permuted symbol IDs.
/// Each residue is one abstract token. The model never receives decimal digits.
pub struct EquationDataset {
    modulus: usize,
    operation: ModularOperation,
    training: Vec<Equation>,
    validation: Vec<Equation>,
    residue_tokens: Vec<u32>,
    tokenizer: CharacterTokenizer,
}

impl EquationDataset {
    pub fn new(
        modulus: usize,
        operation: ModularOperation,
        train_fraction: f64,
        seed: u64,
    ) -> Result<Self> {
        if !(2..=MAX_VOCAB - 2).contains(&modulus)
            || (2..modulus).any(|divisor| modulus.is_multiple_of(divisor))
        {
            return Err(
                "modulus must be a prime in 2..254 (two vocabulary slots are reserved)".into(),
            );
        }
        if !train_fraction.is_finite()
            || !(0.0..1.0).contains(&train_fraction)
            || train_fraction == 0.0
        {
            return Err("train fraction must be finite and strictly between 0 and 1".into());
        }
        let residue_tokens = residue_permutation(modulus, seed);
        let mut characters = vec![' '; modulus + 2];
        for (residue, &token) in residue_tokens.iter().enumerate() {
            // Private-use characters give each abstract symbol an exact identity
            // in existing checkpoints without decimal-number tokenization.
            characters[token as usize] =
                char::from_u32(0xe000 + residue as u32).ok_or("invalid abstract symbol")?;
        }
        characters[modulus] = operation.symbol();
        characters[modulus + 1] = '=';
        let tokenizer = CharacterTokenizer::from_characters(characters)?;
        let mut equations = Vec::new();
        let first_b = usize::from(operation == ModularOperation::Divide);
        for a in 0..modulus {
            for b in first_b..modulus {
                equations.push(Equation {
                    a,
                    b,
                    answer: operation.apply(a, b, modulus),
                });
            }
        }
        let split = (equations.len() as f64 * train_fraction).floor() as usize;
        if split < 2 || split >= equations.len() {
            return Err(
                "split needs at least two training equations and one validation equation".into(),
            );
        }
        let mut splits = Random::new(seed ^ 0x73706c6974730002);
        shuffle(&mut equations, &mut splits);
        let validation = equations.split_off(split);
        Ok(Self {
            modulus,
            operation,
            training: equations,
            validation,
            residue_tokens,
            tokenizer,
        })
    }

    pub fn tokenizer(&self) -> &CharacterTokenizer {
        &self.tokenizer
    }

    pub fn modulus(&self) -> usize {
        self.modulus
    }

    pub fn operation(&self) -> ModularOperation {
        self.operation
    }

    pub fn split(&self, validation: bool) -> &[Equation] {
        if validation {
            &self.validation
        } else {
            &self.training
        }
    }

    pub fn residue_tokens(&self) -> &[u32] {
        &self.residue_tokens
    }

    pub fn symbol_label(&self, token: usize) -> Result<String> {
        if token == self.modulus {
            return Ok(self.operation.name().into());
        }
        if token == self.modulus + 1 {
            return Ok("=".into());
        }
        self.residue_tokens
            .iter()
            .position(|&id| id as usize == token)
            .map(|residue| residue.to_string())
            .ok_or_else(|| "invalid symbol ID".into())
    }

    /// Returns a -> op -> b -> '=' as input, with only the answer supervised.
    /// Indices refer exclusively to the requested split.
    pub fn batch(&self, validation: bool, indices: &[usize]) -> Result<(Vec<u32>, Vec<u32>)> {
        if indices.is_empty() || indices.len() > MAX_ROWS / EQUATION_CONTEXT {
            return Err(format!(
                "equation batches need 1..{} examples",
                MAX_ROWS / EQUATION_CONTEXT
            )
            .into());
        }
        let mut inputs = Vec::with_capacity(indices.len() * EQUATION_CONTEXT);
        let mut targets = Vec::with_capacity(inputs.capacity());
        for &index in indices {
            let equation = self
                .split(validation)
                .get(index)
                .ok_or("equation index is outside the split")?;
            inputs.extend_from_slice(&[
                self.residue_tokens[equation.a],
                self.modulus as u32,
                self.residue_tokens[equation.b],
                (self.modulus + 1) as u32,
            ]);
            targets.extend_from_slice(&[
                IGNORE_TARGET,
                IGNORE_TARGET,
                IGNORE_TARGET,
                self.residue_tokens[equation.answer],
            ]);
        }
        Ok((inputs, targets))
    }
}

pub(crate) fn residue_permutation(modulus: usize, seed: u64) -> Vec<u32> {
    let mut tokens = (0..modulus as u32).collect::<Vec<_>>();
    let mut symbols = Random::new(seed ^ 0x73796d626f6c7301);
    shuffle(&mut tokens, &mut symbols);
    tokens
}

fn shuffle<T>(values: &mut [T], random: &mut Random) {
    for end in (1..values.len()).rev() {
        values.swap(end, random.index(end + 1));
    }
}

/// Shuffled epochs without replacement. The final minibatch may be shorter.
pub struct EquationSampler {
    indices: Vec<usize>,
    random: Random,
    position: usize,
}

impl EquationSampler {
    pub fn new(training_examples: usize, seed: u64) -> Result<Self> {
        if training_examples == 0 || training_examples > MAX_VOCAB * MAX_VOCAB {
            return Err("sampler needs 1..65536 training equations".into());
        }
        let mut sampler = Self {
            indices: (0..training_examples).collect(),
            random: Random::new(seed ^ 0x6261746368657303),
            position: 0,
        };
        shuffle(&mut sampler.indices, &mut sampler.random);
        Ok(sampler)
    }

    pub fn sample(&mut self, batch: usize) -> Result<&[usize]> {
        if batch == 0 || batch > self.indices.len() || batch > MAX_ROWS / EQUATION_CONTEXT {
            return Err("invalid equation minibatch size".into());
        }
        if self.position == self.indices.len() {
            shuffle(&mut self.indices, &mut self.random);
            self.position = 0;
        }
        let start = self.position;
        self.position = (start + batch).min(self.indices.len());
        Ok(&self.indices[start..self.position])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn equations_are_disjoint_complete_reproducible_and_abstract() {
        for operation in [
            ModularOperation::Add,
            ModularOperation::Subtract,
            ModularOperation::Divide,
            ModularOperation::Mixed,
            ModularOperation::SquareSum,
            ModularOperation::Quadratic,
            ModularOperation::QuadraticLinear,
            ModularOperation::CubicProduct,
            ModularOperation::CubicMixed,
        ] {
            let dataset = EquationDataset::new(7, operation, 0.5, 7).unwrap();
            let mut pairs = HashSet::new();
            for equation in dataset.training.iter().chain(&dataset.validation) {
                assert!(pairs.insert((equation.a, equation.b)));
                assert!(equation.answer < 7);
                match operation {
                    ModularOperation::Add => {
                        assert_eq!(equation.answer, (equation.a + equation.b) % 7)
                    }
                    ModularOperation::Subtract => {
                        assert_eq!((equation.answer + equation.b) % 7, equation.a)
                    }
                    ModularOperation::Divide => {
                        assert!(equation.b > 0);
                        assert_eq!(equation.answer * equation.b % 7, equation.a);
                    }
                    _ => {}
                }
            }
            assert_eq!(
                pairs.len(),
                if operation == ModularOperation::Divide {
                    42
                } else {
                    49
                }
            );
            let duplicate = EquationDataset::new(7, operation, 0.5, 7).unwrap();
            assert_eq!(dataset.training, duplicate.training);
            assert_eq!(dataset.residue_tokens, duplicate.residue_tokens);
            assert_ne!(dataset.residue_tokens, (0..7).collect::<Vec<u32>>());
            let (inputs, targets) = dataset.batch(false, &[0, 0, 1]).unwrap();
            for (index, (x, y)) in inputs
                .as_chunks::<4>()
                .0
                .iter()
                .zip(targets.as_chunks::<4>().0.iter())
                .enumerate()
            {
                let equation = dataset.training[usize::from(index == 2)];
                assert_eq!(
                    *x,
                    [
                        dataset.residue_tokens[equation.a],
                        7,
                        dataset.residue_tokens[equation.b],
                        8
                    ]
                );
                assert_eq!(&y[..3], &[IGNORE_TARGET; 3]);
                assert_eq!(y[3], dataset.residue_tokens[equation.answer]);
            }
            for (residue, &token) in dataset.residue_tokens.iter().enumerate() {
                assert_eq!(
                    dataset.symbol_label(token as usize).unwrap(),
                    residue.to_string()
                );
                let symbol = dataset.tokenizer.decode(&[token]).unwrap();
                assert_eq!(dataset.tokenizer.encode(&symbol).unwrap(), [token]);
            }
        }
        let dataset = EquationDataset::new(97, ModularOperation::Add, 0.5, 7).unwrap();
        assert_eq!(dataset.training.len(), 4704);
        assert_eq!(dataset.validation.len(), 4705);
        let changed = EquationDataset::new(97, ModularOperation::Add, 0.5, 8).unwrap();
        assert_ne!(dataset.training, changed.training);
        assert_ne!(dataset.residue_tokens, changed.residue_tokens);
    }

    #[test]
    fn sampler_stays_in_training_and_does_not_duplicate_examples() {
        let mut sampler = EquationSampler::new(24, 7).unwrap();
        let mut duplicate = EquationSampler::new(24, 7).unwrap();
        let mut seen = HashSet::new();
        for _ in 0..100 {
            let indices = sampler.sample(12).unwrap();
            assert_eq!(indices, duplicate.sample(12).unwrap());
            assert_eq!(indices.iter().collect::<HashSet<_>>().len(), 12);
            assert!(indices.iter().all(|&i| i < 24));
            seen.extend(indices.iter().copied());
        }
        assert_eq!(seen.len(), 24);
        assert!(sampler.sample(0).is_err());
        assert!(sampler.sample(25).is_err());
        assert!(EquationSampler::new(0, 7).is_err());
        let mut epoch = EquationSampler::new(25, 7).unwrap();
        let mut seen = HashSet::new();
        for expected in [12, 12, 1] {
            let indices = epoch.sample(12).unwrap();
            assert_eq!(indices.len(), expected);
            for &index in indices {
                assert!(seen.insert(index));
            }
        }
        assert_eq!(seen.len(), 25);
        assert_eq!(epoch.sample(12).unwrap().len(), 12);
    }

    #[test]
    fn equations_reject_invalid_primes_splits_and_batches() {
        for modulus in [0, 1, 4, 255, usize::MAX] {
            assert!(EquationDataset::new(modulus, ModularOperation::Add, 0.5, 7).is_err());
        }
        for fraction in [-0.5, 0.0, 1.0, 1.5, f64::NAN, f64::INFINITY, 0.001] {
            assert!(EquationDataset::new(7, ModularOperation::Add, fraction, 7).is_err());
        }
        let dataset = EquationDataset::new(7, ModularOperation::Add, 0.5, 7).unwrap();
        assert!(dataset.batch(false, &[]).is_err());
        assert!(dataset.batch(false, &[usize::MAX]).is_err());
        assert_eq!(
            dataset
                .batch(false, &vec![0; MAX_ROWS / EQUATION_CONTEXT])
                .unwrap()
                .0
                .len(),
            MAX_ROWS
        );
        assert!(
            dataset
                .batch(false, &vec![0; MAX_ROWS / EQUATION_CONTEXT + 1])
                .is_err()
        );
        assert!(dataset.symbol_label(9).is_err());
    }
}
