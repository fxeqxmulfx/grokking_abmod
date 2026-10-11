//! Geometric token codes and constraints, paired with the theorems in `lean/`.
//! Mesh codes specify its inputs; a table initialized from them has their geometry.

use std::{fmt, str::FromStr};

use crate::{Result, mesh::Random};

use super::{GptConfig, grokking::residue_permutation};

macro_rules! named_enum {
    ($name:ident, $default:ident, {$($variant:ident => $text:literal),+ $(,)?}) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u32)]
        pub enum $name { $($variant),+ }
        impl Default for $name { fn default() -> Self { Self::$default } }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(match self { $(Self::$variant => $text),+ })
            }
        }
        impl FromStr for $name {
            type Err = String;
            fn from_str(text: &str) -> std::result::Result<Self, Self::Err> {
                match text { $($text => Ok(Self::$variant),)+
                    _ => Err(format!("unknown {}: {text}", stringify!($name))) }
            }
        }
        impl $name {
            pub(crate) fn from_code(code: u32) -> Result<Self> {
                $(if code == Self::$variant as u32 { return Ok(Self::$variant); })+
                Err(format!("unknown checkpoint {}: {code}", stringify!($name)).into())
            }
        }
    };
}

named_enum!(EmbeddingForm, Parity, {
    Parity => "parity", Circle => "circle", Fourier => "fourier",
    Orthogonal => "orthogonal", Simplex => "simplex", Equiangular => "equiangular",
    Antipodal => "antipodal", Hypercube => "hypercube",
    SignedPermutations => "signed-permutations", Sphere => "sphere",
});
named_enum!(EmbeddingOrder, Token, {
    Token => "token", Residue => "residue", DiscreteLog => "discrete-log",
});
named_enum!(EmbeddingNorm, None, { None => "none", Unit => "unit", Rms => "rms" });

/// Runtime ablations. Defaults retain the original signed parity codes and training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbeddingGeometry {
    pub form: EmbeddingForm,
    pub order: EmbeddingOrder,
    pub frequency: usize,
    pub harmonics: usize,
    pub overlap: f32,
    pub normalization: EmbeddingNorm,
    pub frozen: bool,
    /// Copy these code coordinates into effective embeddings and preserve them
    /// through every residual block by constraining both final output projections.
    pub preserve_features: usize,
    /// Arithmetic ordering requires the grok dataset's prime and permutation seed.
    pub modulus: usize,
    pub seed: u64,
}

impl Default for EmbeddingGeometry {
    fn default() -> Self {
        Self {
            form: EmbeddingForm::Parity,
            order: EmbeddingOrder::Token,
            frequency: 1,
            harmonics: 1,
            overlap: 0.0,
            normalization: EmbeddingNorm::None,
            frozen: false,
            preserve_features: 0,
            modulus: 0,
            seed: 0,
        }
    }
}

impl EmbeddingGeometry {
    pub(crate) fn validate(self, config: GptConfig) -> Result<()> {
        let (v, w) = (config.vocab_size, config.width);
        if self.frequency == 0
            || self.frequency > u32::MAX as usize
            || self.harmonics == 0
            || self.harmonics > w / 2
        {
            return Err("embedding frequency must be positive and harmonics in 1..width/2".into());
        }
        if !self.overlap.is_finite() || self.preserve_features > w {
            return Err(
                "embedding overlap must be finite; preserved features must fit width".into(),
            );
        }
        if self.modulus > u32::MAX as usize {
            return Err("embedding modulus exceeds checkpoint field capacity".into());
        }
        if self.preserve_features > 0 && self.normalization != EmbeddingNorm::None {
            return Err("preserved coordinates require --embedding-norm none; full-vector normalization would rescale the protected feature".into());
        }
        let fits = match self.form {
            EmbeddingForm::Orthogonal => w >= v,
            EmbeddingForm::Simplex => w >= v - 1,
            EmbeddingForm::Equiangular => {
                let lower = -1.0 / (v - 1) as f32;
                self.overlap >= lower
                    && self.overlap <= 1.0
                    && (self.overlap == 1.0 || w >= v - usize::from(self.overlap == lower))
            }
            EmbeddingForm::Hypercube => w >= (usize::BITS - (v - 1).leading_zeros()) as usize,
            EmbeddingForm::SignedPermutations => w >= 4 && v <= 256,
            _ => true,
        };
        if !fits {
            return Err(format!(
                "{} embedding does not fit vocabulary {v}, width {w}, overlap {}",
                self.form, self.overlap
            )
            .into());
        }
        if self.order != EmbeddingOrder::Token {
            if !(2..=254).contains(&self.modulus)
                || (2..self.modulus).any(|d| self.modulus.is_multiple_of(d))
                || v != self.modulus + 2
            {
                return Err(
                    "arithmetic embedding order requires a grok prime and vocabulary modulus+2"
                        .into(),
                );
            }
            // Reserve independent directions for zero (discrete-log), operation, and '='.
            let special = 2 + usize::from(self.order == EmbeddingOrder::DiscreteLog);
            let pairs = if self.form == EmbeddingForm::Fourier {
                self.harmonics
            } else {
                1
            };
            if matches!(self.form, EmbeddingForm::Circle | EmbeddingForm::Fourier)
                && w < 2 * pairs + special
            {
                return Err(
                    "arithmetic circle/Fourier codes need extra coordinates for special symbols"
                        .into(),
                );
            }
        }
        Ok(())
    }

    pub(crate) fn transformed(self) -> bool {
        self.normalization != EmbeddingNorm::None || self.preserve_features > 0
    }

    pub(crate) fn radius(self, width: usize) -> f32 {
        match self.normalization {
            EmbeddingNorm::Rms => (width as f32).sqrt(),
            _ => 1.0,
        }
    }
}

/// A deterministic primitive root of a validated prime, including p=2.
pub(crate) fn primitive_root(prime: usize) -> usize {
    if prime == 2 {
        return 1;
    }
    (2..prime)
        .find(|&g| {
            let mut seen = vec![false; prime];
            let mut x = 1;
            for _ in 0..prime - 1 {
                if seen[x] {
                    return false;
                }
                seen[x] = true;
                x = x * g % prime;
            }
            x == 1
        })
        .expect("a prime has a primitive root")
}

fn ordered_tokens(config: GptConfig) -> (Vec<usize>, usize) {
    let g = config.geometry;
    if g.order == EmbeddingOrder::Token {
        return ((0..config.vocab_size).collect(), config.vocab_size);
    }
    let ids = residue_permutation(g.modulus, g.seed);
    let mut order = if g.order == EmbeddingOrder::Residue {
        ids.iter().map(|&id| id as usize).collect::<Vec<_>>()
    } else {
        let root = primitive_root(g.modulus);
        let mut residue = 1;
        (0..g.modulus - 1)
            .map(|_| {
                let token = ids[residue] as usize;
                residue = residue * root % g.modulus;
                token
            })
            .collect::<Vec<_>>()
    };
    let cycle = order.len();
    if g.order == EmbeddingOrder::DiscreteLog {
        order.push(ids[0] as usize);
    }
    order.extend([g.modulus, g.modulus + 1]);
    (order, cycle)
}

pub(crate) fn reorder_parity(config: GptConfig, codes: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    if config.geometry.order == EmbeddingOrder::Token {
        return codes;
    }
    let (order, _) = ordered_tokens(config);
    let mut result = vec![Vec::new(); config.vocab_size];
    for (token, code) in order.into_iter().zip(codes) {
        result[token] = code;
    }
    result
}

/// Helmert coordinates of a regular unit simplex in R^(n-1).
fn simplex_coordinate(row: usize, column: usize, n: usize) -> f64 {
    let scale = (n as f64 / (n - 1) as f64).sqrt();
    let denominator = (((column + 1) * (column + 2)) as f64).sqrt();
    if row <= column {
        scale / denominator
    } else if row == column + 1 {
        -scale * (column + 1) as f64 / denominator
    } else {
        0.0
    }
}

pub(crate) fn geometric_codes(config: GptConfig) -> Vec<Vec<f32>> {
    let (v, w, g) = (config.vocab_size, config.width, config.geometry);
    let (order, cycle) = ordered_tokens(config);
    let mut codes = vec![vec![0.0; w]; v];
    let mut random = Random::new(g.seed ^ 0x5350_4845_5245_3031);
    for (row, &token) in order.iter().enumerate() {
        let code = &mut codes[token];
        match g.form {
            EmbeddingForm::Parity => unreachable!("parity codes use the original generator"),
            EmbeddingForm::Circle | EmbeddingForm::Fourier => {
                let pairs = if g.form == EmbeddingForm::Fourier {
                    g.harmonics
                } else {
                    1
                };
                if row >= cycle {
                    code[2 * pairs + row - cycle] = 1.0;
                } else {
                    let scale = 1.0 / (pairs as f64).sqrt();
                    for pair in 0..pairs {
                        // Reduce first so large user frequencies cannot overflow or lose angle precision.
                        let frequency = (g.frequency % cycle + pair % cycle) % cycle;
                        let angle =
                            std::f64::consts::TAU * frequency as f64 * row as f64 / cycle as f64;
                        let (s, c) = angle.sin_cos();
                        code[2 * pair] = (scale * c) as f32;
                        code[2 * pair + 1] = (scale * s) as f32;
                    }
                }
            }
            EmbeddingForm::Orthogonal => code[row] = 1.0,
            EmbeddingForm::Simplex | EmbeddingForm::Equiangular => {
                let rho = if g.form == EmbeddingForm::Simplex {
                    -1.0 / (v - 1) as f64
                } else {
                    f64::from(g.overlap)
                };
                if rho == 1.0 {
                    code[0] = 1.0;
                    continue;
                }
                let a = ((1.0 - rho) * (v - 1) as f64 / v as f64).sqrt();
                for (c, value) in code[..v - 1].iter_mut().enumerate() {
                    *value = (a * simplex_coordinate(row, c, v)) as f32;
                }
                if w >= v {
                    code[v - 1] = ((1.0 + (v - 1) as f64 * rho).max(0.0) / v as f64).sqrt() as f32;
                }
            }
            EmbeddingForm::Antipodal => code[0] = if row % 2 == 0 { 1.0 } else { -1.0 },
            EmbeddingForm::Hypercube => {
                let bits = (usize::BITS - (v - 1).leading_zeros()) as usize;
                for (bit, value) in code[..bits].iter_mut().enumerate() {
                    *value = if row & (1 << bit) == 0 { -1.0 } else { 1.0 } / (bits as f32).sqrt();
                }
            }
            EmbeddingForm::SignedPermutations => {
                // Exact indexing of Transformer.GPTMini.Sparsemax.pairedRecallCode.
                let p = row / 16;
                for (c, value) in code[..4].iter_mut().enumerate() {
                    let slot = if c < 2 {
                        (c + p / 2 % 2) % 2
                    } else {
                        2 + (c - 2 + p % 2) % 2
                    };
                    let magnitude = (1 + (slot + p / 4) % 4) as f32;
                    *value = magnitude * if row & (1 << c) == 0 { 1.0 } else { -1.0 };
                }
            }
            EmbeddingForm::Sphere => {
                let mut squares = 0.0;
                for value in code.iter_mut() {
                    let radius =
                        (-2.0 * f64::from(random.unit().max(1.0 / 16_777_216.0)).ln()).sqrt();
                    let angle = std::f64::consts::TAU * f64::from(random.unit());
                    *value = (radius * angle.cos()) as f32;
                    squares += f64::from(*value).powi(2);
                }
                for value in code {
                    *value = (f64::from(*value) / squares.sqrt()) as f32;
                }
            }
        }
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt::{EquationDataset, ModularOperation, ReferenceGpt, TokenEmbedding};

    fn table(form: EmbeddingForm, overlap: f32) -> GptConfig {
        GptConfig {
            vocab_size: 7,
            width: 8,
            heads: 2,
            layers: 2,
            attention_mesh_depth: 2,
            ffn_mesh_depth: 2,
            embedding: TokenEmbedding::Table,
            geometry: EmbeddingGeometry {
                form,
                overlap,
                ..EmbeddingGeometry::default()
            },
            ..GptConfig::default()
        }
    }

    fn dot(x: &[f32], y: &[f32]) -> f64 {
        x.iter()
            .zip(y)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum()
    }

    #[test]
    fn initialized_tables_realize_the_expected_grams() {
        for (form, rho) in [
            (EmbeddingForm::Orthogonal, 0.0),
            (EmbeddingForm::Simplex, -1.0 / 6.0),
            (EmbeddingForm::Equiangular, 0.25),
            (EmbeddingForm::Equiangular, -1.0 / 6.0),
            (EmbeddingForm::Equiangular, 1.0),
        ] {
            let c = table(form, rho);
            let vectors = ReferenceGpt::new(c, 7).unwrap().token_embeddings().unwrap();
            for (i, x) in vectors.iter().enumerate() {
                for (j, y) in vectors.iter().enumerate() {
                    let expected = if i == j { 1.0 } else { f64::from(rho) };
                    assert!((dot(x, y) - expected).abs() < 3e-7, "{form} ({i},{j})");
                }
            }
        }
        let simplex = GptConfig {
            width: 6,
            heads: 1,
            ..table(EmbeddingForm::Simplex, 0.0)
        };
        assert!(simplex.validate().is_ok());
        assert!(
            GptConfig {
                geometry: EmbeddingGeometry {
                    form: EmbeddingForm::Equiangular,
                    overlap: 0.0,
                    ..EmbeddingGeometry::default()
                },
                ..simplex
            }
            .validate()
            .is_err()
        );
        assert!(
            GptConfig {
                geometry: EmbeddingGeometry {
                    form: EmbeddingForm::Orthogonal,
                    ..EmbeddingGeometry::default()
                },
                ..simplex
            }
            .validate()
            .is_err()
        );
        for rho in [-0.2, 1.1, f32::NAN] {
            assert!(table(EmbeddingForm::Equiangular, rho).validate().is_err());
        }
    }

    #[test]
    fn circle_and_fourier_grams_follow_cosine_identities_in_the_declared_order() {
        for order in [
            EmbeddingOrder::Token,
            EmbeddingOrder::Residue,
            EmbeddingOrder::DiscreteLog,
        ] {
            for form in [EmbeddingForm::Circle, EmbeddingForm::Fourier] {
                let c = GptConfig {
                    width: 12,
                    heads: 2,
                    geometry: EmbeddingGeometry {
                        form,
                        order,
                        harmonics: 3,
                        modulus: if order == EmbeddingOrder::Token { 0 } else { 5 },
                        seed: 17,
                        ..EmbeddingGeometry::default()
                    },
                    ..table(form, 0.0)
                };
                let vectors = c.fixed_token_codes().unwrap();
                let (ids, cycle) = ordered_tokens(c);
                let pairs = if form == EmbeddingForm::Fourier { 3 } else { 1 };
                for i in 0..cycle {
                    for j in 0..cycle {
                        let expected = (0..pairs)
                            .map(|k| {
                                (std::f64::consts::TAU * (k + 1) as f64 * (i as f64 - j as f64)
                                    / cycle as f64)
                                    .cos()
                            })
                            .sum::<f64>()
                            / pairs as f64;
                        assert!((dot(&vectors[ids[i]], &vectors[ids[j]]) - expected).abs() < 2e-7);
                    }
                }
                for &id in &ids[cycle..] {
                    assert_eq!(dot(&vectors[id], &vectors[id]), 1.0);
                    assert_eq!(dot(&vectors[id], &vectors[ids[0]]), 0.0);
                }
                if order != EmbeddingOrder::Token {
                    let dataset = EquationDataset::new(5, ModularOperation::Add, 0.5, 17).unwrap();
                    let first = if order == EmbeddingOrder::Residue {
                        0
                    } else {
                        1
                    };
                    assert_eq!(ids[0], dataset.residue_tokens()[first] as usize);
                }
            }
        }
    }

    #[test]
    fn integer_signed_codes_are_distinct_and_have_squared_norm_thirty() {
        let c = GptConfig {
            vocab_size: 256,
            ..table(EmbeddingForm::SignedPermutations, 0.0)
        };
        let codes = c.fixed_token_codes().unwrap();
        assert_eq!(codes[0][..4], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(codes[1][..4], [-1.0, 2.0, 3.0, 4.0]);
        let mut seen = std::collections::HashSet::new();
        for code in codes {
            assert_eq!(dot(&code, &code), 30.0);
            assert!(seen.insert(code.iter().map(|x| x.to_bits()).collect::<Vec<_>>()));
        }
    }

    #[test]
    fn protected_coordinates_survive_arbitrary_embedding_weights_and_residual_depth() {
        for embedding in [TokenEmbedding::Mesh, TokenEmbedding::Table] {
            let c = GptConfig {
                embedding,
                geometry: EmbeddingGeometry {
                    preserve_features: 3,
                    ..EmbeddingGeometry::default()
                },
                ..table(EmbeddingForm::Parity, 0.0)
            };
            let mut parameters = c.initial_parameters(7).unwrap();
            parameters[..c.embedding_parameter_count()].fill(0.0);
            let model = ReferenceGpt::from_parameters(c, parameters.clone()).unwrap();
            let codes = c.fixed_token_codes().unwrap();
            let input = [0, 1, 2, 3, 4, 5, 6, 0];
            for (state, &token) in model.hidden_states(&input, 4).unwrap().iter().zip(&input) {
                assert_eq!(state[..3], codes[token as usize][..3]);
            }
            let (_, gradient) = model.loss_and_gradient(&input, &[1; 8], 4).unwrap();
            let layout = super::super::config::Layout::new(c).unwrap();
            for id in 0..layout.groups.len() {
                let range = layout.constrained_range(c, id);
                let offset = layout.groups[id].offset;
                assert!(
                    gradient[offset + range.start..offset + range.end]
                        .iter()
                        .all(|&g| g == 0.0)
                );
            }
            let range = layout.constrained_range(c, layout.blocks[0].projection);
            parameters[layout.groups[layout.blocks[0].projection].offset + range.start] = 1.0;
            assert!(ReferenceGpt::from_parameters(c, parameters).is_err());
        }
    }

    #[test]
    fn normalized_embeddings_and_their_autodiff_match_finite_differences() {
        for embedding in [TokenEmbedding::Mesh, TokenEmbedding::Table] {
            for normalization in [EmbeddingNorm::Unit, EmbeddingNorm::Rms] {
                let c = GptConfig {
                    embedding,
                    geometry: EmbeddingGeometry {
                        normalization,
                        ..EmbeddingGeometry::default()
                    },
                    ..table(EmbeddingForm::Parity, 0.0)
                };
                let model = ReferenceGpt::new(c, 7).unwrap();
                for vector in model.token_embeddings().unwrap() {
                    assert!(
                        (dot(&vector, &vector) - f64::from(c.geometry.radius(c.width)).powi(2))
                            .abs()
                            < 1e-6
                    );
                }
                let input = [0, 1, 2, 3];
                let (loss, gradient) = model.loss_and_gradient(&input, &[1; 4], 4).unwrap();
                assert!(loss.is_finite());
                for index in [
                    0,
                    1,
                    c.embedding_parameter_count() - 1,
                    c.embedding_parameter_count() + 3,
                ] {
                    let base = model.parameters()[index];
                    let mut parameters = model.parameters().to_vec();
                    parameters[index] = base + 1e-4;
                    let hi = parameters[index];
                    let plus = ReferenceGpt::from_parameters(c, parameters.clone())
                        .unwrap()
                        .loss_and_gradient(&input, &[1; 4], 4)
                        .unwrap()
                        .0;
                    parameters[index] = base - 1e-4;
                    let lo = parameters[index];
                    let minus = ReferenceGpt::from_parameters(c, parameters)
                        .unwrap()
                        .loss_and_gradient(&input, &[1; 4], 4)
                        .unwrap()
                        .0;
                    let numeric = (plus - minus) / f64::from(hi - lo);
                    assert!((numeric - gradient[index]).abs() < 5e-5 * (1.0 + numeric.abs()));
                }
            }
        }
    }

    #[test]
    fn all_forms_support_both_parameterizations_and_operator_switches() {
        for form in [
            EmbeddingForm::Parity,
            EmbeddingForm::Circle,
            EmbeddingForm::Fourier,
            EmbeddingForm::Orthogonal,
            EmbeddingForm::Simplex,
            EmbeddingForm::Equiangular,
            EmbeddingForm::Antipodal,
            EmbeddingForm::Hypercube,
            EmbeddingForm::SignedPermutations,
            EmbeddingForm::Sphere,
        ] {
            assert_eq!(form.to_string().parse(), Ok(form));
            for embedding in [TokenEmbedding::Mesh, TokenEmbedding::Table] {
                let c = GptConfig {
                    embedding,
                    rope: false,
                    qk_norm: false,
                    xsa: false,
                    ..table(form, 0.25)
                };
                let model = ReferenceGpt::new(c, 7).unwrap();
                assert!(
                    model
                        .loss_and_gradient(&[0, 1, 2, 3], &[1; 4], 4)
                        .unwrap()
                        .0
                        .is_finite()
                );
            }
        }
        for form in [EmbeddingForm::Hypercube, EmbeddingForm::Sphere] {
            let c = table(form, 0.0);
            for code in c.fixed_token_codes().unwrap() {
                assert!((dot(&code, &code) - 1.0).abs() < 2e-7);
            }
        }
        let c = GptConfig {
            geometry: EmbeddingGeometry {
                frozen: true,
                ..EmbeddingGeometry::default()
            },
            ..table(EmbeddingForm::Circle, 0.0)
        };
        let model = ReferenceGpt::new(c, 7).unwrap();
        let (_, gradient) = model.loss_and_gradient(&[0, 1, 2, 3], &[1; 4], 4).unwrap();
        assert!(
            gradient[..c.embedding_parameter_count()]
                .iter()
                .all(|&x| x == 0.0)
        );
        assert!(
            gradient[c.embedding_parameter_count()..]
                .iter()
                .any(|&x| x != 0.0)
        );
    }
}
