use crate::{Result, mesh::validate_finite};

use super::{
    EmbeddingNorm, GptConfig, IGNORE_TARGET, MeshConnectivity, RMSNORM_EPSILON, ROPE_BASE,
    TokenEmbedding, XSA_EPSILON,
    config::{Layout, token_codes},
};

const NONE: usize = usize::MAX;

/// Independent f64 scalar automatic differentiation, used only for verification.
/// The GPU implementation uses hand-written operator derivatives instead.
#[derive(Clone, Debug)]
pub struct ReferenceGpt {
    config: GptConfig,
    layout: Layout,
    parameters: Vec<f32>,
}

impl ReferenceGpt {
    pub fn new(config: GptConfig, seed: u64) -> Result<Self> {
        let layout = Layout::new(config)?;
        let parameters = layout.initialize(config, seed);
        Ok(Self {
            config,
            layout,
            parameters,
        })
    }

    pub fn from_parameters(config: GptConfig, parameters: Vec<f32>) -> Result<Self> {
        let layout = Layout::new(config)?;
        if parameters.len() != layout.count() {
            return Err("parameter count does not match GPT configuration".into());
        }
        validate_finite(&parameters, "parameters")?;
        layout.validate_constraints(config, &parameters)?;
        Ok(Self {
            config,
            layout,
            parameters,
        })
    }

    pub fn parameters(&self) -> &[f32] {
        &self.parameters
    }

    pub fn logits(&self, tokens: &[u32], sequence: usize) -> Result<Vec<Vec<f32>>> {
        self.config.validate_tokens(tokens, sequence)?;
        let (tape, ids, _) = self.forward(tokens, sequence);
        ids.iter()
            .map(|row| {
                let values: Vec<_> = row.iter().map(|&id| tape.values[id] as f32).collect();
                validate_finite(&values, "logits")?;
                Ok(values)
            })
            .collect()
    }

    pub fn loss_and_gradient(
        &self,
        tokens: &[u32],
        targets: &[u32],
        sequence: usize,
    ) -> Result<(f64, Vec<f64>)> {
        let supervised = self.config.validate_batch(tokens, targets, sequence)?;
        let (mut tape, logits, _) = self.forward(tokens, sequence);
        let mut losses = Vec::new();
        for (row, &target) in logits.iter().zip(targets) {
            if target == IGNORE_TARGET {
                continue;
            }
            let (maximum, sum, _) = tape.shifted_exponentials(row);
            let log = tape.log(sum);
            let normalized = tape.shift(log, maximum);
            losses.push(tape.subtract(normalized, row[target as usize]));
        }
        let sum = tape.sum(&losses);
        let loss_id = tape.scale(sum, 1.0 / supervised as f64);
        let loss = tape.values[loss_id];
        let mut gradients = tape.backward(loss_id, self.parameters.len());
        self.layout.mask_gradient(self.config, &mut gradients);
        if !loss.is_finite() || gradients.iter().any(|g| !g.is_finite()) {
            return Err("nonfinite reference loss or gradients".into());
        }
        Ok((loss, gradients))
    }

    /// Effective tied vectors, including normalization and protected code injection.
    pub fn token_embeddings(&self) -> Result<Vec<Vec<f32>>> {
        let mut tape = Tape::new(self.parameters.iter().map(|&p| f64::from(p)));
        let ids = self.embeddings(&mut tape);
        read_features(&tape, &ids)
    }

    /// Residual states before the final RMSNorm, where the protected probe is invariant.
    pub fn hidden_states(&self, tokens: &[u32], sequence: usize) -> Result<Vec<Vec<f32>>> {
        self.config.validate_tokens(tokens, sequence)?;
        let (tape, _, ids) = self.forward(tokens, sequence);
        read_features(&tape, &ids)
    }

    fn embeddings(&self, tape: &mut Tape) -> Vec<Vec<usize>> {
        let config = self.config;
        let width = config.width;
        let offset = self.layout.groups[self.layout.embedding].offset;
        let mut embeddings = match config.embedding {
            TokenEmbedding::Mesh => {
                let codes: Vec<Vec<_>> = token_codes(config)
                    .iter()
                    .map(|row| {
                        row[..width]
                            .iter()
                            .map(|&x| tape.constant(f64::from(x)))
                            .collect()
                    })
                    .collect();
                mesh(
                    tape,
                    &codes,
                    offset,
                    config.attention_mesh_depth,
                    config.connectivity,
                )
            }
            // Parameter values are the first tape nodes, so table entries are their IDs.
            TokenEmbedding::Table => (0..config.vocab_size)
                .map(|token| (0..width).map(|c| offset + token * width + c).collect())
                .collect(),
        };
        if config.geometry.normalization != EmbeddingNorm::None {
            let scale = f64::from(config.geometry.radius(width));
            for row in &mut embeddings {
                let squares = tape.dot(row, row);
                let clamped = tape.clamp_min(squares, f64::from(XSA_EPSILON).powi(2));
                let inverse = tape.inverse_sqrt(clamped);
                let inverse = tape.scale(inverse, scale);
                for id in row {
                    *id = tape.multiply(*id, inverse);
                }
            }
        }
        if config.geometry.preserve_features > 0 {
            for (row, code) in embeddings.iter_mut().zip(token_codes(config)) {
                for channel in 0..config.geometry.preserve_features {
                    row[channel] = tape.constant(f64::from(code[channel]));
                }
            }
        }
        embeddings
    }

    fn forward(&self, tokens: &[u32], sequence: usize) -> (Tape, Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let config = self.config;
        let connectivity = config.connectivity;
        let mut tape = Tape::new(self.parameters.iter().map(|&p| f64::from(p)));
        let embeddings = self.embeddings(&mut tape);
        let mut x: Vec<Vec<_>> = tokens
            .iter()
            .map(|&token| embeddings[token as usize].clone())
            .collect();
        for block in &self.layout.blocks {
            let normalized = rmsnorm(&mut tape, &x, 1);
            let [query, key, value] = [block.query, block.key, block.value].map(|group| {
                mesh(
                    &mut tape,
                    &normalized,
                    self.layout.groups[group].offset,
                    config.attention_mesh_depth,
                    connectivity,
                )
            });
            let query = if config.qk_norm {
                rmsnorm(&mut tape, &query, config.heads)
            } else {
                query
            };
            let key = if config.qk_norm {
                rmsnorm(&mut tape, &key, config.heads)
            } else {
                key
            };
            let query = if config.rope {
                rotary(&mut tape, &query, sequence, config.heads)
            } else {
                query
            };
            let key = if config.rope {
                rotary(&mut tape, &key, sequence, config.heads)
            } else {
                key
            };
            let attended = attention(&mut tape, &query, &key, &value, sequence, config.heads);
            let attended = if config.xsa {
                exclusive(&mut tape, &attended, &value, config.heads)
            } else {
                attended
            };
            let projected = mesh(
                &mut tape,
                &attended,
                self.layout.groups[block.projection].offset,
                config.attention_mesh_depth,
                connectivity,
            );
            x = residual(&mut tape, &x, &projected);
            let normalized = rmsnorm(&mut tape, &x, 1);
            let feed_forward = mesh(
                &mut tape,
                &normalized,
                self.layout.groups[block.ffn].offset,
                config.ffn_mesh_depth,
                connectivity,
            );
            x = residual(&mut tape, &x, &feed_forward);
        }
        let normalized = rmsnorm(&mut tape, &x, 1);
        let logits = normalized
            .iter()
            .map(|row| {
                (0..config.vocab_size)
                    .map(|token| tape.dot(row, &embeddings[token]))
                    .collect()
            })
            .collect();
        (tape, logits, x)
    }
}

fn read_features(tape: &Tape, ids: &[Vec<usize>]) -> Result<Vec<Vec<f32>>> {
    ids.iter()
        .map(|row| {
            let values = row
                .iter()
                .map(|&id| tape.values[id] as f32)
                .collect::<Vec<_>>();
            validate_finite(&values, "features")?;
            Ok(values)
        })
        .collect()
}

struct Tape {
    values: Vec<f64>,
    edges: Vec<[(usize, f64); 2]>,
}

impl Tape {
    fn new(values: impl IntoIterator<Item = f64>) -> Self {
        let values: Vec<_> = values.into_iter().collect();
        Self {
            edges: vec![[(NONE, 0.0); 2]; values.len()],
            values,
        }
    }
    fn node(&mut self, value: f64, edges: [(usize, f64); 2]) -> usize {
        let id = self.values.len();
        self.values.push(value);
        self.edges.push(edges);
        id
    }
    fn constant(&mut self, value: f64) -> usize {
        self.node(value, [(NONE, 0.0); 2])
    }
    fn add(&mut self, a: usize, b: usize) -> usize {
        self.node(self.values[a] + self.values[b], [(a, 1.0), (b, 1.0)])
    }
    fn subtract(&mut self, a: usize, b: usize) -> usize {
        self.node(self.values[a] - self.values[b], [(a, 1.0), (b, -1.0)])
    }
    fn multiply(&mut self, a: usize, b: usize) -> usize {
        self.node(
            self.values[a] * self.values[b],
            [(a, self.values[b]), (b, self.values[a])],
        )
    }
    fn divide(&mut self, a: usize, b: usize) -> usize {
        let reciprocal = 1.0 / self.values[b];
        self.node(
            self.values[a] * reciprocal,
            [
                (a, reciprocal),
                (b, -self.values[a] * reciprocal * reciprocal),
            ],
        )
    }
    fn scale(&mut self, a: usize, scale: f64) -> usize {
        self.node(self.values[a] * scale, [(a, scale), (NONE, 0.0)])
    }
    fn shift(&mut self, a: usize, offset: f64) -> usize {
        self.node(self.values[a] + offset, [(a, 1.0), (NONE, 0.0)])
    }
    fn exp(&mut self, a: usize) -> usize {
        let v = self.values[a].exp();
        self.node(v, [(a, v), (NONE, 0.0)])
    }
    fn log(&mut self, a: usize) -> usize {
        self.node(
            self.values[a].ln(),
            [(a, 1.0 / self.values[a]), (NONE, 0.0)],
        )
    }
    fn inverse_sqrt(&mut self, a: usize) -> usize {
        let v = 1.0 / self.values[a].sqrt();
        self.node(v, [(a, -0.5 * v * v * v), (NONE, 0.0)])
    }
    fn clamp_min(&mut self, a: usize, minimum: f64) -> usize {
        self.node(
            self.values[a].max(minimum),
            [
                (a, if self.values[a] >= minimum { 1.0 } else { 0.0 }),
                (NONE, 0.0),
            ],
        )
    }
    fn elu(&mut self, a: usize) -> usize {
        let x = self.values[a];
        if x >= 0.0 {
            self.scale(a, 1.0)
        } else {
            self.node(x.exp_m1(), [(a, x.exp()), (NONE, 0.0)])
        }
    }
    fn sum(&mut self, ids: &[usize]) -> usize {
        let mut total = ids[0];
        for &id in &ids[1..] {
            total = self.add(total, id);
        }
        total
    }
    fn dot(&mut self, left: &[usize], right: &[usize]) -> usize {
        let products: Vec<_> = left
            .iter()
            .zip(right)
            .map(|(&a, &b)| self.multiply(a, b))
            .collect();
        self.sum(&products)
    }
    fn shifted_exponentials(&mut self, ids: &[usize]) -> (f64, usize, Vec<usize>) {
        let maximum = ids
            .iter()
            .map(|&id| self.values[id])
            .fold(f64::NEG_INFINITY, f64::max);
        let exponentials: Vec<_> = ids
            .iter()
            .map(|&id| {
                let shifted = self.shift(id, -maximum);
                self.exp(shifted)
            })
            .collect();
        (maximum, self.sum(&exponentials), exponentials)
    }
    fn output_and_gradient(
        mut self,
        output: &[usize],
        upstream: &[f64],
        count: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let values = output.iter().map(|&id| self.values[id]).collect();
        let weighted: Vec<_> = output
            .iter()
            .zip(upstream)
            .map(|(&id, &g)| self.scale(id, g))
            .collect();
        let loss = self.sum(&weighted);
        (values, self.backward(loss, count))
    }
    fn backward(&self, loss: usize, count: usize) -> Vec<f64> {
        let mut adjoints = vec![0.0; self.values.len()];
        adjoints[loss] = 1.0;
        for id in (0..self.values.len()).rev() {
            for &(parent, derivative) in &self.edges[id] {
                if parent != NONE {
                    adjoints[parent] += adjoints[id] * derivative;
                }
            }
        }
        adjoints.truncate(count);
        adjoints
    }
}

fn rmsnorm(tape: &mut Tape, x: &[Vec<usize>], heads: usize) -> Vec<Vec<usize>> {
    let head_width = x[0].len() / heads;
    x.iter()
        .map(|row| {
            row.chunks_exact(head_width)
                .flat_map(|head| {
                    let sum = tape.dot(head, head);
                    let mean_square = tape.scale(sum, 1.0 / head_width as f64);
                    let stabilized = tape.shift(mean_square, f64::from(RMSNORM_EPSILON));
                    let inverse = tape.inverse_sqrt(stabilized);
                    head.iter()
                        .map(|&id| tape.multiply(id, inverse))
                        .collect::<Vec<_>>()
                })
                .collect()
        })
        .collect()
}

pub(super) fn rmsnorm_reference(
    input: &[f64],
    gradient: &[f64],
    heads: usize,
) -> (Vec<f64>, Vec<f64>) {
    let width = input.len();
    let mut tape = Tape::new(input.iter().copied());
    let output = rmsnorm(&mut tape, &[(0..width).collect()], heads).remove(0);
    tape.output_and_gradient(&output, gradient, width)
}

fn mesh(
    tape: &mut Tape,
    input: &[Vec<usize>],
    offset: usize,
    depth: usize,
    connectivity: MeshConnectivity,
) -> Vec<Vec<usize>> {
    let width = input[0].len();
    let mut x = input.to_vec();
    for layer in 0..depth {
        let stride = connectivity.stride(width, layer);
        x = x
            .iter()
            .map(|row| {
                (0..width)
                    .map(|channel| {
                        let p = offset + layer * 2 * width + 2 * channel;
                        let left = tape.multiply(row[channel], p);
                        let right = tape.multiply(row[(channel + stride) % width], p + 1);
                        let sum = tape.add(left, right);
                        if layer + 1 == depth {
                            sum
                        } else {
                            tape.elu(sum)
                        }
                    })
                    .collect()
            })
            .collect();
    }
    x
}

fn residual(tape: &mut Tape, left: &[Vec<usize>], right: &[Vec<usize>]) -> Vec<Vec<usize>> {
    left.iter()
        .zip(right)
        .map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| tape.add(x, y)).collect())
        .collect()
}

fn rotary_row(tape: &mut Tape, input: &[usize], position: usize, heads: usize) -> Vec<usize> {
    let head_width = input.len() / heads;
    let half_width = head_width / 2;
    let mut output = input.to_vec();
    for head in 0..heads {
        for pair in 0..half_width {
            let angle =
                position as f64 / f64::from(ROPE_BASE).powf(pair as f64 / half_width as f64);
            let (sine, cosine) = angle.sin_cos();
            let left = head * head_width + pair;
            let right = left + half_width;
            let a = tape.scale(input[left], cosine);
            let b = tape.scale(input[right], sine);
            output[left] = tape.subtract(a, b);
            let a = tape.scale(input[right], cosine);
            let b = tape.scale(input[left], sine);
            output[right] = tape.add(a, b);
        }
    }
    output
}

fn rotary(tape: &mut Tape, input: &[Vec<usize>], sequence: usize, heads: usize) -> Vec<Vec<usize>> {
    input
        .iter()
        .enumerate()
        .map(|(row, x)| rotary_row(tape, x, row % sequence, heads))
        .collect()
}

pub(super) fn rotary_reference(
    input: &[f64],
    gradient: &[f64],
    position: usize,
    heads: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut tape = Tape::new(input.iter().copied());
    let output = rotary_row(
        &mut tape,
        &(0..input.len()).collect::<Vec<_>>(),
        position,
        heads,
    );
    tape.output_and_gradient(&output, gradient, input.len())
}

// A channel is reduced over different key rows, rather than one tensor slice.
#[allow(clippy::needless_range_loop)]
fn attention(
    tape: &mut Tape,
    q: &[Vec<usize>],
    k: &[Vec<usize>],
    v: &[Vec<usize>],
    sequence: usize,
    heads: usize,
) -> Vec<Vec<usize>> {
    let width = q[0].len();
    let head_width = width / heads;
    q.iter()
        .enumerate()
        .map(|(row, query)| {
            let position = row % sequence;
            let first = row - position;
            let mut output = Vec::new();
            for head in 0..heads {
                let begin = head * head_width;
                let scores: Vec<_> = (0..=position)
                    .map(|key| {
                        let sum = tape.dot(
                            &query[begin..begin + head_width],
                            &k[first + key][begin..begin + head_width],
                        );
                        tape.scale(sum, 1.0 / (head_width as f64).sqrt())
                    })
                    .collect();
                let (_, sum, exponentials) = tape.shifted_exponentials(&scores);
                let probabilities: Vec<_> = exponentials
                    .iter()
                    .map(|&id| tape.divide(id, sum))
                    .collect();
                for channel in begin..begin + head_width {
                    let contributions: Vec<_> = probabilities
                        .iter()
                        .enumerate()
                        .map(|(key, &id)| tape.multiply(id, v[first + key][channel]))
                        .collect();
                    output.push(tape.sum(&contributions));
                }
            }
            output
        })
        .collect()
}

pub(super) fn attention_reference(
    inputs: [&[f64]; 3],
    upstream: &[f64],
    width: usize,
    heads: usize,
    sequence: usize,
) -> (Vec<f64>, Vec<f64>) {
    let count = inputs[0].len();
    let mut tape = Tape::new(inputs.into_iter().flatten().copied());
    let ids = [0, 1, 2].map(|tensor| {
        (0..count / width)
            .map(|row| (tensor * count + row * width..tensor * count + (row + 1) * width).collect())
            .collect::<Vec<Vec<_>>>()
    });
    let output = attention(&mut tape, &ids[0], &ids[1], &ids[2], sequence, heads)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    tape.output_and_gradient(&output, upstream, 3 * count)
}

fn exclusive(
    tape: &mut Tape,
    input: &[Vec<usize>],
    value: &[Vec<usize>],
    heads: usize,
) -> Vec<Vec<usize>> {
    let head_width = input[0].len() / heads;
    input
        .iter()
        .zip(value)
        .map(|(y, v)| {
            let mut output = Vec::with_capacity(y.len());
            for (y, v) in y.chunks_exact(head_width).zip(v.chunks_exact(head_width)) {
                let squared_norm = tape.dot(v, v);
                let floored = tape.clamp_min(squared_norm, f64::from(XSA_EPSILON).powi(2));
                let inverse = tape.inverse_sqrt(floored);
                let normalized: Vec<_> = v.iter().map(|&id| tape.multiply(id, inverse)).collect();
                let projection = tape.dot(y, &normalized);
                output.extend(y.iter().zip(normalized).map(|(&a, b)| {
                    let removed = tape.multiply(projection, b);
                    tape.subtract(a, removed)
                }));
            }
            output
        })
        .collect()
}

pub(super) fn exclusive_reference(
    input: &[f64],
    value: &[f64],
    gradient: &[f64],
    heads: usize,
) -> (Vec<f64>, Vec<f64>) {
    let width = input.len();
    let mut tape = Tape::new(input.iter().chain(value).copied());
    let output = exclusive(
        &mut tape,
        &[(0..width).collect()],
        &[(width..2 * width).collect()],
        heads,
    )
    .remove(0);
    tape.output_and_gradient(&output, gradient, 2 * width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_preserves_nonzero_means_and_stabilizes_zero_inputs() {
        let upstream = [0.3, -0.7, 0.5, 1.0];
        let epsilon = f64::from(RMSNORM_EPSILON);
        let (output, _) = rmsnorm_reference(&[2.0; 4], &upstream, 1);
        for &value in &output {
            assert!((value - 2.0 / (4.0 + epsilon).sqrt()).abs() < 1e-12);
        }
        let (output, gradient) = rmsnorm_reference(&[0.0; 4], &upstream, 1);
        assert_eq!(output, [0.0; 4]);
        for (channel, &actual) in gradient.iter().enumerate() {
            let expected = upstream[channel] / epsilon.sqrt();
            assert!((actual - expected).abs() < 1e-12 * (1.0 + expected.abs()));
        }
    }

    #[test]
    fn rmsnorm_gradients_match_finite_differences_for_full_rows_and_heads() {
        let upstream = [-0.3, 0.2, 0.8, -0.4];
        for input in [
            [0.0; 4],
            [1e-7, -2e-7, 3e-7, -4e-7],
            [0.001, -0.002, 0.003, -0.004],
            [1.0, -2.0, 3.0, 4.0],
            [2.0; 4],
        ] {
            for heads in [1, 2] {
                let (_, gradient) = rmsnorm_reference(&input, &upstream, heads);
                for (channel, &analytic) in gradient.iter().enumerate() {
                    let mut plus = input;
                    let mut minus = input;
                    plus[channel] += 1e-7;
                    minus[channel] -= 1e-7;
                    let loss = |values: [f64; 4]| {
                        let (output, _) = rmsnorm_reference(&values, &upstream, heads);
                        output
                            .iter()
                            .zip(upstream)
                            .map(|(&x, g)| x * g)
                            .sum::<f64>()
                    };
                    let numerical = (loss(plus) - loss(minus)) / 2e-7;
                    assert!(
                        (analytic - numerical).abs() < 1e-6 * (1.0 + analytic.abs()),
                        "RMSNorm heads={heads} channel={channel}: {analytic} vs {numerical}"
                    );
                }
            }
        }
    }

    #[test]
    fn qk_norm_normalizes_each_head_independently() {
        let input = [3.0, 4.0, 5.0, 12.0];
        let upstream = [0.3, -0.7, 0.5, 1.0];
        let epsilon = f64::from(RMSNORM_EPSILON);
        let (output, gradient) = rmsnorm_reference(&input, &upstream, 2);
        for (head, values) in input.as_chunks::<2>().0.iter().enumerate() {
            let inverse = 1.0 / (values.iter().map(|x| x * x).sum::<f64>() / 2.0 + epsilon).sqrt();
            for channel in 0..2 {
                assert!((output[2 * head + channel] - values[channel] * inverse).abs() < 1e-12);
            }
        }
        let changed = [3.0, 4.0, -50.0, 0.001];
        let (changed_output, changed_gradient) = rmsnorm_reference(&changed, &upstream, 2);
        assert_eq!(&output[..2], &changed_output[..2]);
        assert_eq!(&gradient[..2], &changed_gradient[..2]);
    }

    #[test]
    fn slice_rope_pairs_opposite_halves_and_preserves_norms() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let (identity, _) = rotary_reference(&input, &[1.0; 4], 0, 1);
        assert_eq!(identity, input);
        let (output, _) = rotary_reference(&input, &[1.0; 4], 1, 1);
        let expected = [
            input[0] * 1.0_f64.cos() - input[2] * 1.0_f64.sin(),
            input[1] * 0.01_f64.cos() - input[3] * 0.01_f64.sin(),
            input[2] * 1.0_f64.cos() + input[0] * 1.0_f64.sin(),
            input[3] * 0.01_f64.cos() + input[1] * 0.01_f64.sin(),
        ];
        for (&a, b) in output.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12);
        }
        let norm = |x: &[f64]| x.iter().map(|x| x * x).sum::<f64>();
        assert!((norm(&output) - norm(&input)).abs() < 1e-12);
        let input = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let (output, _) = rotary_reference(&input, &[1.0; 8], 127, 2);
        for (x, y) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output.as_chunks::<4>().0)
        {
            assert!((norm(x) - norm(y)).abs() < 1e-12);
        }
    }

    #[test]
    fn slice_rope_gradients_and_relative_position_identity() {
        let q = [1.0, -0.5, 2.0, 0.7, -1.0, 0.3, 0.1, -0.9];
        let k = [-0.7, 0.2, 0.5, -1.0, 0.4, 0.8, -0.2, 0.6];
        let loss = |x: &[f64], position| {
            let (z, _) = rotary_reference(x, &k, position, 2);
            z.iter().zip(k).map(|(&z, g)| z * g).sum::<f64>()
        };
        for position in [0, 1, 31, 127] {
            let (_, gradient) = rotary_reference(&q, &k, position, 2);
            for (channel, &analytic) in gradient.iter().enumerate() {
                let mut plus = q;
                let mut minus = q;
                plus[channel] += 1e-6;
                minus[channel] -= 1e-6;
                let numerical = (loss(&plus, position) - loss(&minus, position)) / 2e-6;
                assert!((analytic - numerical).abs() < 1e-8);
            }
        }
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(&a, &b)| a * b).sum::<f64>();
        let (a, _) = rotary_reference(&q, &k, 7, 2);
        let (b, _) = rotary_reference(&k, &q, 11, 2);
        let (a_shifted, _) = rotary_reference(&q, &k, 57, 2);
        let (b_shifted, _) = rotary_reference(&k, &q, 61, 2);
        assert!((dot(&a, &b) - dot(&a_shifted, &b_shifted)).abs() < 1e-12);
    }

    #[test]
    fn exclusive_projection_is_per_head_and_removes_self_alignment() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let v = [1.0, 0.0, 0.0, 2.0];
        let (z, _) = exclusive_reference(&y, &v, &[1.0; 4], 2);
        assert_eq!(z, [0.0, 2.0, 3.0, 0.0]);
        for (z, v) in z.as_chunks::<2>().0.iter().zip(v.as_chunks::<2>().0) {
            let dot: f64 = z.iter().zip(v).map(|(&a, &b)| a * b).sum();
            assert!(dot.abs() < 1e-12);
        }
        // A causal sequence's first position attends only to its own value.
        let (z, _) = exclusive_reference(&v, &v, &[1.0; 4], 2);
        assert!(z.iter().all(|&x| x.abs() < 1e-12));
        let (z, gradient) = exclusive_reference(&y, &[0.0; 4], &[1.0; 4], 2);
        assert_eq!(z, y);
        assert_eq!(&gradient[..4], &[1.0; 4]);
        assert_eq!(&gradient[4..], &[0.0; 4]);
    }

    #[test]
    fn exclusive_gradients_match_finite_differences_including_norm_floor() {
        let input = [1.5, -0.25, 0.8, -1.2];
        let upstream = [0.3, -0.5, 0.7, 0.9];
        let epsilon = f64::from(XSA_EPSILON);
        for (value, heads) in [
            ([2.0, -1.0, 0.3, 0.7], 2),
            ([0.0; 4], 2),
            ([0.2 * epsilon, -0.3 * epsilon, 2.0 * epsilon, -epsilon], 2),
            ([1.0, -2.0, 0.7, 0.8], 4),
        ] {
            let (_, gradient) = exclusive_reference(&input, &value, &upstream, heads);
            for (parameter, &analytic) in gradient.iter().enumerate() {
                let mut plus = [input, value];
                let mut minus = plus;
                let tensor = parameter / 4;
                let channel = parameter % 4;
                let step = if tensor == 1 && value[channel].abs() < 1e-10 {
                    epsilon * 1e-4
                } else {
                    1e-6
                };
                plus[tensor][channel] += step;
                minus[tensor][channel] -= step;
                let loss = |values: [[f64; 4]; 2]| {
                    let (z, _) = exclusive_reference(&values[0], &values[1], &upstream, heads);
                    z.iter().zip(upstream).map(|(&z, g)| z * g).sum::<f64>()
                };
                let numerical = (loss(plus) - loss(minus)) / (2.0 * step);
                assert!(
                    analytic.is_finite()
                        && (analytic - numerical).abs() < 1e-5 * (1.0 + analytic.abs()),
                    "heads={heads} parameter={parameter}: {analytic} vs {numerical}"
                );
            }
        }
    }

    pub(crate) fn tiny_config() -> GptConfig {
        GptConfig {
            vocab_size: 5,
            context: 4,
            width: 4,
            heads: 2,
            layers: 1,
            attention_mesh_depth: 2,
            ffn_mesh_depth: 2,
            ..GptConfig::default()
        }
    }

    /// The default model and all three ablations; width 4 gives butterfly strides 1 and 2.
    fn tiny_variants() -> Vec<GptConfig> {
        [TokenEmbedding::Mesh, TokenEmbedding::Table]
            .into_iter()
            .flat_map(|embedding| {
                [MeshConnectivity::Neighbor, MeshConnectivity::Butterfly].map(|connectivity| {
                    GptConfig {
                        embedding,
                        connectivity,
                        ..tiny_config()
                    }
                })
            })
            .collect()
    }

    #[test]
    fn zero_inputs_do_not_create_mesh_or_head_offsets() {
        let width = 4;
        for depth in [1, 3, 16] {
            for connectivity in [MeshConnectivity::Neighbor, MeshConnectivity::Butterfly] {
                let mut tape = Tape::new(std::iter::repeat_n(0.5, 2 * width * depth));
                let zero = tape.node(0.0, [(NONE, 0.0); 2]);
                let output = mesh(&mut tape, &[vec![zero; width]], 0, depth, connectivity);
                for &id in &output[0] {
                    assert_eq!(tape.values[id], 0.0);
                }
            }
        }
        for config in tiny_variants() {
            let mut parameters = vec![0.5; config.parameter_count().unwrap()];
            parameters[..config.embedding_parameter_count()].fill(0.0);
            let model = ReferenceGpt::from_parameters(config, parameters).unwrap();
            let inputs = [0, 1, 2, 3, 4, 0];
            assert_eq!(
                model.logits(&inputs, 3).unwrap(),
                vec![vec![0.0; config.vocab_size]; inputs.len()]
            );
        }
    }

    #[test]
    fn butterfly_meshes_read_power_of_two_distant_channels() {
        let width = 8;
        let depth = 3;
        let mut tape = Tape::new(std::iter::repeat_n(0.0, 2 * width * depth));
        let input: Vec<_> = (0..width)
            .map(|channel| tape.constant(channel as f64))
            .collect();
        // Isolate each layer's second parent: zero self edges, unit second edges.
        for layer in 0..depth {
            for channel in 0..width {
                tape.values[layer * 2 * width + 2 * channel + 1] = 1.0;
            }
        }
        let output = mesh(&mut tape, &[input], 0, depth, MeshConnectivity::Butterfly);
        for (channel, &id) in output[0].iter().enumerate() {
            // ELU leaves these nonnegative values unchanged; strides are 1, 2 and 4.
            assert_eq!(tape.values[id], ((channel + 7) % width) as f64);
        }
    }

    #[test]
    fn table_embedding_matches_the_tabulated_mesh_model() {
        for connectivity in [MeshConnectivity::Neighbor, MeshConnectivity::Butterfly] {
            let mesh = GptConfig {
                connectivity,
                ..tiny_config()
            };
            let table = GptConfig {
                embedding: TokenEmbedding::Table,
                ..mesh
            };
            let mesh_model = ReferenceGpt::new(mesh, 7).unwrap();
            let table_model = ReferenceGpt::new(table, 7).unwrap();
            assert_eq!(
                table_model.parameters(),
                super::super::tabulate_embedding(mesh, mesh_model.parameters()).unwrap()
            );
            let inputs = [0, 1, 3, 2, 4, 1];
            let expected = mesh_model.logits(&inputs, 3).unwrap();
            let actual = table_model.logits(&inputs, 3).unwrap();
            for (expected, actual) in expected.iter().flatten().zip(actual.iter().flatten()) {
                assert!((expected - actual).abs() < 1e-5 * (1.0 + expected.abs()));
            }
        }
    }

    #[test]
    fn all_transformer_gradients_match_finite_differences() {
        for config in tiny_variants() {
            transformer_gradients_match_finite_differences(config);
        }
    }

    fn transformer_gradients_match_finite_differences(config: GptConfig) {
        let model = ReferenceGpt::new(config, 7).unwrap();
        let x = [0, 1, 3, 2, 4, 1];
        for y in [
            [1, 3, 2, 4, 1, 0],
            [
                IGNORE_TARGET,
                IGNORE_TARGET,
                2,
                IGNORE_TARGET,
                IGNORE_TARGET,
                0,
            ],
        ] {
            let (_, gradients) = model.loss_and_gradient(&x, &y, 3).unwrap();
            for (parameter, &analytic) in gradients.iter().enumerate() {
                let mut plus = model.parameters.clone();
                let mut minus = plus.clone();
                // Small steps resolve normalization and ELU curvature.
                plus[parameter] += 0.0001;
                minus[parameter] -= 0.0001;
                let step = f64::from(plus[parameter]) - f64::from(minus[parameter]);
                let a = ReferenceGpt::from_parameters(model.config, plus)
                    .unwrap()
                    .loss_and_gradient(&x, &y, 3)
                    .unwrap()
                    .0;
                let b = ReferenceGpt::from_parameters(model.config, minus)
                    .unwrap()
                    .loss_and_gradient(&x, &y, 3)
                    .unwrap()
                    .0;
                let numerical = (a - b) / step;
                assert!(
                    (analytic - numerical).abs() < 4e-4 * (1.0 + analytic.abs()),
                    "{config:?} parameter={parameter}: {analytic} vs {numerical}"
                );
            }
        }
    }

    #[test]
    fn answer_only_loss_and_gradients_average_over_supervised_equations() {
        let model = ReferenceGpt::new(tiny_config(), 7).unwrap();
        let inputs = [0, 1, 3, 2, 4, 1];
        let targets = [
            IGNORE_TARGET,
            IGNORE_TARGET,
            2,
            IGNORE_TARGET,
            IGNORE_TARGET,
            0,
        ];
        let (loss, gradient) = model.loss_and_gradient(&inputs, &targets, 3).unwrap();
        let (first_loss, first_gradient) = model
            .loss_and_gradient(&inputs[..3], &targets[..3], 3)
            .unwrap();
        let (second_loss, second_gradient) = model
            .loss_and_gradient(&inputs[3..], &targets[3..], 3)
            .unwrap();
        assert!((loss - (first_loss + second_loss) / 2.0).abs() < 1e-12);
        for ((actual, first), second) in gradient.iter().zip(first_gradient).zip(second_gradient) {
            assert!((actual - (first + second) / 2.0).abs() < 1e-12);
        }
        assert!(
            model
                .loss_and_gradient(&inputs, &[IGNORE_TARGET; 6], 3)
                .is_err()
        );
        assert!(model.loss_and_gradient(&inputs, &[99; 6], 3).is_err());
        assert!(model.logits(&[IGNORE_TARGET], 1).is_err());
    }

    #[test]
    fn future_tokens_and_other_batch_items_cannot_change_prefix_logits() {
        let model = ReferenceGpt::new(tiny_config(), 11).unwrap();
        let a = model.logits(&[0, 1, 2, 3, 1, 2, 3, 4], 4).unwrap();
        let b = model.logits(&[0, 1, 4, 0, 4, 4, 4, 4], 4).unwrap();
        assert_eq!(&a[..2], &b[..2]);
        let prefix = model.logits(&[0, 1], 2).unwrap();
        assert_eq!(&a[..2], &prefix);
    }
}
