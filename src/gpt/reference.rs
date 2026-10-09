use crate::{Result, mesh::validate_finite};

use super::{GptConfig, config::Layout};

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
        let (tape, ids) = self.forward(tokens, sequence);
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
        self.config.validate_batch(tokens, targets, sequence)?;
        let (mut tape, logits) = self.forward(tokens, sequence);
        let mut losses = Vec::new();
        for (row, &target) in logits.iter().zip(targets) {
            let maximum = row
                .iter()
                .map(|&id| tape.values[id])
                .fold(f64::NEG_INFINITY, f64::max);
            let mut exponentials = Vec::new();
            for &logit in row {
                let shifted = tape.shift(logit, -maximum);
                exponentials.push(tape.exp(shifted));
            }
            let sum = tape.sum(&exponentials);
            let log = tape.log(sum);
            let normalized = tape.shift(log, maximum);
            losses.push(tape.subtract(normalized, row[target as usize]));
        }
        let sum = tape.sum(&losses);
        let loss_id = tape.scale(sum, 1.0 / tokens.len() as f64);
        let loss = tape.values[loss_id];
        let gradients = tape.backward(loss_id, self.parameters.len());
        if !loss.is_finite() || gradients.iter().any(|g| !g.is_finite()) {
            return Err("nonfinite reference loss or gradients".into());
        }
        Ok((loss, gradients))
    }

    fn forward(&self, tokens: &[u32], sequence: usize) -> (Tape, Vec<Vec<usize>>) {
        let config = self.config;
        let width = config.width;
        let mut tape = Tape::new(&self.parameters);
        let mut x: Vec<Vec<_>> = tokens
            .iter()
            .enumerate()
            .map(|(row, &token)| {
                (0..width)
                    .map(|channel| {
                        tape.add(
                            token as usize * width + channel,
                            (config.vocab_size + row % sequence) * width + channel,
                        )
                    })
                    .collect()
            })
            .collect();
        for block in &self.layout.blocks {
            let normalized = norm(&mut tape, &x, self.layout.groups[block.norm1].offset);
            let query = mesh(
                &mut tape,
                &normalized,
                self.layout.groups[block.query].offset,
                config.attention_mesh_depth,
            );
            let key = mesh(
                &mut tape,
                &normalized,
                self.layout.groups[block.key].offset,
                config.attention_mesh_depth,
            );
            let value = mesh(
                &mut tape,
                &normalized,
                self.layout.groups[block.value].offset,
                config.attention_mesh_depth,
            );
            let attended = attention(&mut tape, &query, &key, &value, sequence, config.heads);
            let projected = mesh(
                &mut tape,
                &attended,
                self.layout.groups[block.projection].offset,
                config.attention_mesh_depth,
            );
            x = residual(&mut tape, &x, &projected);
            let normalized = norm(&mut tape, &x, self.layout.groups[block.norm2].offset);
            let feed_forward = mesh(
                &mut tape,
                &normalized,
                self.layout.groups[block.ffn].offset,
                config.ffn_mesh_depth,
            );
            x = residual(&mut tape, &x, &feed_forward);
        }
        let normalized = norm(
            &mut tape,
            &x,
            self.layout.groups[self.layout.final_norm].offset,
        );
        let bias = self.layout.groups[self.layout.head_bias].offset;
        let logits = normalized
            .iter()
            .map(|row| {
                (0..config.vocab_size)
                    .map(|token| {
                        let products: Vec<_> = row
                            .iter()
                            .enumerate()
                            .map(|(channel, &id)| tape.multiply(id, token * width + channel))
                            .collect();
                        let sum = tape.sum(&products);
                        tape.add(sum, bias + token)
                    })
                    .collect()
            })
            .collect();
        (tape, logits)
    }
}

struct Tape {
    values: Vec<f64>,
    edges: Vec<[(usize, f64); 2]>,
}

impl Tape {
    fn new(parameters: &[f32]) -> Self {
        Self {
            values: parameters.iter().map(|&p| f64::from(p)).collect(),
            edges: vec![[(NONE, 0.0); 2]; parameters.len()],
        }
    }
    fn node(&mut self, value: f64, edges: [(usize, f64); 2]) -> usize {
        let id = self.values.len();
        self.values.push(value);
        self.edges.push(edges);
        id
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

fn norm(tape: &mut Tape, x: &[Vec<usize>], offset: usize) -> Vec<Vec<usize>> {
    let width = x[0].len();
    x.iter()
        .map(|row| {
            let sum = tape.sum(row);
            let mean = tape.scale(sum, 1.0 / width as f64);
            let centered: Vec<_> = row.iter().map(|&id| tape.subtract(id, mean)).collect();
            let squares: Vec<_> = centered.iter().map(|&id| tape.multiply(id, id)).collect();
            let sum = tape.sum(&squares);
            let variance = tape.scale(sum, 1.0 / width as f64);
            let stabilized = tape.shift(variance, 1e-5);
            let inverse = tape.inverse_sqrt(stabilized);
            centered
                .iter()
                .enumerate()
                .map(|(channel, &id)| {
                    let normalized = tape.multiply(id, inverse);
                    let scaled = tape.multiply(normalized, offset + channel);
                    tape.add(scaled, offset + width + channel)
                })
                .collect()
        })
        .collect()
}

fn mesh(tape: &mut Tape, input: &[Vec<usize>], offset: usize, depth: usize) -> Vec<Vec<usize>> {
    let width = input[0].len();
    let mut x = input.to_vec();
    for layer in 0..depth {
        x = x
            .iter()
            .map(|row| {
                (0..width)
                    .map(|channel| {
                        let p = offset + layer * 3 * width + 3 * channel;
                        let left = tape.multiply(row[channel], p);
                        let right = tape.multiply(row[(channel + 1) % width], p + 1);
                        let sum = tape.add(left, right);
                        let biased = tape.add(sum, p + 2);
                        if layer + 1 == depth {
                            biased
                        } else {
                            tape.elu(biased)
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
                        let products: Vec<_> = (begin..begin + head_width)
                            .map(|c| tape.multiply(query[c], k[first + key][c]))
                            .collect();
                        let sum = tape.sum(&products);
                        tape.scale(sum, 1.0 / (head_width as f64).sqrt())
                    })
                    .collect();
                let maximum = scores
                    .iter()
                    .map(|&id| tape.values[id])
                    .fold(f64::NEG_INFINITY, f64::max);
                let exponentials: Vec<_> = scores
                    .iter()
                    .map(|&id| {
                        let shifted = tape.shift(id, -maximum);
                        tape.exp(shifted)
                    })
                    .collect();
                let sum = tape.sum(&exponentials);
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

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn tiny_config() -> GptConfig {
        GptConfig {
            vocab_size: 5,
            context: 4,
            width: 4,
            heads: 2,
            layers: 1,
            attention_mesh_depth: 2,
            ffn_mesh_depth: 2,
        }
    }

    #[test]
    fn all_transformer_gradients_match_finite_differences() {
        let model = ReferenceGpt::new(tiny_config(), 7).unwrap();
        let x = [0, 1, 3, 2, 4, 1];
        let y = [1, 3, 2, 4, 1, 0];
        let (_, gradients) = model.loss_and_gradient(&x, &y, 3).unwrap();
        for (parameter, &analytic) in gradients.iter().enumerate() {
            let mut plus = model.parameters.clone();
            let mut minus = plus.clone();
            plus[parameter] += 0.001;
            minus[parameter] -= 0.001;
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
                "parameter={parameter}: {analytic} vs {numerical}"
            );
        }
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
