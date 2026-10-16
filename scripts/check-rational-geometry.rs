//! Deterministic representation checks, not neural-network training.
//! Build directly with rustc; this program only needs the standard library.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::TAU;

const DIM: usize = 16;
const FIELD: i64 = 1_000_003;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Rational {
    numerator: u32,
    denominator: u32,
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Rational {
    fn new(numerator: u32, denominator: u32) -> Self {
        assert!(numerator > 0 && denominator > 0);
        let divisor = gcd(numerator, denominator);
        Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        }
    }

    fn divide(self, other: Self) -> Self {
        Self::new(
            self.numerator * other.denominator,
            self.denominator * other.numerator,
        )
    }
}

fn domain(bound: u32) -> Vec<Rational> {
    (1..=bound)
        .flat_map(|a| (1..=bound).map(move |b| Rational::new(a, b)))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            values.swap(i, j);
        }
    }
}

fn primes(bound: u32) -> Vec<u32> {
    (2..=bound)
        .filter(|&p| (2..p).all(|d| p % d != 0))
        .collect()
}

fn valuation(mut value: u32, prime: u32) -> i32 {
    let mut power = 0;
    while value % prime == 0 {
        value /= prime;
        power += 1;
    }
    power
}

type Vector = [[f32; 2]; DIM];

fn prototype(angles: [f64; DIM]) -> Vector {
    angles.map(|angle| [angle.cos() as f32, angle.sin() as f32])
}

fn quotient(a: &Vector, b: &Vector) -> Vector {
    std::array::from_fn(|j| {
        [
            a[j][0] * b[j][0] + a[j][1] * b[j][1],
            a[j][1] * b[j][0] - a[j][0] * b[j][1],
        ]
    })
}

fn score(query: &Vector, candidate: &Vector) -> f32 {
    let mut sum = 0.0_f32;
    for j in 0..DIM {
        sum += query[j][0] * candidate[j][0] + query[j][1] * candidate[j][1];
    }
    sum / DIM as f32
}

fn evaluate(name: &str, vectors: &[Vector], queries: &[(usize, usize, usize)]) {
    let mut correct = 0;
    let mut small_margin = 0;
    let mut minimum_margin = f32::INFINITY;
    for &(a, b, answer) in queries {
        let query = quotient(&vectors[a], &vectors[b]);
        let target_score = score(&query, &vectors[answer]);
        let mut best = (f32::NEG_INFINITY, usize::MAX);
        let mut best_wrong = f32::NEG_INFINITY;
        for (index, candidate) in vectors.iter().enumerate() {
            let value = score(&query, candidate);
            if value > best.0 {
                best = (value, index);
            }
            if index != answer {
                best_wrong = best_wrong.max(value);
            }
        }
        correct += usize::from(best.1 == answer);
        let margin = target_score - best_wrong;
        minimum_margin = minimum_margin.min(margin);
        small_margin += usize::from(margin <= 1e-6);
    }
    println!(
        "    {{\"check\":\"{name}\",\"queries\":{},\"correct\":{correct},\"margin_at_most_1e_6\":{small_margin},\"minimum_margin\":{minimum_margin:e}}},",
        queries.len()
    );
    if name.starts_with("prime_characters") {
        assert_eq!(correct, queries.len());
        assert!(minimum_margin > 1e-6);
    }
}

fn modular_power(mut a: i64, mut power: i64) -> i64 {
    let mut value = 1;
    while power > 0 {
        if power % 2 == 1 {
            value = value * a % FIELD;
        }
        a = a * a % FIELD;
        power /= 2;
    }
    value
}

struct Basis {
    rows: Vec<Option<Vec<i64>>>,
    rank: usize,
}

impl Basis {
    fn new(size: usize) -> Self {
        Self {
            rows: vec![None; size],
            rank: 0,
        }
    }

    fn reduce(&self, row: &mut [i64]) {
        for pivot in 0..row.len() {
            if row[pivot] == 0 {
                continue;
            }
            if let Some(basis_row) = &self.rows[pivot] {
                let multiplier = row[pivot];
                for j in pivot..row.len() {
                    row[j] = (row[j] - multiplier * basis_row[j]).rem_euclid(FIELD);
                }
            }
        }
    }

    fn insert(&mut self, mut row: Vec<i64>) {
        self.reduce(&mut row);
        if let Some(pivot) = row.iter().position(|&value| value != 0) {
            let inverse = modular_power(row[pivot], FIELD - 2);
            for value in &mut row[pivot..] {
                *value = *value * inverse % FIELD;
            }
            self.rows[pivot] = Some(row);
            self.rank += 1;
        }
    }

    fn implies(&self, mut row: Vec<i64>) -> bool {
        self.reduce(&mut row);
        row.iter().all(|&value| value == 0)
    }
}

fn relation(size: usize, a: usize, b: usize, answer: usize) -> Vec<i64> {
    let mut row = vec![0_i64; size];
    row[a] += 1;
    row[b] -= 1;
    row[answer] -= 1;
    row.iter_mut()
        .for_each(|value| *value = value.rem_euclid(FIELD));
    row
}

fn main() {
    assert!(
        (2_i64..)
            .take_while(|divisor| divisor * divisor <= FIELD)
            .all(|divisor| FIELD % divisor != 0)
    );
    let inputs = domain(8);
    let outputs = domain(64);
    let index: BTreeMap<_, _> = outputs.iter().copied().zip(0..).collect();
    let mut queries = Vec::new();
    let mut answers = BTreeSet::new();
    for &a in &inputs {
        for &b in &inputs {
            let c = a.divide(b);
            queries.push((index[&a], index[&b], index[&c]));
            answers.insert(c);
        }
    }
    let one = index[&Rational::new(1, 1)];
    let self_queries: Vec<_> = (0..outputs.len()).map(|i| (i, one, i)).collect();

    let prime_list = primes(64);
    let mut random = Random(20261016);
    let prime_angles: Vec<[f64; DIM]> = prime_list
        .iter()
        .map(|_| std::array::from_fn(|_| TAU * random.uniform()))
        .collect();
    let characters: Vec<_> = outputs
        .iter()
        .map(|q| {
            let mut angles = [0.0; DIM];
            for (p, generator) in prime_list.iter().zip(&prime_angles) {
                let power = valuation(q.numerator, *p) - valuation(q.denominator, *p);
                for j in 0..DIM {
                    angles[j] += f64::from(power) * generator[j];
                }
            }
            prototype(angles)
        })
        .collect();
    let mut random = Random(20261016);
    let log_frequencies: [f64; DIM] = std::array::from_fn(|_| 2.0 * random.uniform() - 1.0);
    let logarithms: Vec<_> = outputs
        .iter()
        .map(|q| {
            let value = (f64::from(q.numerator) / f64::from(q.denominator)).ln();
            prototype(log_frequencies.map(|frequency| frequency * value))
        })
        .collect();

    println!("{{");
    println!("  \"kind\":\"constructed_representations_and_exact_algebra_no_training\",");
    println!(
        "  \"phase_coordinates\":{DIM},\"stored_float32_components\":{},",
        DIM * 2
    );
    println!(
        "  \"positive_inputs\":{},\"positive_output_candidates\":{},\"distinct_quotients\":{},",
        inputs.len(),
        outputs.len(),
        answers.len()
    );
    println!("  \"representation_checks\":[");
    evaluate("prime_characters_division", &characters, &queries);
    evaluate(
        "prime_characters_self_retrieval",
        &characters,
        &self_queries,
    );
    evaluate("low_frequency_log_division", &logarithms, &queries);
    evaluate(
        "low_frequency_log_self_retrieval",
        &logarithms,
        &self_queries,
    );
    println!(
        "    {{\"note\":\"Low-frequency log features are one control, not every logarithmic encoding.\"}}"
    );
    println!("  ],");

    let small_index: BTreeMap<_, _> = inputs.iter().copied().zip(0..).collect();
    let mut equations = Vec::new();
    for (a, &x) in inputs.iter().enumerate() {
        for (b, &y) in inputs.iter().enumerate() {
            if let Some(&answer) = small_index.get(&x.divide(y)) {
                equations.push((a, b, answer));
            }
        }
    }
    let mut full = Basis::new(inputs.len());
    for &(a, b, c) in &equations {
        full.insert(relation(inputs.len(), a, b, c));
    }
    assert_eq!(inputs.len() - full.rank, primes(8).len());
    println!(
        "  \"phase_grid_field\":{FIELD},\"bounded_equations\":{},\"full_rank\":{},\"full_nullity\":{},",
        equations.len(),
        full.rank,
        inputs.len() - full.rank
    );
    println!("  \"constraint_checks\":[");
    let mut first = true;
    for seed in [11, 29, 47] {
        let mut shuffled = equations.clone();
        Random(seed).shuffle(&mut shuffled);
        for percent in [10, 20, 30, 50] {
            let split = equations.len() * percent / 100;
            let mut basis = Basis::new(inputs.len());
            let mut known = BTreeSet::new();
            let mut known_labels = BTreeSet::new();
            for &(a, b, c) in &shuffled[..split] {
                basis.insert(relation(inputs.len(), a, b, c));
                known.extend([a, b, c]);
                known_labels.insert(c);
            }
            let implied = shuffled[split..]
                .iter()
                .filter(|&&(a, b, c)| basis.implies(relation(inputs.len(), a, b, c)))
                .count();
            let covered = shuffled[split..]
                .iter()
                .filter(|&&(a, b, c)| {
                    known.contains(&a) && known.contains(&b) && known_labels.contains(&c)
                })
                .count();
            if !first {
                println!(",");
            }
            first = false;
            print!(
                "    {{\"seed\":{seed},\"train_percent\":{percent},\"train_equations\":{split},\"rank\":{},\"held_out\":{},\"implied_held_out\":{implied},\"held_out_with_seen_identities_and_positive_label\":{covered}}}",
                basis.rank,
                equations.len() - split
            );
        }
    }
    println!("\n  ],");

    let mut paths = BTreeSet::new();
    let mut maximum_depth = 0;
    for q in &outputs {
        let (mut a, mut b) = (q.numerator, q.denominator);
        let mut path = Vec::new();
        while a != b {
            if a < b {
                b -= a;
                path.push(b'L');
            } else {
                a -= b;
                path.push(b'R');
            }
        }
        assert_eq!((a, b), (1, 1));
        path.reverse();
        let (mut a, mut b) = (1, 1);
        for &step in &path {
            if step == b'L' {
                b += a;
            } else {
                a += b;
            }
        }
        assert_eq!(Rational::new(a, b), *q);
        maximum_depth = maximum_depth.max(path.len());
        assert!(paths.insert(path));
    }
    println!(
        "  \"rational_tree\":{{\"unique_roundtrips\":{},\"maximum_uncompressed_depth\":{maximum_depth}}}",
        paths.len()
    );
    println!("}}");
}
