//! Full-table operand interaction checks using the independent f64 reference.

use std::{collections::BTreeMap, env, fs, io::Write, path::Path};

use grokking_abmod::{
    Result,
    gpt::{Equation, EquationDataset},
    simple::{Architecture, ReferenceSimpleModel, SimpleConfig},
};

struct Interaction {
    weighted_fraction: f64,
    mean_class_fraction: f64,
    variances: Vec<f64>,
    fractions: Vec<f64>,
    normalized_pair_residuals: Vec<f64>,
}

// Exact two-factor ANOVA on a complete, equally weighted a-by-b grid.
fn interaction(side: usize, channels: usize, values: &[f64]) -> Interaction {
    assert_eq!(values.len(), side * side * channels);
    let mut a_mean = vec![0.0; side * channels];
    let mut b_mean = a_mean.clone();
    let mut mean = vec![0.0; channels];
    for a in 0..side {
        for b in 0..side {
            for q in 0..channels {
                let v = values[(a * side + b) * channels + q];
                a_mean[a * channels + q] += v / side as f64;
                b_mean[b * channels + q] += v / side as f64;
                mean[q] += v / (side * side) as f64;
            }
        }
    }
    let mut total = vec![0.0; channels];
    let mut joint = total.clone();
    let mut residuals = vec![0.0; values.len()];
    for a in 0..side {
        for b in 0..side {
            for q in 0..channels {
                let index = (a * side + b) * channels + q;
                let value = values[index];
                let residual =
                    value - a_mean[a * channels + q] - b_mean[b * channels + q] + mean[q];
                residuals[index] = residual;
                total[q] += (value - mean[q]).powi(2);
                joint[q] += residual.powi(2);
            }
        }
    }
    let active = total.iter().filter(|&&v| v > 1e-20).count();
    let fractions: Vec<_> = total
        .iter()
        .zip(&joint)
        .map(|(&t, &j)| {
            if t > 1e-20 {
                (j / t).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect();
    let total_sum = total.iter().sum::<f64>();
    let normalized_pair_residuals = residuals
        .chunks_exact(channels)
        .map(|row| {
            if active == 0 {
                return 0.0;
            }
            row.iter()
                .zip(&total)
                .filter(|(_, t)| **t > 1e-20)
                .map(|(&r, &t)| r * r * (side * side) as f64 / t)
                .sum::<f64>()
                / active as f64
        })
        .collect();
    Interaction {
        weighted_fraction: if total_sum == 0.0 {
            0.0
        } else {
            joint.iter().sum::<f64>() / total_sum
        },
        mean_class_fraction: if active == 0 {
            0.0
        } else {
            fractions.iter().sum::<f64>() / active as f64
        },
        variances: total.iter().map(|t| t / (side * side) as f64).collect(),
        fractions,
        normalized_pair_residuals,
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: shared-neighbor-interactions TRIAL_DIR OUTPUT_DIR".into());
    }
    let trial = Path::new(&args[0]);
    let output = Path::new(&args[1]);
    let settings: BTreeMap<_, _> = fs::read_to_string(trial.join("run.txt"))?
        .lines()
        .filter_map(|line| {
            line.split_once('=')
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
        })
        .collect();
    let value = |key: &str| -> Result<&str> {
        settings
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("missing {key}").into())
    };
    if value("architecture")? != "shared-neighbor-mesh-mlp"
        || value("operation")? != "div"
        || value("normalization")? != "pre_rmsnorm_blocks_and_final"
        || value("status")? != "complete"
    {
        return Err("require a complete corrected shared Neighbor division run".into());
    }
    let config = SimpleConfig {
        architecture: Architecture::SharedNeighborMeshMlp,
        width: value("width")?.parse()?,
        modulus: value("modulus")?.parse()?,
        depth: value("depth")?.parse()?,
        embedding_mesh_depth: value("embedding_mesh_depth")?.parse()?,
        activation: value("activation")?.parse()?,
        encoding: value("encoding")?.parse()?,
        connectivity: value("connectivity")?.parse()?,
        full_mesh_initialization: value("mesh_initialization")?.parse()?,
        ..SimpleConfig::default()
    };
    config.validate()?;
    let dataset = EquationDataset::new(
        config.modulus,
        value("operation")?.parse()?,
        value("train_fraction")?.parse()?,
        value("seed")?.parse()?,
    )?;
    let side = config.modulus - 1;
    let mut equations = vec![
        Equation {
            a: 0,
            b: 0,
            answer: 0
        };
        side * side
    ];
    for eq in dataset.split(false).iter().chain(dataset.split(true)) {
        if eq.a != 0 {
            equations[(eq.a - 1) * side + eq.b - 1] = *eq;
        }
    }
    let mut ideal = vec![0.0; side * side * side];
    for (index, eq) in equations.iter().enumerate() {
        ideal[index * side + eq.answer - 1] = 1.0;
    }
    let ideal = interaction(side, side, &ideal);
    if (ideal.mean_class_fraction - 1.0).abs() > 1e-12 {
        return Err("ideal division interaction check failed".into());
    }
    fs::create_dir_all(output)?;
    let mut summary = fs::File::create(output.join("joint-interaction.csv"))?;
    let mut classes = fs::File::create(output.join("joint-interaction-classes.csv"))?;
    let mut distances = fs::File::create(output.join("joint-interaction-distance.csv"))?;
    writeln!(
        summary,
        "phase,core_connectivity,weighted_interaction_fraction,mean_class_interaction_fraction,nonzero_grid_accuracy"
    )?;
    writeln!(classes, "phase,class,logit_variance,interaction_fraction")?;
    writeln!(
        distances,
        "phase,source_distance,pairs,mean_class_normalized_joint_energy"
    )?;
    let mut verification = fs::File::create(output.join("interaction-verification.txt"))?;
    writeln!(
        verification,
        "scope=All {} nonzero a/b pairs; exclude trivial a=0 and answer class 0\nsource=Independent ReferenceSimpleModel f64 logits\ncentering=Subtract each row's mean across nonzero class logits\ndecomposition=f(a,b)=grand_mean+a_effect(a)+b_effect(b)+joint_residual(a,b)\nclass_variance_floor=1e-20\nideal_division_mean_class_interaction_fraction={:.12}",
        equations.len(),
        ideal.mean_class_fraction
    )?;
    for (phase, filename) in [("initial", "initial-model.bin"), ("trained", "model.bin")] {
        let bytes = fs::read(trial.join(filename))?;
        if bytes.len() != 4 * config.parameter_count() {
            return Err("wrong checkpoint size".into());
        }
        let parameters = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        let model = ReferenceSimpleModel::from_parameters(config, parameters)?;
        let mut scores = Vec::with_capacity(equations.len() * side);
        let mut correct = 0;
        for chunk in equations.chunks(8) {
            let (features, targets) = config.encode(chunk)?;
            for (row, target) in model.logits(&features)?.into_iter().zip(targets) {
                if row.iter().any(|v| !v.is_finite()) {
                    return Err("nonfinite reference logits".into());
                }
                let mut prediction = 0;
                for q in 1..row.len() {
                    if row[q] > row[prediction] {
                        prediction = q;
                    }
                }
                correct += usize::from(prediction == target as usize);
                let mean = row[1..].iter().sum::<f64>() / side as f64;
                scores.extend(row[1..].iter().map(|v| v - mean));
            }
        }
        let stats = interaction(side, side, &scores);
        writeln!(
            summary,
            "{phase},{},{:.12},{:.12},{:.12}",
            config.connectivity,
            stats.weighted_fraction,
            stats.mean_class_fraction,
            correct as f64 / equations.len() as f64
        )?;
        for q in 0..side {
            writeln!(
                classes,
                "{phase},{},{:.12e},{:.12}",
                q + 1,
                stats.variances[q],
                stats.fractions[q]
            )?;
        }
        let mut sums = [0.0; 4];
        let mut counts = [0; 4];
        for (eq, energy) in equations.iter().zip(&stats.normalized_pair_residuals) {
            let b = if eq.b.is_multiple_of(2) {
                eq.b + 1
            } else {
                eq.b - 1
            };
            let raw = eq.a.abs_diff(b);
            let distance = raw.min(config.width - raw);
            let bin = if distance < 8 {
                0
            } else if distance < 16 {
                1
            } else if distance < 32 {
                2
            } else {
                3
            };
            sums[bin] += energy;
            counts[bin] += 1;
        }
        for (i, label) in ["0-7", "8-15", "16-31", "32+"].iter().enumerate() {
            if counts[i] > 0 {
                writeln!(
                    distances,
                    "{phase},{label},{},{:.12}",
                    counts[i],
                    sums[i] / counts[i] as f64
                )?;
            }
        }
        println!(
            "{phase}: mean per-class interaction {:.6}; accuracy {:.6}",
            stats.mean_class_fraction,
            correct as f64 / equations.len() as f64
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::interaction;

    #[test]
    fn additive_grid_has_no_joint_component() {
        let stats = interaction(2, 1, &[3.0, 5.0, 7.0, 9.0]);
        assert_eq!(stats.weighted_fraction, 0.0);
        assert_eq!(stats.mean_class_fraction, 0.0);
    }

    #[test]
    fn balanced_xor_grid_is_entirely_joint() {
        let stats = interaction(2, 1, &[0.0, 1.0, 1.0, 0.0]);
        assert!((stats.mean_class_fraction - 1.0).abs() < 1e-12);
        assert!((stats.weighted_fraction - 1.0).abs() < 1e-12);
    }

    #[test]
    fn constant_grid_has_zero_variance_and_no_nan() {
        let stats = interaction(2, 1, &[4.0; 4]);
        assert_eq!(stats.mean_class_fraction, 0.0);
        assert!(stats.normalized_pair_residuals.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn mixed_additive_and_joint_grid_has_known_fraction() {
        let stats = interaction(2, 1, &[-1.0, -1.0, -1.0, 3.0]);
        assert!((stats.mean_class_fraction - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn class_average_is_not_dominated_by_output_scale() {
        let stats = interaction(2, 2, &[-1.0, -200.0, 1.0, 0.0, 1.0, 0.0, -1.0, 200.0]);
        assert!((stats.mean_class_fraction - 0.5).abs() < 1e-12);
        assert!((stats.weighted_fraction - 1.0 / 20001.0).abs() < 1e-12);
    }
}
