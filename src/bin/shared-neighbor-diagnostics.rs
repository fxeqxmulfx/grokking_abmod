//! Inspect signal mixing and activation variation in saved shared Neighbor runs.

use std::{collections::BTreeMap, env, fs, io::Write, path::Path};

use grokking_abmod::{
    Result,
    gpt::EquationDataset,
    simple::{Activation, Architecture, ReferenceSimpleModel, SimpleConfig},
};

fn normalize(values: &[f64]) -> Vec<f64> {
    // Match the reference's FP32 epsilon promoted to f64 exactly.
    let inverse = (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64
        + f64::from(1e-5_f32))
    .sqrt()
    .recip();
    values.iter().map(|v| inverse * v).collect()
}

fn softmax(values: &[f64]) -> Vec<f64> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<_> = values.iter().map(|v| (v - maximum).exp()).collect();
    let total = exponentials.iter().sum::<f64>();
    exponentials.iter().map(|v| v / total).collect()
}

fn trace(
    file: &mut fs::File,
    phase: &str,
    stage: &str,
    rows: &[Vec<f64>],
    active: Option<f64>,
) -> Result<()> {
    let width = rows[0].len();
    let mut mean = vec![0.0; width];
    let mut mean_direction = vec![0.0; width];
    let mut energy = 0.0;
    let mut rms = 0.0;
    let mut nonzero = 0;
    for row in rows {
        let square = row.iter().map(|v| v * v).sum::<f64>();
        energy += square;
        rms += (square / width as f64).sqrt();
        nonzero += usize::from(square > 0.0);
        for (c, &value) in row.iter().enumerate() {
            mean[c] += value;
            if square > 0.0 {
                mean_direction[c] += value / square.sqrt();
            }
        }
    }
    let count = rows.len() as f64;
    let common = mean.iter().map(|v| v * v).sum::<f64>() / count;
    let variation = if energy == 0.0 {
        0.0
    } else {
        (1.0 - common / energy).clamp(0.0, 1.0)
    };
    let direction_variation = if nonzero == 0 {
        0.0
    } else {
        (1.0 - mean_direction.iter().map(|v| v * v).sum::<f64>() / (nonzero as f64).powi(2))
            .clamp(0.0, 1.0)
    };
    writeln!(
        file,
        "{phase},{stage},{},{:.9},{variation:.9},{direction_variation:.9},{:.9},{}",
        rows.len(),
        rms / count,
        1.0 - nonzero as f64 / count,
        active.map_or_else(String::new, |v| format!("{v:.9}"))
    )?;
    Ok(())
}

fn embedding(config: SimpleConfig, parameters: &[f32], index: usize) -> Vec<f64> {
    let mut row = vec![0.0; config.width];
    row[index] = 1.0;
    for layer in 0..config.embedding_mesh_depth {
        row = (0..config.width)
            .map(|c| {
                let unit = 2 * (layer * config.width + c);
                f64::from(parameters[unit]) * row[c]
                    + f64::from(parameters[unit + 1]) * row[(c + 1) % config.width]
            })
            .collect();
    }
    row
}

fn checkpoint(path: &Path, count: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    if bytes.len() != 4 * count {
        return Err("unexpected checkpoint length".into());
    }
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite checkpoint".into());
    }
    Ok(values)
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: shared-neighbor-diagnostics TRIAL_DIR OUTPUT_DIR".into());
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
        || value("normalization")? != "pre_rmsnorm_blocks_and_final"
        || value("rmsnorm_epsilon")? != "1e-5"
        || value("activation")? != "relu"
    {
        return Err(
            "diagnostics require the corrected pre-RMSNorm/ReLU shared Neighbor model".into(),
        );
    }
    let config = SimpleConfig {
        architecture: Architecture::SharedNeighborMeshMlp,
        activation: Activation::Relu,
        width: value("width")?.parse()?,
        modulus: value("modulus")?.parse()?,
        depth: value("depth")?.parse()?,
        embedding_mesh_depth: value("embedding_mesh_depth")?.parse()?,
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
    let equations = &dataset.split(false)[..512.min(dataset.split(false).len())];
    fs::create_dir_all(output)?;
    let mut traces = fs::File::create(output.join("activation-traces.csv"))?;
    let mut embeddings = fs::File::create(output.join("embedding-spread.csv"))?;
    let mut weights = fs::File::create(output.join("parameter-gradients.csv"))?;
    let mut verification = fs::File::create(output.join("verification.txt"))?;
    writeln!(
        traces,
        "phase,stage,rows,mean_rms,variable_energy_fraction,direction_variation_fraction,zero_vector_fraction,branch_active_fraction"
    )?;
    writeln!(
        embeddings,
        "phase,residue,l2,effective_channels_energy_participation,channels_above_relative_0_001"
    )?;
    writeln!(
        weights,
        "phase,group,layer,parameter_rms,gradient_rms,nonzero_gradient_fraction"
    )?;
    writeln!(
        verification,
        "scope=First {} seeded training equations; f64 scalar diagnostics. Gradients use the first 16 rows.\nvariable_energy_fraction=1 minus squared batch mean divided by mean squared activation norm\ndirection_variation_fraction=1 minus squared mean of unit activation vectors\neffective_channels_energy_participation=squared total energy divided by sum of squared coordinate energies\nrelative_channel_threshold=0.001 of each embedding vector's largest absolute value",
        equations.len()
    )?;
    for (phase, name) in [("initial", "initial-model.bin"), ("trained", "model.bin")] {
        let parameters = checkpoint(&trial.join(name), config.parameter_count())?;
        let vectors: Vec<_> = (0..config.modulus)
            .map(|i| embedding(config, &parameters, i))
            .collect();
        for (index, row) in vectors.iter().enumerate() {
            let energy = row.iter().map(|v| v * v).sum::<f64>();
            let fourth = row.iter().map(|v| v.powi(4)).sum::<f64>();
            let maximum = row.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let effective = if fourth == 0.0 {
                0.0
            } else {
                energy * energy / fourth
            };
            let above = row.iter().filter(|v| v.abs() > 0.001 * maximum).count();
            writeln!(
                embeddings,
                "{phase},{index},{:.9},{effective:.9},{above}",
                energy.sqrt()
            )?;
        }
        let mut hidden: Vec<_> = equations
            .iter()
            .map(|eq| {
                (0..config.width)
                    .map(|c| {
                        vectors[eq.a][c]
                            + if c.is_multiple_of(2) {
                                -vectors[eq.b][c + 1]
                            } else {
                                vectors[eq.b][c - 1]
                            }
                    })
                    .collect()
            })
            .collect();
        trace(&mut traces, phase, "joined", &hidden, None)?;
        for (layer, shape) in config.two_parent_layers().into_iter().enumerate() {
            let offset = 2 * config.width * config.embedding_mesh_depth + 3 * config.width * layer;
            let mut active = 0usize;
            for row in &mut hidden {
                let normalized = normalize(row);
                for (c, value) in row.iter_mut().enumerate() {
                    let [a, b] = shape.parents(c);
                    let unit = offset + 3 * c;
                    let branch = f64::from(parameters[unit]) * normalized[a]
                        + f64::from(parameters[unit + 1]) * normalized[b]
                        + f64::from(parameters[unit + 2]);
                    active += usize::from(branch > 0.0);
                    *value += branch.max(0.0);
                }
            }
            trace(
                &mut traces,
                phase,
                &format!("core_{}", layer + 1),
                &hidden,
                Some(active as f64 / (equations.len() * config.width) as f64),
            )?;
        }
        hidden = hidden.iter().map(|row| normalize(row)).collect();
        trace(&mut traces, phase, "final_rmsnorm", &hidden, None)?;
        for layer in (0..config.embedding_mesh_depth).rev() {
            hidden = hidden
                .iter()
                .map(|row| {
                    (0..config.width)
                        .map(|c| {
                            let previous = (c + config.width - 1) % config.width;
                            f64::from(parameters[2 * (layer * config.width + c)]) * row[c]
                                + f64::from(parameters[2 * (layer * config.width + previous) + 1])
                                    * row[previous]
                        })
                        .collect()
                })
                .collect();
        }
        for row in &mut hidden {
            row.truncate(config.modulus);
        }
        trace(&mut traces, phase, "logits", &hidden, None)?;
        // Softmax ignores a common shift of every class logit. Inspect centered
        // logits and answer probabilities before interpreting raw-logit similarity.
        let centered: Vec<_> = hidden
            .iter()
            .map(|row| {
                let mean = row.iter().sum::<f64>() / row.len() as f64;
                row.iter().map(|v| v - mean).collect()
            })
            .collect();
        trace(&mut traces, phase, "centered_logits", &centered, None)?;
        let probabilities: Vec<_> = hidden.iter().map(|row| softmax(row)).collect();
        trace(
            &mut traces,
            phase,
            "answer_probabilities",
            &probabilities,
            None,
        )?;
        // Division with a=0 has the fixed answer 0 and develops an unusually
        // large class-0 embedding. Inspect nonzero inputs/classes separately.
        let nonzero_logits: Vec<_> = hidden
            .iter()
            .zip(equations)
            .filter(|(_, eq)| eq.a != 0)
            .map(|(row, _)| {
                let mean = row[1..].iter().sum::<f64>() / (row.len() - 1) as f64;
                row[1..].iter().map(|v| v - mean).collect()
            })
            .collect();
        if !nonzero_logits.is_empty() {
            trace(
                &mut traces,
                phase,
                "nonzero_a_centered_logits",
                &nonzero_logits,
                None,
            )?;
            let conditional: Vec<_> = nonzero_logits.iter().map(|row| softmax(row)).collect();
            trace(
                &mut traces,
                phase,
                "nonzero_a_answer_probabilities",
                &conditional,
                None,
            )?;
        }
        let reference = ReferenceSimpleModel::from_parameters(config, parameters.clone())?;
        let mut maximum_error: f64 = 0.0;
        for (index, equation) in equations.iter().take(8).enumerate() {
            let (features, _) = config.encode(std::slice::from_ref(equation))?;
            let logits = reference.logits(&features)?;
            for (&actual, &expected) in hidden[index].iter().zip(&logits[0]) {
                maximum_error = maximum_error.max((actual - expected).abs());
            }
        }
        if maximum_error > 1e-9 {
            return Err(
                format!("diagnostic forward differs from f64 reference: {maximum_error}").into(),
            );
        }
        writeln!(
            verification,
            "{phase}_reference_logit_maximum_absolute_error={maximum_error:.12e}"
        )?;
        let (features, targets) = config.encode(&equations[..16.min(equations.len())])?;
        let (loss, gradients) = reference.loss_and_gradient(&features, &targets)?;
        writeln!(verification, "{phase}_gradient_batch_loss={loss:.9}")?;
        let mut offset = 0;
        for (group, depth, count) in [
            ("embedding", config.embedding_mesh_depth, 2 * config.width),
            ("core", config.depth, 3 * config.width),
        ] {
            for layer in 0..depth {
                let p = &parameters[offset..offset + count];
                let g = &gradients[offset..offset + count];
                writeln!(
                    weights,
                    "{phase},{group},{},{:.9},{:.12e},{:.9}",
                    layer + 1,
                    (p.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / count as f64).sqrt(),
                    (g.iter().map(|v| v * v).sum::<f64>() / count as f64).sqrt(),
                    g.iter().filter(|&&v| v != 0.0).count() as f64 / count as f64
                )?;
                offset += count;
            }
        }
    }
    println!("Saved checked f64 diagnostics to {}", output.display());
    Ok(())
}
