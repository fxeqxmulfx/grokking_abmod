//! Audit paired unit-norm shared Neighbor initializers before CUDA training.

use std::{env, fs, io::Write, path::Path};

use grokking_abmod::{
    Result,
    simple::{Architecture, FullMeshInitialization, SharedEmbeddingInitialization, SimpleConfig},
};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("usage: shared-embedding-initialization OUTPUT_DIR".into());
    }
    let output = Path::new(&args[0]);
    fs::create_dir_all(output)?;
    let mut vectors = fs::File::create(output.join("initial-embedding-spread.csv"))?;
    let mut summary = fs::File::create(output.join("initialization-summary.csv"))?;
    writeln!(
        vectors,
        "mode,seed,residue,l2,effective_channels,channels_covering_99pct_energy"
    )?;
    writeln!(
        summary,
        "mode,seed,parameters,min_norm,max_norm,mean_effective_channels,mean_channels_covering_99pct_energy,effective_gram_rank"
    )?;
    let base = SimpleConfig {
        architecture: Architecture::SharedNeighborMeshMlp,
        modulus: 97,
        width: 128,
        depth: 64,
        embedding_mesh_depth: 32,
        full_mesh_initialization: FullMeshInitialization::IdentityHidden,
        ..SimpleConfig::default()
    };
    let offset = 2 * base.width * base.embedding_mesh_depth;
    for seed in [11, 17, 23] {
        let legacy = base.initial_parameters(seed);
        for mode in [
            SharedEmbeddingInitialization::UnitNormWeak,
            SharedEmbeddingInitialization::UnitNormStrong,
        ] {
            let config = SimpleConfig {
                shared_embedding_initialization: mode,
                ..base
            };
            config.validate()?;
            let parameters = config.initial_parameters(seed);
            if parameters[offset..] != legacy[offset..] {
                return Err("paired core weights differ".into());
            }
            let directory = output.join(format!("{mode}-seed{seed}"));
            fs::create_dir_all(&directory)?;
            let mut file = fs::File::create(directory.join("initial-model.bin"))?;
            for p in &parameters {
                file.write_all(&p.to_le_bytes())?;
            }
            let mut rows = Vec::with_capacity(config.modulus);
            let (mut minimum, mut maximum, mut effective_sum, mut cover_sum) =
                (f64::INFINITY, 0.0_f64, 0.0, 0usize);
            for class in 0..config.modulus {
                let mut row = vec![0.0_f64; config.width];
                row[class] = 1.0;
                for layer in 0..config.embedding_mesh_depth {
                    row = (0..config.width)
                        .map(|c| {
                            let unit = 2 * (layer * config.width + c);
                            f64::from(parameters[unit]) * row[c]
                                + f64::from(parameters[unit + 1]) * row[(c + 1) % config.width]
                        })
                        .collect();
                }
                let energy = row.iter().map(|v| v * v).sum::<f64>();
                let norm = energy.sqrt();
                if !norm.is_finite() || (norm - 1.0).abs() > 1e-6 {
                    return Err(format!(
                        "bad initial norm: {mode}, seed {seed}, class {class}, {norm}"
                    )
                    .into());
                }
                let effective = energy * energy / row.iter().map(|v| v.powi(4)).sum::<f64>();
                let mut energies: Vec<_> = row.iter().map(|v| v * v).collect();
                energies.sort_by(|a, b| b.total_cmp(a));
                let (mut covered, mut count) = (0.0, 0);
                for value in energies {
                    covered += value;
                    count += 1;
                    if covered >= 0.99 * energy {
                        break;
                    }
                }
                minimum = minimum.min(norm);
                maximum = maximum.max(norm);
                effective_sum += effective;
                cover_sum += count;
                writeln!(
                    vectors,
                    "{mode},{seed},{class},{norm:.12},{effective:.9},{count}"
                )?;
                rows.push(row);
            }
            let trace = rows
                .iter()
                .flat_map(|row| row.iter())
                .map(|v| v * v)
                .sum::<f64>();
            let frobenius = rows
                .iter()
                .flat_map(|a| {
                    rows.iter()
                        .map(move |b| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>().powi(2))
                })
                .sum::<f64>();
            let rank = trace * trace / frobenius;
            writeln!(
                summary,
                "{mode},{seed},{},{minimum:.12},{maximum:.12},{:.9},{:.9},{rank:.9}",
                config.parameter_count(),
                effective_sum / config.modulus as f64,
                cover_sum as f64 / config.modulus as f64
            )?;
            println!(
                "{mode}, seed {seed}: mean effective channels {:.3}, norm {minimum:.8}..{maximum:.8}, Gram energy rank {rank:.3}",
                effective_sum / config.modulus as f64
            );
        }
    }
    Ok(())
}
