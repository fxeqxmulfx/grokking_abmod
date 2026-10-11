//! Shared geometry flags for text training and modular arithmetic experiments.

use crate::cli::value;
use grokking_abmod::{Result, gpt::GptConfig};

pub const HELP: &str = "\
    --embedding mesh|table  Tied embedding parameterization (default: mesh)\n\
    --connectivity neighbor|butterfly  Mesh parents (default: neighbor)\n\
    --embedding-form NAME  parity (default), circle, fourier, orthogonal, simplex,\n\
                           equiangular, antipodal, hypercube, signed-permutations, sphere\n\
    --embedding-order NAME token (default), residue, discrete-log; arithmetic order is grok-only\n\
    --embedding-frequency N  First cyclic frequency (default: 1)\n\
    --embedding-harmonics N  Consecutive Fourier pairs (default: 1)\n\
    --equiangular-overlap X  Off-diagonal cosine, -1/(V-1)<=X<=1 (default: 0)\n\
    --embedding-norm NAME  none (default), unit, rms; effective tied vectors\n\
    --freeze-embedding     Freeze embedding parameters throughout AdamW\n\
    --preserve-features N  Inject N code coordinates and protect them through residual blocks\n\
    --geometry-seed N      Sphere-code seed (default: 0; grok arithmetic order uses --seed)\n\
    --rope 0|1            Slice RoPE (default: 1)\n\
    --qk-norm 0|1         Per-head RMS QK normalization (default: 1)\n\
    --xsa 0|1             Exclusive projection after attention (default: 1)";

pub fn parse_option(
    config: &mut GptConfig,
    flag: &str,
    arguments: &mut impl Iterator<Item = String>,
) -> Result<bool> {
    match flag {
        "--embedding" => config.embedding = value(arguments, flag)?,
        "--connectivity" => config.connectivity = value(arguments, flag)?,
        "--embedding-form" => config.geometry.form = value(arguments, flag)?,
        "--embedding-order" => config.geometry.order = value(arguments, flag)?,
        "--embedding-frequency" => config.geometry.frequency = value(arguments, flag)?,
        "--embedding-harmonics" => config.geometry.harmonics = value(arguments, flag)?,
        "--equiangular-overlap" => config.geometry.overlap = value(arguments, flag)?,
        "--embedding-norm" => config.geometry.normalization = value(arguments, flag)?,
        "--freeze-embedding" => config.geometry.frozen = true,
        "--preserve-features" => config.geometry.preserve_features = value(arguments, flag)?,
        "--geometry-seed" => config.geometry.seed = value(arguments, flag)?,
        "--rope" => config.rope = switch(arguments, flag)?,
        "--qk-norm" => config.qk_norm = switch(arguments, flag)?,
        "--xsa" => config.xsa = switch(arguments, flag)?,
        _ => return Ok(false),
    }
    Ok(true)
}

fn switch(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<bool> {
    match value::<u32>(arguments, flag)? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(format!("{flag} requires 0 or 1").into()),
    }
}

pub fn metadata(config: GptConfig) -> String {
    let g = config.geometry;
    format!(
        "embedding_form={}\nembedding_order={}\nembedding_frequency={}\nembedding_harmonics={}\nequiangular_overlap={}\nembedding_norm={}\nfreeze_embedding={}\npreserve_features={}\ngeometry_modulus={}\ngeometry_seed={}\nrope={}\nqk_norm_enabled={}\nxsa_enabled={}\ngeometry_supplies_arithmetic_order={}\n",
        g.form,
        g.order,
        g.frequency,
        g.harmonics,
        g.overlap,
        g.normalization,
        g.frozen,
        g.preserve_features,
        g.modulus,
        g.seed,
        config.rope,
        config.qk_norm,
        config.xsa,
        g.order != grokking_abmod::gpt::EmbeddingOrder::Token
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_flags_reject_malformed_switches_and_record_their_meaning() {
        let mut config = GptConfig::default();
        for (flag, text) in [
            ("--embedding", "table"),
            ("--embedding-form", "fourier"),
            ("--embedding-order", "discrete-log"),
            ("--embedding-harmonics", "3"),
            ("--preserve-features", "2"),
            ("--geometry-seed", "18446744073709551615"),
            ("--rope", "0"),
            ("--qk-norm", "0"),
            ("--xsa", "0"),
        ] {
            assert!(parse_option(&mut config, flag, &mut [text.to_string()].into_iter()).unwrap());
        }
        assert!(parse_option(&mut config, "--freeze-embedding", &mut std::iter::empty()).unwrap());
        assert!(config.geometry.frozen && !config.rope && !config.qk_norm && !config.xsa);
        assert_eq!(config.geometry.seed, u64::MAX);
        assert!(metadata(config).contains("geometry_supplies_arithmetic_order=true\n"));
        assert!(parse_option(&mut config, "--rope", &mut ["2".into()].into_iter()).is_err());
        assert!(
            parse_option(
                &mut config,
                "--embedding-form",
                &mut ["unknown".into()].into_iter()
            )
            .is_err()
        );
        assert!(parse_option(&mut config, "--embedding-norm", &mut std::iter::empty()).is_err());
        assert!(!parse_option(&mut config, "--unknown", &mut std::iter::empty()).unwrap());
    }
}
