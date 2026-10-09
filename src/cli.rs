//! Shared value parsing for the example and benchmark binaries.

use std::str::FromStr;

use crate::Result;

pub fn value<T: FromStr>(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<T> {
    let text = arguments
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))?;
    text.parse()
        .map_err(|_| format!("invalid value for {flag}: {text}").into())
}
