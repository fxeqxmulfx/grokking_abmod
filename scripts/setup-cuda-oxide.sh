#!/usr/bin/env bash
set -euo pipefail

oxide_project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$oxide_project_root"

for oxide_command in cargo rustup git python3 clang-23 clang++-23 llc-23; do
    if ! command -v "$oxide_command" >/dev/null 2>&1; then
        printf 'Required command is missing: %s\n' "$oxide_command" >&2
        exit 1
    fi
done

mkdir -p target
cargo metadata --format-version 1 --locked > target/cuda-oxide-metadata.json
oxide_upstream_root=$(python3 - target/cuda-oxide-metadata.json <<'PY'
import json
import pathlib
import sys

with open(sys.argv[1]) as metadata_file:
    metadata = json.load(metadata_file)
device = next(package for package in metadata["packages"] if package["name"] == "cuda-device")
root = pathlib.Path(device["manifest_path"]).parent
while not (root / "cuda-oxide/crates/rustc-codegen-cuda/Cargo.toml").is_file():
    if root == root.parent:
        raise SystemExit("Cannot locate the cuda-rust checkout resolved by Cargo")
    root = root.parent
print(root)
PY
)
oxide_revision=$(git -C "$oxide_upstream_root" rev-parse HEAD)
oxide_backend_source="$oxide_project_root/target/cuda-oxide-source"
oxide_backend_target="$oxide_project_root/target/cuda-oxide-backend"

if [ ! -d "$oxide_backend_source" ]; then
    git clone --no-hardlinks --no-checkout "$oxide_upstream_root" "$oxide_backend_source"
fi
# This is a generated checkout under target/, refreshed on dependency updates.
git -C "$oxide_backend_source" fetch --no-tags "$oxide_upstream_root" "$oxide_revision"
git -C "$oxide_backend_source" -c advice.detachedHead=false checkout --force --detach "$oxide_revision"

cargo build \
    --manifest-path "$oxide_backend_source/cuda-oxide/crates/rustc-codegen-cuda/Cargo.toml" \
    --target-dir "$oxide_backend_target" \
    --release --locked
cargo install \
    --path "$oxide_backend_source/cuda-oxide/crates/cargo-oxide" \
    --target-dir "$oxide_backend_target" \
    --locked --force
rustc --version --verbose > "$oxide_backend_target/rustc-version.txt"
printf '%s\n' "$oxide_revision" > "$oxide_backend_target/source-rev.txt"

printf '\nCUDA Oxide ready at %s.\n' "$oxide_revision"
printf 'Build: cargo oxide build --arch sm_86\nRun: cargo oxide run\n'
