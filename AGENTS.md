# Agent instructions

## Repository language

Keep repository content, documentation, code comments, and commit messages in English.

## Python execution

Always run Python scripts through `uv`, including temporary analysis scripts.
For repository scripts, declare dependencies in PEP 723 inline metadata, keep
their `.py.lock` files in Git, and run `uv run --locked scripts/<name>.py`.
Do not rely on packages installed in the system Python environment.

## Optional local instructions

If `optional.md` is present, read and follow it. It contains the instructions
for changing the sibling CUDA Rust repository, publishing its patches, and
improving safe client APIs. This file is intentionally ignored by Git.
