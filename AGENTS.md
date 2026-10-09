# Agent instructions

## Repository language

Keep repository content, documentation, code comments, and commit messages in English.

## Changes to CUDA Rust

The sibling repository is `../cuda-rust` (`/home/misha/rust_projects/cuda-rust`).
You are authorized to edit it on the `experiments` branch to fix bugs or make
improvements needed for this project. Routine edits within this scope do not
require separate user confirmation.

Before editing, verify that the sibling checkout is on `experiments`, read its
applicable `AGENTS.md` instructions, and preserve unrelated user changes.
Run checks from the workspace that owns the affected CUDA Rust component.

When validating sibling changes, make this project's dependencies and codegen
backend use the same modified local checkout. Git dependencies pinned in
`Cargo.lock` do not automatically pick up local edits to the sibling repository.

## Safe client code

Actively improve CUDA Rust when doing so allows this project's client code to
use sound, safe Rust APIs with fewer `unsafe` call sites. Treat such improvements
as a priority, even when the existing client code already works.

Prefer library APIs that enforce safety through ownership, borrowing, typed
handles, checked launch contracts, and validation. When a sound library change
can remove client-side `unsafe`, implement it in `../cuda-rust` and adapt the
client to use it.

A safe API must enforce all safety preconditions itself. Wrapping unchecked
operations in a safe function or removing an `unsafe` annotation is not enough.
Keep `unsafe` explicit where the library cannot establish the required guarantees.

Validate safety changes with targeted tests, including invalid usage where
applicable, and build and run this project's CUDA example against the changed
library.
