// Shared scaffolding for the demo examples.
//
// Cargo examples cannot depend on one another, so this module is pulled into each
// demo with `#[path = "support/mod.rs"] mod support;` — the same idiom
// `examples/fidelity_dump.rs` uses for `tests/support/`. A directory under
// `examples/` with no `main.rs` is not auto-discovered as a target, so this
// compiles only where it is explicitly included.
//
// Every demo compiles the whole module, so most of it is dead code in any given
// target; the allow below is what keeps `cargo clippy --all-targets` clean.
#![allow(dead_code)]

pub mod args;
pub mod encode;
pub mod encoder_probe;
pub mod env;
pub mod metrics;
pub mod report;
pub mod rng;

#[cfg(feature = "macroquad-demos")]
pub mod viz;
