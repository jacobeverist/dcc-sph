//! Emit the shared fidelity scenario's output as JSON.
//!
//! Used to inspect the Rust side of the functional-fidelity harness and to compare
//! against the PyAOgmaNeo-generated golden vector. Run single-threaded for the
//! faithful-regime parity contract:
//!
//! ```text
//! RAYON_NUM_THREADS=1 cargo run -p dcc_sph --release --example fidelity_dump
//! ```
//!
//! The C++ golden fixture is produced by `fidelity/cpp/generate_golden.cpp`, compiled
//! against an AOgmaNeo checkout at commit `645a54a` by `fidelity/build_and_generate.sh`.
//! See `fidelity/README.md`. (This note used to name a `generate_golden.py` driving
//! PyAOgmaNeo — that approach was replaced by compiling the C++ directly.)

#[path = "../tests/support/fidelity_scenario.rs"]
mod scenario;

fn main() {
    let out = scenario::run_scenario();
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
