//! Functional-fidelity test: Rust dcc_sph vs. upstream AOgmaNeo C++.
//!
//! Runs the shared deterministic scenario (`tests/support/fidelity_scenario.rs`)
//! in-process and diffs it against a committed golden vector generated from
//! **PyAOgmaNeo rebuilt at AOgmaNeo `645a54a` with `USE_STD_MATH`, single-threaded**
//! (see `fidelity/README.md`). Because both sides use the *identical* PCG32 RNG and
//! the scenario pins the faithful regime (`leak=0`, `ticks=1`, single-thread), the
//! integer CSDR streams are expected to match **exactly**:
//!
//! - `prediction_cis` and `hidden_cis` per step  → exact equality (ULP-immune).
//! - `final_prediction_acts`                      → tolerance comparison only
//!   (floats; the actor-init FP-op-order caveat and any residual math differences
//!   live here — see `doc/MethodFidelity.md`).
//!
//! Run single-threaded to honor the parity contract:
//! ```text
//! RAYON_NUM_THREADS=1 cargo test -p dcc_sph --test fidelity
//! ```
//!
//! If the golden fixture is absent (PyAOgmaNeo not yet rebuilt on this machine), the
//! test prints a SKIP notice and passes, so CI stays green without a C++ toolchain.
//! To (re)generate the fixture, follow `fidelity/README.md`.

#[path = "support/fidelity_scenario.rs"]
mod scenario;

use std::path::PathBuf;

use scenario::ScenarioOutput;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wave_fidelity_golden.json")
}

#[test]
fn matches_cpp_golden() {
    let actual = scenario::run_scenario();

    let path = fixture_path();
    let Ok(golden_str) = std::fs::read_to_string(&path) else {
        eprintln!(
            "SKIP matches_cpp_golden: golden fixture {} not present.\n\
             Rebuild PyAOgmaNeo @645a54a and regenerate — see fidelity/README.md.",
            path.display()
        );
        return;
    };
    let golden: ScenarioOutput =
        serde_json::from_str(&golden_str).expect("golden fixture is valid ScenarioOutput JSON");

    assert_eq!(
        actual.steps.len(),
        golden.steps.len(),
        "step count differs (Rust {} vs C++ golden {})",
        actual.steps.len(),
        golden.steps.len(),
    );

    // Integer CSDR streams: exact per-step equality, reporting the FIRST divergence
    // so a failure pinpoints where fidelity breaks.
    for (t, (a, g)) in actual.steps.iter().zip(&golden.steps).enumerate() {
        assert_eq!(
            a.prediction_cis, g.prediction_cis,
            "prediction_cis diverged at step {t}\n  rust = {:?}\n  cpp  = {:?}",
            a.prediction_cis, g.prediction_cis,
        );
        assert_eq!(
            a.hidden_cis, g.hidden_cis,
            "hidden_cis diverged at step {t}\n  rust = {:?}\n  cpp  = {:?}",
            a.hidden_cis, g.hidden_cis,
        );
    }

    // Float acts: tolerance only. Empirically the max abs difference is ~8.4e-3
    // (measured against the C++ golden), even though every integer CSDR above matches
    // exactly — the softmax/transcendental float ops accumulate small ordering
    // differences that don't change the arg-max. The tolerance is set above that
    // observed noise floor so it still catches a *gross* activation divergence.
    assert_eq!(
        actual.final_prediction_acts.len(),
        golden.final_prediction_acts.len(),
        "final act vector length differs",
    );
    let max_diff = actual
        .final_prediction_acts
        .iter()
        .zip(&golden.final_prediction_acts)
        .map(|(a, g)| (a - g).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 2e-2,
        "final_prediction_acts diverged beyond float-noise tolerance: max abs diff {max_diff}",
    );
}

/// Determinism guard: the Rust scenario is fully reproducible in-process (the global
/// RNG reset in `run_scenario` makes back-to-back runs identical).
#[test]
fn scenario_is_deterministic() {
    let a = scenario::run_scenario();
    let b = scenario::run_scenario();
    assert_eq!(a, b, "same scenario must produce identical output across runs");
}
