//! Shared deterministic AOgmaNeo scenario for the functional-fidelity harness.
//!
//! This file is compiled into BOTH the `fidelity_dump` example (which prints the
//! scenario output as JSON, for producing/inspecting golden vectors) and the
//! `fidelity` integration test (which diffs the in-process Rust run against the
//! committed C++ golden). It lives under `tests/support/` — a subdirectory, so Cargo
//! does NOT treat it as its own test target — and is pulled in via `#[path = ...]`.
//!
//! ## The parity contract (must match the PyAOgmaNeo generator exactly)
//!
//! The scenario deliberately runs in the *faithful regime* where the Rust port is
//! expected to match upstream AOgmaNeo `645a54a` (see `doc/Divergences.md`):
//!
//! - **`leak = 0.0`** on every decoder/actor (crate default is 0.01 — a genuine
//!   numeric divergence; upstream uses plain softplus).
//! - **`ticks_per_update = 1`** on every layer — tick-gating is the deferred
//!   Rust-only divergence absent from `645a54a`; ticks=1 makes it a no-op.
//! - **single-thread** — the C++ reference must run with `set_num_threads(1)` and the
//!   Rust side with `RAYON_NUM_THREADS=1`, because C++ learn kernels share one global
//!   RNG under OpenMP and are otherwise not bit-reproducible.
//! - **global RNG reset** to `rand_get_state(12345)` (the shared default) before
//!   `init_random`, so weight init draws the same PCG32 stream on both sides.
//!
//! The input is the same `wave(t)` waveform + `unorm8_to_csdr` encoding as
//! `wave_prediction.rs` / `wavy_line_prediction.py`.

use dcc_sph::helpers::{rand_get_state, set_global_state, Int3};
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};
use serde::{Deserialize, Serialize};

/// Number of learning steps in the scenario. Kept modest so the golden fixture is
/// small and the test is fast, but long enough to exercise learning dynamics.
pub const STEPS: usize = 200;

/// Encoder layers (all `ticks_per_update = 1`). Two layers is enough to exercise
/// up/down passes and recurrence-off behavior without a large fixture.
pub const NUM_LAYERS: usize = 2;

/// One timestep's integer observation surface — all exact-comparable.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct StepRecord {
    /// The input CSDR fed this step.
    pub input_cis: Vec<i32>,
    /// The IO-0 prediction CSDR after the step (`get_prediction_cis(0)`).
    pub prediction_cis: Vec<i32>,
    /// The top encoder's hidden CSDR after the step (`get_encoder(top).get_hidden_cis()`).
    pub hidden_cis: Vec<i32>,
}

/// Full scenario output: the per-step integer streams plus the final float
/// prediction activations (compared with tolerance, never exactly).
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct ScenarioOutput {
    pub steps: Vec<StepRecord>,
    /// `get_prediction_acts(0)` at the final step — float, tolerance-compared.
    pub final_prediction_acts: Vec<f32>,
}

/// Target waveform: 1.0 whenever `t` is divisible by 20 or 7, else 0.0.
fn wave(t: usize) -> f32 {
    if t % 20 == 0 || t % 7 == 0 {
        1.0
    } else {
        0.0
    }
}

/// Encode a float in [0, 1] as two 4-bit nibbles (2 columns × 16 cells).
fn unorm8_to_csdr(x: f32) -> [i32; 2] {
    let i = (x * 255.0 + 0.5) as u8 as i32;
    [i & 0x0f, (i >> 4) & 0x0f]
}

/// Build the faithful-regime hierarchy, run `STEPS` learning steps on the fixed
/// waveform, and capture the integer streams + final float acts.
pub fn run_scenario() -> ScenarioOutput {
    // Deterministic RNG: return the global stream to its default so repeated
    // in-process runs are identical and aligned with the C++ reference seed.
    set_global_state(rand_get_state(12345));

    let io_descs = vec![IoDesc {
        size: Int3::new(1, 2, 16),
        io_type: IoType::Prediction,
        num_dendrites_per_cell: 4,
        up_radius: 2,
        down_radius: 2,
        value_size: 64,
        value_num_dendrites_per_cell: 4,
        history_capacity: 64,
    }];

    let layer_descs: Vec<LayerDesc> = (0..NUM_LAYERS)
        .map(|_| LayerDesc {
            hidden_size: Int3::new(5, 5, 64),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: -1, // recurrence off
            down_radius: 2,
            ticks_per_update: 1, // faithful regime: tick-gating disabled
            top_feedback: false, // faithful regime: no goal path exists in the C++
        })
        .collect();

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    // Faithful regime: plain softplus (leak = 0) on every decoder/actor.
    for lp in h.params.layers.iter_mut() {
        lp.decoder.leak = 0.0;
    }
    for ip in h.params.ios.iter_mut() {
        ip.decoder.leak = 0.0;
        ip.actor.leak = 0.0;
    }

    let mut steps = Vec::with_capacity(STEPS);
    let mut input_cis = vec![0i32; 2];

    for t in 0..STEPS {
        let csdr = unorm8_to_csdr(wave(t));
        input_cis[0] = csdr[0];
        input_cis[1] = csdr[1];

        h.step(&[&input_cis], true, 0.0, 0.0);

        steps.push(StepRecord {
            input_cis: input_cis.clone(),
            prediction_cis: h.get_prediction_cis(0).to_vec(),
            hidden_cis: h.get_encoder(NUM_LAYERS - 1).get_hidden_cis().to_vec(),
        });
    }

    ScenarioOutput {
        steps,
        final_prediction_acts: h.get_prediction_acts(0).to_vec(),
    }
}
