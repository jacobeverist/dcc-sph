//! Assertions about whether the library **learns**, kept out of CI on purpose.
//!
//! ```bash
//! cargo test --release --test learning -- --nocapture
//! ```
//!
//! `test = false` in `Cargo.toml` keeps this target out of `cargo test`,
//! `cargo test --all-targets` and `cargo test --release`. Naming it with
//! `--test learning` is the only way to run it, so it cannot be re-enabled by
//! deleting an attribute or forgetting a flag. CI still *compiles* it
//! (`cargo test --no-run --test learning`), so it cannot rot unnoticed — it just
//! never runs there.
//!
//! ## Why this file exists
//!
//! Experimentation is the point of this repository, and a learning threshold in CI
//! turns every experiment into a build break. Worse, it puts quiet pressure on the
//! measurements to be flattering, which is the exact failure the demos' baselines
//! exist to prevent. Whether a configuration learns belongs in an offline
//! `--repeat` / `--sweep` run with someone reading the spread.
//!
//! ## What stayed in CI, and why that is not the same thing
//!
//! `tests/goal_conditioned.rs` still runs on every push, including
//! `default_path_is_bit_identical`, which trains for 60 steps and hashes the
//! result. That is not a learning assertion: it asserts *sameness* against
//! constants measured before the goal path landed, and it fails only if behaviour
//! changed. The distinction is between "did this stay identical" — deterministic,
//! and exactly what CI is for — and "did this get good enough", which is a
//! judgement about a configuration and belongs here.
//!
//! Every test below is deterministic at a fixed seed. They are slow and
//! threshold-bearing, not flaky.

use dcc_sph::helpers::{rand_get_state, rand_step, set_global_state, Int3};
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

// ---------------------------------------------------------------------------
// Goal conditioning
// ---------------------------------------------------------------------------

/// Two goal CSDRs as far apart as the top hidden layer allows.
fn goal_pair(h: &Hierarchy) -> (Vec<i32>, Vec<i32>) {
    let size = h.get_top_hidden_size();
    let columns = (size.x * size.y) as usize;
    (vec![0i32; columns], vec![size.z - 1; columns])
}

fn goal_hierarchy() -> Hierarchy {
    set_global_state(rand_get_state(7));

    let io_descs = vec![IoDesc {
        size: Int3::new(1, 1, 2),
        io_type: IoType::Prediction,
        ..Default::default()
    }];
    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(4, 4, 16),
        top_feedback: true,
        ..Default::default()
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);
    h
}

/// **The load-bearing test for the whole goal path.** Every test in
/// `tests/goal_conditioned.rs` would still pass against an implementation that
/// accepted a goal and quietly discarded it; this one would not.
///
/// The task is built so the goal is the *only* route to the answer. At each step it
/// picks a bit, passes the goal encoding that bit, and makes the *next* observation
/// carry it. The observation fed to the encoder therefore encodes the *previous*
/// bit and says nothing about the current one — only the goal does. A hierarchy
/// ignoring the goal is left guessing, and 50% is its ceiling.
///
/// The one-step offset is not incidental: the decoder learns from the goal that was
/// current when it made the prediction being corrected, the same pairing every
/// lower layer uses for feedback from the layer above.
#[test]
fn the_goal_reaches_the_decoder() {
    let mut h = goal_hierarchy();
    let (lo, hi) = goal_pair(&h);

    let mut state = rand_get_state(99);
    let mut input = vec![0i32; 1];
    let mut correct = 0usize;
    let mut scored = 0usize;

    const STEPS: usize = 4000;
    const SCORE_FROM: usize = 3000;

    for t in 0..STEPS {
        let bit = (rand_step(&mut state) % 2) as i32;
        let goal = if bit == 1 { &hi } else { &lo };

        h.step_with_goal(&[&input], goal, true, 0.0, 0.0);

        // The activate pass just ran against *this* goal, so the prediction on the
        // table is the one for the observation this goal is about to produce.
        if t >= SCORE_FROM {
            scored += 1;
            if h.get_prediction_cis(0)[0] == bit {
                correct += 1;
            }
        }

        input[0] = bit;
    }

    let accuracy = correct as f64 / scored as f64;
    println!("goal-conditioned accuracy over the last {scored} steps: {accuracy:.3}");
    assert!(
        accuracy > 0.9,
        "goal-conditioned prediction accuracy {accuracy:.3} over the last {scored} \
         steps; the observation carries only the previous bit, so anything near 0.5 \
         means the goal never reached the decoder"
    );
}

/// The goal must change the *current* prediction, not merely correlate with it over
/// training. Same trained hierarchy, same history, two different goals.
#[test]
fn switching_the_goal_switches_the_prediction() {
    let mut h = goal_hierarchy();
    let (lo, hi) = goal_pair(&h);

    let mut state = rand_get_state(99);
    let mut input = vec![0i32; 1];

    for _ in 0..4000 {
        let bit = (rand_step(&mut state) % 2) as i32;
        let goal = if bit == 1 { &hi } else { &lo };
        h.step_with_goal(&[&input], goal, true, 0.0, 0.0);
        input[0] = bit;
    }

    // Fork the trained hierarchy and drive the two copies apart with the goal
    // alone — identical weights, identical history, identical observation.
    let mut with_lo = h.clone();
    let mut with_hi = h.clone();
    with_lo.step_with_goal(&[&input], &lo, false, 0.0, 0.0);
    with_hi.step_with_goal(&[&input], &hi, false, 0.0, 0.0);

    assert_eq!(with_lo.get_prediction_cis(0)[0], 0);
    assert_eq!(with_hi.get_prediction_cis(0)[0], 1);
}

// ---------------------------------------------------------------------------
// Ordinary sequence prediction
// ---------------------------------------------------------------------------

/// Prediction accuracy on a fixed repeating sequence must not regress with more
/// training. Moved here from `tests/smoke_test.rs`, where it predated the demos: it
/// is the same kind of claim as the two above, and belongs under the same rule.
#[test]
fn prediction_improves_on_a_repeating_sequence() {
    let io_size = Int3::new(4, 4, 8);
    let io_descs = vec![IoDesc {
        size: io_size,
        io_type: IoType::Prediction,
        num_dendrites_per_cell: 4,
        up_radius: 2,
        down_radius: 2,
        value_size: 64,
        value_num_dendrites_per_cell: 4,
        history_capacity: 64,
    }];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(4, 4, 16),
        num_dendrites_per_cell: 4,
        up_radius: 2,
        recurrent_radius: -1,
        down_radius: 2,
        ticks_per_update: 1,
        top_feedback: false,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    let num_cols = (io_size.x * io_size.y) as usize;
    let col_size = io_size.z as usize;

    // A repeating sequence of four distinct patterns.
    let patterns: Vec<Vec<i32>> = (0..4)
        .map(|p| (0..num_cols).map(|c| ((p + c) % col_size) as i32).collect())
        .collect();

    let score = |h: &mut Hierarchy| {
        let mut correct = 0usize;
        for (i, pattern) in patterns.iter().enumerate() {
            let next = &patterns[(i + 1) % patterns.len()];
            h.step(&[pattern], false, 0.0, 0.0);
            correct += h
                .get_prediction_cis(0)
                .iter()
                .zip(next.iter())
                .filter(|(p, n)| p == n)
                .count();
        }
        correct
    };

    let train = |h: &mut Hierarchy, passes: usize| {
        for _ in 0..passes {
            for pattern in &patterns {
                h.step(&[pattern], true, 0.0, 0.0);
            }
        }
    };

    train(&mut h, 50);
    let correct_early = score(&mut h);
    train(&mut h, 100);
    let correct_late = score(&mut h);

    let total = num_cols * patterns.len();
    println!("repeating sequence: early {correct_early}/{total}, late {correct_late}/{total}");
    assert!(
        correct_late >= correct_early,
        "prediction accuracy regressed: early={correct_early} late={correct_late} out of {total}"
    );
}
