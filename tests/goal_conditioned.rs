//! Tests for the RUST-ONLY goal-conditioned step (`LayerDesc::top_feedback` and
//! `Hierarchy::step_with_goal`).
//!
//! **The fidelity harness cannot check any of this.** `tests/fidelity.rs` diffs
//! against a golden fixture generated from AOgmaNeo `645a54a`, and that C++ has no
//! goal path at all — nothing to generate a golden from. So this file carries the
//! whole verification burden for the feature, and it is deliberately split in two:
//!
//! - `default_path_is_bit_identical` guards what the change must *not* do. The two
//!   constants were measured against the tree immediately before the feature
//!   landed, so they are a real before/after comparison rather than a snapshot of
//!   whatever the code happens to do now.
//! - Everything below it guards what the feature must actually do — above all
//!   `the_goal_reaches_the_decoder`, without which every other test here would
//!   still pass against an implementation that accepted a goal and ignored it.

use dcc_sph::helpers::{
    rand_get_state, set_global_state, Int3, SliceReader, StreamWriter, VecWriter,
};
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

fn fnv1a(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// A hierarchy exercising both decoder and actor paths, feedback between two
/// layers, and the anticipation pass — i.e. every branch the goal work touched.
fn reference_hierarchy() -> Hierarchy {
    set_global_state(rand_get_state(20260812));

    let io_descs = vec![
        IoDesc { size: Int3::new(4, 4, 8), io_type: IoType::Prediction, ..Default::default() },
        IoDesc { size: Int3::new(1, 1, 5), io_type: IoType::Action, ..Default::default() },
    ];
    let layer_descs = vec![
        LayerDesc { hidden_size: Int3::new(5, 5, 16), ..Default::default() },
        LayerDesc { hidden_size: Int3::new(4, 4, 16), ..Default::default() },
    ];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);
    h
}

/// The default (non-goal-conditioned) path must be bit-identical to the tree before
/// `top_feedback` existed.
///
/// Both constants were captured by running exactly this scenario against the parent
/// commit. The behaviour hash covers 60 steps of both ports' outputs; the weight
/// hash covers everything learning touched. `write` is deliberately *not* hashed —
/// version 2 inserts the structural flag into that stream, so its bytes are
/// expected to move and would mask a real behavioural change behind an explicable
/// one.
#[test]
fn default_path_is_bit_identical() {
    const BEHAVIOUR: u64 = 0x42ae_e08f_6222_ed82;
    const WEIGHTS: u64 = 0x7196_7e53_0834_ebe5;

    let mut h = reference_hierarchy();
    assert!(!h.has_top_feedback());

    let mut hash = FNV_OFFSET;
    let mut obs = vec![0i32; 16];
    let mut act = vec![0i32; 1];

    for t in 0..60usize {
        for i in 0..16 {
            obs[i] = ((t * 7 + i * 3) % 8) as i32;
        }
        h.step(&[&obs, &act], true, ((t % 5) as f32) * 0.25 - 0.5, 0.0);

        for &c in h.get_prediction_cis(0) {
            hash = fnv1a(&c.to_le_bytes(), hash);
        }
        for &c in h.get_prediction_cis(1) {
            hash = fnv1a(&c.to_le_bytes(), hash);
        }
        act[0] = h.get_prediction_cis(1)[0];
    }

    assert_eq!(
        hash, BEHAVIOUR,
        "the default step path changed behaviour; `top_feedback: false` must be \
         bit-identical to the tree before the goal path existed"
    );

    let mut w = VecWriter::new();
    h.write_weights(&mut w);
    assert_eq!(fnv1a(&w.data, FNV_OFFSET), WEIGHTS, "learned weights changed on the default path");
}

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

/// **The load-bearing test.** Everything else here would still pass against an
/// implementation that accepted a goal and quietly discarded it.
///
/// The task is built so the goal is the *only* route to the answer. At each step
/// the demo picks a bit, passes the goal that encodes it, and makes the *next*
/// observation carry that bit. The observation fed to the encoder therefore encodes
/// the *previous* bit and says nothing about the current one; only the goal does.
/// A hierarchy ignoring the goal is left guessing, and 50% is the ceiling.
///
/// Note the one-step offset is not incidental: the decoder learns from the goal
/// that was current when it made the prediction being corrected, which is the same
/// pairing every lower layer uses for feedback from the layer above.
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
        let bit = (dcc_sph::helpers::rand_step(&mut state) % 2) as i32;
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
        let bit = (dcc_sph::helpers::rand_step(&mut state) % 2) as i32;
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

#[test]
fn top_feedback_adds_a_visible_layer_only_at_the_top() {
    set_global_state(rand_get_state(3));
    let io_descs = vec![IoDesc { size: Int3::new(2, 2, 4), ..Default::default() }];
    let plain = vec![
        LayerDesc { hidden_size: Int3::new(4, 4, 8), ..Default::default() },
        LayerDesc { hidden_size: Int3::new(4, 4, 8), ..Default::default() },
    ];
    let mut goal = plain.clone();
    goal[1].top_feedback = true;

    let mut a = Hierarchy::new();
    a.init_random(&io_descs, &plain);
    let mut b = Hierarchy::new();
    b.init_random(&io_descs, &goal);

    // Layer 0 has feedback from layer 1 either way.
    assert_eq!(a.get_decoder(0, 0).get_num_visible_layers(), 2);
    assert_eq!(b.get_decoder(0, 0).get_num_visible_layers(), 2);
    // Only the top layer's arity moves.
    assert_eq!(a.get_decoder(1, 0).get_num_visible_layers(), 1);
    assert_eq!(b.get_decoder(1, 0).get_num_visible_layers(), 2);

    assert!(!a.has_top_feedback());
    assert!(b.has_top_feedback());
}

/// A single-layer hierarchy is its own top layer, so the goal lands on the IO
/// decoders directly — the configuration both stacking demos use.
#[test]
fn a_single_layer_hierarchy_is_its_own_top() {
    let h = goal_hierarchy();
    assert_eq!(h.get_num_layers(), 1);
    assert_eq!(h.get_decoder(0, 0).get_num_visible_layers(), 2);
    assert_eq!(h.get_top_hidden_size(), Int3::new(4, 4, 16));
    assert_eq!(h.get_top_hidden_cis().len(), 16);
}

#[test]
#[should_panic(expected = "only meaningful on the topmost layer")]
fn top_feedback_below_the_top_is_rejected() {
    set_global_state(rand_get_state(3));
    let io_descs = vec![IoDesc { size: Int3::new(2, 2, 4), ..Default::default() }];
    let layer_descs = vec![
        LayerDesc { hidden_size: Int3::new(4, 4, 8), top_feedback: true, ..Default::default() },
        LayerDesc { hidden_size: Int3::new(4, 4, 8), ..Default::default() },
    ];
    Hierarchy::new().init_random(&io_descs, &layer_descs);
}

#[test]
#[should_panic(expected = "call `step_with_goal`")]
fn plain_step_on_a_goal_conditioned_hierarchy_panics() {
    let mut h = goal_hierarchy();
    h.step(&[&[0i32][..]], true, 0.0, 0.0);
}

#[test]
#[should_panic(expected = "nowhere to put a goal")]
fn a_goal_on_a_plain_hierarchy_panics() {
    let mut h = reference_hierarchy();
    let obs = vec![0i32; 16];
    let act = vec![0i32; 1];
    h.step_with_goal(&[&obs, &act], &[0i32; 16], true, 0.0, 0.0);
}

#[test]
#[should_panic(expected = "expected a 16-column goal")]
fn a_wrong_length_goal_panics() {
    let mut h = goal_hierarchy();
    h.step_with_goal(&[&[0i32][..]], &[0i32; 4], true, 0.0, 0.0);
}

#[test]
fn serialisation_round_trips_the_goal_path() {
    let mut h = goal_hierarchy();
    let (_, hi) = goal_pair(&h);
    let input = vec![0i32; 1];
    for _ in 0..50 {
        h.step_with_goal(&[&input], &hi, true, 0.0, 0.0);
    }

    let mut w = VecWriter::new();
    h.write(&mut w);

    let mut restored = Hierarchy::new();
    restored.read(&mut SliceReader::new(&w.data));

    assert!(restored.has_top_feedback());
    assert_eq!(restored.get_decoder(0, 0).get_num_visible_layers(), 2);

    // The reloaded hierarchy must be able to keep going, and re-serialise the same.
    let mut w2 = VecWriter::new();
    restored.write(&mut w2);
    assert_eq!(w.data, w2.data, "round-trip is not byte-stable");
}

/// Version 1 files predate the flag, and the flag is structural — it decides the
/// top layer's decoder arity, so it cannot be defaulted on read. Rejecting them is
/// the point of the version bump.
#[test]
#[should_panic(expected = "Unsupported AOgmaNeo file version: 1")]
fn version_1_files_are_rejected() {
    let mut w = VecWriter::new();
    w.write_u32(0x4d47_4f41); // "AOGM"
    w.write_u32(1);
    Hierarchy::new().read(&mut SliceReader::new(&w.data));
}
