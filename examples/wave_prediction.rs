// Wave prediction example using AOgmaNeo Sparse Predictive Hierarchies (Rust port)
//
// Port of python_ref/wavy_line_prediction.py
//
// Demonstrates pure sequence prediction: SPH learns a structured non-sinusoidal
// waveform during a training phase, then runs autonomously on its own predictions
// during a recall phase.
//
// Run with: cargo run --release --example wave_prediction

use dcc_sph::helpers::Int3;
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

#[path = "support/mod.rs"]
mod support;

// These three used to be defined here; they moved into the shared demo
// scaffolding when the OgmaNeoDemos suite landed, since several demos need them.
use support::encode::{csdr_to_unorm8, unorm8_to_csdr};
use support::report::ascii_bar;

/// The target waveform: 1.0 whenever t is divisible by 20 or 7, else 0.0.
fn wave(t: usize) -> f32 {
    if t % 20 == 0 || t % 7 == 0 { 1.0 } else { 0.0 }
}

fn main() {
    // --- Hierarchy configuration ---

    // One IO: 2 columns × 16 cells, prediction type
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

    // Three layers with exponential temporal memory (ticks 1, 2, 4)
    let layer_descs = vec![
        LayerDesc {
            hidden_size: Int3::new(5, 5, 64),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: -1,
            down_radius: 2,
            ticks_per_update: 1,
            top_feedback: false,
        },
        LayerDesc {
            hidden_size: Int3::new(5, 5, 64),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: -1,
            down_radius: 2,
            ticks_per_update: 2,
            top_feedback: false,
        },
        LayerDesc {
            hidden_size: Int3::new(5, 5, 64),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: -1,
            down_radius: 2,
            ticks_per_update: 4,
            top_feedback: false,
        },
    ];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    // --- Training phase ---

    let train_steps = 10_000;
    println!("Training for {train_steps} steps...");

    let mut input_cis = vec![0i32; 2];

    for t in 0..train_steps {
        let v = wave(t);
        let csdr = unorm8_to_csdr(v);
        input_cis[0] = csdr[0];
        input_cis[1] = csdr[1];

        h.step(&[&input_cis], true, 0.0, 0.0);

        if t % 1000 == 0 {
            println!("  step {t}");
        }
    }

    // --- Recall phase ---

    let recall_steps = 1_000;
    println!("\nRecall phase ({recall_steps} steps) — feeding own predictions back:");
    println!("{:<8} {:<20} {:<20} match", "t", "actual", "predicted");
    println!("{}", "-".repeat(58));

    let mut matches = 0usize;
    let threshold = 0.5f32;

    for t2 in 0..recall_steps {
        let t = t2 + train_steps;
        let actual_v = wave(t);

        // Feed the hierarchy's own prediction back as input (learning disabled)
        let pred_cis: Vec<i32> = h.get_prediction_cis(0).to_vec();
        h.step(&[&pred_cis], false, 0.0, 0.0);

        // Decode the latest prediction
        let pred_raw = h.get_prediction_cis(0);
        let pred_v = csdr_to_unorm8(pred_raw);

        let is_match = (actual_v >= threshold) == (pred_v >= threshold);
        if is_match {
            matches += 1;
        }

        let match_str = if is_match { "match" } else { "MISS" };
        println!(
            "t={:<5} [{}] [{}] {}",
            t,
            ascii_bar(actual_v),
            ascii_bar(pred_v),
            match_str
        );
    }

    let pct = matches as f32 / recall_steps as f32 * 100.0;
    println!("\nRecall accuracy: {matches}/{recall_steps} ({pct:.1}%)");
}
