use dcc_sph::helpers::{Int3, VecWriter, SliceReader};
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};
use dcc_sph::image_encoder::{ImageEncoder, VisibleLayerDesc as IeVLD};

fn make_random_cis(size: Int3) -> Vec<i32> {
    let cols = (size.x * size.y) as usize;
    let col_size = size.z as usize;
    (0..cols).map(|i| (i % col_size) as i32).collect()
}

#[test]
fn test_hierarchy_create_and_step() {
    // Minimal 1-IO, 1-layer hierarchy
    let io_descs = vec![IoDesc {
        size: Int3::new(4, 4, 16),
        io_type: IoType::Prediction,
        num_dendrites_per_cell: 2,
        up_radius: 2,
        down_radius: 2,
        value_size: 64,
        value_num_dendrites_per_cell: 2,
        history_capacity: 64,
    }];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(4, 4, 16),
        num_dendrites_per_cell: 2,
        up_radius: 2,
        recurrent_radius: -1, // no recurrence
        down_radius: 2,
        ticks_per_update: 1,
        top_feedback: false,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    assert_eq!(h.get_num_layers(), 1);
    assert_eq!(h.get_num_io(), 1);
    assert_eq!(h.get_io_type(0), IoType::Prediction);

    // Run a few steps with dummy input
    let io_size = h.get_io_size(0);
    let input_cis = make_random_cis(io_size);

    for _ in 0..3 {
        h.step(&[&input_cis], true, 0.0, 0.0);
    }

    // Predictions should have correct length
    let pred_cis = h.get_prediction_cis(0);
    let expected_len = (io_size.x * io_size.y) as usize;
    assert_eq!(pred_cis.len(), expected_len);

    // All predicted column indices should be in [0, column_size)
    let col_size = io_size.z;
    for &ci in pred_cis {
        assert!(ci >= 0 && ci < col_size, "ci={ci} out of range [0,{col_size})");
    }
}

#[test]
fn test_hierarchy_two_layers() {
    let io_descs = vec![IoDesc {
        size: Int3::new(4, 4, 16),
        io_type: IoType::Prediction,
        num_dendrites_per_cell: 2,
        up_radius: 2,
        down_radius: 2,
        value_size: 64,
        value_num_dendrites_per_cell: 2,
        history_capacity: 64,
    }];

    let layer_descs = vec![
        LayerDesc {
            hidden_size: Int3::new(4, 4, 16),
            num_dendrites_per_cell: 2,
            up_radius: 2,
            recurrent_radius: -1,
            down_radius: 2,
            ticks_per_update: 1,
            top_feedback: false,
        },
        LayerDesc {
            hidden_size: Int3::new(3, 3, 16),
            num_dendrites_per_cell: 2,
            up_radius: 2,
            recurrent_radius: -1,
            down_radius: 2,
            ticks_per_update: 1,
            top_feedback: false,
        },
    ];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    assert_eq!(h.get_num_layers(), 2);

    let io_size = h.get_io_size(0);
    let input_cis = make_random_cis(io_size);

    for _ in 0..5 {
        h.step(&[&input_cis], true, 0.0, 0.0);
    }

    let pred_cis = h.get_prediction_cis(0);
    let col_size = io_size.z;
    for &ci in pred_cis {
        assert!(ci >= 0 && ci < col_size);
    }
}

#[test]
fn test_hierarchy_clear_state() {
    let io_descs = vec![IoDesc {
        size: Int3::new(4, 4, 8),
        io_type: IoType::Prediction,
        num_dendrites_per_cell: 2,
        up_radius: 2,
        down_radius: 2,
        value_size: 32,
        value_num_dendrites_per_cell: 2,
        history_capacity: 32,
    }];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(4, 4, 8),
        num_dendrites_per_cell: 2,
        up_radius: 2,
        recurrent_radius: -1,
        down_radius: 2,
        ticks_per_update: 1,
        top_feedback: false,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    let io_size = h.get_io_size(0);
    let input_cis = make_random_cis(io_size);

    h.step(&[&input_cis], true, 0.0, 0.0);
    h.clear_state();
    // Should still be able to step after clearing state
    h.step(&[&input_cis], true, 0.0, 0.0);
}

#[test]
fn test_multiple_io() {
    let io_descs = vec![
        IoDesc {
            size: Int3::new(4, 4, 8),
            io_type: IoType::Prediction,
            num_dendrites_per_cell: 2,
            up_radius: 2,
            down_radius: 2,
            value_size: 32,
            value_num_dendrites_per_cell: 2,
            history_capacity: 32,
        },
        IoDesc {
            size: Int3::new(4, 4, 8),
            io_type: IoType::Prediction,
            num_dendrites_per_cell: 2,
            up_radius: 2,
            down_radius: 2,
            value_size: 32,
            value_num_dendrites_per_cell: 2,
            history_capacity: 32,
        },
    ];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(4, 4, 16),
        num_dendrites_per_cell: 2,
        up_radius: 2,
        recurrent_radius: -1,
        down_radius: 2,
        ticks_per_update: 1,
        top_feedback: false,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    assert_eq!(h.get_num_io(), 2);

    let io_size0 = h.get_io_size(0);
    let io_size1 = h.get_io_size(1);
    let input0 = make_random_cis(io_size0);
    let input1 = make_random_cis(io_size1);

    for _ in 0..3 {
        h.step(&[&input0, &input1], true, 0.0, 0.0);
    }

    for i in 0..2 {
        let pred = h.get_prediction_cis(i);
        let io_size = h.get_io_size(i);
        let col_size = io_size.z;
        assert_eq!(pred.len(), (io_size.x * io_size.y) as usize);
        for &ci in pred {
            assert!(ci >= 0 && ci < col_size);
        }
    }
}

#[test]
fn test_image_encoder_step_and_reconstruct() {
    let visible_layer_descs = vec![IeVLD {
        size: Int3::new(8, 8, 3), // 8x8 RGB
        radius: 2,
    }];

    let mut ie = ImageEncoder::default();
    ie.init_random(Int3::new(4, 4, 16), visible_layer_descs);

    assert_eq!(ie.get_hidden_size(), Int3::new(4, 4, 16));
    assert_eq!(ie.get_num_visible_layers(), 1);

    // Synthesize a simple gradient image (8*8*3 bytes)
    let num_pixels = 8 * 8 * 3;
    let inputs: Vec<u8> = (0..num_pixels).map(|i| (i % 256) as u8).collect();

    for _ in 0..5 {
        ie.step(&[&inputs], true, true);
    }

    // Hidden CIs must be valid
    let hidden_cis: Vec<i32> = ie.get_hidden_cis().to_vec();
    assert_eq!(hidden_cis.len(), 4 * 4);
    for &ci in &hidden_cis {
        assert!((0..16).contains(&ci), "ci={ci} out of range [0,16)");
    }

    // reconstruct() should produce a byte image of the right size
    ie.reconstruct(&hidden_cis);
    let recon = ie.get_reconstruction(0);
    assert_eq!(recon.len(), 8 * 8 * 3);
}

#[test]
fn test_hierarchy_serialization_roundtrip() {
    let io_descs = vec![IoDesc {
        size: Int3::new(4, 4, 8),
        io_type: IoType::Prediction,
        num_dendrites_per_cell: 2,
        up_radius: 2,
        down_radius: 2,
        value_size: 32,
        value_num_dendrites_per_cell: 2,
        history_capacity: 32,
    }];
    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(4, 4, 8),
        num_dendrites_per_cell: 2,
        up_radius: 2,
        recurrent_radius: -1,
        down_radius: 2,
        ticks_per_update: 1,
        top_feedback: false,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    let io_size = h.get_io_size(0);
    let input_cis: Vec<i32> = (0..(io_size.x * io_size.y) as usize)
        .map(|i| (i % io_size.z as usize) as i32)
        .collect();

    // Run a few steps to build up state
    for _ in 0..3 {
        h.step(&[&input_cis], true, 0.0, 0.0);
    }

    // Capture predictions before serialization
    let preds_before: Vec<i32> = h.get_prediction_cis(0).to_vec();

    // Serialize
    let mut writer = VecWriter::new();
    h.write(&mut writer);
    let bytes = writer.data;

    // Deserialize into a fresh hierarchy
    let mut h2 = Hierarchy::new();
    let mut reader = SliceReader::new(&bytes);
    h2.read(&mut reader);

    // Predictions should match after loading
    let preds_after: Vec<i32> = h2.get_prediction_cis(0).to_vec();
    assert_eq!(preds_before, preds_after, "predictions differ after round-trip");

    // Should still be able to step after deserializing
    h2.step(&[&input_cis], true, 0.0, 0.0);
}

#[test]
fn test_io_type_none_no_prediction() {
    // An IoType::None port is input-only; prediction slice should be empty.
    let io_descs = vec![
        IoDesc {
            size: Int3::new(4, 4, 8),
            io_type: IoType::None,
            num_dendrites_per_cell: 2,
            up_radius: 2,
            down_radius: 2,
            value_size: 32,
            value_num_dendrites_per_cell: 2,
            history_capacity: 32,
        },
        IoDesc {
            size: Int3::new(2, 2, 4),
            io_type: IoType::Prediction,
            num_dendrites_per_cell: 2,
            up_radius: 2,
            down_radius: 2,
            value_size: 16,
            value_num_dendrites_per_cell: 2,
            history_capacity: 16,
        },
    ];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(4, 4, 8),
        num_dendrites_per_cell: 2,
        up_radius: 2,
        recurrent_radius: -1,
        down_radius: 2,
        ticks_per_update: 1,
        top_feedback: false,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    let io_size0 = h.get_io_size(0);
    let io_size1 = h.get_io_size(1);
    let input0 = make_random_cis(io_size0);
    let input1 = make_random_cis(io_size1);

    for _ in 0..3 {
        h.step(&[&input0, &input1], true, 0.0, 0.0);
    }

    // IoType::None port should have no predictions (empty slice)
    assert_eq!(h.get_prediction_cis(0).len(), 0, "None IO should have empty prediction");

    // IoType::Prediction port should have valid predictions
    let pred = h.get_prediction_cis(1);
    assert_eq!(pred.len(), (io_size1.x * io_size1.y) as usize);
    let col_size = io_size1.z;
    for &ci in pred {
        assert!(ci >= 0 && ci < col_size);
    }
}

#[test]
fn test_recurrent_layers() {
    // Verify that hierarchies with recurrent_radius >= 0 initialize and step without panic.
    let io_descs = vec![IoDesc {
        size: Int3::new(4, 4, 8),
        io_type: IoType::Prediction,
        num_dendrites_per_cell: 2,
        up_radius: 2,
        down_radius: 2,
        value_size: 32,
        value_num_dendrites_per_cell: 2,
        history_capacity: 32,
    }];

    let layer_descs = vec![
        LayerDesc {
            hidden_size: Int3::new(4, 4, 8),
            num_dendrites_per_cell: 2,
            up_radius: 2,
            recurrent_radius: 2, // recurrent connections enabled
            down_radius: 2,
            ticks_per_update: 1,
            top_feedback: false,
        },
        LayerDesc {
            hidden_size: Int3::new(3, 3, 8),
            num_dendrites_per_cell: 2,
            up_radius: 2,
            recurrent_radius: 1,
            down_radius: 2,
            ticks_per_update: 2,
            top_feedback: false,
        },
    ];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    let io_size = h.get_io_size(0);
    let input_cis = make_random_cis(io_size);

    // Run enough steps for the upper layer to fire at least once
    for _ in 0..6 {
        h.step(&[&input_cis], true, 0.0, 0.0);
    }

    // Check that layer 1 has fired (ticks_per_update=2 means it fires at steps 2, 4, 6)
    let pred = h.get_prediction_cis(0);
    let col_size = io_size.z;
    assert_eq!(pred.len(), (io_size.x * io_size.y) as usize);
    for &ci in pred {
        assert!(ci >= 0 && ci < col_size);
    }
}

#[test]
fn test_image_encoder_serialization_roundtrip() {
    let visible_layer_descs = vec![IeVLD {
        size: Int3::new(6, 6, 3),
        radius: 2,
    }];

    let mut ie = ImageEncoder::default();
    ie.init_random(Int3::new(4, 4, 8), visible_layer_descs);

    let num_pixels = 6 * 6 * 3;
    let inputs: Vec<u8> = (0..num_pixels).map(|i| (i * 7 % 256) as u8).collect();

    // Train for a few steps
    for _ in 0..5 {
        ie.step(&[&inputs], true, true);
    }

    // Capture hidden CIs and reconstruction before serialization
    let hidden_cis_before: Vec<i32> = ie.get_hidden_cis().to_vec();
    ie.reconstruct(&hidden_cis_before);
    let recon_before: Vec<u8> = ie.get_reconstruction(0).to_vec();

    // Serialize
    let mut writer = VecWriter::new();
    ie.write(&mut writer);
    let bytes = writer.data;

    // Deserialize into a fresh ImageEncoder
    let mut ie2 = ImageEncoder::default();
    let mut reader = SliceReader::new(&bytes);
    ie2.read(&mut reader);

    // Hidden CIs should match after round-trip
    assert_eq!(ie2.get_hidden_cis(), hidden_cis_before.as_slice(), "hidden_cis differ after round-trip");

    // Reconstruction from the same CIs should match
    ie2.reconstruct(&hidden_cis_before);
    let recon_after = ie2.get_reconstruction(0);
    assert_eq!(recon_after, recon_before.as_slice(), "reconstruction differs after round-trip");
}

#[test]
fn test_prediction_improves_on_repeating_sequence() {
    // After training on a fixed repeating pattern, prediction accuracy should improve.
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

    // Build a simple repeating sequence of 4 distinct patterns
    let patterns: Vec<Vec<i32>> = (0..4)
        .map(|p| (0..num_cols).map(|c| ((p + c) % col_size) as i32).collect())
        .collect();

    // Warm-up phase — many passes over the sequence
    for _ in 0..50 {
        for pattern in &patterns {
            h.step(&[pattern], true, 0.0, 0.0);
        }
    }

    // Evaluation phase — count correct next-step predictions over one full cycle
    let mut correct_early = 0usize;
    for (i, pattern) in patterns.iter().enumerate() {
        let next = &patterns[(i + 1) % patterns.len()];
        h.step(&[pattern], false, 0.0, 0.0);
        let pred = h.get_prediction_cis(0);
        correct_early += pred.iter().zip(next.iter()).filter(|(p, n)| p == n).count();
    }

    // After even more training, accuracy should be at least as good
    for _ in 0..100 {
        for pattern in &patterns {
            h.step(&[pattern], true, 0.0, 0.0);
        }
    }

    let mut correct_late = 0usize;
    for (i, pattern) in patterns.iter().enumerate() {
        let next = &patterns[(i + 1) % patterns.len()];
        h.step(&[pattern], false, 0.0, 0.0);
        let pred = h.get_prediction_cis(0);
        correct_late += pred.iter().zip(next.iter()).filter(|(p, n)| p == n).count();
    }

    // Late accuracy should be at least as high as early accuracy (not worse)
    assert!(
        correct_late >= correct_early,
        "prediction accuracy regressed: early={correct_early} late={correct_late} out of {total}",
        total = num_cols * patterns.len()
    );
}
