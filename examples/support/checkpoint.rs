// Saving and loading what a demo learned.
//
// Every upstream demo has an `S`-to-save key. Nothing in this repository persisted
// a model until now: `Hierarchy::write`/`read` appeared only in one smoke test, and
// `write_weights`/`read_weights` were called from nowhere at all.
//
//     --save run.ohr          write the model after training
//     --load run.ohr          start from a saved model instead of random weights
//     --save-weights w.ohw    weights only, without the running state
//     --load-weights w.ohw
//
// Two things this makes possible that were awkward before: resuming a long `runner`
// run, and shipping a trained checkpoint so the windowed viewer opens on a
// competent agent rather than a random one.
//
// `helpers::FileWriter` / `FileReader` already existed for exactly this and were
// likewise unused.

use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::Path;

use dcc_sph::helpers::{FileReader, FileWriter, StreamReader, StreamWriter};
use dcc_sph::hierarchy::Hierarchy;
use dcc_sph::image_encoder::ImageEncoder;

use crate::support::args::Args;

/// Run `f` against a buffered writer over `path`.
pub fn write_to(path: &Path, f: impl FnOnce(&mut dyn StreamWriter)) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = FileWriter::new(BufWriter::new(file));
    f(&mut writer as &mut dyn StreamWriter);
    Ok(())
}

/// Run `f` against a buffered reader over `path`.
pub fn read_from(path: &Path, f: impl FnOnce(&mut dyn StreamReader)) -> io::Result<()> {
    let file = File::open(path)?;
    let mut reader = FileReader::new(BufReader::new(file));
    f(&mut reader as &mut dyn StreamReader);
    Ok(())
}

pub fn save_hierarchy(h: &Hierarchy, path: &Path) -> io::Result<()> {
    write_to(path, |w| h.write(w))
}

pub fn load_hierarchy(h: &mut Hierarchy, path: &Path) -> io::Result<()> {
    read_from(path, |r| h.read(r))
}

/// Weights only — no hidden state, no tick counters.
///
/// Smaller than a full save and portable across runs that differ in where they
/// happened to be in a sequence. It is *not* a substitute for `save_hierarchy`:
/// loading weights into a freshly constructed hierarchy gives a model that knows
/// what it learned but not where it was.
pub fn save_weights(h: &Hierarchy, path: &Path) -> io::Result<()> {
    write_to(path, |w| h.write_weights(w))
}

pub fn load_weights(h: &mut Hierarchy, path: &Path) -> io::Result<()> {
    read_from(path, |r| h.read_weights(r))
}

pub fn save_image_encoder(e: &ImageEncoder, path: &Path) -> io::Result<()> {
    write_to(path, |w| e.write(w))
}

pub fn load_image_encoder(e: &mut ImageEncoder, path: &Path) -> io::Result<()> {
    read_from(path, |r| e.read(r))
}

/// Apply `--load` / `--load-weights` if either was given. Returns true if anything
/// was loaded.
///
/// A missing or unreadable checkpoint is fatal rather than a warning: a run that
/// silently trained from scratch when it was told to resume would waste the time it
/// was meant to save.
pub fn maybe_load(h: &mut Hierarchy, args: &Args) -> bool {
    if let Some(path) = args.str("load") {
        load_hierarchy(h, Path::new(path))
            .unwrap_or_else(|e| panic!("--load {path}: {e}"));
        println!("Loaded model from {path}");
        return true;
    }
    if let Some(path) = args.str("load-weights") {
        load_weights(h, Path::new(path))
            .unwrap_or_else(|e| panic!("--load-weights {path}: {e}"));
        println!("Loaded weights from {path}");
        return true;
    }
    false
}

/// Apply `--save` / `--save-weights` if either was given.
pub fn maybe_save(h: &Hierarchy, args: &Args) {
    if let Some(path) = args.str("save") {
        save_hierarchy(h, Path::new(path))
            .unwrap_or_else(|e| panic!("--save {path}: {e}"));
        println!("Saved model to {path}");
    }
    if let Some(path) = args.str("save-weights") {
        save_weights(h, Path::new(path))
            .unwrap_or_else(|e| panic!("--save-weights {path}: {e}"));
        println!("Saved weights to {path}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dcc_sph::helpers::{rand_get_state, set_global_state, Int3};
    use dcc_sph::hierarchy::{IoDesc, IoType, LayerDesc};

    fn build(seed: u64) -> Hierarchy {
        set_global_state(rand_get_state(seed));
        let io_descs = vec![IoDesc {
            size: Int3::new(1, 1, 16),
            io_type: IoType::Prediction,
            ..Default::default()
        }];
        let layer_descs = vec![LayerDesc { hidden_size: Int3::new(4, 4, 16), ..Default::default() }];
        let mut h = Hierarchy::new();
        h.init_random(&io_descs, &layer_descs);
        h
    }

    /// Drive a short repeating sequence so the model has something to have learned.
    fn train(h: &mut Hierarchy, steps: usize) {
        for t in 0..steps {
            let cis = vec![(t % 4) as i32];
            h.step(&[&cis], true, 0.0, 0.0);
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("dcc_sph_ckpt_{name}_{}", std::process::id()));
        p
    }

    #[test]
    fn a_full_save_reproduces_the_next_prediction_exactly() {
        let path = tmp("full");

        let mut a = build(7);
        train(&mut a, 200);
        save_hierarchy(&a, &path).unwrap();

        // What the saved model would predict next.
        let expected = a.get_prediction_cis(0).to_vec();

        let mut b = build(7);
        load_hierarchy(&mut b, &path).unwrap();
        assert_eq!(b.get_prediction_cis(0), expected.as_slice());

        // And it must keep agreeing as both are driven identically.
        for t in 0..50i32 {
            let cis = vec![t % 4];
            a.step(&[&cis], false, 0.0, 0.0);
            b.step(&[&cis], false, 0.0, 0.0);
            assert_eq!(a.get_prediction_cis(0), b.get_prediction_cis(0), "diverged at {t}");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_weights_only_save_round_trips() {
        let path = tmp("weights");

        let mut a = build(11);
        train(&mut a, 200);
        save_weights(&a, &path).unwrap();

        let mut b = build(11);
        load_weights(&mut b, &path).unwrap();

        // Weights carry no hidden state, so drive both from a cleared state to
        // compare what they *know* rather than where they were.
        a.clear_state();
        b.clear_state();
        for t in 0..50i32 {
            let cis = vec![t % 4];
            a.step(&[&cis], false, 0.0, 0.0);
            b.step(&[&cis], false, 0.0, 0.0);
            assert_eq!(a.get_prediction_cis(0), b.get_prediction_cis(0), "diverged at {t}");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_checkpoint_actually_contains_what_was_learned() {
        // Guards against a round trip that "passes" because nothing was written:
        // two empty files also compare equal.
        //
        // Comparing a single prediction would not do it — one column of 16 cells
        // agrees by chance often enough to be useless, and an untrained decoder
        // tends to emit cell 0 regardless. Comparing the serialised bytes tests the
        // property directly.
        let trained_path = tmp("trained");
        let fresh_path = tmp("fresh");

        let mut trained = build(3);
        train(&mut trained, 400);
        save_hierarchy(&trained, &trained_path).unwrap();

        let fresh = build(3);
        save_hierarchy(&fresh, &fresh_path).unwrap();

        let a = std::fs::read(&trained_path).unwrap();
        let b = std::fs::read(&fresh_path).unwrap();

        assert!(!a.is_empty(), "the checkpoint is empty");
        assert_eq!(a.len(), b.len(), "same architecture should serialise to the same size");
        assert_ne!(a, b, "a trained model serialises identically to an untrained one");

        let _ = std::fs::remove_file(&trained_path);
        let _ = std::fs::remove_file(&fresh_path);
    }

    #[test]
    fn a_missing_checkpoint_is_an_error_not_a_silent_fresh_start() {
        let mut h = build(1);
        let err = load_hierarchy(&mut h, Path::new("/nonexistent/path/model.ohr"));
        assert!(err.is_err());
    }
}
