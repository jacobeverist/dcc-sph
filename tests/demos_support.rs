// Runs the unit tests that live inside `examples/support/`.
//
// Cargo builds examples during `cargo test` but does not run `#[cfg(test)]` code
// inside them (example targets default to `test = false`). Pulling the shared
// demo scaffolding into an integration test compiles it in test configuration, so
// its own `mod tests` blocks run as part of the normal suite — the binning,
// reporting and environment helpers are shared by nine demos and worth covering.
//
// The module must be named `support` here: `examples/support/env/*.rs` refers to
// its siblings through `crate::support::…`, which has to resolve the same way in
// every crate that includes the tree.

#[path = "../examples/support/mod.rs"]
mod support;

// Reference each module so an unused-import lint cannot quietly drop one from the
// build and take its tests with it.
#[test]
fn support_modules_are_wired_in() {
    let a = support::args::Args::from_iter(["--steps".to_string(), "3".to_string()]);
    assert_eq!(a.get::<usize>("steps", 0), 3);

    assert_eq!(support::encode::bin_unit(1.0, 16), 15);
    assert_eq!(support::report::ascii_bar(0.0).chars().count(), 16);

    let mut rng = support::rng::Rng::new(1);
    assert!((0.0..1.0).contains(&rng.unit()));

    let mut w = support::env::wavy::WavyLine::new(1);
    assert_eq!(w.advance(&mut rng).len(), 1);
}
