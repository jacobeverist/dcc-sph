// Environments for the demos. Most are ported from OgmaNeoDemos; `noise` is
// RUST-ONLY and is marked as such in its own header.
//
// Each module holds only the world simulation and its encoding — no hierarchy
// setup, no reporting, no drawing. That split is what lets a demo run identically
// headless and under the windowed viewer in `examples-viz`.

pub mod ball;
pub mod catmouse;
pub mod cluster;
pub mod noise;
pub mod pusher;
pub mod racing;
pub mod runner;
pub mod stacking;
pub mod text;
pub mod video;
pub mod wavy;
