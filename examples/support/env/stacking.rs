// Block-stacking world, shared by `stacking_rl` and `stacking_prog`.
//
// Port of the world in `demos/Stacking_RL.cpp` and `demos/Stacking_Prog.cpp` from
// jacobeverist/OgmaNeoDemos @ aogmaneo. See `doc/Demos.md` for the deviations.
//
// Three blocks sit in three columns at most three high. A grabber occupies one
// column and is either empty or holding a block. Four actions: do nothing, grab or
// place, step left, step right. That is the whole simulation — the interest is
// entirely in how the two demos specify *which* configuration is wanted.

use dcc_sph::helpers::Int3;
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

use crate::support::rng::Rng;

/// Number of columns. Upstream's `width`.
pub const WIDTH: usize = 3;
/// Maximum blocks in one column. Upstream's `height`.
pub const HEIGHT: usize = 3;
/// Blocks in play. Conserved: a held block is off the grid but still counted.
pub const NUM_BLOCKS: usize = 3;
/// do-nothing, grab-or-place, left, right.
pub const NUM_ACTIONS: usize = 4;
/// Columns in the grid CSDR.
pub const STATE_SIZE: usize = WIDTH * HEIGHT;

/// Encode a stack-height vector as the grid CSDR both demos feed to IO port 0.
///
/// Index `j + i * HEIGHT`, matching upstream's `actualStates[j + i * height]` — the
/// `y + x * h` layout this crate uses for a single-channel `Int3(x, y, z)` surface,
/// so it maps across unchanged.
pub fn fill_cis(stacks: &[usize], out: &mut [i32]) {
    out.fill(0);
    for i in 0..WIDTH {
        for j in 0..stacks[i].min(HEIGHT) {
            out[j + i * HEIGHT] = 1;
        }
    }
}

/// Fraction of grid cells that agree. This is upstream's reward before its cubing,
/// and the headline metric for both demos: 1.0 means the configuration was built.
pub fn match_fraction(a: &[i32], b: &[i32]) -> f32 {
    let same = a.iter().zip(b).filter(|(x, y)| x == y).count();
    same as f32 / a.len() as f32
}

/// `NUM_BLOCKS` blocks dropped into random columns, as upstream initialises both
/// the world and the target.
pub fn random_stacks(rng: &mut Rng) -> Vec<usize> {
    let mut stacks = vec![0usize; WIDTH];
    for _ in 0..NUM_BLOCKS {
        stacks[rng.below(WIDTH)] += 1;
    }
    stacks
}

/// The grabber and the blocks.
#[derive(Clone, Debug)]
pub struct StackingWorld {
    pub stacks: Vec<usize>,
    pub holding: bool,
    pub position: usize,
}

impl StackingWorld {
    pub fn new(rng: &mut Rng) -> Self {
        Self { stacks: random_stacks(rng), holding: false, position: 0 }
    }

    /// Apply one action. Illegal moves are no-ops rather than errors, exactly as
    /// upstream: grabbing from an empty column, placing onto a full one, or walking
    /// off either end all simply do nothing.
    pub fn apply(&mut self, action: i32) {
        match action {
            1 => {
                if !self.holding && self.stacks[self.position] > 0 {
                    self.stacks[self.position] -= 1;
                    self.holding = true;
                } else if self.holding && self.stacks[self.position] < HEIGHT {
                    self.stacks[self.position] += 1;
                    self.holding = false;
                }
            }
            2 => self.position = self.position.saturating_sub(1),
            3 => {
                if self.position + 1 < WIDTH {
                    self.position += 1;
                }
            }
            _ => {}
        }
    }

    pub fn fill_state(&self, out: &mut [i32]) {
        fill_cis(&self.stacks, out);
    }

    /// The action a competent player would take to reach `target`.
    ///
    /// RUST-ONLY, and not a controller the hierarchy ever sees: upstream has no
    /// solver. It exists so a demo can say what score is actually reachable, and so
    /// `stacking_prog` can train on demonstrations rather than on noise — see
    /// `doc/Demos.md`. Blocks are conserved, so when the grabber is holding one
    /// exactly one column is short of its target and the move is never ambiguous.
    pub fn scripted_action(&self, target: &[usize]) -> i32 {
        let want = if self.holding {
            // Holding a block: take it to the leftmost column that is short one.
            (0..WIDTH).find(|&i| self.stacks[i] < target[i])
        } else {
            // Empty-handed: take a block off the leftmost column with too many.
            (0..WIDTH).find(|&i| self.stacks[i] > target[i])
        };

        match want {
            None => 0, // already built, and not mid-carry
            Some(i) if i < self.position => 2,
            Some(i) if i > self.position => 3,
            Some(_) => 1,
        }
    }
}

/// How `stacking_rl` tells the hierarchy which configuration it wants.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum GoalMode {
    /// A fourth IO port carrying the target grid, as an ordinary input the encoder
    /// sees alongside the actual grid. This is the closest mainline expression of
    /// upstream's per-IO-port goal array.
    Port,
    /// The target grid's distilled top-layer CSDR, fed through
    /// `Hierarchy::step_with_goal`. Uses the RUST-ONLY goal path.
    Top,
}

impl std::str::FromStr for GoalMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "port" => Ok(GoalMode::Port),
            "top" => Ok(GoalMode::Top),
            other => Err(format!("unknown --goal-mode {other:?} (expected `port` or `top`)")),
        }
    }
}

/// `stacking_rl`'s hierarchy: grid, action, grabber position, and the goal by
/// whichever route `mode` selects.
///
/// Upstream additionally sets `h.params.ios[2].actor.discount = 0.9f` on IO port 2,
/// which is a **Prediction** port and has no actor. That line is inert and is not
/// replicated.
pub fn build_rl_hierarchy(mode: GoalMode) -> Hierarchy {
    let mut io_descs = vec![
        IoDesc {
            size: Int3::new(WIDTH as i32, HEIGHT as i32, 2),
            io_type: IoType::Prediction,
            ..Default::default()
        },
        IoDesc {
            size: Int3::new(1, 1, NUM_ACTIONS as i32),
            io_type: IoType::Action,
            ..Default::default()
        },
        IoDesc {
            size: Int3::new(1, 1, WIDTH as i32),
            io_type: IoType::Prediction,
            ..Default::default()
        },
    ];

    if mode == GoalMode::Port {
        // Input-only: the goal is something to condition on, never something to
        // predict. Same role IO port 0 plays in `cat_mouse`.
        io_descs.push(IoDesc {
            size: Int3::new(WIDTH as i32, HEIGHT as i32, 2),
            io_type: IoType::None,
            ..Default::default()
        });
    }

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(5, 5, 32),
        num_dendrites_per_cell: 4,
        up_radius: 2,
        recurrent_radius: 0,
        down_radius: 2,
        ticks_per_update: 1,
        top_feedback: mode == GoalMode::Top,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);
    h
}

/// `stacking_prog`'s hierarchy: the same three ports, but **all** of them
/// Prediction — there is no actor, and the action is whatever the action port
/// predicts. Goal-conditioned, since the whole demo is about the goal CSDR.
pub fn build_prog_hierarchy() -> Hierarchy {
    let io_descs = vec![
        IoDesc {
            size: Int3::new(WIDTH as i32, HEIGHT as i32, 2),
            io_type: IoType::Prediction,
            ..Default::default()
        },
        IoDesc {
            size: Int3::new(1, 1, NUM_ACTIONS as i32),
            io_type: IoType::Prediction,
            ..Default::default()
        },
        IoDesc {
            size: Int3::new(1, 1, WIDTH as i32),
            io_type: IoType::Prediction,
            ..Default::default()
        },
    ];

    let layer_descs = vec![LayerDesc {
        hidden_size: Int3::new(5, 5, 32),
        num_dendrites_per_cell: 4,
        up_radius: 2,
        recurrent_radius: 0,
        down_radius: 2,
        ticks_per_update: 1,
        top_feedback: true,
    }];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);
    h
}

/// Distil a "program": the top-layer CSDR the hierarchy settles into when shown the
/// goal configuration and nothing else.
///
/// This is upstream's `stateSaved` branch. It clones the hierarchy so the rollout
/// leaves no trace on the live one, shows the clone the goal grid for `iters` steps
/// with learning off and null action and position, and takes the top hidden CSDR
/// after each — the fixed point that settles out is the goal expressed in the
/// hierarchy's own most abstract vocabulary, which is exactly the form
/// `step_with_goal` wants.
pub fn distil_program(h: &Hierarchy, goal_cis: &[i32], iters: usize) -> Vec<i32> {
    let mut copy = h.clone();
    let mut program = copy.get_top_hidden_cis().to_vec();
    let no_action = [0i32];
    let no_position = [0i32];

    for _ in 0..iters {
        copy.step_with_goal(&[goal_cis, &no_action, &no_position], &program, false, 0.0, 0.0);
        program = copy.get_top_hidden_cis().to_vec();
    }

    program
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(stacks: &[usize], position: usize, holding: bool) -> StackingWorld {
        StackingWorld { stacks: stacks.to_vec(), holding, position }
    }

    #[test]
    fn grid_encoding_matches_upstream_indexing() {
        let mut cis = vec![0i32; STATE_SIZE];
        fill_cis(&[0, 2, 1], &mut cis);
        // Column 1 has two blocks at j = 0, 1 → indices 3 and 4; column 2 has one at
        // index 6.
        assert_eq!(cis, vec![0, 0, 0, 1, 1, 0, 1, 0, 0]);
    }

    #[test]
    fn blocks_are_conserved_under_any_action_sequence() {
        let mut rng = Rng::new(4);
        let mut w = StackingWorld::new(&mut rng);
        for _ in 0..2000 {
            w.apply(rng.below(NUM_ACTIONS) as i32);
            let total = w.stacks.iter().sum::<usize>() + usize::from(w.holding);
            assert_eq!(total, NUM_BLOCKS);
            assert!(w.position < WIDTH);
            assert!(w.stacks.iter().all(|&s| s <= HEIGHT));
        }
    }

    #[test]
    fn scripted_solver_reaches_any_reachable_target() {
        let mut rng = Rng::new(11);
        let mut state = vec![0i32; STATE_SIZE];
        let mut target_cis = vec![0i32; STATE_SIZE];

        for _ in 0..200 {
            let mut w = StackingWorld::new(&mut rng);
            let target = random_stacks(&mut rng);
            fill_cis(&target, &mut target_cis);

            // Six blocks to shift at worst, each needing at most two walks plus a
            // grab and a place.
            for _ in 0..64 {
                let a = w.scripted_action(&target);
                if a == 0 && !w.holding {
                    break;
                }
                w.apply(a);
            }

            w.fill_state(&mut state);
            assert_eq!(match_fraction(&state, &target_cis), 1.0, "failed to build {target:?}");
        }
    }

    #[test]
    fn scripted_solver_is_a_no_op_once_built() {
        let w = world(&[1, 1, 1], 2, false);
        assert_eq!(w.scripted_action(&[1, 1, 1]), 0);
    }
}
