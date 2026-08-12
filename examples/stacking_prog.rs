// Stacking Prog — a goal as a distilled top-layer CSDR.
//
// Port of `demos/Stacking_Prog.cpp` from jacobeverist/OgmaNeoDemos @ aogmaneo.
// See `doc/Demos.md` for the deviations from upstream.
//
// Same world as `stacking_rl`, but there is **no actor and no reward**. All three
// IO ports are Prediction ports, and the action executed is simply whatever the
// action port predicts. The only thing steering it is the goal:
//
//   1. Train: drive the world under some policy, feeding a goal CSDR at the top.
//   2. Distil: clone the hierarchy, show the clone the target configuration for 32
//      steps with learning off, and take its top hidden CSDR. That is the
//      "program" — the target written in the hierarchy's own most abstract
//      vocabulary.
//   3. Execute: run the live hierarchy with that program as its goal, with learning
//      off, and do what the action port says.
//
// This is the demo the RUST-ONLY `Hierarchy::step_with_goal` exists for. Upstream's
// three-argument `step` taking a single top-layer buffer is published on AOgmaNeo's
// `ubl3_recurrent` branch and nowhere in mainline; `doc/Divergences.md` records what
// it cost to add.
//
// `--train-policy` is the load-bearing switch. Upstream trains on **random** actions
// under **random** goals, which is faithful and which this demo reproduces — and
// which does not work, for a reason the demo reports rather than hides. See
// `doc/Demos.md`.
//
//   cargo run --release --example stacking_prog
//   cargo run --release --example stacking_prog -- --sweep train-policy=random,scripted --repeat 5

#[path = "support/mod.rs"]
mod support;

use support::args::Args;
use support::checkpoint;
use support::env::stacking::{
    build_prog_hierarchy, distil_program, fill_cis, match_fraction, random_stacks, StackingWorld,
    HEIGHT, NUM_ACTIONS, STATE_SIZE, WIDTH,
};
use support::metrics::{Recorder, Summary};
use support::report::Rolling;
use support::rng::{seed_everything, Rng};
use support::sweep;
use dcc_sph::hierarchy::Hierarchy;

fn main() {
    let args = Args::parse();

    let mut rec = Recorder::from_args("stacking_prog", &args);
    sweep::drive(&args, &mut rec, run);
    rec.finish();
}

fn run(args: &Args, seed: u64, rec: &mut Recorder) -> Summary {
    let steps: usize = args.get("steps", 300_000);
    let trials: usize = args.get("trials", 200);
    let trial_steps: usize = args.get("trial-steps", 96);
    let every: usize = args.get("every", 50_000);
    let distil_iters: usize = args.get("distil-iters", 32);
    // Upstream re-draws its random goal CSDR with probability 0.1 per frame.
    let goal_hold: usize = args.get("goal-hold", 300);

    let policy = match args.str("train-policy") {
        None | Some("random") => TrainPolicy::Random,
        Some("scripted") => TrainPolicy::Scripted,
        Some(other) => panic!("unknown --train-policy {other:?} (expected `random` or `scripted`)"),
    };

    let silent = args.flag("silent");
    let quiet = silent || args.flag("quiet");

    macro_rules! say {
        ($($arg:tt)*) => { if !silent { println!($($arg)*); } };
    }

    let mut rng = seed_everything(seed);

    rec.config("steps", steps);
    rec.config("trials", trials);
    rec.config("trial_steps", trial_steps);
    rec.config("distil_iters", distil_iters);
    rec.config("train_policy", match policy {
        TrainPolicy::Random => "random",
        TrainPolicy::Scripted => "scripted",
    });

    let mut h = build_prog_hierarchy();
    checkpoint::maybe_load(&mut h, args);

    say!("Stacking Prog — {steps} training steps, seed {seed}");
    say!("  {WIDTH} columns x {HEIGHT} high, 1 layer 5x5x32, all three IO ports Prediction");
    say!(
        "  training policy: {}",
        match policy {
            TrainPolicy::Random => "random actions under random goal CSDRs (upstream)",
            TrainPolicy::Scripted => "a scripted solver under the goal it is actually building",
        }
    );
    say!("  goal distilled over {distil_iters} rollout steps on a clone");
    say!();

    // --- Training ---

    let mut world = StackingWorld::new(&mut rng);
    let mut target = random_stacks(&mut rng);

    let mut state_cis = vec![0i32; STATE_SIZE];
    let mut goal_cis = vec![0i32; STATE_SIZE];
    let mut action_cis = vec![0i32; 1];
    let mut position_cis = vec![0i32; 1];

    fill_cis(&target, &mut goal_cis);

    let top = h.get_top_hidden_size();
    let top_columns = (top.x * top.y) as usize;
    let mut goal: Vec<i32> = h.get_top_hidden_cis().to_vec();
    if policy == TrainPolicy::Scripted {
        goal = distil_program(&h, &goal_cis, distil_iters);
    }

    let mut action_hit = Rolling::new(20_000, 0.0002);

    for t in 0..steps {
        // When to draw a new target. Upstream re-randomises on a timer, which is
        // right for a demonstrator that never finishes anything — but a scripted
        // solver builds the target in about twenty steps and then idles, so a
        // 300-step timer would make ~93% of the training data "do nothing" and
        // teach the hierarchy to freeze. Under a solver the target is therefore
        // redrawn the moment it is built, which keeps the demonstrator working.
        let built = policy == TrainPolicy::Scripted
            && !world.holding
            && world.scripted_action(&target) == 0;
        let expired = t > 0 && t % goal_hold == 0;

        if built || (policy == TrainPolicy::Random && expired) {
            target = random_stacks(&mut rng);
            fill_cis(&target, &mut goal_cis);

            match policy {
                // Upstream: a fresh *random* CSDR, unrelated to any configuration.
                TrainPolicy::Random => {
                    for g in goal.iter_mut() {
                        *g = rng.below(top.z as usize) as i32;
                    }
                }
                // The goal the demonstrator is actually building, in the same form
                // execution will supply it.
                TrainPolicy::Scripted => {
                    goal = distil_program(&h, &goal_cis, distil_iters);
                }
            }
        }

        world.fill_state(&mut state_cis);
        position_cis[0] = world.position as i32;

        h.step_with_goal(&[&state_cis, &action_cis, &position_cis], &goal, true, 0.0, 0.0);

        // Whether the predicted action matched the one actually taken — the only
        // learning signal in this demo, and the one that tells you whether training
        // is converging at all.
        let predicted = h.get_prediction_cis(1)[0];

        let action = match policy {
            TrainPolicy::Random => rng.below(NUM_ACTIONS) as i32,
            TrainPolicy::Scripted => world.scripted_action(&target),
        };
        // Idle steps are excluded. A demonstrator that has finished emits 0 until
        // the target changes, so scoring every step measures how often the world is
        // already built rather than how well the policy was cloned — an easy 90%+
        // for a hierarchy that has learned nothing but "do nothing".
        if action != 0 {
            action_hit.push(if predicted == action { 1.0 } else { 0.0 });
        }

        action_cis[0] = action;
        world.apply(action);

        if every > 0 && (t + 1) % every == 0 {
            rec.sample(t as u64 + 1, &[("action_agreement", action_hit.mean() as f64)]);
            if !quiet {
                say!(
                    "  step {:>8} / {steps} | predicted action matches demonstrator {:.1}% of the time (idle steps excluded)",
                    t + 1,
                    action_hit.mean() * 100.0
                );
            }
        }
    }

    // --- Execution ---
    //
    // Learning is off from here. Each trial draws a fresh target, distils it into a
    // program, and lets the hierarchy run on that program alone.

    say!();
    say!("Executing {trials} distilled programs, {trial_steps} steps each, learning off:");

    let mut agent = TrialScore::default();
    let mut random_goal = TrialScore::default();
    let mut random_action = TrialScore::default();
    let mut scripted = TrialScore::default();

    for _ in 0..trials {
        let start = StackingWorld::new(&mut rng);
        let target = random_stacks(&mut rng);
        fill_cis(&target, &mut goal_cis);

        // The real thing.
        let program = distil_program(&h, &goal_cis, distil_iters);
        agent.add(run_trial(&h, &program, start.clone(), &goal_cis, trial_steps));

        // Control 1: an arbitrary program. If this scores the same, the distillation
        // carried no information and the demo has shown nothing.
        let mut noise = vec![0i32; top_columns];
        for g in noise.iter_mut() {
            *g = rng.below(top.z as usize) as i32;
        }
        random_goal.add(run_trial(&h, &noise, start.clone(), &goal_cis, trial_steps));

        // Controls 2 and 3: the floor and the ceiling.
        random_action.add(run_reference(start.clone(), &target, &goal_cis, trial_steps, false, &mut rng));
        scripted.add(run_reference(start, &target, &goal_cis, trial_steps, true, &mut rng));
    }

    say!("                      held at target   best match reached");
    for (label, s) in [
        ("distilled program", &agent),
        ("random program", &random_goal),
        ("random actions", &random_action),
        ("scripted solver", &scripted),
    ] {
        say!(
            "  {label:<18}      {:>6.1}%           {:.3}",
            s.mean_held() * 100.0,
            s.mean_best()
        );
    }

    let mut summary = Summary::new();
    summary.push("held", agent.mean_held() as f64);
    summary.push("random_program_held", random_goal.mean_held() as f64);
    summary.push("baseline_held", random_action.mean_held() as f64);
    summary.push("scripted_held", scripted.mean_held() as f64);
    summary.push("best_match", agent.mean_best() as f64);
    summary.push("baseline_best_match", random_action.mean_best() as f64);
    summary.push("action_agreement", action_hit.mean() as f64);

    // The comparison that matters is against the *random program*, not against
    // random actions: both run the same trained hierarchy on the same world, and
    // differ only in whether the goal handed to it means anything. Anything else
    // could be explained by the hierarchy having learned to move blocks around at
    // all.
    let informative = agent.mean_held() - random_goal.mean_held();
    summary.push("program_advantage", informative as f64);

    say!();
    if informative > 0.05 {
        say!(
            "The distilled program carries the target: it holds it {:.1} points more of the\n\
             time than an arbitrary program does, on the same hierarchy and the same worlds.",
            informative * 100.0
        );
        summary.verdict(true, "the distilled program outperforms an arbitrary one");
    } else {
        say!(
            "The distilled program is worth no more than an arbitrary one ({:+.1} points held).",
            informative * 100.0
        );
        say!(
            "  With --train-policy random this is the expected result, and it is upstream's own\n  \
             configuration: training never shows the hierarchy a goal that means anything, so at\n  \
             execution time a distilled program is out of distribution. Try --train-policy scripted."
        );
        summary.verdict(false, "a distilled program is no better than an arbitrary one");
    }

    checkpoint::maybe_save(&h, args);
    rec.finish_summary(&summary);
    summary
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum TrainPolicy {
    Random,
    Scripted,
}

/// One trial's score.
///
/// `held` is the headline and `best` is not, which is the opposite of the obvious
/// choice and the whole reason this demo reports anything meaningful. There are only
/// ten reachable configurations of three blocks in three columns, so a random walk
/// stumbles onto any given one within a few dozen steps: scored on the best match it
/// ever touched, flailing beats every policy here, including the trained one. Scored
/// on how much of the settled second half of the trial is spent *at* the target,
/// flailing gets the ~1/10 it deserves and only a policy that arrives and stays
/// scores well.
#[derive(Clone, Copy, Default)]
struct Trial {
    best: f32,
    held: f32,
}

#[derive(Default)]
struct TrialScore {
    best: f64,
    held: f64,
    n: u64,
}

impl TrialScore {
    fn add(&mut self, t: Trial) {
        self.best += t.best as f64;
        self.held += t.held as f64;
        self.n += 1;
    }
    fn mean_best(&self) -> f32 {
        if self.n == 0 { 0.0 } else { (self.best / self.n as f64) as f32 }
    }
    fn mean_held(&self) -> f32 {
        if self.n == 0 { 0.0 } else { (self.held / self.n as f64) as f32 }
    }
}

/// Run one trial on a clone of the trained hierarchy, driven only by `program`.
///
/// The clone is what keeps trials independent: execution is a `step_with_goal` like
/// any other and still advances the hierarchy's recurrent state, so without it every
/// trial would inherit the last one's history. Learning is off throughout.
///
/// Scored on the *best* match reached rather than the final one — the hierarchy has
/// no notion of stopping, so a policy that builds the configuration and then keeps
/// fidgeting would otherwise score as a failure.
fn run_trial(
    h: &Hierarchy,
    program: &[i32],
    mut world: StackingWorld,
    goal_cis: &[i32],
    steps: usize,
) -> Trial {
    let mut copy = h.clone();
    let mut state_cis = vec![0i32; STATE_SIZE];
    let mut action_cis = vec![0i32; 1];
    let mut position_cis = vec![0i32; 1];
    let mut score = Score::new(steps);

    for t in 0..steps {
        world.fill_state(&mut state_cis);
        position_cis[0] = world.position as i32;
        score.observe(t, match_fraction(&state_cis, goal_cis));

        copy.step_with_goal(&[&state_cis, &action_cis, &position_cis], program, false, 0.0, 0.0);

        action_cis[0] = copy.get_prediction_cis(1)[0];
        world.apply(action_cis[0]);
    }

    world.fill_state(&mut state_cis);
    score.observe(steps, match_fraction(&state_cis, goal_cis));
    score.finish()
}

/// Accumulates a trial's best match and its time-at-target over the settled second
/// half. The halfway point is the settling allowance: the scripted solver needs
/// roughly twenty steps to build a target from the worst start, so half of a
/// 96-step trial is ample for any policy that is going to arrive at all.
struct Score {
    from: usize,
    best: f32,
    at_target: usize,
    counted: usize,
}

impl Score {
    fn new(steps: usize) -> Self {
        Self { from: steps / 2, best: 0.0, at_target: 0, counted: 0 }
    }

    fn observe(&mut self, t: usize, m: f32) {
        self.best = self.best.max(m);
        if t >= self.from {
            self.counted += 1;
            if m == 1.0 {
                self.at_target += 1;
            }
        }
    }

    fn finish(self) -> Trial {
        let held =
            if self.counted == 0 { 0.0 } else { self.at_target as f32 / self.counted as f32 };
        Trial { best: self.best, held }
    }
}

/// The same trial under a reference policy: scripted when `scripted` is `Some`,
/// uniformly random otherwise.
fn run_reference(
    mut world: StackingWorld,
    target: &[usize],
    goal_cis: &[i32],
    steps: usize,
    scripted: bool,
    rng: &mut Rng,
) -> Trial {
    let mut state_cis = vec![0i32; STATE_SIZE];
    let mut score = Score::new(steps);

    for t in 0..steps {
        world.fill_state(&mut state_cis);
        score.observe(t, match_fraction(&state_cis, goal_cis));

        let action = if scripted {
            world.scripted_action(target)
        } else {
            rng.below(NUM_ACTIONS) as i32
        };
        world.apply(action);
    }

    world.fill_state(&mut state_cis);
    score.observe(steps, match_fraction(&state_cis, goal_cis));
    score.finish()
}
