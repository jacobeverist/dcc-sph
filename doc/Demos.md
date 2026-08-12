# Demos

Nine demos ported from [`jacobeverist/OgmaNeoDemos`](https://github.com/jacobeverist/OgmaNeoDemos/tree/aogmaneo) (branch `aogmaneo`), Ogma Intelligent Systems Corp, CC BY-NC-SA 4.0 — the same licence as this crate. The attribution required by §3(a) is in [`PROVENANCE.md`](../PROVENANCE.md); this file is the engineering half, recording what each demo does and where it departs from its source.

They all **run headless and text-only with no features enabled**. That is the default path and the one CI builds. The optional `macroquad-demos` feature opens a window for the demos where motion is the point; it exists so a demo can be eyeballed quickly, not as instrumentation — that is dcc-dashboard's job.

## Running them

```bash
cargo run --release --example wavy_line
cargo run --release --example wavy_line -- --steps 40000 --ahead 8
```

Every demo takes `--steps` (or `--episodes`), `--seed`, `--every` (report interval) and `--quiet`. `--seed` fully determines a run: it seeds both the library's global RNG and the demo's own environment stream (`examples/support/rng.rs`).

## Getting the numbers out

`--metrics <path>` writes machine-readable records alongside the usual text. Stdout is unchanged — this is strictly additive, and `--quiet` still governs printing only.

```bash
cargo run --release --example pusher -- --steps 300000 --metrics run.jsonl
```

JSONL is the default: one self-describing object per line, so a consumer can tail it without knowing which demo produced it.

```
{"kind":"run","demo":"pusher","seed":12345,"run":0,"config":{"exploration":0.05,"steps":300000,"timeout":500}}
{"kind":"sample","demo":"pusher","seed":12345,"run":0,"step":50000,"metrics":{"reward_ema":0.021,"goals_per_100k":24.0}}
{"kind":"summary","demo":"pusher","seed":12345,"run":0,"metrics":{"goals_per_100k":16.0,"baseline_goals_per_100k":2.0,"goals_vs_random":8.0}}
{"kind":"verdict","demo":"pusher","seed":12345,"run":0,"learned":true,"note":"reaching the goal far more often than random action"}
```

Every record carries `demo`, `seed` and `run`, so files from different demos or different seeds can be concatenated and still make sense. `--metrics-format csv` emits long format instead — `demo,seed,run,kind,step,metric,value`, one row per number.

Two properties CI checks, because everything else rests on them: the file parses, and **the same seed produces byte-identical metrics**. Non-finite values are written as `null` rather than `NaN`, so a file always parses even when a metric is undefined.

Where a demo reports a baseline it also records the ratio (`goals_vs_random`, `furthest_vs_random`). That is the figure worth comparing across runs — absolute counts move with `--steps`, but "how many times better than random" does not.

Without `--metrics` the recorder is inert: no file, no buffer, and the call returns before touching its arguments.

## Running a demo more than once

A single run of an RL demo does not tell you much, and this suite has already been misled by one: see the `wavy_classify` section below, where three separate claims in this file turned out to rest on single seeds.

```bash
--repeat 5                      # five seeds, reported as mean ± sample stddev
--sweep layers=2,3,4,5          # once per value of --layers
--sweep layers=2,3 --repeat 5   # both: five seeds at each of two settings
```

`--sweep` works by overriding one argument and re-running, so no demo needs to know it is being swept — it reads its knobs off `Args` as usual, and any argument is sweepable. Each sweep point sees the same seed sequence, so points are compared like for like.

The output is a table of mean ± stddev per metric per point, plus a `learned` row giving the fraction of seeds whose verdict was positive. **That fraction is often the more informative number**: a task that works a third of the time and fails otherwise has a meaningless mean.

In matrix mode each individual run is silenced (`--silent`, which suppresses the final report as well as the periodic lines that `--quiet` covers) — twenty runs of scatter plots and ASCII frames would bury the comparison. Every run still writes its own records to `--metrics`, tagged with its seed and sweep point, so nothing is lost to aggregation.

A difference smaller than the spread is not a difference. The table prints that reminder under itself for a reason.

## Saving and resuming

```bash
cargo run --release --example runner -- --steps 300000 --save runner.ohr
cargo run --release --example runner -- --steps 300000 --load runner.ohr   # carry on
```

`--save` / `--load` persist the whole model; `--save-weights` / `--load-weights` persist weights without the running state, which is smaller and portable across runs that differ in where they happened to be in a sequence. `cat_mouse` has two agents, so it writes the mouse to `<path>.mouse` alongside the cat.

A missing or unreadable checkpoint is fatal rather than a warning. A run that silently trained from scratch when it was told to resume would waste exactly the time it was meant to save.

This is upstream's `S`-to-save key, and it covers `Hierarchy::write`/`read` and `write_weights`/`read_weights` — the latter pair having previously been called from nowhere in the repository at all.

## What the actor is thinking

The RL demos record two numbers that explain *why* a run is or is not learning, and that nothing in this repository could previously observe — `Hierarchy::get_actor` was never called, so the critic and the replay buffer were unreachable:

- **`critic_value`** — the actor's mean value estimate. It should move as the policy finds reward.
- **`history_fill`** — how full the credit-assignment history is, as a fraction. **Learning does not begin until this passes `actor::Params::min_steps`**, so an early run that looks dead is often just waiting for it. In `pusher` it reaches 1.0 by about step 1000.

`examples/support/probe.rs` also offers `prediction_confidence` (mean peak softmax over a port — it rises before any accuracy metric moves), `action_values`, and `layer_updates`. Each is guarded: `get_actor` and `get_prediction_values` index through `d_indices`, which holds a *decoder* index for a Prediction port and `-1` for a `None` port, so calling them unguarded reads the wrong actor or panics.

## Driving a demo programmatically

Each demo's hierarchy lives in its environment module — `env::pusher::build_hierarchy()`, `env::ball::build()`, `env::wavy::build_line_hierarchy()`, and so on — rather than inline in `main`. The windowed viewer and the headless demo therefore cannot drift apart, and a caller can construct exactly the configuration a demo uses.

Each demo is also split into `run(args, seed, rec) -> Summary` and a `main` that drives it once. `Summary` is a list of named metrics plus the verdict, so a caller can aggregate across runs without knowing what any particular demo measures.

| Demo | Upstream source | What it exercises |
|---|---|---|
| `wavy_line` | `demos/Wavy_Line.cpp` | Sequence prediction; `write_state`/`read_state` round trip |
| `wavy_classify` | `demos/Wavy_Classify.cpp` | Two Prediction ports, per-port `importance`, `ticks_per_update` |
| `ball_physics` | `demos/Ball_Physics.cpp` | `ImageEncoder` + `reconstruct()`, closed-loop generation |
| `pusher` | `demos/Pusher.cpp` | `Actor`, multi-column action port, shaped reward |
| `cat_mouse` | `demos/Cat_Mouse.cpp`, `demos/catmouse/CatMouseEnv.cpp` | **Two hierarchies**, zero-sum reward, `IoType::None` observation |
| `car_racing` | `demos/Car_Racing.cpp` | `Actor` steering, raycast sensors, a real track asset |
| `runner` | `demos/Runner_Run.cpp`, `demos/runner/Runner.cpp` | 8-motor articulated body, 24-column observation port |
| `enc_vis` | `demos/EncVis.cpp` | Bare `Encoder`, receptive-field readout |
| `topo_test` | `demos/Topo_Test_AON.cpp` | `Encoder` topology preservation |

## Layout

`examples/support/` holds everything shared: argument parsing, CSDR encoding, text reporting, the RNG wrapper, the encoder probe, and one module per environment. Cargo examples cannot depend on each other, so each demo pulls it in with `#[path = "support/mod.rs"] mod support;` — the idiom `examples/fidelity_dump.rs` already uses for `tests/support/`. A directory under `examples/` with no `main.rs` is not auto-discovered, so `support/` is not itself a target.

Example targets default to `test = false`, so `#[cfg(test)]` code inside them never runs. `tests/demos_support.rs` includes the same tree, which compiles it in test configuration and runs its unit tests as part of `cargo test` — 63 of them, covering the encoding helpers, the reporting primitives, every environment's physics and the encoder probe.

## Cross-cutting decisions

**`ticks_per_update: 1` everywhere except `wavy_classify`.** The tick-gating mechanism is an addition of this crate that AOgmaNeo `645a54a` does not have (see [`Divergences.md`](Divergences.md)), so keeping it at 1 makes the demos behave as upstream does. `wavy_classify` is the exception and needs to be — see below.

**`mimic = 0.0`.** Upstream calls `h.step(inputs, learn, reward)`; this crate's fourth parameter has no upstream counterpart.

**Every RL demo runs a random-action baseline first.** Goals per 100k steps, crashes per 1000 frames and metres travelled are meaningless in isolation: the pusher's object can drift onto its goal unaided, a car that never steers still covers ground, and a flailing runner still moves. Each headline number is reported against what random play achieves on the same world and seed.

**Assets.** Only `car_racing` needs one. `assets/racingCollision.png` and `assets/racingCheckpoints.png` are upstream's, 26 KB together. The background, foreground and car sprite are skipped: 590 KB that only ever gets drawn. Two further assets referenced by upstream demos — `resources/map0.png` and `resources/density_image5.png` — **are absent from the upstream repository**, so `cat_mouse` and `enc_vis` generate theirs.

## Deviations, demo by demo

### `wavy_line`

- **Both prediction horizons are reported.** Upstream computes the true 1-step prediction and then plots `mPredIndice` — the N-step value — for *both* its "1-step" and "N-step" curves, so the 1-step prediction is never displayed.
- **Baselines are horizon-matched.** Persistence is measured at the same horizon as the prediction it judges. These signals are smooth and heavily oversampled, so 1-step persistence sits near the encoder's quantisation floor and is nearly unbeatable; scoring an N-step prediction against it would say nothing.
- **The encoding clamps.** Upstream's `simpleFloat2CSDR` does not, so a large enough noise spike produces an out-of-range column index there.
- `--check-state` verifies every rollout's `write_state`/`read_state` round trip leaves the predictions bit-identical, rather than assuming it. This is the demo's main reason for existing.

Typical result over 30k steps: 1-step MAE 0.0113 against 0.0359 for persistence; 5-step 0.0230 against 0.1754. The 1-step figure sits just above the encoder's mean quantisation error of 0.0099, so it is at the resolution limit of the encoding rather than of the model.

### `wavy_classify`

- **`ios[1].importance` defaults to 0.0, not upstream's 0.1.** `importance` weights an IO port on the *encoder input* side only; the decoder predicts that port from the hidden state regardless. At 0.1 the true label reaches the hidden state during training, and since the label is constant for `--hold` steps at a time, "predict the next label" is solved by the identity — copy the label just given. The decoder never learns to infer class from the signal, and at inference, when the label is withheld and the port is fed the model's own prediction, that identity latches onto whatever it emitted first: the confusion matrix collapses into a single column. At 0.0 the label cannot reach the hidden state at all and the decoder has to do real classification.
- **The layer stack uses `ticks_per_update` 1, 2, 4, 8.** Telling these classes apart means measuring frequency — class 1 has a period of ~22 steps, class 0 of 80, and classes 3 and 4 differ only by a 40-step component — which needs tens of samples of context. A flat stack has only self-recurrence to work with. Upstream's `//lds[i].ticks_per_update = 2;` is commented out: the mechanism did not exist in that AOgmaNeo revision.
- **Accuracy is measured.** Upstream never measures it at all — there is no counter anywhere in the file, and it is judged by eye from two overlaid curves. This reports a confusion matrix with per-class recall, both overall and excluding a settling window after each class switch.
- **The `USE_SENSOR_DATA` path is not ported.** It needs `resources/training_camdataDetrend.txt`, 32,173 rows and 2.5 MB.
- The demo also prints an *online* training accuracy, explicitly labelled optimistic: the decoder is updated toward the current target immediately before the activation that reads it, so it runs near 100% even when nothing generalisable has been learned. It is there to separate a shortcut from an inseparable task, which is how both problems above were found.

**This demo is strongly seed-dependent, and the numbers previously recorded here were a single lucky seed.** Running it under `--repeat` says so plainly:

```bash
cargo run --release --example wavy_classify -- --train-steps 60000 --test-steps 15000 --sweep layers=2,4 --repeat 3
```

| metric | layers=2 | layers=4 |
|---|---|---|
| settled_accuracy | 0.3104 ± 0.2446 | 0.3143 ± 0.2421 |
| learned | 33% of runs | 33% of runs |

Three things follow, and all three contradict what this file used to claim.

**Only about a third of seeds learn.** The rest sit at chance. A seed that works reaches roughly 66% settled accuracy against 20% chance — which is where the previously quoted figure came from — but quoting it alone was misleading, because the spread across seeds (±0.24) is larger than the mean.

**Layer count makes no measurable difference between 2 and 5.** All four settings land within 0.01 of each other against a seed-to-seed spread of 0.24 to 0.25. The earlier claim that "four layers beats five (75% vs 67%)" was two single-seed runs at different step counts, and does not survive repetition.

**The `label_importance` difference is real in mechanism but not clear in aggregate.** Sweeping `label-importance=0.0,0.1` over four seeds gives 0.3147 ± 0.1977 against 0.2611 ± 0.2553 — a gap far smaller than the spread. What *is* visible is the predicted failure mode: at 0.1, classes 1 and 2 are never predicted at all (`recall_c1` 0.007, `recall_c2` 0.000), which is the identity-shortcut collapse described above. So the reasoning holds and the default stays at 0.0, but it should not be presented as a measured accuracy win.

The honest summary is that this is a hard, high-variance task where the model either finds the structure or does not. Report it that way rather than quoting a best case — and note that the `learned` row of a sweep, which counts the fraction of seeds that succeeded, is the more informative number here than any mean.

### `ball_physics`

- **The physics is hand-written; Box2D is not a dependency.** One circle under gravity in an axis-aligned box needs no solver. The geometry constants are upstream's Box2D body definitions translated into the surfaces they produce: ground top at `y = 0`, wall inner faces at `x = ±7.5`, ball radius 1.4, restitution 0.82.
- **The scene is rasterised in software** into the 64×64 buffer the `ImageEncoder` reads. One subtlety worth recording: upstream draws through a default-constructed `sf::View`, whose size is SFML's default 1000×1000 view-pixels, so the visible region is 15.6 m across — not the 1 m it looks like from the 64 px target at 64 px/m.
- **Position error is the headline metric, not frame MSE.** MSE prefers a *blank* frame to a slightly misplaced ball, because a misplaced ball is wrong twice over — once where it is drawn and once where it should have been. An undertrained model that collapsed to empty space scored 0.026 against 0.043 for a trained one that keeps a ball alive. The demo recovers the ball's position from each frame by subtracting the static background, and reports tracking error bucketed by how long the loop has run unaided.

Typical result: the ball persists in 100% of generated frames at 3.4 m mean error against 5.3 m for a frozen frame, in a 15.6 m view. It learns the dynamics but decorrelates from the true trajectory, which the per-horizon breakdown shows directly.

### `pusher`

- **An episode timeout was added, and the demo does not work without one.** Upstream has no episode limit. The object only ever moves while the pusher overlaps it, so a policy that stops moving freezes the world: no reward, no termination, nothing to learn from, for ever. Standing still is a perfect local optimum worth exactly 0, and the actor finds it within about 50k steps and never leaves — goals and losses both drop to zero and stay there. `--timeout 0` reproduces upstream.
- **Exploration defaults to 0.05.** Upstream wires up an exploration hook and leaves it at 0, relying on the actor's own stochastic policy; that is not enough to escape the local optimum above.

Typical result: 16 goals per 100k steps against 2 for random, losing the object less often than random does.

### `cat_mouse`

- **The maze is generated.** `resources/map0.png` is absent from the upstream repository. It is a randomised depth-first maze, then *braided* — extra walls reopened to create loops. A perfect maze gives the mouse nowhere to dodge and the cat a guaranteed corner, so neither agent has anything to learn; upstream's hand-drawn map has open rooms for the same reason.
- **The default maze is 5×5 cells.** Bigger looks better but makes capture so rare that nothing is measurable: at 8×8 a random cat catches the mouse in about 9% of episodes, and a 40k-step run produces two captures total.
- **An episode timeout was added.** Without one, a mouse that simply outruns the cat produces an episode that never ends and mean time-to-capture is unmeasurable.
- **The curiosity reward is not ported.** It compares observations against `get_prediction_cis(0)`, and this crate returns an **empty slice** for an `IoType::None` port, so it would panic. It is commented out of the reward upstream anyway.

Typical result over 250k decisions: capture rate 32.8% against 21.1% for random play, with mean steps-to-capture falling monotonically — the cat out-learning the mouse.

### `car_racing`

- **Only the collision mask and checkpoint ring are vendored**, as above. The `png` dev-dependency decodes them.
- Otherwise a close port: 12 rays at `0.16·(s − 6) + rotation` cast in 2-pixel increments, the same drag-and-accelerate car model, and upstream's reward — speed projected onto the direction of the track, so going fast the wrong way scores negative.

Typical result over 100k frames: 13 laps, crashes down from 28 to 1.7 per 1000 frames, 4× the distance of random steering. The strongest result of the RL demos.

### `runner`

- **`rapier2d` stands in for Box2D**, as a dev-dependency. This is the only demo whose physics is not hand-written: four limbs of two segments, eight revolute motors with angle limits and torque caps, contact sensing and world raycasts need a constraint solver.
- **Joint limits are offset by each joint's assembly angle.** Box2D lets a revolute joint carry a reference frame — upstream sets `frameA.q = relativeAngle` — so limits and the reported angle are measured from the pose the limb was assembled in. rapier's 2D `RevoluteJointBuilder` has no such frame: limits apply to the raw relative rotation. Since the segments are assembled at −0.75π and +0.5π, applying `[-1.1, 1.1]` directly constructs every joint already outside its own limit; the solver then locks the whole body rigid and the runner cannot move at all, under any policy, at any torque. It fails silently — nothing errors, the demo just reports zero distance, and even the random baseline reaches 0.00 m. There is a regression test for it.
- **`MotorModel::ForceBased`.** Box2D's `maxMotorTorque` is a torque; `AccelerationBased` would scale the cap by each segment's inertia and let the motors overpower their own angle limits.
- **The four foot-contact flags go to fixed slots.** Upstream's `runner/Runner.cpp` writes `state[si++] = 1.0f` *inside* the "is this foot touching" conditional, so the write index only advances when a contact is found — the whisker and IMU readings shift position in the vector by however many feet happen to be on the ground, and the tail is left at zero. Reproducing that is possible but makes the sensor layout contact-count dependent for no benefit, and it changes the learning problem either way.
- Upstream's co-located limbs are preserved: both the "back" and "front" limbs attach at the *same* hip point on each side, so this is two legs per hip rather than fore and aft legs.
- The IMU reports per-frame velocity *differences*, not accelerations — upstream never divides by dt, and the sensor scaling depends on it.

This is by far the hardest problem in the suite: a gait has to be discovered from a sparse velocity signal across eight coupled motors. Expect it to need far more steps than the other demos — and it is the slowest to run, being the only one simulating rigid bodies.

Typical result over 200k control steps, with the furthest point reached in each window: 3.8 m → 6.3 m → 14.8 m → 20.2 m, mean velocity −0.11 → +0.70 m/s, resets falling from 6.1 to 1.7 per 1000 steps. Against a random baseline of 0.99 m. Almost all resets are hurdle collisions rather than stalls, which is the signature of a body that is actually travelling.

### `enc_vis` and `topo_test`

Both upstream demos read a stored `vl.means` scalar per cell — already a normalised position. **This crate's `encoder::VisibleLayer` has no `means`.** It ports a byte-weight ART formulation whose weights are indexed by the *input cell* as well, so instead of a scalar there is a learned histogram over the input column, and the position is its weighted centroid. `examples/support/encoder_probe.rs` does that decoding.

Two things about it matter:

- **The centroid is taken over supra-threshold weights only.** A 64-cell input column carries ~224 units of uniform initialisation noise spread evenly across it against a single 255-unit spike, so a raw centroid is dragged almost halfway to the middle of the range — it reports roughly 0.36 for a cell trained solely at 0.25. Weights start in `0..8` and learning drives winners to 255, so thresholding at 128 is unambiguous.
- **The probe also reports `compactness`**, the fraction of a cell's index span that is actually learned. ART never requires a committed set to be contiguous, and upstream's stored mean cannot represent a split set at all.

`enc_vis` additionally generates its density field procedurally, since `resources/density_image5.png` is missing upstream, and exposes `--vigilance`. `topo_test` samples all clusters uniformly, where upstream feeds only the cluster selected with the number keys, and skips the `class Enc` at the top of the upstream file — a hand-written reference learner that is never instantiated in `main`.

**Both report negative results, and both are correct to.** `Encoder` cells commit to scattered input sets rather than contiguous bands (compactness ~0.22); raising vigilance to 0.99 makes them far more selective — about 2 input levels instead of 17 — without making them contiguous. And adjacent cells within a column are no closer in input space than randomly paired ones (0.3495 against 0.3436).

More training cannot change this. `Encoder` has **no topology-forming mechanism**: its only neighbourhood parameter, `Params::l_radius`, drives lateral inhibition — it decides whether a column may learn by counting how many neighbours scored higher — and never updates a neighbour's weights. Learning touches the winning cell and nothing else, so a cell's index within a column carries no spatial meaning. `ImageEncoder` is the SOM here: it carries `Params::falloff` and `Params::n_radius`, and updates cells at distance `d` from the winner at `rate * falloff^d`. That is worth knowing before reaching for `Encoder` expecting a map. The upstream demos probe an AOgmaNeo revision whose encoder stored `vl.means`, a different formulation; the difference is algorithmic, not a port defect.

`enc_vis` prints the raw weight profiles alongside its summary so the claim can be checked rather than taken on trust.

## Not ported

| Upstream | Why |
|---|---|
| `Loop_Counter`, `Loop_Mapper`, `Single_Lap_Mapper`, `TrackSOM`, `Car_Tracker`, `Donkey_Playback`, `FP_Playback`, `SDC_Controller_Test` | **Blocked on missing data, not on code.** They need `resources/data/video0.avi`, `resources/video0.avi`, `resources/singlelap.avi` and matching `control0.txt` / `racevit_controls.txt` throttle-and-steer logs. None of those files exists on *any* branch of the upstream repository — `Bullfinch192.mp4` and `Tesseract.wmv` are the only video files that were ever committed. `Single_Lap_Mapper` and `TrackSOM` also link no `aogmaneo/` header at all. |
| `Fluid` | Pure inference from pretrained `resources/fluidsim.oenc` / `.ohr`, which exist on no branch, and the file contains no training loop to regenerate them. It also feeds the image encoder `F32_Array` float pixels, which neither this crate nor upstream `master` accepts. Permanently blocked. |
| `Image_Encoder_Test` | Uses a **DCT-based** `Image_Encoder` (`init(Int2)`, `get_dct_size()`, `get_dct_bases()`, `get_encoding_size()`, `encode()`, `get_encoded_cis()`). This is not something that could be ported: all 404 branches of `ogmacorp/AOgmaNeo` were searched, plus `CLOgmaNeo`, `EOgmaNeo`, `OgmaNeo2`, `PyAOgmaNeo` and `TiOgmaNeo`, and there are **zero** occurrences of `dct`, `get_encoded_cis` or `get_encoding_size` anywhere. The demo targets an unpublished working copy. Writing it would be invention rather than porting — the coefficient-to-column-index quantiser is a free design parameter with no reference to validate against — and it would unlock one demo whose payoff is `ball_physics` with a fixed transform instead of a learned SOM. |
| `Ball_Physics_Vec`, `VSA_Char`, `VSA_Tests_Comp` | Use a templated `Hierarchy<S, L>` over **hypervectors**, with `Int2` hidden sizes and `get_prediction_vecs()`. This is a *different algorithm*, not a newer version of this one: upstream `master`'s `IO_Desc` is field-for-field identical to this crate's `IoDesc` (same eight fields, same defaults) and its `step` signature matches too — the only mainline drift since `645a54a` is `Params::anticipation` and `Layer_Desc::num_dendrites_per_cell`, both of which this crate already has. The `Int2` API lives in the **`SVECTOR` branch family** (20 of the 404 branches), roughly 63 KB of headers implementing a second learner, and the exact revision these two demos want is unpublished even there. |
| `C_Test`, `Wavy_Classify_Old`, `Cat_Mouse_Torch`, `Pusher_Goal` | Variants or earlier drafts of demos already covered. `C_Test` is additionally broken as checked in: `num_dendrites_per_cell = 0` on two ports, and a four-entry direction table turned with `% 3`, so one direction is unreachable and turning is asymmetric. |
| ~40 others (`ARTTest`, `ART_Visualizer`, `Basic`, `Basic2`, `Evo`, `STDP`, `SOMGridCells`, `Topo_Test`, `VSA_Tests*`, `Generative`, `Marcher`, `TEM`, `Swarm_*`, `Stacking`, `Stacking_BP`, `Car_Racing_RL`, `Reacher`, `NaviGraph`, …) | Include no `aogmaneo/` headers at all — standalone research sketches sharing the repository. Note in particular that **none of the MNIST demos uses AOgmaNeo**: `ART_Visualizer`, `Basic`, `Basic2`, `Generative` and `NaviGraph` are a hand-written ART implementation, a diffusion sketch and a grid-cell graph. An MNIST loader would let this repository run someone else's algorithms, not `dcc_sph`. |

Of the 69 top-level demos upstream, 27 link AOgmaNeo.

## Graphics

The whole of the `macroquad-demos` feature is one extra target, `examples/viewer.rs`:

```bash
cargo run --release --example viewer --features macroquad-demos -- --demo car_racing
```

`--demo` takes `ball_physics`, `cat_mouse` or `car_racing` — the three where motion is the point. It trains live and draws what is happening, the way the upstream SFML demos do. Space fast-forwards; Escape quits; in `ball_physics`, `G` closes the loop so the hierarchy generates from its own predictions.

The other six demos are text-only by design. A scrolling plot, an ASCII frame or a scatter says everything a window would for them, and they already print it.

Two things keep this honest. The viewer builds its hierarchies through the same `build_hierarchy` functions in `support/env/` that the headless demos use, so the two configurations cannot drift apart. And `support/viz.rs` is five functions — `View`, `blit_gray`, `plot_series`, `scatter`, `hud`. If it starts growing state, options or layout logic, that is the signal the work belongs in dcc-dashboard rather than here.

The feature is optional and kept out of `--all-features` in CI, for the same reason as `gymnasium-examples`: a CI runner has no display. CI checks it still compiles; opening a window is not CI's job.

## What CI does

`cargo build --all-targets`, `cargo test --all-targets`, `cargo clippy --all-targets -- -D warnings`, and then **runs all nine demos** at small step counts. Building is not enough — a demo that panics on step one still compiles.
