# Demos

Fifteen demos ported from [`jacobeverist/OgmaNeoDemos`](https://github.com/jacobeverist/OgmaNeoDemos/tree/aogmaneo) (branch `aogmaneo`), Ogma Intelligent Systems Corp, CC BY-NC-SA 4.0 — the same licence as this crate. The attribution required by §3(a) is in [`PROVENANCE.md`](../PROVENANCE.md); this file is the engineering half, recording what each demo does and where it departs from its source.

They all **run headless and text-only with no features enabled**. That is the default path and the one CI builds. A windowed viewer lives in the separate `examples-viz` crate for the demos where motion is the point; it exists so a demo can be eyeballed quickly, not as instrumentation — that is dcc-dashboard's job.

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
| `video_prediction` | `demos/Video_Prediction.cpp` | RGB `ImageEncoder`, multi-pass training, generated video |
| `pusher` | `demos/Pusher.cpp` | `Actor`, multi-column action port, shaped reward |
| `cat_mouse` | `demos/Cat_Mouse.cpp`, `demos/catmouse/CatMouseEnv.cpp` | **Two hierarchies**, zero-sum reward, `IoType::None` observation |
| `cat_mouse_pos` | `demos/Cat_Mouse_Pos.cpp` | A port's own prediction fed back as its next input |
| `explore` | `demos/Explore.cpp` | Curiosity as the whole reward — prediction error |
| `car_racing` | `demos/Car_Racing.cpp` | `Actor` steering, raycast sensors, a real track asset |
| `runner` | `demos/Runner_Run.cpp`, `demos/runner/Runner.cpp` | 8-motor articulated body, 24-column observation port |
| `vsa_char` | `demos/VSA_Char_Single.cpp` | Hypervector coding; a CSDR fed with no encoder |
| `enc_vis` | `demos/EncVis.cpp` | Bare `Encoder`, receptive-field readout |
| `topo_test` | `demos/Topo_Test_AON.cpp` | `Encoder` topology preservation |
| `stacking_rl` | `demos/Stacking_RL.cpp` | Goal-conditioned RL; two routes for delivering the goal, compared |
| `stacking_prog` | `demos/Stacking_Prog.cpp` | `step_with_goal`, and a goal distilled into a top-layer CSDR |

## Layout

`examples/support/` holds everything shared: argument parsing, CSDR encoding, text reporting, the RNG wrapper, the encoder probe, and one module per environment. Cargo examples cannot depend on each other, so each demo pulls it in with `#[path = "support/mod.rs"] mod support;` — the idiom `examples/fidelity_dump.rs` already uses for `tests/support/`. A directory under `examples/` with no `main.rs` is not auto-discovered, so `support/` is not itself a target.

Example targets default to `test = false`, so `#[cfg(test)]` code inside them never runs. `tests/demos_support.rs` includes the same tree, which compiles it in test configuration and runs its unit tests as part of `cargo test` — 126 of them, covering the encoding helpers, the reporting primitives, the hypervector algebra, every environment's physics and the encoder probe.

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

### `video_prediction`

An `ImageEncoder` compresses each RGB frame to a CSDR, a hierarchy predicts the next one, and after several passes the loop closes: the hierarchy is fed its own predictions with learning off and generates the rest unaided.

- **No video decoder, and no committed clip.** Upstream uses OpenCV but *only* for `cv::VideoCapture` frame reading — no `resize`, no `cvtColor`, and its rescale runs at scale 1.0, so it is a no-op. What the demo needs is a sequence of RGB frames, which is not a reason to take a video-decoding dependency. The default source is **procedural** (drifting shapes with parallax), so the demo runs out of the box and in CI. `--frames <dir>` points it at real extracted frames, decoded with the existing `png` dev-dependency:

  ```bash
  ffmpeg -i resources/Bullfinch192.mp4 -vf scale=64:64 frames/%04d.png
  cargo run --release --example video_prediction -- --frames frames/
  ```

  That also sidesteps redistributing a derivative of Ogma's 2.7 MB clip.
- Note the buffer layout for a 3-channel visible layer is `channel + 3 * (y + h * x)`, which differs from the single-channel `y + x * h` that `ball_physics` uses.

**The verdict needs two checks, and neither alone would do.** Frame MSE can be beaten by *hedging* — emitting the blurry average of everywhere the scene might be — which scores well while having learned nothing about the motion. So the demo also reports **detail**: the standard deviation of pixel intensity in the generated frame against the real one. Hedging drives that toward zero. Retaining detail alone would not be enough either, since echoing the last frame verbatim retains all of it, which is exactly what the frozen baseline does.

Typical result over 6 passes of a 90-frame procedural clip: generated MSE 0.021 against 0.041 for a frozen frame, retaining 79% of the real detail. Both conditions met, so it is generating rather than hedging.

This is the same lesson `ball_physics` produced from the other direction — there, MSE preferred a blank frame to a slightly misplaced ball. Frame-space error is a poor judge of generative video in both directions, and each demo now carries a metric that is not fooled in its own way.

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

### `cat_mouse_pos`

The same chase as `cat_mouse` plus a third Prediction port whose output is never compared against anything. Each prediction nudges an accumulator, and what is fed back into the port next step is the **residual** between that accumulator and the prediction — so the port has to emit whatever increment keeps the estimate consistent with what the agent sees. Dead reckoning, learned end to end rather than supplied.

This is the only demo where a port's own prediction becomes its next input, and the only one where the hierarchy maintains state outside itself.

- **The `IO_Desc` arity had to be re-mapped.** Upstream passes six positional arguments — `IO_Desc(size, type, 4, 8, 2, 2)` — which is a 6-field variant, not mainline's 8-field order. Read against mainline that would set `value_size = 2`, which is nonsense. Mapped to `num_dendrites_per_cell: 4, up_radius: 8, down_radius: 2` with the RL fields left at their defaults.
- Upstream's `map0.png` is missing, so the maze is generated as in `cat_mouse`.

Typical result over 150k decisions: capture rate 35.9% against 11.8% for random movement, mean steps-to-capture falling from 313 to 243. The chase clearly works.

**The `memory_error` figure needs reading narrowly.** It compares the estimate's first two components against the cat's normalised position, and it sits at 0.378 against a random-guess distance of 0.3826 — that is, at chance. But *nothing constrains the port to encode position in the frame that metric assumes*: a rotated, permuted, reflected or offset encoding would be just as useful to the agent and would still score at chance. The measurement shows the estimate is not a drop-in world position; it does not show the estimate carries no positional information. Upstream does not measure this at all, so there is no reference figure to compare against.

The verdict is therefore decided on the chase, which depends on no such assumption. Whether the extra port *helps* is a third question again — run plain `cat_mouse` at the same seeds and compare `capture_rate`.

The random-guess constant is derived rather than guessed: each wrapped axis difference is uniform on `[0, 0.5]`, so the expected distance is `0.5 · E[hypot(u, v)]` for `u, v ~ U(0, 1)`, which has the closed form `(√2 + ln(1 + √2)) / 3`, giving 0.3826. A 400k-sample simulation agrees to four places.

### `explore`

No goal and no external reward: the agent's reward **is** its own prediction error, the fraction of observation columns the hierarchy got wrong. Predicting your surroundings well earns nothing, so the only way to score is to go somewhere unmodelled.

This works only because the observation port is `IoType::Prediction`. It is exactly the term that had to be dropped from `cat_mouse`, whose observation port is `IoType::None` and returns an empty slice from `get_prediction_cis` — the two demos are the same world with that one difference, which is why `explore` reuses `env/catmouse.rs` rather than porting a near-duplicate of upstream's `ExploreEnv`. Upstream's `map_test.png` is missing, so the maze is generated as elsewhere.

**Measuring it took two corrections worth recording.** The first version compared 60k agent decisions against a 10k random walk and reported 100% coverage against 60% — but coverage is *monotonic*, so that gap was almost entirely the extra time. The baseline now runs for the same number of decisions by default.

With that fixed the comparison collapsed to 92.1% against 89.5%, because on a small maze a random walk eventually reaches nearly everything: **final coverage saturates and cannot discriminate**. So the demo also reports time-to-coverage, which still can. Typical result over 60k decisions on a 13×13 maze:

| | agent | random walk |
|---|---|---|
| final coverage | 92.1% | 89.5% |
| steps to 50% | 7438 | 5149 |
| steps to 90% | 10870 | never reached |

The random walk is *faster* to 50% — curiosity takes time to bootstrap, since an untrained model is surprised by everything and the signal carries no direction. It is the tail that separates them.

Note a milestone the baseline never reaches counts as a win rather than as missing data. Treating that `NaN` as a failure, which the first version did, reported the opposite of what happened.

Curiosity is a *vanishing* signal — it pays only while the model is still wrong, so it fades exactly where the agent has already been. That is what makes it interesting and also what makes it slow; `--cells 10` and longer runs show it more clearly.

The demo also accumulates upstream's hypervector "map": the observation bound to a positional vector and bundled into one `Bundle` standing for the whole space, which `support/vec.rs` makes nearly free.

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

### `stacking_rl` and `stacking_prog`

Three blocks, three columns, a grabber that carries one at a time, and four actions: do nothing, grab-or-place, left, right. A target configuration is shown and the job is to build it. The world is thirty lines and deliberately trivial — the interest is entirely in *how the target is communicated*, which is what both demos are about and what nothing else in the suite does.

Both upstream files call a three-argument `step` that mainline AOgmaNeo does not have. Reading the two sources side by side, they do not call the *same* one:

| | Upstream's goal argument | Status |
|---|---|---|
| `Stacking_Prog.cpp` | `Int_Buffer goalCIs = h.get_top_hidden_cis();` — a single buffer over the top layer's hidden columns | Published, on AOgmaNeo's `ubl3_recurrent` branch |
| `Stacking_RL.cpp` | `Array<S32_Array_View> goalCIs(ioDescs.size());` — one goal buffer **per IO port**, of which only `goalCIs[0]` is filled | On no published branch; even the `S32_Array` naming belongs to a revision mainline has never used |

So only one of them can be ported faithfully. `Stacking_Prog`'s signature became [`Hierarchy::step_with_goal`](Divergences.md) — the RUST-ONLY goal path, and the only reason it exists. `Stacking_RL`'s has to be approximated, and rather than pick an approximation silently the demo offers both and makes it a measurement.

#### `stacking_rl`

An actor on IO port 1, reward = (fraction of grid cells matching the target)³, exactly upstream's `reward *= reward * reward`. `--goal-mode` selects how the target is delivered:

- **`port`** (default) — a fourth IO port carrying the target grid, `IoType::None` so it is conditioned on and never predicted. This is the closest mainline expression of a per-IO-port goal array, and it needs no goal path at all.
- **`top`** — the target grid distilled to a top-layer CSDR and passed to `step_with_goal`, the same distillation `stacking_prog` uses.

Deviations:

- **The target is re-drawn on a fixed schedule (`--episode`, default 300 steps)** rather than upstream's 10%-per-rendered-frame. Same idea, but it makes "how well was this target built" a per-episode number instead of a smear.
- **The world is not reset between targets**, as upstream never resets it: the agent rebuilds from wherever the last target left the blocks.
- **A scripted solver was added** — RUST-ONLY, never visible to the hierarchy — purely to report the ceiling. It is load-bearing for reading the result at all: **random actions already score 0.70 on match fraction**, because most cells are empty in both grids and agree by default. A headline of "0.80 match" sounds like competence and is mostly arithmetic. The demo therefore reports `gap_closed`, the fraction of the distance between random play and a perfect solver that was actually covered.
- **Upstream's `h.params.ios[2].actor.discount = 0.9f` is not replicated.** IO port 2 is a Prediction port and has no actor; the line is inert.

Both routes work. Over 100k steps and 3 seeds, `--sweep goal-mode=port,top --repeat 3`:

| | match | gap closed | targets built | random | scripted |
|---|---|---|---|---|---|
| `--goal-mode port` | 0.848 ± 0.031 | 0.530 ± 0.123 | 27.1% ± 8.6 | 0.679 | 0.996 |
| `--goal-mode top` | 0.814 ± 0.021 | 0.428 ± 0.041 | 23.3% ± 3.5 | 0.679 | 0.996 |

Learned on 3 of 3 seeds either way, and **the two are not distinguishable**: the 0.10 gap between them sits inside `port`'s own 0.12 spread. A single seed at 60k steps had `top` ahead, which is exactly the kind of reading `--repeat` exists to overturn. Both deliver the goal; nothing here says which is better, and the demo should not be quoted as if it did.

#### `stacking_prog`

No actor and no reward. All three IO ports are Prediction ports and the action executed is simply whatever the action port predicts. The mechanism is upstream's, in three parts:

1. **Train** the hierarchy while feeding a goal CSDR at the top.
2. **Distil** a program: clone the hierarchy, show the clone the target grid for 32 steps with learning off and null action and position, and take the top hidden CSDR that settles out. That is the target expressed in the hierarchy's own most abstract vocabulary — the form `step_with_goal` wants. (`Hierarchy` had to gain `Clone` for this; C++ gets `aon::Hierarchy copy = h;` for free.)
3. **Execute** the live hierarchy on that program with learning off.

**`--train-policy` is the load-bearing switch, and the faithful setting does not work.** Upstream trains on *random actions* under *random goal CSDRs*, and this demo reproduces that as its default. It cannot work, and the demo says so rather than hiding it: the action port is being asked to predict a uniformly random action, so it learns the marginal and nothing else, and at execution time a distilled program is out of distribution because training never showed the hierarchy a goal that meant anything. Measured over 3 seeds: action agreement 0.244 ± 0.005 — chance for four actions — and a distilled program beats an arbitrary one by 0.036 ± 0.015, which is inside its own spread.

`--train-policy scripted` is the RUST-ONLY variant that makes the mechanism testable: the same scripted solver from `stacking_rl` demonstrates, under the goal it is actually building, distilled in exactly the form execution will supply. That is goal-conditioned behavioural cloning through a top-layer CSDR, and it works decisively — 0.588 ± 0.173 advantage, on 3 of 3 seeds.

Two metric decisions here are worth more than the code, because the obvious choice is wrong in both cases:

- **Scored on time *held* at the target, not the best match reached.** There are only ten reachable configurations of three blocks in three columns, so a random walk stumbles onto any given one within a few dozen steps. Scored on the best match it ever touched, flailing beats every policy in the demo including the trained one — random actions scored 0.859 against the trained agent's 0.819, which reads as failure and is an artefact. Scored on how much of the settled second half of a trial is spent *at* the target, random gets the ~1/10 it deserves.
- **Action agreement excludes idle steps.** A scripted demonstrator builds the target in about twenty steps and then emits "do nothing" until it changes. On a 300-step timer that is 93% of the training data, so scoring every step measures how often the world is already built — an easy 99% for a hierarchy that has learned nothing but to freeze. The same reasoning is why the scripted training regime re-draws the target the moment it is built rather than on a timer: it keeps the demonstrator working.
- **The control that matters is the *random program*, not random actions.** Same trained hierarchy, same worlds, same starting positions — differing only in whether the goal handed to it means anything. Beating random actions could be explained by having learned to move blocks at all; beating an arbitrary program cannot.

### `vsa_char`

Each character is bound to a positional vector and the results bundled, so a whole word collapses into a single `SegVec<256, 8>`. Written out one-hot that vector **is** a CSDR — 256 columns of 8 cells — so it feeds a `Hierarchy` directly with no encoding step at all. The hierarchy learns to predict the next word's vector, and reading the answer means unbinding each position and cleaning up against the alphabet: the *decoding* is algebra, not a learned decoder.

`examples/support/vec.rs` ports `demos/vec.h` — segmented hypervectors with bind (`(a+b) mod L`), unbind, bundle, permute and `thin()`. It is renamed from upstream's `Vec` because shadowing `std::vec::Vec` in Rust would be a permanent nuisance, but the `*` and `/` operators are provided too so ported code reads the same. `thin()`'s context-dependent tie-breaking is preserved and is not decoration: an *empty* bundle has every value tied at zero, and without it every segment would collapse to 0.

Upstream's `resources/ts_snippet.txt` is missing, so the corpus is generated — a small vocabulary walked by a sparse Markov chain — and `--text <path>` reads any ASCII file. Generating it also makes the difficulty a dial: `--successors 1` gives a deterministic chain, higher values a correspondingly harder one.

**The demo judges itself against the sequence's own predictability ceiling, not against chance.** That distinction is the whole result:

| | next-word accuracy | chance | ceiling | of ceiling |
|---|---|---|---|---|
| `--successors 2` (default) | 50.2% | 8.3% | 52.5% | **95.7%** |
| `--successors 1` | 100.0% | 8.3% | 100.0% | **100%** |

Judged against chance alone, the first row reads as a mediocre 50%. Judged against what the sequence actually permits — two roughly equally likely continuations — it is essentially optimal. The ceiling is measured from the corpus (for each word, how often its most common successor actually follows), not assumed.

The demo also reports **encoding fidelity** before training: how much of a word survives the encode/decode round trip with no learning involved. Bundling `k` pairs into one vector is lossy, so this is the ceiling on character accuracy, and it separates "the hierarchy did not learn" from "the representation could not hold the word in the first place". At 4-character words it is 100%.

### `enc_vis` and `topo_test`

Both upstream demos read a stored `vl.means` scalar per cell — already a normalised position. **This crate's `encoder::VisibleLayer` has no `means`.** It ports a byte-weight ART formulation whose weights are indexed by the *input cell* as well, so instead of a scalar there is a learned histogram over the input column, and the position is its weighted centroid. `examples/support/probe.rs` does that decoding.

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

The whole of the graphics story is one extra target, `examples-viz/examples/viewer.rs`, **in its own crate**:

```bash
cargo run --release -p dcc_sph_viz_examples --example viewer -- --demo car_racing
```

That separation is required rather than stylistic. R16 of dcc-core's import contract (see [`Conformance.md`](Conformance.md)) says local applications belong in a separate crate, not behind an optional feature — an optional dependency is still a real `[dependencies]` entry that lands in the lockfile and constrains resolution for every consumer. `macroquad` therefore does not appear in the library's manifest at all, and `r10_runtime_dependencies_stay_minimal` fails the build if it comes back. The Gymnasium runners are making the same move for `pyo3`.

`[workspace] default-members = ["."]` is what keeps plain `cargo build` and `cargo test` from pulling a GL stack in anyway.

`--demo` takes `ball_physics`, `cat_mouse` or `car_racing` — the three where motion is the point. It trains live and draws what is happening, the way the upstream SFML demos do. Space fast-forwards; Escape quits; in `ball_physics`, `G` closes the loop so the hierarchy generates from its own predictions.

The other twelve demos are text-only by design. A scrolling plot, an ASCII frame or a scatter says everything a window would for them, and they already print it.

Two things keep this honest. The viewer builds its hierarchies through the same `build_hierarchy` functions in `support/env/` that the headless demos use, so the two configurations cannot drift apart. And `support/viz.rs` is five functions — `View`, `blit_gray`, `plot_series`, `scatter`, `hud`. If it starts growing state, options or layout logic, that is the signal the work belongs in dcc-dashboard rather than here.

CI builds the crate in its own job and no further: opening a window is not CI's job, and a runner has no display to open one on.

## What CI does

`cargo build --all-targets`, `cargo test --all-targets`, `cargo clippy --all-targets -- -D warnings`, and then **runs every demo** at small step counts. Building is not enough — a demo that panics on step one still compiles. `stacking_rl` and `stacking_prog` are each run twice, once per mode: `--goal-mode top` and `--train-policy scripted` are the only demo paths through `step_with_goal`, and a build that broke the goal path would otherwise still pass.
