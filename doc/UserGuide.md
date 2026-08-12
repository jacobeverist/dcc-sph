# AOgmaNeo User Guide

*Rust port of the AOgmaNeo Sparse Predictive Hierarchies library.*

---

## 1. Introduction

### 1.1 What is AOgmaNeo?

AOgmaNeo is a biologically-inspired online machine learning system. It differentiates itself from Deep Learning (DL) through a very low compute footprint and the ability to learn online without forgetting. The underlying theory is called **SPH** (Sparse Predictive Hierarchies).

This guide describes AOgmaNeo well enough to understand what is happening and to use it in your own projects. It does not go into deep detail on the internal algorithms.

### 1.2 When Should I Use AOgmaNeo?

AOgmaNeo is suited to any task that involves predicting a stream of data points one step ahead of time. Given sample X_t, AOgmaNeo predicts X_{t+1}. It learns to model a stream as a time series, and includes short-term memory and optional reinforcement learning for action-selection tasks.

---

## 2. Basic Concepts

### 2.1 The CSDR

A **CSDR** (Columnar Sparse Distributed Representation) is the data format used throughout AOgmaNeo. All subsystems communicate using CSDRs.

A CSDR is a flat `Vec<i32>`. Each integer is a **column index** — it selects the single active cell within that column (one-hot). All columns have the same number of cells, specified by the user.

Although stored as a 1D array, a CSDR represents a 2D grid of columns (making it 3D overall — two spatial dimensions plus the column/cell dimension). Columns are addressed in row-major order:

```
index = col_y + col_x * size_y
```

And in reverse:

```
col_x = index / size_y      (integer division)
col_y = index % size_y
```

This is analogous to a grayscale image: a 2D array of integers stored in a 1D array, where the "pixels" are column indices rather than intensity values.

In Rust, a CSDR is `Vec<i32>` (or a `&[i32]` slice when passed to an API). Layer sizes are given as `Int3 { x, y, z }`, where `x` and `y` are the spatial grid dimensions and `z` is the column size (number of cells per column).

### 2.2 The Hierarchy

An AOgmaNeo system is a **hierarchy** (stack) of layers. Each layer contains at least one encoder and one decoder. Input and output both happen at the bottom of the hierarchy through one or more **IO ports** (also called IO layers).

Each layer's hidden representation is the CSDR output of that layer's encoder. These can be retrieved at any time to inspect the system's internal state.

IO ports accept CSDRs and produce CSDRs. Each port's output is either:
- A **prediction** — the t+1 prediction of what that port will receive next.
- An **action** — a prediction modified by a reinforcement learning actor to maximize reward.
- **Nothing** — if the IO type is `None` (input-only).

AOgmaNeo hierarchies are bidirectional. Each timestep has an **up** pass (encoding inputs through the encoder stack) followed by a **down** pass (generating predictions through the decoder stack).

### 2.3 Encoders

The encoder produces a sparse hidden CSDR from its inputs using **Adaptive Resonance Theory** (ART). The key properties are:

- Inputs from the visible layer(s) below are compared against learned weight templates.
- A vigilance threshold controls how strictly a new input must match an existing template before that template is updated (vs. a new one being allocated).
- Lateral inhibition limits how many columns activate simultaneously, enforcing sparsity.
- 8-bit unsigned weights (`ByteBuffer = Vec<u8>`) keep memory usage low.

The encoder's output (its hidden CIS) is the primary representation that propagates up the hierarchy.

### 2.4 Decoders

Decoders perform predictive reconstruction using **multi-dendrite perceptrons**. Each decoder:

- Receives the encoder's hidden CIS from the same or higher layer as input.
- Produces a prediction CSDR for the layer below it.
- Learns online by comparing predictions to the actual inputs at the next timestep.
- Uses 8-bit signed weights (`SByteBuffer = Vec<i8>`).

Decoders at the bottom of the hierarchy (layer 0) produce the user-visible prediction CSDRs.

### 2.5 Actors

An actor is used for **reinforcement learning**. It behaves like a decoder but:

- Instead of minimizing prediction error, it modifies its predictions to maximize the scalar reward signal.
- It maintains a circular history buffer of past states, inputs, and rewards for temporal credit assignment.
- It has two heads: a **policy** (which action/column to select) and a **value function** (estimate of future discounted reward).
- Float32 weights (not quantized to 8-bit), since the RL gradient updates require more precision.

### 2.6 Exponential Memory

AOgmaNeo uses **Exponential Memory** (a form of Clockwork RNN) for short-term/working memory. Each layer in the hierarchy clocks at a slower rate than the layer below it (typically 2×). This means:

- Layer 0 processes every timestep.
- Layer 1 processes every 2 timesteps.
- Layer 2 processes every 4 timesteps, etc.

With a 2× stride and L layers, the effective memory horizon grows as 2^L timesteps. This exponential scaling is why just a few layers can cover long temporal dependencies.

A useful property: the total compute cost barely grows as more layers are added. Since each additional layer runs half as often, the series 1 + 1/2 + 1/4 + ... converges to 2, meaning the total compute never exceeds 2× the cost of the first layer.

> **Note:** The `ticks_per_update` and `temporal_horizon` fields from the original C++ `Layer_Desc` are not yet implemented in the Rust port. The current Rust implementation processes all layers every step.

### 2.7 Online Learning

AOgmaNeo learns from a data stream in temporal order, one sample at a time — no replay buffer, no i.i.d. shuffling. This is possible because CSDRs are **sparse**: most entries are zero (inactive), with only a few cells active per column. Sparse representations give:

- **Non-interference**: only a small part of the network updates at each step, so previous patterns are not overwritten.
- **Near-orthogonality**: different inputs tend to activate different cells, making representations easy to distinguish.

Together these properties enable online learning without catastrophic forgetting.

### 2.8 Performance

AOgmaNeo is fast relative to the number of parameters it uses. The main reasons:

- **Sparsity**: since CSDRs index the active cells directly, zero entries are never iterated — the loop only touches active weights.
- **Online learning**: each sample is seen once; no re-visiting or i.i.d. shuffling required.
- **Exponential memory**: higher layers run proportionally less often, so added depth is nearly free.
- **Parallelism**: the Rust port uses `rayon` for data-parallel forward passes across columns (encoder and decoder). Actor and ImageEncoder remain sequential.

### 2.9 Receptive Fields

Connectivity in AOgmaNeo is **local**: each column only receives input from columns within a square neighbourhood on the layer below. This neighbourhood is called the **receptive field**. Its size is set by a radius parameter:

```
diameter = radius * 2 + 1
area     = diameter²
```

Larger receptive fields bridge larger spatial gaps but increase compute proportionally. The default radius of `2` gives a 5×5 receptive field (area = 25).

### 2.10 Pre-Encoders and Pre-Decoders

The hierarchy only speaks CSDRs. To use it with your own data, you need to convert your data to a CSDR (**pre-encoder**) and convert predictions back (**pre-decoder**). These are application-specific; you define them yourself.

The Rust port ships with one built-in pre-encoder/pre-decoder: **`ImageEncoder`** (`src/image_encoder.rs`). It maps between raw byte images and CSDRs using a Self-Organizing Map (SOM), and supports `reconstruct()` to decode a hidden CSDR back to pixel data.

**Design tip**: make your pre-encoder locally sensitive — small changes in the input should change only a few column indices; large changes should change many. This helps the hierarchy generalize efficiently.

---

## 3. Using AOgmaNeo in Rust

Add the crate to your project:

```toml
[dependencies]
dcc_sph = { path = "." }
```

### 3.1 Imports

```rust
use dcc_sph::helpers::{Int3, VecWriter, SliceReader};
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};
use dcc_sph::image_encoder::{ImageEncoder, VisibleLayerDesc as IeVLD};
```

### 3.2 Creating a Hierarchy

Configure IO ports with `IoDesc` and hidden layers with `LayerDesc`, then call `init_random`:

```rust
let io_descs = vec![IoDesc {
    size: Int3::new(4, 4, 16),       // 4×4 grid, 16 cells per column
    io_type: IoType::Prediction,
    num_dendrites_per_cell: 4,
    up_radius: 2,
    down_radius: 2,
    value_size: 128,                 // only used for Action IO
    value_num_dendrites_per_cell: 2, // only used for Action IO
    history_capacity: 512,           // only used for Action IO
}];

let layer_descs = vec![LayerDesc {
    hidden_size: Int3::new(4, 4, 16),
    num_dendrites_per_cell: 4,
    up_radius: 2,
    recurrent_radius: -1, // -1 disables recurrent connections
    down_radius: 2,
}];

let mut h = Hierarchy::new();
h.init_random(&io_descs, &layer_descs);
```

For reinforcement learning, set `io_type: IoType::Action` on the relevant IO port and supply a non-zero `reward` to `step()`.

For input-only ports (no prediction needed), set `io_type: IoType::None`.

### 3.3 Stepping

Call `step` once per timestep with your current input CSDRs:

```rust
// input_cis: one Vec<i32> per IO port, length = size.x * size.y,
//            values in [0, size.z)
h.step(
    &[&input_cis],   // slice of CSDR slices, one per IO port
    true,            // learn_enabled
    0.0,             // reward (used by Action ports)
    0.0,             // mimic strength (0.0 = pure RL, > 0.0 blends in supervised)
);
```

After stepping, read the predictions:

```rust
let pred_cis: &[i32] = h.get_prediction_cis(0); // IO port index 0
```

For `Action` IO, you can also read the value estimate:

```rust
let values: &[f32] = h.get_prediction_values(0);
```

### 3.4 Multiple IO Ports

Pass one CSDR per port, in the same order as the `io_descs` slice:

```rust
h.step(&[&input_a, &input_b], true, 0.0, 0.0);
let pred_a = h.get_prediction_cis(0);
let pred_b = h.get_prediction_cis(1);
```

### 3.5 Inspecting Hidden State

```rust
let layer = 0;
let hidden_cis: &[i32] = h.get_encoder(layer).get_hidden_cis();
let hidden_size: Int3 = h.get_encoder(layer).get_hidden_size();
```

### 3.6 Adjusting Parameters at Runtime

Runtime parameters live in `h.params`. They can be changed between any two `step()` calls:

```rust
// Encoder parameters for hidden layer 0
h.params.layers[0].encoder.lr = 0.5;
h.params.layers[0].encoder.vigilance = 0.9;

// Decoder parameters for IO port 0
h.params.ios[0].decoder.lr = 0.1;

// Actor parameters for a reinforcement learning IO port
h.params.ios[0].actor.vlr = 0.1;
h.params.ios[0].actor.plr = 0.01;
h.params.ios[0].actor.discount = 0.99;

// Importance scaling of an IO port's input to the encoder
h.params.ios[0].importance = 1.0;

// Recurrent connection importance for a layer
h.params.layers[0].recurrent_importance = 0.5;

// Anticipation mode (default: true)
h.params.anticipation = true;
```

See `doc/Tuning.md` for guidance on what values to use.

### 3.7 Clearing State

To reset the system's hidden state without discarding learned weights:

```rust
h.clear_state();
```

### 3.8 Serialization

Save and load the full hierarchy (weights + state + params) using `VecWriter` and `SliceReader`:

```rust
// Save
let mut writer = VecWriter::new();
h.write(&mut writer);
let bytes: Vec<u8> = writer.data;

// Load into a fresh hierarchy
let mut h2 = Hierarchy::new();
let mut reader = SliceReader::new(&bytes);
h2.read(&mut reader);
```

You can also save/load state (activations, no weights) or weights (no state) separately:

```rust
// State only
let mut w = VecWriter::new();
h.write_state(&mut w);

// Weights only
let mut w = VecWriter::new();
h.write_weights(&mut w);
```

The serialized format is field-by-field little-endian binary. It is **not** compatible with the C++ binary format.

---

## 4. Using the ImageEncoder

`ImageEncoder` converts raw byte images to CSDRs and back. Its inputs are `&[u8]` (raw pixel bytes, channel-last), not CSDRs.

```rust
use dcc_sph::image_encoder::{ImageEncoder, VisibleLayerDesc};
use dcc_sph::helpers::Int3;

let visible_layer_descs = vec![VisibleLayerDesc {
    size: Int3::new(8, 8, 3), // 8×8 image, 3 channels (e.g. RGB)
    radius: 2,
}];

let mut ie = ImageEncoder::default();
ie.init_random(Int3::new(4, 4, 16), visible_layer_descs);

// Step (learn = true enables weight updates, learn_recon = true enables reconstruction weight updates)
let pixels: Vec<u8> = vec![128u8; 8 * 8 * 3];
ie.step(&[&pixels], true, true);

// Get the hidden CSDR
let hidden_cis: Vec<i32> = ie.get_hidden_cis().to_vec(); // clone before reconstruct

// Reconstruct the image from a CSDR
ie.reconstruct(&hidden_cis);
let recon: &[u8] = ie.get_reconstruction(0); // 8 * 8 * 3 bytes
```

> **Note**: Clone `get_hidden_cis()` before calling `reconstruct()` because both borrow `ie`.

Runtime parameters for `ImageEncoder` are on `ie.params` directly (not via a hierarchy):

```rust
ie.params.lr = 0.1;      // SOM learning rate
ie.params.falloff = 0.9; // neighbourhood falloff
ie.params.n_radius = 1;  // neighbourhood radius
```

---

## 5. Reinforcement Learning Example

The `examples/cartpole.rs` file demonstrates a full RL loop. The pattern is:

1. Encode the environment state into one or more CSDRs (pre-encoder).
2. Call `h.step(...)` with the current reward and the encoded state. For action ports, pass the _previous_ action as the `input_cis` for that port (the hierarchy predicts what action to take next).
3. Read the action from `h.get_prediction_cis(action_port_index)`.
4. Apply the action to the environment.
5. Observe the new state and reward, go to step 1.

```bash
cargo run --release --example cartpole
```

---

## 6. Thread Count

By default, `rayon` uses all available CPU threads. You can limit this:

```rust
use dcc_sph::helpers::set_num_threads;
set_num_threads(4);
```
