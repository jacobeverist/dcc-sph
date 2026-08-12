# AOgmaNeo parameter tuning guide

## Structural parameters

Structural parameters are set in the descriptor structs when calling `init_random`. They cannot be changed after creation. There are two descriptor types: `IoDesc` and `LayerDesc`.

**Note on radii:** Receptive fields are square. Given `radius`, the diameter is `2 * radius + 1` and the area is `diameter²` columns.

### IoDesc

Describes one input/output port. One per IO port in the `io_descs` slice passed to `Hierarchy::init_random`.

- **size: Int3** — `(width, height, column_size)` of this IO's CSDR. `width * height` is the number of columns; `column_size` is the number of cells per column (i.e. the number of distinct values a column can take).

- **io_type: IoType** — `IoType::None` (input only, no prediction), `IoType::Prediction` (encoder + decoder), or `IoType::Action` (encoder + actor, for RL).

- **num_dendrites_per_cell: usize** — number of dendrites per output cell for the decoder or actor policy head. More dendrites increase representational capacity at the cost of memory and compute. Default: `4`.

- **value_num_dendrites_per_cell: usize** — dendrites per cell for the actor value head. Only used when `io_type` is `Action`. Default: `2`.

- **up_radius: i32** — receptive field radius from this IO port up to the first encoder layer. Default: `2`.

- **down_radius: i32** — receptive field radius from the first encoder layer down to this IO port's decoder or actor. Default: `2`.

- **value_size: usize** — number of bins in the actor's value distribution. Only used when `io_type` is `Action`. Larger values give finer-grained value estimates. Default: `128`.

- **history_capacity: usize** — credit-assignment horizon for the actor, in timesteps. Only used when `io_type` is `Action`. Larger values allow learning from more distant rewards but use more memory. Default: `512`.

### LayerDesc

Describes one hidden (non-IO) layer. One per element in the `layer_descs` slice passed to `Hierarchy::init_random`.

- **hidden_size: Int3** — `(width, height, column_size)` of this layer's encoder output. Typical width/height: `4`–`16`. Typical column_size: `16`–`64`. Larger column sizes increase representational capacity but also compute.

- **num_dendrites_per_cell: usize** — dendrites per cell for this layer's inter-layer decoder. Default: `4`.

- **up_radius: i32** — receptive field radius from the layer below up to this layer's encoder. Default: `2`.

- **recurrent_radius: i32** — receptive field radius for the layer's recurrent self-connection (the encoder's own previous hidden CIS fed back as a visible input). Set to `-1` to disable recurrence. Default: `0` (recurrence enabled with radius 0, i.e. same-position only).

- **down_radius: i32** — receptive field radius from this layer's encoder down to the decoder of the layer below. Default: `2`.

---

## Runtime-adjustable parameters

Runtime parameters live in `h.params` and can be changed between any two `step()` calls.

```rust
// Layer parameters (for hidden layer l)
h.params.layers[l].encoder.lr = 0.5;
h.params.layers[l].decoder.lr = 0.1;
h.params.layers[l].recurrent_importance = 0.5;

// IO parameters (for IO port i)
h.params.ios[i].decoder.lr = 0.1;
h.params.ios[i].actor.vlr = 0.1;
h.params.ios[i].actor.plr = 0.01;
h.params.ios[i].importance = 1.0;

// Global
h.params.anticipation = true;
```

### LayerParams

- **encoder: encoder::Params** — encoder parameters (see below).
- **decoder: decoder::Params** — decoder parameters (see below). Used for inter-layer decoders.
- **recurrent_importance: f32** — scaling of the recurrent visible layer's input to the encoder, relative to the primary input. Default `0.5`. Only has effect when `recurrent_radius >= 0`.

### IoParams

- **decoder: decoder::Params** — decoder parameters for the IO port's prediction decoder.
- **actor: actor::Params** — actor parameters (only used when `io_type` is `Action`).
- **importance: f32** — scaling of this IO port's contribution to the first-layer encoder. Default `1.0`. Useful when you have multiple IO ports and want to weight their relative influence on the encoder.

### encoder::Params

- **choice: f32** — ART choice parameter (alpha). Must be `> 0.0`. Smaller values make the encoder prefer larger (more complete) matches. Default `0.01`.

- **vigilance: f32** — ART vigilance threshold, in `[0.0, 1.0]`. Higher values make the encoder pickier: a new input must closely match an existing cluster before that cluster gets updated (otherwise a new cluster is allocated). Default `0.9`.

- **lr: f32** — encoder learning rate in `[0.0, 1.0]`. Controls how much committed clusters move towards the new input. Default `0.5`.

- **active_ratio: f32** — fraction of columns within a lateral inhibition neighbourhood allowed to learn per step. Smaller values enforce sparser updates. Constraint: `(l_radius + 1)^-2 <= active_ratio`. Default `0.1`.

- **l_radius: i32** — lateral inhibition neighbourhood radius. Must be large enough to satisfy the `active_ratio` constraint. Default `2`.

### decoder::Params

- **scale: f32** — range scaling factor applied to 8-bit signed weights when computing activations. Unlikely to need adjustment. Default `8.0`.

- **lr: f32** — decoder learning rate in `[0.0, 1.0]`. Default `0.1`.

### actor::Params

- **vlr: f32** — value function learning rate. Around `0.01`–`0.1`. Must be `<= 1.0`. Default `0.1`.

- **plr: f32** — policy learning rate. Around `0.01`. Can exceed `1.0` in rare cases. Default `0.01`.

- **smoothing: f32** — exponential smoothing coefficient mixed into the TD return computation. Default `0.02`.

- **discount: f32** — RL discounting factor (lambda). Must be in `[0.0, 1.0)`. Higher values prioritize distant rewards; lower values are better when reward variance is high or rewards are dense. Default `0.99`.

- **td_scale_decay: f32** — decay rate for the running TD-error scale normalization. Default `0.999`.

- **value_range: f32** — the `symlog`/`symexp` range for encoding values. Default `10.0`.

- **min_steps: usize** — minimum number of history timesteps accumulated before the actor begins learning. Default `16`.

- **history_iters: usize** — number of past timesteps replayed during each learn call. Default `8`.

---

## Assorted tips

- **Receptive field coverage:** The default radius of `2` covers a 5×5 area. If your input layers are large relative to your hidden layers, increase the radius or add more hidden layers to bridge the spatial gap.

- **Default parameters are a good starting point.** Only adjust them if training is visibly too slow, unstable, or not converging.

- **Mimic mode:** When using `mimic > 0.0` in `step()` (imitation learning rather than pure RL), consider increasing `actor.plr`. The default policy learning rate is tuned for RL signals, which are typically weak; imitation signals are stronger and can accommodate a larger rate.

- **Memory vs. compute trade-off:** Adding more layers is generally more efficient than increasing `recurrent_radius` on a single layer for extending temporal memory, since higher layers clock less frequently (once exponential memory is implemented).

- **Sparse rewards:** If the environment has sparse or delayed rewards, increase `discount` (closer to `1.0`) and `history_capacity` to allow credit to propagate further back in time.

- **Dense rewards or high variance:** Decrease `discount` to focus on immediate rewards and reduce gradient variance.

- **Multiple IO ports and importance:** If some inputs are more informative than others, use `h.params.ios[i].importance` to up-weight the important ones relative to the encoder.

- **Recurrent connections:** Set `recurrent_radius = -1` to disable recurrence entirely if your task has no temporal dependencies, which reduces both memory and compute. If recurrence causes instability, reduce `recurrent_importance`.
