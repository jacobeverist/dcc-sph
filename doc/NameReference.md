# AOgmaNeo variable naming reference

This document describes the variable and field naming conventions used throughout the Rust source (`src/`).

## Encoder / Decoder / Actor / ImageEncoder

**vl** — visible layer. The input-side of an encoder, decoder, actor, or image encoder. Stored as `Vec<VisibleLayer>` (each module defines its own `VisibleLayer` type).

**vld** — visible layer descriptor. The corresponding configuration struct `VisibleLayerDesc { size: Int3, radius: i32 }`.

**hc** — index of a cell within one hidden column. Range `[0, hidden_size.z)`.

**vc** — index of a cell within one visible/input column. Range `[0, vld.size.z)`.

**hidden_column_index** — flat 2D index of a column in the hidden layer, derived via `address2(column_pos, ...)`.

**hidden_cells_start** — first flat index of the cells belonging to one hidden column: `hidden_column_index * hidden_size.z`.

**count** — number of visible columns that fall inside a receptive field, used for normalization.

**radius** — receptive field radius (stored in `VisibleLayerDesc::radius`).

**diam** — receptive field diameter: `2 * radius + 1`.

**area** — receptive field area: `diam * diam`.

**h_to_v** — `Float2` scale factors to project a hidden column position to a visible layer position, computed as `(vld.size.x / hidden_size.x, vld.size.y / hidden_size.y)`.

**visible_center** — the center of the receptive field on the visible layer, computed via `project(column_pos, h_to_v)`.

**field_lower_bound** — lower corner of the receptive field before edge clamping: `visible_center - radius`.

**iter_lower_bound / iter_upper_bound** — clamped iteration bounds, keeping the receptive field within the visible layer boundaries.

**in_ci** — the input column index: the active cell index read from the visible CSDR for a given visible column.

**offset** — 2D position within the receptive field `[0, diam)` for both x and y.

**wi** — weight index: the final flat index into a weight buffer.

**wi_start** — partially computed weight index, missing only the final `hc` (hidden cell) term.

**wi_start_partial / wi_offset** — further partial weight index computations used for indexing weight tensors with multiple strides (e.g. in decoders, which also have a dendrite dimension).

**max_index** — the column index (hc value) of the winning cell (highest activation).

**max_activation** — the activation value of the winning cell.

**total** — float normalization accumulator, typically a softmax denominator.

**delta** — a weight increment (integer, before clamping).

**state** — a local `u64` PCG32 RNG state for a single column's computation.

**base_state** — a per-step seed drawn from the global RNG, used to derive deterministic per-column sub-seeds via `rand_get_state(base_state + column * RAND_SUBSEED_OFFSET)`.

## Encoder-specific

**hidden_totals** — per-hidden-cell running sum of the weights from all active inputs. Used by ART vigilance matching. Stored in `VisibleLayer::hidden_totals: IntBuffer`.

**hidden_totals_snapshot** — a clone of `hidden_totals` taken before the parallel forward pass begins, so all columns read a consistent snapshot. The actual `hidden_totals` are updated sequentially during the learn pass.

**hidden_committed_flags** — per-hidden-cell flag (`ByteBuffer`, 0 or 1). A cell is "committed" once it has been selected and learned at least once. Uncommitted cells are treated as vigilance-pass candidates regardless of their match score.

**hidden_learn_flags** — per-hidden-column flag indicating whether the winning cell for that column passed vigilance and should participate in the learn pass.

**hidden_comparisons** — per-hidden-column activation value of the winning cell, used for lateral inhibition comparison across neighbouring columns.

**l_radius** — lateral inhibition radius. Defines the neighbourhood over which a column's activation must be among the top `active_ratio` fraction in order to trigger weight updates.

**active_ratio** — the fraction of columns within a lateral inhibition neighbourhood that are allowed to learn each step.

## Decoder-specific

**num_dendrites_per_cell** — number of dendrites attached to each hidden cell. Dendrites form two equal groups: the first half are inhibitory and the second half are excitatory.

**dendrite_acts** — per-dendrite pre-activation accumulator, then reused to store the sigmoid of that pre-activation (for use in the backward pass).

**dendrite_deltas** — per-dendrite integer weight gradient, computed during the learn pass.

**hidden_acts** — per-hidden-cell softmax probability, stored after the forward pass for use in the learn pass.

## Actor-specific

**value_size** — number of cells in the value head's output column. The value function is represented as a distribution over `value_size` bins.

**value_num_dendrites_per_cell** — dendrites per cell for the value head.

**policy_num_dendrites_per_cell** — dendrites per cell for the policy head.

**hidden_values** — per-hidden-column scalar value estimate (decoded from the value head's softmax distribution via `symexpf`).

**hidden_td_scales** — per-column running maximum |TD error|, used to normalize the TD error for stable policy gradient updates.

**history_samples** — `CircleBuffer<HistorySample>`. Each `HistorySample` stores `input_cis`, `hidden_target_cis_prev`, `hidden_values`, and `reward` for one past timestep.

**vlr** — value function learning rate.

**plr** — policy learning rate.

**smoothing** — exponential smoothing factor mixed into the TD return.

**discount** — RL discount factor (lambda). Must be in `[0.0, 1.0)`.

**td_scale_decay** — decay rate for the running `hidden_td_scales` normalization term.

**value_range** — range of the `symlog`/`symexp` value encoding.

**min_steps** — minimum number of history steps before learning begins.

**history_iters** — number of history timesteps replayed per learn call.

**mimic** — passed to `step()`; a `f32` that blends supervised (imitation) signal into the policy gradient. `0.0` is pure RL.

## Hierarchy

**ticks** — per-layer clock counters for the exponential-memory clockwork: a higher layer runs only once every `ticks_per_update` bottom-layer steps. **Implemented** (`hierarchy.rs` — `self.ticks[l]` incremented each step, layer runs when it reaches `ticks_per_update`). ⚠ *Not in the referenced upstream `645a54a`* (which has 0 references to `ticks_per_update` and uses recurrence only — `recurrent_radius` + `feedback_cis_prev`). This crate carries ticks as an **extra** mechanism alongside recurrence — see [Divergences.md](Divergences.md).

**ticks_per_update** — number of bottom-layer ticks between a layer's updates (default `1` = every step). **Implemented** via `LayerDesc.ticks_per_update`.

**i_indices** — flat buffer of length `num_io * 2`. The first `num_io` entries map decoder index → IO index. The second `num_io` entries map actor index → IO index.

**d_indices** — per-IO-port decoder/actor index. `-1` if the port has no decoder or actor (`IoType::None`).

**updates** — per-layer boolean indicating whether a layer ticked (ran) this step. **Implemented** (`self.updates[l]` gates the forward pass in `hierarchy.rs`).

**io_types** — `Vec<u8>` storing the `IoType` discriminant for each IO port.

**io_sizes** — `Vec<Int3>` storing the CSDR size for each IO port.

**hidden_cis_prev** — the encoder hidden CIS from the previous timestep, used as the decoder's input during the learning pass.

**feedback_cis_prev** — the decoder prediction from the layer above, from the previous timestep, used as the second input during the learning pass.

**recurrent_importance** — scaling factor applied to a layer's recurrent input (the encoder's own previous hidden CIS). Set via `h.params.layers[l].recurrent_importance`.

**anticipation** — when `true` (default), the hierarchy also runs a second forward+learn pass on decoders using the *current* encoder output (as opposed to only the previous-step encoder output). Stored in `h.params.anticipation`.

## Suffixes and Prefixes

**prev** — value from the previous timestep.

**next** — value intended for the next timestep.

**base** — a cached seed value used to derive per-column RNG sub-seeds.

**acts** — activations buffer.

**cis** — column indices (a CSDR buffer).

**probs** — probabilities buffer (post-softmax activations).

**size** — usually the 3D `Int3` size of a layer, or occasionally the byte count of a serialized object.

## Module-level letters (shorthand)

**e** — encoder (in Hierarchy, `self.encoders`).

**d** — decoder (in Hierarchy, `self.decoders`; also `d_index` for decoder index within a layer).

**a** — actor (in Hierarchy, `self.actors`).

**i** — IO port index.

**l** — layer index.

**t** — time index (into history buffer in Actor).

**vli** — visible layer index (loop variable over `vl` / `vld` pairs).

## Misc

**params** — runtime-adjustable parameters, kept separate from structural state. Accessed via `h.params.layers[l]` and `h.params.ios[i]`.

**importance** — scaling factor for a visible layer's contribution to encoder activations. Default `1.0`. Adjusted via `h.params.ios[i].importance` (IO inputs) or `h.params.layers[l].recurrent_importance` (recurrent input).
