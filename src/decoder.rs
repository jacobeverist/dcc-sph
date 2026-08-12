// AOgmaNeo Rust port - Decoder (predictive reconstruction with multi-dendrite perceptrons)
#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;
use crate::helpers::*;

/// Re-export the shared visible-layer descriptor for decoder use.
pub use crate::helpers::VisibleLayerDesc;

/// One column's disjoint slice of the mutable learn state: `(column index, its
/// dendrite deltas, one weight slice per visible layer)`.
///
/// Named because this is what makes `learn` parallelisable — `chunks_mut` hands out
/// non-overlapping slices, so each column can be updated independently and the result
/// stays bit-identical to the sequential order. Written inline it is an unreadable
/// four-level type that says nothing about why it exists.
type ColumnWork<'a> = (usize, &'a mut [i32], Vec<&'a mut [i8]>);

/// Internal state of one visible (input) connection to the decoder.
#[derive(Clone, Debug, Default)]
pub struct VisibleLayer {
    /// Synaptic weights, stored as `i8` in `[-127, 127]`.
    ///
    /// Layout: `weights[di + num_dendrites * (hc + num_hc * (dy + diam * (dx + diam * (in_ci + z * col))))]`
    pub weights: SByteBuffer,
}

/// Hyperparameters for the [`Decoder`].
#[derive(Clone, Debug)]
pub struct Params {
    /// Input scale applied before the softplus nonlinearity. Larger values
    /// sharpen the dendrite responses.
    /// Range: `(0.0, ∞)`. Default: `8.0`.
    pub scale: f32,
    /// Learning rate for weight updates.
    /// Range: `[0.0, 1.0]`. Default: `0.1`.
    pub lr: f32,
    /// Leaky-ReLU coefficient applied to the negative softplus branch of each
    /// dendrite: `activation = softplus(x) - leak * softplus(-x)`.
    /// A value of `0.0` gives standard (non-leaky) softplus.
    /// Range: `[0.0, 1.0]`. Default: `0.01`.
    pub leak: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            scale: 8.0,
            lr: 0.1,
            leak: 0.01,
        }
    }
}

/// Multi-dendrite perceptron decoder for predicting discrete column indices.
///
/// Each hidden column has `num_dendrites_per_cell` dendrites per cell. The
/// first half of dendrites contribute positively and the second half negatively
/// (push-pull arrangement), which enables a form of divisive normalisation.
///
/// The forward pass produces a softmax probability distribution over the
/// `hidden_size.z` possible column indices, from which the argmax is taken as
/// the prediction.
///
/// The forward pass is parallelised across hidden columns using [`rayon`].
#[derive(Clone, Debug, Default)]
pub struct Decoder {
    hidden_size: Int3,
    num_dendrites_per_cell: usize,
    hidden_cis: IntBuffer,
    hidden_acts: FloatBuffer,
    dendrite_acts: FloatBuffer,
    dendrite_deltas: IntBuffer,
    /// Internal state for each visible connection.
    pub visible_layers: Vec<VisibleLayer>,
    /// Structural descriptors (size, radius) for each visible connection.
    pub visible_layer_descs: Vec<VisibleLayerDesc>,
}

// Per-column result of forward pass
struct ForwardResult {
    hidden_ci: i32,
    // dendrite_acts and hidden_acts written at column offset
    dendrite_acts: Vec<f32>,
    hidden_acts: Vec<f32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn compute_forward(
        column_pos: Int2,
        hidden_size: Int3,
        num_dendrites_per_cell: usize,
        visible_layers: &[VisibleLayer],
        visible_layer_descs: &[VisibleLayerDesc],
        input_cis: &[&[i32]],
        params: &Params,
    ) -> ForwardResult {
        let hidden_column_index = address2(column_pos, Int2::new(hidden_size.x, hidden_size.y));

        let num_hc = hidden_size.z as usize;
        let num_dendrites = num_hc * num_dendrites_per_cell;

        let mut dendrite_acts = vec![0.0f32; num_dendrites];
        let mut hidden_acts_col = vec![0.0f32; num_hc];

        let mut count = 0usize;

        for vli in 0..visible_layers.len() {
            let vl = &visible_layers[vli];
            let vld = &visible_layer_descs[vli];

            let diam = vld.radius * 2 + 1;
            let h_to_v = Float2::new(
                vld.size.x as f32 / hidden_size.x as f32,
                vld.size.y as f32 / hidden_size.y as f32,
            );
            let visible_center = project(column_pos, h_to_v);
            let field_lower_bound = Int2::new(
                visible_center.x - vld.radius,
                visible_center.y - vld.radius,
            );
            let iter_lower_bound =
                Int2::new(field_lower_bound.x.max(0), field_lower_bound.y.max(0));
            let iter_upper_bound = Int2::new(
                (visible_center.x + vld.radius).min(vld.size.x - 1),
                (visible_center.y + vld.radius).min(vld.size.y - 1),
            );

            count += ((iter_upper_bound.x - iter_lower_bound.x + 1)
                * (iter_upper_bound.y - iter_lower_bound.y + 1)) as usize;

            let vl_input_cis = input_cis[vli];

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let visible_column_index =
                        address2(Int2::new(ix, iy), Int2::new(vld.size.x, vld.size.y));
                    let in_ci = vl_input_cis[visible_column_index] as usize;
                    let offset = Int2::new(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    let wi_start_partial = num_hc
                        * (offset.y as usize
                            + diam as usize
                                * (offset.x as usize
                                    + diam as usize
                                        * (in_ci + vld.size.z as usize * hidden_column_index)));

                    for hc in 0..num_hc {
                        let dendrites_start = num_dendrites_per_cell * hc;
                        let wi_start = num_dendrites_per_cell * (hc + wi_start_partial);

                        for di in 0..num_dendrites_per_cell {
                            dendrite_acts[dendrites_start + di] +=
                                vl.weights[di + wi_start] as f32;
                        }
                    }
                }
            }
        }

        let half_num = num_dendrites_per_cell / 2;
        let dendrite_scale = (1.0f32 / count as f32).sqrt() / 127.0 * params.scale;
        let activation_scale = (1.0f32 / num_dendrites_per_cell as f32).sqrt();

        let mut max_index = 0usize;
        let mut max_activation = LIMIT_MIN;

        for hc in 0..num_hc {
            let dendrites_start = num_dendrites_per_cell * hc;

            let mut activation = 0.0f32;

            for di in 0..num_dendrites_per_cell {
                let act = dendrite_acts[dendrites_start + di] * dendrite_scale;
                // Store sigmoid(act) for use during learning (derivative)
                dendrite_acts[dendrites_start + di] = sigmoidf(act);
                let leaky = softplusf(act) - params.leak * softplusf(-act);
                activation += leaky * (if di >= half_num { 2.0 } else { 0.0 } - 1.0);
            }

            activation *= activation_scale;
            hidden_acts_col[hc] = activation;

            if activation > max_activation {
                max_activation = activation;
                max_index = hc;
            }
        }

        // softmax
        let mut total = 0.0f32;
        for hc in 0..num_hc {
            hidden_acts_col[hc] = (hidden_acts_col[hc] - max_activation).exp();
            total += hidden_acts_col[hc];
        }
        let total_inv = 1.0 / LIMIT_SMALL.max(total);
        for hc in 0..num_hc {
            hidden_acts_col[hc] *= total_inv;
        }

        ForwardResult {
            hidden_ci: max_index as i32,
            dendrite_acts,
            hidden_acts: hidden_acts_col,
        }
    }

    /// Initialise the decoder with random weights.
    ///
    /// - `hidden_size` — spatial grid and vocabulary size of the predicted layer.
    /// - `num_dendrites_per_cell` — number of dendrites per cell (must be even;
    ///   first half are positive, second half are negative).
    /// - `visible_layer_descs` — descriptors for the input connections.
    pub fn init_random(
        &mut self,
        hidden_size: Int3,
        num_dendrites_per_cell: usize,
        visible_layer_descs: Vec<VisibleLayerDesc>,
    ) {
        self.visible_layer_descs = visible_layer_descs;
        self.hidden_size = hidden_size;
        self.num_dendrites_per_cell = num_dendrites_per_cell;

        let num_hidden_columns = (hidden_size.x * hidden_size.y) as usize;
        let num_hidden_cells = num_hidden_columns * hidden_size.z as usize;
        let num_dendrites = num_hidden_cells * num_dendrites_per_cell;

        self.visible_layers = self
            .visible_layer_descs
            .iter()
            .map(|vld| {
                let diam = vld.radius * 2 + 1;
                let area = (diam * diam) as usize;
                let weights_size = num_dendrites * area * vld.size.z as usize;

                let weights: SByteBuffer = (0..weights_size)
                    .map(|_| {
                        ((global_rand() % (INIT_WEIGHT_NOISEI + 1)) as i32
                            - INIT_WEIGHT_NOISEI as i32 / 2) as i8
                    })
                    .collect();

                VisibleLayer { weights }
            })
            .collect();

        self.hidden_cis = vec![0i32; num_hidden_columns];
        self.hidden_acts = vec![0.0f32; num_hidden_cells];
        self.dendrite_acts = vec![0.0f32; num_dendrites];
        self.dendrite_deltas = vec![0i32; num_dendrites];
    }

    /// Run the forward pass, updating `hidden_cis` and `hidden_acts`.
    ///
    /// Call [`Decoder::learn`] immediately afterwards to update weights.
    pub fn activate(&mut self, input_cis: &[&[i32]], params: &Params) {
        let num_hidden_columns = (self.hidden_size.x * self.hidden_size.y) as usize;
        let hidden_size = self.hidden_size;
        let num_dendrites_per_cell = self.num_dendrites_per_cell;

        let results: Vec<ForwardResult> = (0..num_hidden_columns)
            .into_par_iter()
            .map(|i| {
                let column_pos = Int2::new(
                    (i / hidden_size.y as usize) as i32,
                    (i % hidden_size.y as usize) as i32,
                );
                Self::compute_forward(
                    column_pos,
                    hidden_size,
                    num_dendrites_per_cell,
                    &self.visible_layers,
                    &self.visible_layer_descs,
                    input_cis,
                    params,
                )
            })
            .collect();

        for (i, res) in results.into_iter().enumerate() {
            self.hidden_cis[i] = res.hidden_ci;
            let cells_start = i * hidden_size.z as usize;
            let dend_start = i * hidden_size.z as usize * num_dendrites_per_cell;
            self.hidden_acts[cells_start..cells_start + hidden_size.z as usize]
                .copy_from_slice(&res.hidden_acts);
            self.dendrite_acts[dend_start..dend_start + hidden_size.z as usize * num_dendrites_per_cell]
                .copy_from_slice(&res.dendrite_acts);
        }
    }

    /// Run one learning step using the targets in `hidden_target_cis`.
    ///
    /// [`activate`](Self::activate) should be called before `learn` to populate
    /// the dendrite activations used for weight updates.
    pub fn learn(
        &mut self,
        input_cis: &[&[i32]],
        hidden_target_cis: &[i32],
        params: &Params,
    ) {
        let num_hidden_columns = (self.hidden_size.x * self.hidden_size.y) as usize;
        let hidden_size = self.hidden_size;
        let ndpc = self.num_dendrites_per_cell;
        let num_hc = hidden_size.z as usize;
        let base_state = global_rand() as u64;

        // Each hidden column owns a *contiguous* block of every visible layer's weight
        // array (the weight index is `column_stride * hidden_column_index + …`) and of
        // `dendrite_deltas`, so we split those arrays into per-column mutable chunks and
        // run the columns in parallel — matching the C++ OpenMP `PARALLEL_FOR`. The
        // per-column RNG seed (`base_state + i·offset`) is order-independent, so the
        // result is identical to the serial loop regardless of thread scheduling (the
        // fidelity harness stays bit-exact). Indexing below is *chunk-relative* (the
        // `hidden_column_index` term is dropped from write indices; reads into the
        // shared `hidden_acts`/`dendrite_acts`/`hidden_target_cis` stay global).
        let Decoder {
            visible_layers,
            visible_layer_descs,
            hidden_acts,
            dendrite_acts,
            dendrite_deltas,
            ..
        } = self;

        let delta_stride = ndpc * num_hc;
        let w_strides: Vec<usize> = visible_layer_descs
            .iter()
            .map(|vld| {
                let diam = (vld.radius * 2 + 1) as usize;
                ndpc * num_hc * diam * diam * vld.size.z as usize
            })
            .collect();

        // Gather per-column disjoint mutable slices (deltas + one weight slice per
        // visible layer). `chunks_mut` yields non-overlapping slices in order.
        let mut delta_it = dendrite_deltas.chunks_mut(delta_stride);
        let mut w_its: Vec<std::slice::ChunksMut<i8>> = visible_layers
            .iter_mut()
            .enumerate()
            .map(|(vli, vl)| vl.weights.chunks_mut(w_strides[vli]))
            .collect();
        let work: Vec<ColumnWork<'_>> = (0..num_hidden_columns)
            .map(|col| {
                let d = delta_it.next().unwrap();
                let ws: Vec<&mut [i8]> = w_its.iter_mut().map(|it| it.next().unwrap()).collect();
                (col, d, ws)
            })
            .collect();

        // Small hierarchies don't amortise the rayon task-spawn overhead — run them
        // serially. Both paths call the same per-column kernel, so the result is
        // identical either way.
        const PARALLEL_THRESHOLD: usize = 64;
        let go = |col: usize, deltas: &mut [i32], ws: &mut [&mut [i8]]| {
            Self::learn_one_column(
                col, deltas, ws, hidden_size, ndpc, num_hc, base_state, hidden_acts,
                dendrite_acts, hidden_target_cis, input_cis, visible_layer_descs, params,
            );
        };
        if num_hidden_columns >= PARALLEL_THRESHOLD {
            work.into_par_iter().for_each(|(col, deltas, mut ws)| go(col, deltas, &mut ws));
        } else {
            work.into_iter().for_each(|(col, deltas, mut ws)| go(col, deltas, &mut ws));
        }
    }

    /// The per-column learn kernel (delta compute + weight apply) operating on this
    /// column's disjoint `deltas` / `ws` weight chunks. Chunk-relative write indexing;
    /// global reads into the shared `hidden_acts`/`dendrite_acts`/`hidden_target_cis`.
    #[allow(clippy::too_many_arguments)]
    fn learn_one_column(
        col: usize,
        deltas: &mut [i32],
        ws: &mut [&mut [i8]],
        hidden_size: Int3,
        ndpc: usize,
        num_hc: usize,
        base_state: u64,
        hidden_acts: &[f32],
        dendrite_acts: &[f32],
        hidden_target_cis: &[i32],
        input_cis: &[&[i32]],
        visible_layer_descs: &[VisibleLayerDesc],
        params: &Params,
    ) {
        let column_pos = Int2::new(
            (col / hidden_size.y as usize) as i32,
            (col % hidden_size.y as usize) as i32,
        );
        let mut state = rand_get_state(base_state + col as u64 * RAND_SUBSEED_OFFSET);

        let hidden_column_index = address2(column_pos, Int2::new(hidden_size.x, hidden_size.y));
        let hidden_cells_start = hidden_column_index * num_hc;
        let target_ci = hidden_target_cis[hidden_column_index] as usize;
        let half_num = ndpc / 2;

        // compute deltas (chunk-relative writes; global reads)
        for hc in 0..num_hc {
            let hidden_cell_index = hc + hidden_cells_start;
            let g_dend = ndpc * hidden_cell_index; // global dendrite base (read)
            let l_dend = ndpc * hc; // chunk-relative dendrite base (write)
            let error =
                params.lr * 127.0 * ((hc == target_ci) as i32 as f32 - hidden_acts[hidden_cell_index]);
            for di in 0..ndpc {
                let sign = if di >= half_num { 2.0f32 } else { 0.0 } - 1.0;
                deltas[l_dend + di] =
                    rand_roundf_step(error * sign * dendrite_acts[g_dend + di], &mut state);
            }
        }

        // apply deltas to this column's weight chunk of each visible layer
        for vli in 0..ws.len() {
            let vld = &visible_layer_descs[vli];
            let diam = vld.radius * 2 + 1;
            let h_to_v = Float2::new(
                vld.size.x as f32 / hidden_size.x as f32,
                vld.size.y as f32 / hidden_size.y as f32,
            );
            let visible_center = project(column_pos, h_to_v);
            let field_lower_bound =
                Int2::new(visible_center.x - vld.radius, visible_center.y - vld.radius);
            let iter_lower_bound =
                Int2::new(field_lower_bound.x.max(0), field_lower_bound.y.max(0));
            let iter_upper_bound = Int2::new(
                (visible_center.x + vld.radius).min(vld.size.x - 1),
                (visible_center.y + vld.radius).min(vld.size.y - 1),
            );
            let vl_input_cis = input_cis[vli];
            let w = &mut ws[vli];

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let visible_column_index =
                        address2(Int2::new(ix, iy), Int2::new(vld.size.x, vld.size.y));
                    let in_ci = vl_input_cis[visible_column_index] as usize;
                    let offset = Int2::new(ix - field_lower_bound.x, iy - field_lower_bound.y);
                    // chunk-relative: the `size.z * hidden_column_index` term is dropped.
                    let wi_start_partial = num_hc
                        * (offset.y as usize + diam as usize * (offset.x as usize + diam as usize * in_ci));
                    for hc in 0..num_hc {
                        let l_dend = ndpc * hc;
                        let wi_start = ndpc * (hc + wi_start_partial);
                        for di in 0..ndpc {
                            let delta = deltas[l_dend + di];
                            w[di + wi_start] = (w[di + wi_start] as i32 + delta).clamp(-127, 127) as i8;
                        }
                    }
                }
            }
        }
    }

    /// Reset hidden CIs and activations to zero.
    pub fn clear_state(&mut self) {
        self.hidden_cis.fill(0);
        self.hidden_acts.fill(0.0);
    }

    /// Return the current hidden column-index predictions.
    pub fn get_hidden_cis(&self) -> &[i32] {
        &self.hidden_cis
    }

    /// Return the current softmax activation probabilities (one per hidden cell).
    pub fn get_hidden_acts(&self) -> &[f32] {
        &self.hidden_acts
    }

    /// Return the spatial size of the hidden (predicted) layer.
    pub fn get_hidden_size(&self) -> Int3 {
        self.hidden_size
    }

    /// Return the number of visible (input) connections.
    pub fn get_num_visible_layers(&self) -> usize {
        self.visible_layers.len()
    }

    /// Return a reference to visible layer `i`.
    pub fn get_visible_layer(&self, i: usize) -> &VisibleLayer {
        &self.visible_layers[i]
    }

    /// Return the descriptor for visible layer `i`.
    pub fn get_visible_layer_desc(&self, i: usize) -> &VisibleLayerDesc {
        &self.visible_layer_descs[i]
    }

    // Serialization

    /// Serialise the full decoder (weights + state) to a [`StreamWriter`].
    pub fn write(&self, writer: &mut dyn StreamWriter) {
        writer.write_int3(self.hidden_size);
        writer.write_i32(self.num_dendrites_per_cell as i32);
        writer.write_i32_slice(&self.hidden_cis);
        writer.write_f32_slice(&self.hidden_acts);
        writer.write_f32_slice(&self.dendrite_acts);
        writer.write_i32(self.visible_layers.len() as i32);

        for (vl, vld) in self.visible_layers.iter().zip(self.visible_layer_descs.iter()) {
            writer.write_int3(vld.size);
            writer.write_i32(vld.radius);
            writer.write_i8_slice(&vl.weights);
        }
    }

    /// Deserialise the decoder from a [`StreamReader`].
    pub fn read(&mut self, reader: &mut dyn StreamReader) {
        self.hidden_size = reader.read_int3();
        self.num_dendrites_per_cell = reader.read_i32() as usize;

        let num_hidden_columns = (self.hidden_size.x * self.hidden_size.y) as usize;
        let num_hidden_cells = num_hidden_columns * self.hidden_size.z as usize;
        let num_dendrites = num_hidden_cells * self.num_dendrites_per_cell;

        self.hidden_cis = vec![0i32; num_hidden_columns];
        reader.read_i32_slice(&mut self.hidden_cis);

        self.hidden_acts = vec![0.0f32; num_hidden_cells];
        reader.read_f32_slice(&mut self.hidden_acts);

        self.dendrite_acts = vec![0.0f32; num_dendrites];
        reader.read_f32_slice(&mut self.dendrite_acts);

        self.dendrite_deltas = vec![0i32; num_dendrites];

        let num_visible_layers = reader.read_i32() as usize;
        self.visible_layers = Vec::with_capacity(num_visible_layers);
        self.visible_layer_descs = Vec::with_capacity(num_visible_layers);

        for _ in 0..num_visible_layers {
            let size = reader.read_int3();
            let radius = reader.read_i32();
            let vld = VisibleLayerDesc { size, radius };

            let diam = vld.radius * 2 + 1;
            let area = (diam * diam) as usize;
            let weights_size = num_dendrites * area * vld.size.z as usize;

            let mut weights = vec![0i8; weights_size];
            reader.read_i8_slice(&mut weights);

            self.visible_layers.push(VisibleLayer { weights });
            self.visible_layer_descs.push(vld);
        }
    }

    /// Serialise only the hidden state (CIs and activations).
    pub fn write_state(&self, writer: &mut dyn StreamWriter) {
        writer.write_i32_slice(&self.hidden_cis);
        writer.write_f32_slice(&self.hidden_acts);
        writer.write_f32_slice(&self.dendrite_acts);
    }

    /// Deserialise only the hidden state.
    pub fn read_state(&mut self, reader: &mut dyn StreamReader) {
        reader.read_i32_slice(&mut self.hidden_cis);
        reader.read_f32_slice(&mut self.hidden_acts);
        reader.read_f32_slice(&mut self.dendrite_acts);
    }

    /// Serialise only the synaptic weights.
    pub fn write_weights(&self, writer: &mut dyn StreamWriter) {
        for vl in &self.visible_layers {
            writer.write_i8_slice(&vl.weights);
        }
    }

    /// Deserialise only the synaptic weights.
    pub fn read_weights(&mut self, reader: &mut dyn StreamReader) {
        for vl in &mut self.visible_layers {
            reader.read_i8_slice(&mut vl.weights);
        }
    }
}
