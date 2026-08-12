// AOgmaNeo Rust port - Encoder (sparse coder with ART)
#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;
use crate::helpers::*;

/// Re-export the shared visible-layer descriptor for encoder use.
pub use crate::helpers::VisibleLayerDesc;

/// Internal state of one visible (input) connection to the encoder.
#[derive(Clone, Debug, Default)]
pub struct VisibleLayer {
    /// Synaptic weights, stored as `u8` in `[0, 255]`.
    ///
    /// Layout: `weights[hc + num_hc * (dy + diam * (dx + diam * (in_ci + z * hidden_col)))]`
    pub weights: ByteBuffer,
    /// Running sum of weights for each hidden cell (used for ART match computation).
    pub hidden_totals: IntBuffer,
    /// Relative importance of this visible layer in the combined activation.
    /// Default: `1.0`. Higher values give this input more influence.
    pub importance: f32,
}

impl VisibleLayer {
    /// Create a new, empty [`VisibleLayer`] with importance `1.0`.
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            hidden_totals: Vec::new(),
            importance: 1.0,
        }
    }
}

/// Hyperparameters for the [`Encoder`].
#[derive(Clone, Debug)]
pub struct Params {
    /// ART choice function denominator bias. Prevents division by zero and
    /// controls the preference between uncommitted and committed nodes.
    /// Range: `(0.0, ∞)`. Default: `0.01`.
    pub choice: f32,
    /// ART vigilance threshold. A committed node is only selected if its match
    /// ratio (overlap / expected) exceeds this value.
    /// Range: `[0.0, 1.0]`. Default: `0.9`.
    pub vigilance: f32,
    /// Learning rate applied to committed nodes. Uncommitted nodes always learn
    /// at rate `1.0` (fast commit).
    /// Range: `[0.0, 1.0]`. Default: `0.5`.
    pub lr: f32,
    /// Maximum fraction of the lateral neighbourhood that may have a higher
    /// match score than the winner. Controls effective sparsity.
    /// Range: `[0.0, 1.0]`. Default: `0.1`.
    pub active_ratio: f32,
    /// Radius (in hidden columns) of the lateral inhibition neighbourhood.
    /// Default: `2`.
    pub l_radius: i32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            choice: 0.01,
            vigilance: 0.9,
            lr: 0.5,
            active_ratio: 0.1,
            l_radius: 2,
        }
    }
}

// Result of one column's forward pass
struct ForwardResult {
    hidden_ci: i32,
    learn_flag: u8,
    comparison: f32,
}

/// Adaptive Resonance Theory (ART) sparse coder.
///
/// The encoder maps a collection of discrete column-index (CI) inputs from one
/// or more visible layers to a single discrete CI output per hidden column.
/// Learning is unsupervised: each column independently performs competitive
/// learning with a vigilance gate that prevents over-generalisation.
///
/// The forward pass is parallelised across hidden columns using [`rayon`].
#[derive(Clone, Debug, Default)]
pub struct Encoder {
    hidden_size: Int3,
    hidden_cis: IntBuffer,
    hidden_learn_flags: ByteBuffer,
    hidden_committed_flags: ByteBuffer,
    hidden_comparisons: FloatBuffer,
    /// Internal state for each visible connection.
    pub visible_layers: Vec<VisibleLayer>,
    /// Structural descriptors (size, radius) for each visible connection.
    pub visible_layer_descs: Vec<VisibleLayerDesc>,
}

impl Encoder {
    // Compute forward for a single column; returns (ci, learn_flag, comparison)
    #[allow(clippy::too_many_arguments)]
    fn compute_forward(
        column_pos: Int2,
        hidden_size: Int3,
        visible_layers: &[VisibleLayer],
        visible_layer_descs: &[VisibleLayerDesc],
        input_cis: &[&[i32]],
        hidden_committed_flags: &[u8],
        hidden_totals_snapshot: &[Vec<i32>], // snapshot per vli
        params: &Params,
    ) -> ForwardResult {
        let hidden_column_index = address2(column_pos, Int2::new(hidden_size.x, hidden_size.y));
        let hidden_cells_start = hidden_column_index * hidden_size.z as usize;

        let mut count_except = 0.0f32;
        let mut count_all = 0.0f32;

        let num_hc = hidden_size.z as usize;
        let num_vl = visible_layers.len();

        // local sums[vli][hc]
        let mut local_sums: Vec<i32> = vec![0i32; num_vl * num_hc];

        for vli in 0..num_vl {
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

            let sub_count = (iter_upper_bound.x - iter_lower_bound.x + 1)
                * (iter_upper_bound.y - iter_lower_bound.y + 1);
            count_except += vl.importance * sub_count as f32 * (vld.size.z - 1) as f32;
            count_all += vl.importance * sub_count as f32 * vld.size.z as f32;

            let vl_input_cis = input_cis[vli];

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let visible_column_index =
                        address2(Int2::new(ix, iy), Int2::new(vld.size.x, vld.size.y));
                    let in_ci = vl_input_cis[visible_column_index] as usize;
                    let offset = Int2::new(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    let wi_start = num_hc
                        * (offset.y as usize
                            + diam as usize
                                * (offset.x as usize
                                    + diam as usize
                                        * (in_ci
                                            + vld.size.z as usize * hidden_column_index)));

                    for hc in 0..num_hc {
                        local_sums[vli * num_hc + hc] += vl.weights[hc + wi_start] as i32;
                    }
                }
            }
        }

        let mut max_index: i32 = -1;
        let mut max_activation = 0.0f32;
        let mut max_complete_index = 0usize;
        let mut max_complete_activation = 0.0f32;

        let byte_inv = 1.0f32 / 255.0;

        for hc in 0..num_hc {
            let hidden_cell_index = hc + hidden_cells_start;

            let mut sum = 0.0f32;
            let mut total = 0.0f32;

            for vli in 0..num_vl {
                let vl = &visible_layers[vli];
                let influence = vl.importance * byte_inv;
                sum += local_sums[vli * num_hc + hc] as f32 * influence;
                total += hidden_totals_snapshot[vli][hidden_cell_index] as f32 * influence;
            }

            let complemented = sum - total + count_except;
            let match_val = if count_except > 0.0 { complemented / count_except } else { 0.0 };
            let activation = complemented / (params.choice + count_all - total);

            let committed = hidden_committed_flags[hidden_cell_index] != 0;

            if (!committed || match_val >= params.vigilance) && activation > max_activation {
                max_activation = activation;
                max_index = hc as i32;
            }

            if activation > max_complete_activation {
                max_complete_activation = activation;
                max_complete_index = hc;
            }
        }

        ForwardResult {
            hidden_ci: if max_index == -1 { max_complete_index as i32 } else { max_index },
            learn_flag: if max_index != -1 { 1 } else { 0 },
            comparison: if max_index == -1 { 0.0 } else { max_complete_activation },
        }
    }

    /// Initialise the encoder with random weights.
    ///
    /// - `hidden_size` — spatial grid size `(x, y)` and column vocabulary `z`.
    /// - `visible_layer_descs` — one descriptor per visible (input) connection.
    pub fn init_random(
        &mut self,
        hidden_size: Int3,
        visible_layer_descs: Vec<VisibleLayerDesc>,
    ) {
        self.visible_layer_descs = visible_layer_descs;
        self.hidden_size = hidden_size;

        let num_hidden_columns = (hidden_size.x * hidden_size.y) as usize;
        let num_hidden_cells = num_hidden_columns * hidden_size.z as usize;

        self.visible_layers = self
            .visible_layer_descs
            .iter()
            .map(|vld| {
                let diam = vld.radius * 2 + 1;
                let area = (diam * diam) as usize;
                let weights_size = num_hidden_cells * area * vld.size.z as usize;

                let weights: ByteBuffer = (0..weights_size)
                    .map(|_| (global_rand() % INIT_WEIGHT_NOISEI) as u8)
                    .collect();

                VisibleLayer {
                    weights,
                    hidden_totals: vec![0i32; num_hidden_cells],
                    importance: 1.0,
                }
            })
            .collect();

        self.hidden_cis = vec![0i32; num_hidden_columns];
        self.hidden_learn_flags = vec![0u8; num_hidden_columns];
        self.hidden_committed_flags = vec![0u8; num_hidden_cells];
        self.hidden_comparisons = vec![0.0f32; num_hidden_columns];
    }

    /// Run one forward (and optionally learn) step.
    ///
    /// - `input_cis` — one CI slice per visible layer, in the same order as
    ///   `visible_layer_descs`.
    /// - `learn_enabled` — if `false`, only the forward pass runs (no weight updates).
    /// - `params` — hyperparameters for this step.
    pub fn step(&mut self, input_cis: &[&[i32]], learn_enabled: bool, params: &Params) {
        let num_hidden_columns = (self.hidden_size.x * self.hidden_size.y) as usize;
        let hidden_size = self.hidden_size;

        // Take snapshots of hidden_totals for the parallel forward pass
        // (forward only reads them; learn updates them sequentially after)
        let hidden_totals_snapshot: Vec<Vec<i32>> = self
            .visible_layers
            .iter()
            .map(|vl| vl.hidden_totals.clone())
            .collect();

        // --- Parallel forward pass ---
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
                    &self.visible_layers,
                    &self.visible_layer_descs,
                    input_cis,
                    &self.hidden_committed_flags,
                    &hidden_totals_snapshot,
                    params,
                )
            })
            .collect();

        for (i, res) in results.into_iter().enumerate() {
            self.hidden_cis[i] = res.hidden_ci;
            self.hidden_learn_flags[i] = res.learn_flag;
            self.hidden_comparisons[i] = res.comparison;
        }

        if !learn_enabled {
            return;
        }

        // --- Sequential learn pass ---
        for i in 0..num_hidden_columns {
            let column_pos = Int2::new(
                (i / hidden_size.y as usize) as i32,
                (i % hidden_size.y as usize) as i32,
            );

            let hidden_column_index =
                address2(column_pos, Int2::new(hidden_size.x, hidden_size.y));
            let hidden_cells_start = hidden_column_index * hidden_size.z as usize;

            if self.hidden_learn_flags[hidden_column_index] == 0 {
                continue;
            }

            let hidden_ci = self.hidden_cis[hidden_column_index] as usize;
            let hidden_max = self.hidden_comparisons[hidden_column_index];

            // lateral inhibition check
            let mut num_higher = 0usize;
            let mut count = 1usize;

            for dcx in -params.l_radius..=params.l_radius {
                for dcy in -params.l_radius..=params.l_radius {
                    if dcx == 0 && dcy == 0 {
                        continue;
                    }
                    let other_pos = Int2::new(column_pos.x + dcx, column_pos.y + dcy);
                    if in_bounds0(other_pos, Int2::new(hidden_size.x, hidden_size.y)) {
                        let other_idx =
                            address2(other_pos, Int2::new(hidden_size.x, hidden_size.y));
                        if self.hidden_comparisons[other_idx] >= hidden_max {
                            num_higher += 1;
                        }
                        count += 1;
                    }
                }
            }

            if num_higher as f32 / count as f32 > params.active_ratio {
                continue;
            }

            let hidden_cell_index_max = hidden_ci + hidden_cells_start;
            let committed = self.hidden_committed_flags[hidden_cell_index_max] != 0;
            let rate = if committed { params.lr } else { 1.0 };

            for vli in 0..self.visible_layers.len() {
                let vld_size = self.visible_layer_descs[vli].size;
                let vld_radius = self.visible_layer_descs[vli].radius;
                let diam = vld_radius * 2 + 1;

                let h_to_v = Float2::new(
                    vld_size.x as f32 / hidden_size.x as f32,
                    vld_size.y as f32 / hidden_size.y as f32,
                );
                let visible_center = project(column_pos, h_to_v);
                let field_lower_bound = Int2::new(
                    visible_center.x - vld_radius,
                    visible_center.y - vld_radius,
                );
                let iter_lower_bound = Int2::new(
                    field_lower_bound.x.max(0),
                    field_lower_bound.y.max(0),
                );
                let iter_upper_bound = Int2::new(
                    (visible_center.x + vld_radius).min(vld_size.x - 1),
                    (visible_center.y + vld_radius).min(vld_size.y - 1),
                );

                let vl_input_cis = input_cis[vli];

                for ix in iter_lower_bound.x..=iter_upper_bound.x {
                    for iy in iter_lower_bound.y..=iter_upper_bound.y {
                        let visible_column_index =
                            address2(Int2::new(ix, iy), Int2::new(vld_size.x, vld_size.y));
                        let in_ci = vl_input_cis[visible_column_index] as usize;
                        let offset = Int2::new(
                            ix - field_lower_bound.x,
                            iy - field_lower_bound.y,
                        );

                        let wi = hidden_ci
                            + hidden_size.z as usize
                                * (offset.y as usize
                                    + diam as usize
                                        * (offset.x as usize
                                            + diam as usize
                                                * (in_ci
                                                    + vld_size.z as usize * hidden_column_index)));

                        let vl = &mut self.visible_layers[vli];
                        let w_old = vl.weights[wi];
                        let delta = ceilf_to_i32(rate * (255.0 - w_old as f32));
                        vl.weights[wi] = (w_old as i32 + delta).min(255) as u8;
                        vl.hidden_totals[hidden_cell_index_max] +=
                            vl.weights[wi] as i32 - w_old as i32;
                    }
                }
            }

            self.hidden_committed_flags[hidden_cell_index_max] = 1;
        }
    }

    /// Reset the hidden CIs to zero (column 0 in each column).
    pub fn clear_state(&mut self) {
        self.hidden_cis.fill(0);
    }

    /// Return the current hidden column-index array (one entry per hidden column).
    pub fn get_hidden_cis(&self) -> &[i32] {
        &self.hidden_cis
    }

    /// Return the spatial size of the hidden layer.
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

    /// Return a mutable reference to visible layer `i`.
    pub fn get_visible_layer_mut(&mut self, i: usize) -> &mut VisibleLayer {
        &mut self.visible_layers[i]
    }

    /// Return the descriptor for visible layer `i`.
    pub fn get_visible_layer_desc(&self, i: usize) -> &VisibleLayerDesc {
        &self.visible_layer_descs[i]
    }

    // Serialization

    /// Serialise the full encoder (weights + state) to a [`StreamWriter`].
    pub fn write(&self, writer: &mut dyn StreamWriter) {
        writer.write_int3(self.hidden_size);
        writer.write_i32_slice(&self.hidden_cis);
        writer.write_u8_slice(&self.hidden_committed_flags);
        writer.write_i32(self.visible_layers.len() as i32);

        for (vl, vld) in self.visible_layers.iter().zip(self.visible_layer_descs.iter()) {
            writer.write_int3(vld.size);
            writer.write_i32(vld.radius);
            writer.write_u8_slice(&vl.weights);
            writer.write_i32_slice(&vl.hidden_totals);
            writer.write_f32(vl.importance);
        }
    }

    /// Deserialise the encoder from a [`StreamReader`].
    pub fn read(&mut self, reader: &mut dyn StreamReader) {
        self.hidden_size = reader.read_int3();

        let num_hidden_columns = (self.hidden_size.x * self.hidden_size.y) as usize;
        let num_hidden_cells = num_hidden_columns * self.hidden_size.z as usize;

        self.hidden_cis = vec![0i32; num_hidden_columns];
        reader.read_i32_slice(&mut self.hidden_cis);

        self.hidden_learn_flags = vec![0u8; num_hidden_columns];
        self.hidden_committed_flags = vec![0u8; num_hidden_cells];
        reader.read_u8_slice(&mut self.hidden_committed_flags);
        self.hidden_comparisons = vec![0.0f32; num_hidden_columns];

        let num_visible_layers = reader.read_i32() as usize;
        self.visible_layers = Vec::with_capacity(num_visible_layers);
        self.visible_layer_descs = Vec::with_capacity(num_visible_layers);

        for _ in 0..num_visible_layers {
            let size = reader.read_int3();
            let radius = reader.read_i32();
            let vld = VisibleLayerDesc { size, radius };

            let diam = vld.radius * 2 + 1;
            let area = (diam * diam) as usize;
            let weights_size = num_hidden_cells * area * vld.size.z as usize;

            let mut weights = vec![0u8; weights_size];
            reader.read_u8_slice(&mut weights);

            let mut hidden_totals = vec![0i32; num_hidden_cells];
            reader.read_i32_slice(&mut hidden_totals);

            let importance = reader.read_f32();

            self.visible_layers.push(VisibleLayer {
                weights,
                hidden_totals,
                importance,
            });
            self.visible_layer_descs.push(vld);
        }
    }

    /// Serialise only the hidden CIs (fast partial save).
    pub fn write_state(&self, writer: &mut dyn StreamWriter) {
        writer.write_i32_slice(&self.hidden_cis);
    }

    /// Deserialise only the hidden CIs.
    pub fn read_state(&mut self, reader: &mut dyn StreamReader) {
        reader.read_i32_slice(&mut self.hidden_cis);
    }

    /// Serialise only the synaptic weights.
    pub fn write_weights(&self, writer: &mut dyn StreamWriter) {
        for vl in &self.visible_layers {
            writer.write_u8_slice(&vl.weights);
        }
    }

    /// Deserialise only the synaptic weights.
    pub fn read_weights(&mut self, reader: &mut dyn StreamReader) {
        for vl in &mut self.visible_layers {
            reader.read_u8_slice(&mut vl.weights);
        }
    }
}
