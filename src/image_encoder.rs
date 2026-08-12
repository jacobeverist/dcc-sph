// AOgmaNeo Rust port - ImageEncoder (SOM for images with reconstruction)
#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;
use crate::helpers::*;

/// Re-export the shared visible-layer descriptor for image-encoder use.
pub use crate::helpers::VisibleLayerDesc;

/// Internal state of one visible (input) connection to the image encoder.
#[derive(Clone, Debug, Default)]
pub struct VisibleLayer {
    /// Forward SOM weights, stored as `u8` in `[0, 255]`.
    /// Interpreted as `w ∈ [-1, 1]` via `w_f = byte / 255.0 * 2.0 - 1.0`.
    pub weights: ByteBuffer,
    /// Reconstruction weights (reverse projection), stored as `u8` in `[0, 255]`.
    pub recon_weights: ByteBuffer,
    /// Latest reconstructed image for this visible layer (byte pixels).
    pub reconstruction: ByteBuffer,
}

/// Hyperparameters for the [`ImageEncoder`].
#[derive(Clone, Debug)]
pub struct Params {
    /// Neighbourhood falloff factor: the learning rate for cells at distance
    /// `d` from the winner is `rate * falloff^d`.
    /// Range: `[0.0, 1.0]`. Default: `0.9`.
    pub falloff: f32,
    /// Learning rate for SOM weight updates.
    /// Range: `[0.0, 1.0]`. Default: `0.1`.
    pub lr: f32,
    /// Scale applied during reconstruction to stretch the linear response
    /// back into `[0, 1]`.
    /// Default: `2.0`.
    pub scale: f32,
    /// Reconstruction learning rate.
    /// Range: `[0.0, 1.0]`. Default: `0.02`.
    pub rr: f32,
    /// Radius of the winner neighbourhood (in hidden-cell units within a column).
    /// Default: `1`.
    pub n_radius: i32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            falloff: 0.9,
            lr: 0.1,
            scale: 2.0,
            rr: 0.02,
            n_radius: 1,
        }
    }
}

// Per-column result of the parallel forward pass
struct ForwardColumnResult {
    /// Winning hidden cell index for this column.
    hidden_ci: i32,
    /// Activation values for all hidden cells in this column.
    hidden_acts: Vec<f32>,
    /// Total (normalisation) values for all hidden cells in this column.
    hidden_totals: Vec<f32>,
}

/// Self-Organising Map (SOM) encoder for byte-valued image inputs.
///
/// Supports multiple visible layers (e.g., separate colour channels or image
/// patches). After training, [`reconstruct`](Self::reconstruct) can be used to
/// decode hidden CIs back into approximate pixel values.
///
/// The forward pass is parallelised across hidden columns using [`rayon`].
/// The SOM learning pass and reconstruction learning remain sequential.
#[derive(Clone, Debug, Default)]
pub struct ImageEncoder {
    hidden_size: Int3,
    hidden_cis: IntBuffer,
    hidden_acts: FloatBuffer,
    hidden_totals: FloatBuffer,
    hidden_resources: FloatBuffer,
    /// Internal state for each visible connection.
    pub visible_layers: Vec<VisibleLayer>,
    /// Structural descriptors (size, radius) for each visible connection.
    pub visible_layer_descs: Vec<VisibleLayerDesc>,
    /// Runtime hyperparameters (may be adjusted between steps).
    pub params: Params,
}

impl ImageEncoder {
    // Pure forward computation for one hidden column.
    // Returns per-column results without mutating self.
    fn compute_forward_column(
        column_pos: Int2,
        hidden_size: Int3,
        visible_layers: &[VisibleLayer],
        visible_layer_descs: &[VisibleLayerDesc],
        inputs: &[&[u8]],
    ) -> ForwardColumnResult {
        let hidden_column_index = address2(column_pos, Int2::new(hidden_size.x, hidden_size.y));

        let byte_inv = 1.0f32 / 255.0;

        // ---- Pass 1: compute the mean input value (center) ----
        let mut center = 0.0f32;
        let mut count = 0usize;

        // Cache receptive field bounds per visible layer so we compute them once.
        let mut rf: Vec<(Int2, Int2, Int2)> = Vec::with_capacity(visible_layer_descs.len()); // (field_lower_bound, iter_lower, iter_upper)

        for vli in 0..visible_layer_descs.len() {
            let vld = &visible_layer_descs[vli];
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

            rf.push((field_lower_bound, iter_lower_bound, iter_upper_bound));

            count += ((iter_upper_bound.x - iter_lower_bound.x + 1)
                * (iter_upper_bound.y - iter_lower_bound.y + 1)) as usize
                * vld.size.z as usize;

            let vl_inputs = inputs[vli];

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let visible_cells_start =
                        vld.size.z as usize * (iy as usize + ix as usize * vld.size.y as usize);

                    for vc in 0..vld.size.z as usize {
                        center += vl_inputs[vc + visible_cells_start] as f32 * byte_inv;
                    }
                }
            }
        }

        center /= count as f32;

        // ---- Pass 2: accumulate SOM activations ----
        let num_hc = hidden_size.z as usize;
        let mut hidden_acts_col = vec![0.0f32; num_hc];
        let mut hidden_totals_col = vec![0.0f32; num_hc];

        for vli in 0..visible_layer_descs.len() {
            let vld = &visible_layer_descs[vli];
            let diam = vld.radius * 2 + 1;
            let (field_lower_bound, iter_lower_bound, iter_upper_bound) = rf[vli];

            let vl_inputs = inputs[vli];
            let vl = &visible_layers[vli];

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let offset = Int2::new(ix - field_lower_bound.x, iy - field_lower_bound.y);
                    let wi_start_partial = vld.size.z as usize
                        * (offset.y as usize
                            + diam as usize
                                * (offset.x as usize
                                    + diam as usize * hidden_column_index));
                    let visible_cells_start =
                        vld.size.z as usize * (iy as usize + ix as usize * vld.size.y as usize);

                    for vc in 0..vld.size.z as usize {
                        let wi_start = hidden_size.z as usize * (vc + wi_start_partial);
                        let input_centered =
                            (vl_inputs[vc + visible_cells_start] as f32 * byte_inv - center) * 2.0;

                        for hc in 0..num_hc {
                            let w = vl.weights[hc + wi_start] as f32 * byte_inv * 2.0 - 1.0;
                            hidden_acts_col[hc] += input_centered * w;
                            hidden_totals_col[hc] += w * w;
                        }
                    }
                }
            }
        }

        // Normalise and find winner
        let mut max_index = 0usize;
        let mut max_activation = LIMIT_MIN;

        for hc in 0..num_hc {
            hidden_acts_col[hc] /= LIMIT_SMALL.max(hidden_totals_col[hc].sqrt());

            if hidden_acts_col[hc] > max_activation {
                max_activation = hidden_acts_col[hc];
                max_index = hc;
            }
        }

        ForwardColumnResult {
            hidden_ci: max_index as i32,
            hidden_acts: hidden_acts_col,
            hidden_totals: hidden_totals_col,
        }
    }

    // Learn SOM weights for one hidden column (sequential, writes to visible_layers).
    fn learn_column(
        &mut self,
        column_pos: Int2,
        inputs: &[&[u8]],
    ) {
        let hidden_size = self.hidden_size;
        let hidden_column_index = address2(column_pos, Int2::new(hidden_size.x, hidden_size.y));
        let hidden_cells_start = hidden_column_index * hidden_size.z as usize;
        let max_index = self.hidden_cis[hidden_column_index] as usize;

        let byte_inv = 1.0f32 / 255.0;

        // Recompute center for this column
        let mut center = 0.0f32;
        let mut total_count = 0usize;

        for vli in 0..self.visible_layers.len() {
            let vld = &self.visible_layer_descs[vli];
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

            total_count += ((iter_upper_bound.x - iter_lower_bound.x + 1)
                * (iter_upper_bound.y - iter_lower_bound.y + 1)) as usize
                * vld.size.z as usize;

            let vl_inputs = inputs[vli];
            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let visible_cells_start =
                        vld.size.z as usize * (iy as usize + ix as usize * vld.size.y as usize);
                    for vc in 0..vld.size.z as usize {
                        center += vl_inputs[vc + visible_cells_start] as f32 * byte_inv;
                    }
                }
            }
        }
        center /= total_count as f32;

        for dhc in -self.params.n_radius..=self.params.n_radius {
            let hc = max_index as i32 + dhc;
            if hc < 0 || hc >= hidden_size.z {
                continue;
            }
            let hc = hc as usize;
            let hidden_cell_index = hc + hidden_cells_start;

            let rate = self.hidden_resources[hidden_cell_index]
                * self.params.falloff.powi(dhc.unsigned_abs() as i32);

            for vli in 0..self.visible_layers.len() {
                let vld = &self.visible_layer_descs[vli];
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

                let vl_inputs = inputs[vli];
                let vl = &mut self.visible_layers[vli];

                for ix in iter_lower_bound.x..=iter_upper_bound.x {
                    for iy in iter_lower_bound.y..=iter_upper_bound.y {
                        let visible_column_index =
                            address2(Int2::new(ix, iy), Int2::new(vld.size.x, vld.size.y));
                        let visible_cells_start = visible_column_index * vld.size.z as usize;
                        let offset = Int2::new(
                            ix - field_lower_bound.x,
                            iy - field_lower_bound.y,
                        );
                        let wi_start_partial = vld.size.z as usize
                            * (offset.y as usize
                                + diam as usize
                                    * (offset.x as usize
                                        + diam as usize * hidden_column_index));

                        for vc in 0..vld.size.z as usize {
                            let wi = hc
                                + hidden_size.z as usize * (vc + wi_start_partial);
                            let input_centered =
                                vl_inputs[vc + visible_cells_start] as f32 * byte_inv - center;
                            let w = vl.weights[wi] as f32 * byte_inv * 2.0 - 1.0;

                            let delta = roundf2i(rate * 255.0 * (input_centered - w));
                            vl.weights[wi] = (vl.weights[wi] as i32 + delta)
                                .clamp(0, 255) as u8;
                        }
                    }
                }
            }

            self.hidden_resources[hidden_cell_index] -= self.params.lr * rate;
        }
    }

    fn learn_reconstruction_column(
        &mut self,
        column_pos: Int2,
        inputs: &[u8],
        vli: usize,
        state: &mut u64,
    ) {
        let hidden_size = self.hidden_size;
        let vld_size = self.visible_layer_descs[vli].size;
        let vld_radius = self.visible_layer_descs[vli].radius;
        let diam = vld_radius * 2 + 1;

        let visible_column_index =
            address2(column_pos, Int2::new(vld_size.x, vld_size.y));
        let visible_cells_start = visible_column_index * vld_size.z as usize;

        let v_to_h = Float2::new(
            hidden_size.x as f32 / vld_size.x as f32,
            hidden_size.y as f32 / vld_size.y as f32,
        );
        let h_to_v = Float2::new(
            vld_size.x as f32 / hidden_size.x as f32,
            vld_size.y as f32 / hidden_size.y as f32,
        );

        let reverse_radii = Int2::new(
            ceilf_to_i32(v_to_h.x * (vld_radius * 2 + 1) as f32 * 0.5),
            ceilf_to_i32(v_to_h.y * (vld_radius * 2 + 1) as f32 * 0.5),
        );

        let hidden_center = project(column_pos, v_to_h);
        let field_lower_bound = Int2::new(
            hidden_center.x - reverse_radii.x,
            hidden_center.y - reverse_radii.y,
        );
        let iter_lower_bound =
            Int2::new(field_lower_bound.x.max(0), field_lower_bound.y.max(0));
        let iter_upper_bound = Int2::new(
            (hidden_center.x + reverse_radii.x).min(hidden_size.x - 1),
            (hidden_center.y + reverse_radii.y).min(hidden_size.y - 1),
        );

        let byte_inv = 1.0f32 / 255.0;

        for vc in 0..vld_size.z as usize {
            let visible_cell_index = vc + visible_cells_start;
            let mut sum = 0.0f32;
            let mut count = 0usize;

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let hidden_pos = Int2::new(ix, iy);
                    let hidden_column_index =
                        address2(hidden_pos, Int2::new(hidden_size.x, hidden_size.y));
                    let visible_center = project(hidden_pos, h_to_v);

                    if in_bounds(
                        column_pos,
                        Int2::new(visible_center.x - vld_radius, visible_center.y - vld_radius),
                        Int2::new(
                            visible_center.x + vld_radius + 1,
                            visible_center.y + vld_radius + 1,
                        ),
                    ) {
                        let hidden_cell_index = self.hidden_cis[hidden_column_index] as usize
                            + hidden_column_index * hidden_size.z as usize;
                        let offset = Int2::new(
                            column_pos.x - visible_center.x + vld_radius,
                            column_pos.y - visible_center.y + vld_radius,
                        );
                        let wi = vc
                            + vld_size.z as usize
                                * (offset.y as usize
                                    + diam as usize
                                        * (offset.x as usize + diam as usize * hidden_cell_index));

                        sum += self.visible_layers[vli].recon_weights[wi] as f32;
                        count += 1;
                    }
                }
            }

            sum /= count.max(1) as f32 * 255.0;

            let target = inputs[visible_cell_index] as f32 * byte_inv;
            let reconstructed =
                ((sum - 0.5) * 2.0 * self.params.scale + 0.5).clamp(0.0, 1.0);
            let delta = rand_roundf_step(self.params.rr * (target - reconstructed) * 255.0, state);

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let hidden_pos = Int2::new(ix, iy);
                    let hidden_column_index =
                        address2(hidden_pos, Int2::new(hidden_size.x, hidden_size.y));
                    let visible_center = project(hidden_pos, h_to_v);

                    if in_bounds(
                        column_pos,
                        Int2::new(visible_center.x - vld_radius, visible_center.y - vld_radius),
                        Int2::new(
                            visible_center.x + vld_radius + 1,
                            visible_center.y + vld_radius + 1,
                        ),
                    ) {
                        let hidden_cell_index = self.hidden_cis[hidden_column_index] as usize
                            + hidden_column_index * hidden_size.z as usize;
                        let offset = Int2::new(
                            column_pos.x - visible_center.x + vld_radius,
                            column_pos.y - visible_center.y + vld_radius,
                        );
                        let wi = vc
                            + vld_size.z as usize
                                * (offset.y as usize
                                    + diam as usize
                                        * (offset.x as usize + diam as usize * hidden_cell_index));

                        self.visible_layers[vli].recon_weights[wi] =
                            (self.visible_layers[vli].recon_weights[wi] as i32 + delta)
                                .clamp(0, 255) as u8;
                    }
                }
            }
        }
    }

    fn reconstruct_column(&mut self, column_pos: Int2, recon_cis: &[i32], vli: usize) {
        let hidden_size = self.hidden_size;
        let vld_size = self.visible_layer_descs[vli].size;
        let vld_radius = self.visible_layer_descs[vli].radius;
        let diam = vld_radius * 2 + 1;

        let visible_column_index =
            address2(column_pos, Int2::new(vld_size.x, vld_size.y));
        let visible_cells_start = visible_column_index * vld_size.z as usize;

        let v_to_h = Float2::new(
            hidden_size.x as f32 / vld_size.x as f32,
            hidden_size.y as f32 / vld_size.y as f32,
        );
        let h_to_v = Float2::new(
            vld_size.x as f32 / hidden_size.x as f32,
            vld_size.y as f32 / hidden_size.y as f32,
        );

        let reverse_radii = Int2::new(
            ceilf_to_i32(v_to_h.x * (vld_radius * 2 + 1) as f32 * 0.5),
            ceilf_to_i32(v_to_h.y * (vld_radius * 2 + 1) as f32 * 0.5),
        );

        let hidden_center = project(column_pos, v_to_h);
        let field_lower_bound = Int2::new(
            hidden_center.x - reverse_radii.x,
            hidden_center.y - reverse_radii.y,
        );
        let iter_lower_bound =
            Int2::new(field_lower_bound.x.max(0), field_lower_bound.y.max(0));
        let iter_upper_bound = Int2::new(
            (hidden_center.x + reverse_radii.x).min(hidden_size.x - 1),
            (hidden_center.y + reverse_radii.y).min(hidden_size.y - 1),
        );

        for vc in 0..vld_size.z as usize {
            let visible_cell_index = vc + visible_cells_start;
            let mut sum = 0.0f32;
            let mut count = 0usize;

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let hidden_pos = Int2::new(ix, iy);
                    let hidden_column_index =
                        address2(hidden_pos, Int2::new(hidden_size.x, hidden_size.y));
                    let visible_center = project(hidden_pos, h_to_v);

                    if in_bounds(
                        column_pos,
                        Int2::new(visible_center.x - vld_radius, visible_center.y - vld_radius),
                        Int2::new(
                            visible_center.x + vld_radius + 1,
                            visible_center.y + vld_radius + 1,
                        ),
                    ) {
                        let hidden_cell_index = recon_cis[hidden_column_index] as usize
                            + hidden_column_index * hidden_size.z as usize;
                        let offset = Int2::new(
                            column_pos.x - visible_center.x + vld_radius,
                            column_pos.y - visible_center.y + vld_radius,
                        );
                        let wi = vc
                            + vld_size.z as usize
                                * (offset.y as usize
                                    + diam as usize
                                        * (offset.x as usize + diam as usize * hidden_cell_index));

                        sum += self.visible_layers[vli].recon_weights[wi] as f32;
                        count += 1;
                    }
                }
            }

            sum /= count.max(1) as f32 * 255.0;

            self.visible_layers[vli].reconstruction[visible_cell_index] = roundf2b(
                ((sum - 0.5) * 2.0 * self.params.scale + 0.5).clamp(0.0, 1.0) * 255.0,
            );
        }
    }

    /// Initialise the image encoder with random weights.
    ///
    /// - `hidden_size` — spatial grid `(x, y)` and code book size `z`.
    /// - `visible_layer_descs` — descriptors for the visible (image) inputs.
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
                let num_visible_columns = (vld.size.x * vld.size.y) as usize;
                let num_visible_cells = num_visible_columns * vld.size.z as usize;
                let diam = vld.radius * 2 + 1;
                let area = (diam * diam) as usize;
                let weights_size = num_hidden_cells * area * vld.size.z as usize;

                let weights: ByteBuffer = (0..weights_size)
                    .map(|_| (global_rand() % 256) as u8)
                    .collect();
                let recon_weights: ByteBuffer = vec![128u8; weights_size];
                let reconstruction: ByteBuffer = vec![0u8; num_visible_cells];

                VisibleLayer {
                    weights,
                    recon_weights,
                    reconstruction,
                }
            })
            .collect();

        self.hidden_cis = vec![0i32; num_hidden_columns];
        self.hidden_acts = vec![0.0f32; num_hidden_cells];
        self.hidden_totals = vec![0.0f32; num_hidden_cells];
        self.hidden_resources = vec![0.5f32; num_hidden_cells];
    }

    /// Run one encode step, updating `hidden_cis` and optionally learning.
    ///
    /// - `inputs` — one `&[u8]` slice per visible layer (raw pixel bytes).
    /// - `learn_enabled` — if `true`, SOM weights are updated towards the input.
    /// - `learn_recon` — if `true` and `learn_enabled`, reconstruction weights
    ///   are updated from the error between the decoded image and `inputs`.
    pub fn step(&mut self, inputs: &[&[u8]], learn_enabled: bool, learn_recon: bool) {
        let num_hidden_columns = (self.hidden_size.x * self.hidden_size.y) as usize;
        let hidden_size = self.hidden_size;

        // ---- Parallel forward pass ----
        let results: Vec<ForwardColumnResult> = (0..num_hidden_columns)
            .into_par_iter()
            .map(|i| {
                let column_pos = Int2::new(
                    (i / hidden_size.y as usize) as i32,
                    (i % hidden_size.y as usize) as i32,
                );
                Self::compute_forward_column(
                    column_pos,
                    hidden_size,
                    &self.visible_layers,
                    &self.visible_layer_descs,
                    inputs,
                )
            })
            .collect();

        // Write-back
        for (i, res) in results.into_iter().enumerate() {
            let cells_start = i * hidden_size.z as usize;
            self.hidden_cis[i] = res.hidden_ci;
            self.hidden_acts[cells_start..cells_start + hidden_size.z as usize]
                .copy_from_slice(&res.hidden_acts);
            self.hidden_totals[cells_start..cells_start + hidden_size.z as usize]
                .copy_from_slice(&res.hidden_totals);
        }

        // ---- Sequential SOM learning pass ----
        if learn_enabled {
            for i in 0..num_hidden_columns {
                let column_pos = Int2::new(
                    (i / hidden_size.y as usize) as i32,
                    (i % hidden_size.y as usize) as i32,
                );
                self.learn_column(column_pos, inputs);
            }

            if learn_recon {
                let base_state = global_rand() as u64;

                for vli in 0..self.visible_layers.len() {
                    let num_visible_columns =
                        (self.visible_layer_descs[vli].size.x * self.visible_layer_descs[vli].size.y)
                            as usize;

                    for i in 0..num_visible_columns {
                        let column_pos = Int2::new(
                            (i / self.visible_layer_descs[vli].size.y as usize) as i32,
                            (i % self.visible_layer_descs[vli].size.y as usize) as i32,
                        );
                        let mut state = rand_get_state(base_state + i as u64 * RAND_SUBSEED_OFFSET);
                        // Pass inputs[vli] directly — no unnecessary heap allocation
                        self.learn_reconstruction_column(column_pos, inputs[vli], vli, &mut state);
                    }
                }
            }
        }
    }

    /// Decode the given hidden CIs back into approximate pixel values.
    ///
    /// Results are stored in [`VisibleLayer::reconstruction`] and accessible
    /// via [`get_reconstruction`](Self::get_reconstruction).
    pub fn reconstruct(&mut self, recon_cis: &[i32]) {
        for vli in 0..self.visible_layers.len() {
            let num_visible_columns =
                (self.visible_layer_descs[vli].size.x * self.visible_layer_descs[vli].size.y)
                    as usize;

            for i in 0..num_visible_columns {
                let column_pos = Int2::new(
                    (i / self.visible_layer_descs[vli].size.y as usize) as i32,
                    (i % self.visible_layer_descs[vli].size.y as usize) as i32,
                );
                self.reconstruct_column(column_pos, recon_cis, vli);
            }
        }
    }

    /// Return the decoded pixel byte array for visible layer `vli`.
    pub fn get_reconstruction(&self, vli: usize) -> &[u8] {
        &self.visible_layers[vli].reconstruction
    }

    /// Return the current hidden column-index array.
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

    /// Return a reference to visible layer `vli`.
    pub fn get_visible_layer(&self, vli: usize) -> &VisibleLayer {
        &self.visible_layers[vli]
    }

    /// Return the descriptor for visible layer `vli`.
    pub fn get_visible_layer_desc(&self, vli: usize) -> &VisibleLayerDesc {
        &self.visible_layer_descs[vli]
    }

    // Serialization

    /// Serialise the full image encoder (weights + state + params) to a [`StreamWriter`].
    pub fn write(&self, writer: &mut dyn StreamWriter) {
        writer.write_int3(self.hidden_size);
        // params
        writer.write_f32(self.params.falloff);
        writer.write_f32(self.params.lr);
        writer.write_f32(self.params.scale);
        writer.write_f32(self.params.rr);
        writer.write_i32(self.params.n_radius);

        writer.write_i32_slice(&self.hidden_cis);
        writer.write_f32_slice(&self.hidden_resources);
        writer.write_i32(self.visible_layers.len() as i32);

        for (vl, vld) in self.visible_layers.iter().zip(self.visible_layer_descs.iter()) {
            writer.write_int3(vld.size);
            writer.write_i32(vld.radius);
            writer.write_u8_slice(&vl.weights);
            writer.write_u8_slice(&vl.recon_weights);
        }
    }

    /// Deserialise the image encoder from a [`StreamReader`].
    pub fn read(&mut self, reader: &mut dyn StreamReader) {
        self.hidden_size = reader.read_int3();

        self.params.falloff = reader.read_f32();
        self.params.lr = reader.read_f32();
        self.params.scale = reader.read_f32();
        self.params.rr = reader.read_f32();
        self.params.n_radius = reader.read_i32();

        let num_hidden_columns = (self.hidden_size.x * self.hidden_size.y) as usize;
        let num_hidden_cells = num_hidden_columns * self.hidden_size.z as usize;

        self.hidden_cis = vec![0i32; num_hidden_columns];
        reader.read_i32_slice(&mut self.hidden_cis);

        self.hidden_acts = vec![0.0f32; num_hidden_cells];
        self.hidden_totals = vec![0.0f32; num_hidden_cells];

        self.hidden_resources = vec![0.0f32; num_hidden_cells];
        reader.read_f32_slice(&mut self.hidden_resources);

        let num_visible_layers = reader.read_i32() as usize;
        self.visible_layers = Vec::with_capacity(num_visible_layers);
        self.visible_layer_descs = Vec::with_capacity(num_visible_layers);

        for _ in 0..num_visible_layers {
            let size = reader.read_int3();
            let radius = reader.read_i32();
            let vld = VisibleLayerDesc { size, radius };

            let num_visible_columns = (vld.size.x * vld.size.y) as usize;
            let num_visible_cells = num_visible_columns * vld.size.z as usize;
            let diam = vld.radius * 2 + 1;
            let area = (diam * diam) as usize;
            let weights_size = num_hidden_cells * area * vld.size.z as usize;

            let mut weights = vec![0u8; weights_size];
            reader.read_u8_slice(&mut weights);

            let mut recon_weights = vec![0u8; weights_size];
            reader.read_u8_slice(&mut recon_weights);

            let reconstruction = vec![0u8; num_visible_cells];

            self.visible_layers.push(VisibleLayer {
                weights,
                recon_weights,
                reconstruction,
            });
            self.visible_layer_descs.push(vld);
        }
    }

    /// Serialise only the hidden CIs.
    pub fn write_state(&self, writer: &mut dyn StreamWriter) {
        writer.write_i32_slice(&self.hidden_cis);
    }

    /// Deserialise only the hidden CIs.
    pub fn read_state(&mut self, reader: &mut dyn StreamReader) {
        reader.read_i32_slice(&mut self.hidden_cis);
    }

    /// Serialise only the SOM and reconstruction weights.
    pub fn write_weights(&self, writer: &mut dyn StreamWriter) {
        for vl in &self.visible_layers {
            writer.write_u8_slice(&vl.weights);
            writer.write_u8_slice(&vl.recon_weights);
        }
    }

    /// Deserialise only the SOM and reconstruction weights.
    pub fn read_weights(&mut self, reader: &mut dyn StreamReader) {
        for vl in &mut self.visible_layers {
            reader.read_u8_slice(&mut vl.weights);
            reader.read_u8_slice(&mut vl.recon_weights);
        }
    }
}
