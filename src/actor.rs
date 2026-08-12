// AOgmaNeo Rust port - Actor (RL with actor-critic, eligibility traces)
#![allow(clippy::needless_range_loop)]

use crate::helpers::*;

/// Re-export the shared visible-layer descriptor for actor use.
pub use crate::helpers::VisibleLayerDesc;

/// Internal state of one visible (input) connection to the actor.
#[derive(Clone, Debug, Default)]
pub struct VisibleLayer {
    /// Float weights for the value (critic) network.
    pub value_weights: FloatBuffer,
    /// Float weights for the policy (actor) network.
    pub policy_weights: FloatBuffer,
}

/// One element in the actor's experience replay buffer.
#[derive(Clone, Debug, Default)]
pub struct HistorySample {
    /// Input CIs at this timestep, one slice per visible layer.
    pub input_cis: Vec<IntBuffer>,
    /// Previous timestep's target (taken) action CIs.
    pub hidden_target_cis_prev: IntBuffer,
    /// Value estimates at this timestep (one per hidden column).
    pub hidden_values: FloatBuffer,
    /// Reward received at this timestep.
    pub reward: f32,
}

/// Hyperparameters for the [`Actor`].
#[derive(Clone, Debug)]
pub struct Params {
    /// Value (critic) learning rate.
    /// Range: `[0.0, 1.0]`. Default: `0.1`.
    pub vlr: f32,
    /// Policy (actor) learning rate.
    /// Range: `[0.0, 1.0]`. Default: `0.01`.
    pub plr: f32,
    /// Smoothing factor for the TD return blend. Controls how much weight is
    /// given to the multi-step return vs. the current value estimate.
    /// Range: `[0.0, 1.0]`. Default: `0.02`.
    pub smoothing: f32,
    /// Discount factor γ for future rewards.
    /// Range: `[0.0, 1.0]`. Default: `0.99`.
    pub discount: f32,
    /// Per-step decay of the TD-error scale normaliser.
    /// Range: `[0.0, 1.0]`. Default: `0.999`.
    pub td_scale_decay: f32,
    /// Symmetric log / exp value range. Rewards are mapped into
    /// `[-value_range, value_range]` via [`symlogf`].
    /// Default: `10.0`.
    pub value_range: f32,
    /// Minimum number of history steps before learning starts.
    /// Default: `16`.
    pub min_steps: usize,
    /// Number of random history samples used per learning call.
    /// Default: `8`.
    pub history_iters: usize,
    /// Leaky-ReLU coefficient for dendrite activations:
    /// `activation = softplus(x) - leak * softplus(-x)`.
    /// Range: `[0.0, 1.0]`. Default: `0.01`.
    pub leak: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            vlr: 0.1,
            plr: 0.01,
            smoothing: 0.02,
            discount: 0.99,
            td_scale_decay: 0.999,
            value_range: 10.0,
            min_steps: 16,
            history_iters: 8,
            leak: 0.01,
        }
    }
}

/// Actor-critic reinforcement learning module.
///
/// Combines a policy network (actor) and a value network (critic), both
/// implemented as multi-dendrite perceptrons, inside a single module.
/// Learning uses a temporal-difference return with a circular replay buffer.
///
/// Actions are sampled from the policy's softmax distribution; during mimic
/// learning, an external target CI is provided via the `mimic` signal in
/// [`step`](Self::step).
///
/// The forward pass is sequential (the actor is not parallelised with rayon).
#[derive(Clone, Debug, Default)]
pub struct Actor {
    hidden_size: Int3,
    value_size: usize,
    value_num_dendrites_per_cell: usize,
    policy_num_dendrites_per_cell: usize,
    history_size: usize,
    hidden_cis: IntBuffer,
    hidden_value_acts: FloatBuffer,
    hidden_policy_acts: FloatBuffer,
    value_dendrite_acts: FloatBuffer,
    policy_dendrite_acts: FloatBuffer,
    hidden_values: FloatBuffer,
    hidden_td_scales: FloatBuffer,
    history_samples: CircleBuffer<HistorySample>,
    /// Internal state for each visible connection.
    pub visible_layers: Vec<VisibleLayer>,
    /// Structural descriptors (size, radius) for each visible connection.
    pub visible_layer_descs: Vec<VisibleLayerDesc>,
}

impl Actor {
    fn forward_column(
        &mut self,
        column_pos: Int2,
        input_cis: &[&[i32]],
        state: &mut u64,
        params: &Params,
    ) {
        let hidden_size = self.hidden_size;
        let hidden_column_index = address2(column_pos, Int2::new(hidden_size.x, hidden_size.y));
        let hidden_cells_start = hidden_column_index * hidden_size.z as usize;
        let value_cells_start = hidden_column_index * self.value_size;

        // zero dendrite accumulators
        for vac in 0..self.value_size {
            let value_dendrites_start =
                self.value_num_dendrites_per_cell * (vac + value_cells_start);
            for di in 0..self.value_num_dendrites_per_cell {
                self.value_dendrite_acts[di + value_dendrites_start] = 0.0;
            }
        }
        for hc in 0..hidden_size.z as usize {
            let policy_dendrites_start =
                self.policy_num_dendrites_per_cell * (hc + hidden_cells_start);
            for di in 0..self.policy_num_dendrites_per_cell {
                self.policy_dendrite_acts[di + policy_dendrites_start] = 0.0;
            }
        }

        let mut count = 0usize;

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

            count += ((iter_upper_bound.x - iter_lower_bound.x + 1)
                * (iter_upper_bound.y - iter_lower_bound.y + 1)) as usize;

            let vl_input_cis = input_cis[vli];
            let vl = &self.visible_layers[vli];

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let visible_column_index =
                        address2(Int2::new(ix, iy), Int2::new(vld.size.x, vld.size.y));
                    let in_ci = vl_input_cis[visible_column_index] as usize;
                    let offset = Int2::new(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    let wi_start_partial = offset.y as usize
                        + diam as usize
                            * (offset.x as usize
                                + diam as usize
                                    * (in_ci + vld.size.z as usize * hidden_column_index));
                    let wi_value_partial = self.value_size * wi_start_partial;
                    let wi_policy_partial = hidden_size.z as usize * wi_start_partial;

                    for vac in 0..self.value_size {
                        let value_dendrites_start =
                            self.value_num_dendrites_per_cell * (vac + value_cells_start);
                        let wi_start =
                            self.value_num_dendrites_per_cell * (vac + wi_value_partial);
                        for di in 0..self.value_num_dendrites_per_cell {
                            self.value_dendrite_acts[di + value_dendrites_start] +=
                                vl.value_weights[di + wi_start];
                        }
                    }

                    for hc in 0..hidden_size.z as usize {
                        let policy_dendrites_start =
                            self.policy_num_dendrites_per_cell * (hc + hidden_cells_start);
                        let wi_start =
                            self.policy_num_dendrites_per_cell * (hc + wi_policy_partial);
                        for di in 0..self.policy_num_dendrites_per_cell {
                            self.policy_dendrite_acts[di + policy_dendrites_start] +=
                                vl.policy_weights[di + wi_start];
                        }
                    }
                }
            }
        }

        let half_value_num = self.value_num_dendrites_per_cell / 2;
        let half_policy_num = self.policy_num_dendrites_per_cell / 2;
        let dendrite_scale = (1.0f32 / count as f32).sqrt();
        let value_activation_scale = (1.0f32 / self.value_num_dendrites_per_cell as f32).sqrt();
        let policy_activation_scale =
            (1.0f32 / self.policy_num_dendrites_per_cell as f32).sqrt();

        let mut max_value_activation = LIMIT_MIN;
        for vac in 0..self.value_size {
            let value_dendrites_start =
                self.value_num_dendrites_per_cell * (vac + value_cells_start);
            let mut activation = 0.0f32;

            for di in 0..self.value_num_dendrites_per_cell {
                let act = self.value_dendrite_acts[di + value_dendrites_start] * dendrite_scale;
                let leaky = softplusf(act) - params.leak * softplusf(-act);
                activation += leaky * (if di >= half_value_num { 2.0 } else { 0.0 } - 1.0);
            }

            activation *= value_activation_scale;
            self.hidden_value_acts[vac + value_cells_start] = activation;

            if activation > max_value_activation {
                max_value_activation = activation;
            }
        }

        // softmax for value
        let mut total = 0.0f32;
        for vac in 0..self.value_size {
            let idx = vac + value_cells_start;
            self.hidden_value_acts[idx] =
                (self.hidden_value_acts[idx] - max_value_activation).exp();
            total += self.hidden_value_acts[idx];
        }
        let total_inv = 1.0 / LIMIT_SMALL.max(total);
        let mut smooth_max_value_index = 0.0f32;
        for vac in 0..self.value_size {
            let idx = vac + value_cells_start;
            self.hidden_value_acts[idx] *= total_inv;
            smooth_max_value_index += self.hidden_value_acts[idx] * vac as f32;
        }

        let value = symexpf(
            (smooth_max_value_index / (self.value_size - 1) as f32 * 2.0 - 1.0) * params.value_range,
        );
        self.hidden_values[hidden_column_index] = value;

        // policy
        let mut max_activation = LIMIT_MIN;
        for hc in 0..hidden_size.z as usize {
            let policy_dendrites_start =
                self.policy_num_dendrites_per_cell * (hc + hidden_cells_start);
            let mut activation = 0.0f32;

            for di in 0..self.policy_num_dendrites_per_cell {
                let act = self.policy_dendrite_acts[di + policy_dendrites_start] * dendrite_scale;
                let leaky = softplusf(act) - params.leak * softplusf(-act);
                activation += leaky * (if di >= half_policy_num { 2.0 } else { 0.0 } - 1.0);
            }

            activation *= policy_activation_scale;
            self.hidden_policy_acts[hc + hidden_cells_start] = activation;
            max_activation = max_activation.max(activation);
        }

        // softmax for policy
        let mut total = 0.0f32;
        for hc in 0..hidden_size.z as usize {
            let idx = hc + hidden_cells_start;
            self.hidden_policy_acts[idx] = (self.hidden_policy_acts[idx] - max_activation).exp();
            total += self.hidden_policy_acts[idx];
        }
        let total_inv = 1.0 / LIMIT_SMALL.max(total);
        for hc in 0..hidden_size.z as usize {
            self.hidden_policy_acts[hc + hidden_cells_start] *= total_inv;
        }

        // sample action
        let cusp = randf_step(state);
        let mut select_index = 0usize;
        let mut sum_so_far = 0.0f32;
        for hc in 0..hidden_size.z as usize {
            sum_so_far += self.hidden_policy_acts[hc + hidden_cells_start];
            if sum_so_far >= cusp {
                select_index = hc;
                break;
            }
        }
        self.hidden_cis[hidden_column_index] = select_index as i32;
    }

    fn learn_column(&mut self, column_pos: Int2, t: usize, mimic: f32, params: &Params) {
        let hidden_size = self.hidden_size;
        let hidden_column_index = address2(column_pos, Int2::new(hidden_size.x, hidden_size.y));
        let hidden_cells_start = hidden_column_index * hidden_size.z as usize;
        let value_cells_start = hidden_column_index * self.value_size;

        let target_ci = self.history_samples.get(t - 1).hidden_target_cis_prev
            [hidden_column_index] as usize;

        let mut new_value = self.hidden_values[hidden_column_index];

        // TD(lambda)-like return
        for t2 in 1..=t {
            let h_v = self.history_samples.get(t2).hidden_values[hidden_column_index];
            let r = self.history_samples.get(t2 - 1).reward;
            new_value = params.smoothing * h_v
                + (1.0 - params.smoothing) * (r + params.discount * new_value);
        }

        // zero dendrite accumulators
        for vac in 0..self.value_size {
            let value_dendrites_start =
                self.value_num_dendrites_per_cell * (vac + value_cells_start);
            for di in 0..self.value_num_dendrites_per_cell {
                self.value_dendrite_acts[di + value_dendrites_start] = 0.0;
            }
        }
        for hc in 0..hidden_size.z as usize {
            let policy_dendrites_start =
                self.policy_num_dendrites_per_cell * (hc + hidden_cells_start);
            for di in 0..self.policy_num_dendrites_per_cell {
                self.policy_dendrite_acts[di + policy_dendrites_start] = 0.0;
            }
        }

        let mut count = 0usize;

        // NOTE: We need to borrow history_samples immutably for input_cis,
        // but can't do that while self is borrowed mutably.
        // Collect input_cis data first.
        let input_cis_owned: Vec<Vec<i32>> = (0..self.visible_layers.len())
            .map(|vli| self.history_samples.get(t).input_cis[vli].clone())
            .collect();

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

            count += ((iter_upper_bound.x - iter_lower_bound.x + 1)
                * (iter_upper_bound.y - iter_lower_bound.y + 1)) as usize;

            let vl = &self.visible_layers[vli];
            let vl_input_cis = &input_cis_owned[vli];

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let visible_column_index =
                        address2(Int2::new(ix, iy), Int2::new(vld.size.x, vld.size.y));
                    let in_ci = vl_input_cis[visible_column_index] as usize;
                    let offset = Int2::new(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    let wi_start_partial = offset.y as usize
                        + diam as usize
                            * (offset.x as usize
                                + diam as usize
                                    * (in_ci + vld.size.z as usize * hidden_column_index));
                    let wi_value_partial = self.value_size * wi_start_partial;
                    let wi_policy_partial = hidden_size.z as usize * wi_start_partial;

                    for vac in 0..self.value_size {
                        let value_dendrites_start =
                            self.value_num_dendrites_per_cell * (vac + value_cells_start);
                        let wi_start =
                            self.value_num_dendrites_per_cell * (vac + wi_value_partial);
                        for di in 0..self.value_num_dendrites_per_cell {
                            self.value_dendrite_acts[di + value_dendrites_start] +=
                                vl.value_weights[di + wi_start];
                        }
                    }

                    for hc in 0..hidden_size.z as usize {
                        let policy_dendrites_start =
                            self.policy_num_dendrites_per_cell * (hc + hidden_cells_start);
                        let wi_start =
                            self.policy_num_dendrites_per_cell * (hc + wi_policy_partial);
                        for di in 0..self.policy_num_dendrites_per_cell {
                            self.policy_dendrite_acts[di + policy_dendrites_start] +=
                                vl.policy_weights[di + wi_start];
                        }
                    }
                }
            }
        }

        let half_value_num = self.value_num_dendrites_per_cell / 2;
        let half_policy_num = self.policy_num_dendrites_per_cell / 2;
        let dendrite_scale = (1.0f32 / count as f32).sqrt();
        let value_activation_scale = (1.0f32 / self.value_num_dendrites_per_cell as f32).sqrt();
        let policy_activation_scale =
            (1.0f32 / self.policy_num_dendrites_per_cell as f32).sqrt();

        let mut max_value_activation = LIMIT_MIN;

        for vac in 0..self.value_size {
            let value_dendrites_start =
                self.value_num_dendrites_per_cell * (vac + value_cells_start);
            let mut activation = 0.0f32;

            for di in 0..self.value_num_dendrites_per_cell {
                let act = self.value_dendrite_acts[di + value_dendrites_start] * dendrite_scale;
                self.value_dendrite_acts[di + value_dendrites_start] = sigmoidf(act); // store deriv
                let leaky = softplusf(act) - params.leak * softplusf(-act);
                activation += leaky * (if di >= half_value_num { 2.0 } else { 0.0 } - 1.0);
            }

            activation *= value_activation_scale;
            self.hidden_value_acts[vac + value_cells_start] = activation;

            if activation > max_value_activation {
                max_value_activation = activation;
            }
        }

        let mut total = 0.0f32;
        for vac in 0..self.value_size {
            let idx = vac + value_cells_start;
            self.hidden_value_acts[idx] =
                (self.hidden_value_acts[idx] - max_value_activation).exp();
            total += self.hidden_value_acts[idx];
        }
        let total_inv = 1.0 / LIMIT_SMALL.max(total);
        let mut smooth_max_value_index = 0.0f32;
        for vac in 0..self.value_size {
            let idx = vac + value_cells_start;
            self.hidden_value_acts[idx] *= total_inv;
            smooth_max_value_index += self.hidden_value_acts[idx] * vac as f32;
        }

        let value = symexpf(
            (smooth_max_value_index / (self.value_size - 1) as f32 * 2.0 - 1.0) * params.value_range,
        );

        let mut max_activation = LIMIT_MIN;

        for hc in 0..hidden_size.z as usize {
            let policy_dendrites_start =
                self.policy_num_dendrites_per_cell * (hc + hidden_cells_start);
            let mut activation = 0.0f32;

            for di in 0..self.policy_num_dendrites_per_cell {
                let act =
                    self.policy_dendrite_acts[di + policy_dendrites_start] * dendrite_scale;
                self.policy_dendrite_acts[di + policy_dendrites_start] = sigmoidf(act);
                let leaky = softplusf(act) - params.leak * softplusf(-act);
                activation += leaky * (if di >= half_policy_num { 2.0 } else { 0.0 } - 1.0);
            }

            activation *= policy_activation_scale;
            self.hidden_policy_acts[hc + hidden_cells_start] = activation;
            max_activation = max_activation.max(activation);
        }

        let mut total = 0.0f32;
        for hc in 0..hidden_size.z as usize {
            let idx = hc + hidden_cells_start;
            self.hidden_policy_acts[idx] = (self.hidden_policy_acts[idx] - max_activation).exp();
            total += self.hidden_policy_acts[idx];
        }
        let total_inv = 1.0 / LIMIT_SMALL.max(total);
        for hc in 0..hidden_size.z as usize {
            self.hidden_policy_acts[hc + hidden_cells_start] *= total_inv;
        }

        let td_error = new_value - value;
        let scale = &mut self.hidden_td_scales[hidden_column_index];
        *scale = (*scale * params.td_scale_decay).max(td_error.abs());
        let scaled_td_error = td_error / LIMIT_SMALL.max(*scale);

        let policy_error_partial = params.plr * scaled_td_error + mimic;

        let smooth_new_value_index =
            (symlogf(new_value) / params.value_range * 0.5 + 0.5)
                .clamp(0.0, 1.0)
                * (self.value_size - 1) as f32;

        // compute value weight deltas (re-use value_dendrite_acts)
        for vac in 0..self.value_size {
            let value_dendrites_start =
                self.value_num_dendrites_per_cell * (vac + value_cells_start);
            let target = (1.0 - (vac as f32 - smooth_new_value_index).abs()).max(0.0);
            let error = params.vlr * (target - self.hidden_value_acts[vac + value_cells_start]);

            for di in 0..self.value_num_dendrites_per_cell {
                self.value_dendrite_acts[di + value_dendrites_start] *=
                    error * (if di >= half_value_num { 2.0 } else { 0.0 } - 1.0);
            }
        }

        // compute policy weight deltas (re-use policy_dendrite_acts)
        for hc in 0..hidden_size.z as usize {
            let policy_dendrites_start =
                self.policy_num_dendrites_per_cell * (hc + hidden_cells_start);
            let error = policy_error_partial
                * ((hc == target_ci) as i32 as f32
                    - self.hidden_policy_acts[hc + hidden_cells_start]);

            for di in 0..self.policy_num_dendrites_per_cell {
                self.policy_dendrite_acts[di + policy_dendrites_start] *=
                    error * (if di >= half_policy_num { 2.0 } else { 0.0 } - 1.0);
            }
        }

        // apply deltas
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

            let vl_input_cis = &input_cis_owned[vli];
            let vl = &mut self.visible_layers[vli];

            for ix in iter_lower_bound.x..=iter_upper_bound.x {
                for iy in iter_lower_bound.y..=iter_upper_bound.y {
                    let visible_column_index =
                        address2(Int2::new(ix, iy), Int2::new(vld.size.x, vld.size.y));
                    let in_ci = vl_input_cis[visible_column_index] as usize;
                    let offset = Int2::new(ix - field_lower_bound.x, iy - field_lower_bound.y);

                    let wi_start_partial = offset.y as usize
                        + diam as usize
                            * (offset.x as usize
                                + diam as usize
                                    * (in_ci + vld.size.z as usize * hidden_column_index));
                    let wi_value_partial = self.value_size * wi_start_partial;
                    let wi_policy_partial = hidden_size.z as usize * wi_start_partial;

                    for vac in 0..self.value_size {
                        let value_dendrites_start =
                            self.value_num_dendrites_per_cell * (vac + value_cells_start);
                        let wi_start =
                            self.value_num_dendrites_per_cell * (vac + wi_value_partial);
                        for di in 0..self.value_num_dendrites_per_cell {
                            vl.value_weights[di + wi_start] +=
                                self.value_dendrite_acts[di + value_dendrites_start];
                        }
                    }

                    for hc in 0..hidden_size.z as usize {
                        let policy_dendrites_start =
                            self.policy_num_dendrites_per_cell * (hc + hidden_cells_start);
                        let wi_start =
                            self.policy_num_dendrites_per_cell * (hc + wi_policy_partial);
                        for di in 0..self.policy_num_dendrites_per_cell {
                            vl.policy_weights[di + wi_start] +=
                                self.policy_dendrite_acts[di + policy_dendrites_start];
                        }
                    }
                }
            }
        }
    }

    /// Initialise the actor with random weights.
    ///
    /// - `hidden_size` — spatial grid and action vocabulary size.
    /// - `value_size` — number of discrete value bins for the critic head.
    /// - `value_num_dendrites_per_cell` — dendrites per value cell (must be even).
    /// - `policy_num_dendrites_per_cell` — dendrites per policy cell (must be even).
    /// - `history_capacity` — number of past timesteps to keep for replay.
    /// - `visible_layer_descs` — descriptors for the input connections.
    pub fn init_random(
        &mut self,
        hidden_size: Int3,
        value_size: usize,
        value_num_dendrites_per_cell: usize,
        policy_num_dendrites_per_cell: usize,
        history_capacity: usize,
        visible_layer_descs: Vec<VisibleLayerDesc>,
    ) {
        self.visible_layer_descs = visible_layer_descs;
        self.hidden_size = hidden_size;
        self.value_size = value_size;
        self.value_num_dendrites_per_cell = value_num_dendrites_per_cell;
        self.policy_num_dendrites_per_cell = policy_num_dendrites_per_cell;

        let num_hidden_columns = (hidden_size.x * hidden_size.y) as usize;
        let num_hidden_cells = num_hidden_columns * hidden_size.z as usize;
        let num_value_cells = num_hidden_columns * value_size;
        let value_num_dendrites = num_value_cells * value_num_dendrites_per_cell;
        let policy_num_dendrites = num_hidden_cells * policy_num_dendrites_per_cell;

        // Use the shared global RNG (same as encoder/decoder/image_encoder)
        self.visible_layers = self
            .visible_layer_descs
            .iter()
            .map(|vld| {
                let diam = vld.radius * 2 + 1;
                let area = (diam * diam) as usize;

                let value_weights: FloatBuffer = (0..value_num_dendrites * area * vld.size.z as usize)
                    .map(|_| (global_randf() * 2.0 - 1.0) * INIT_WEIGHT_NOISEF)
                    .collect();

                let policy_weights: FloatBuffer = (0..policy_num_dendrites * area * vld.size.z as usize)
                    .map(|_| (global_randf() * 2.0 - 1.0) * INIT_WEIGHT_NOISEF)
                    .collect();

                VisibleLayer {
                    value_weights,
                    policy_weights,
                }
            })
            .collect();

        self.hidden_cis = vec![0i32; num_hidden_columns];
        self.hidden_values = vec![0.0f32; num_hidden_columns];
        self.hidden_td_scales = vec![0.0f32; num_hidden_columns];
        self.value_dendrite_acts = vec![0.0f32; value_num_dendrites];
        self.policy_dendrite_acts = vec![0.0f32; policy_num_dendrites];
        self.hidden_value_acts = vec![0.0f32; num_value_cells];
        self.hidden_policy_acts = vec![0.0f32; num_hidden_cells];

        self.history_size = 0;
        self.history_samples.resize(history_capacity);

        for i in 0..history_capacity {
            let sample = self.history_samples.get_mut(i);
            sample.input_cis = self
                .visible_layer_descs
                .iter()
                .map(|vld| {
                    let num_visible_columns = (vld.size.x * vld.size.y) as usize;
                    vec![0i32; num_visible_columns]
                })
                .collect();
            sample.hidden_target_cis_prev = vec![0i32; num_hidden_columns];
            sample.hidden_values = vec![0.0f32; num_hidden_columns];
            sample.reward = 0.0;
        }
    }

    /// Run one actor step: forward pass, history push, and optional learning.
    ///
    /// - `input_cis` — current observation as CI slices (one per visible layer).
    /// - `hidden_target_cis_prev` — the action taken at the previous timestep
    ///   (used as the learning target).
    /// - `learn_enabled` — if `false`, learning is skipped.
    /// - `reward` — scalar reward received at this timestep.
    /// - `mimic` — non-zero to apply an additional cross-entropy loss towards
    ///   `hidden_target_cis_prev` (imitation learning signal). Pass `0.0` for
    ///   pure RL.
    /// - `params` — hyperparameters for this step.
    pub fn step(
        &mut self,
        input_cis: &[&[i32]],
        hidden_target_cis_prev: &[i32],
        learn_enabled: bool,
        reward: f32,
        mimic: f32,
        params: &Params,
    ) {
        let num_hidden_columns = (self.hidden_size.x * self.hidden_size.y) as usize;
        let base_state = global_rand() as u64;

        // forward
        for i in 0..num_hidden_columns {
            let column_pos = Int2::new(
                (i / self.hidden_size.y as usize) as i32,
                (i % self.hidden_size.y as usize) as i32,
            );
            let mut state = rand_get_state(base_state + i as u64 * RAND_SUBSEED_OFFSET);
            self.forward_column(column_pos, input_cis, &mut state, params);
        }

        self.history_samples.push_front();
        if self.history_size < self.history_samples.len() {
            self.history_size += 1;
        }

        // store new sample
        {
            let s = self.history_samples.get_mut(0);
            for vli in 0..input_cis.len() {
                let src = input_cis[vli];
                s.input_cis[vli].clear();
                s.input_cis[vli].extend_from_slice(src);
            }
            s.hidden_target_cis_prev.clear();
            s.hidden_target_cis_prev.extend_from_slice(hidden_target_cis_prev);
            s.hidden_values.clear();
            s.hidden_values.extend_from_slice(&self.hidden_values);
            s.reward = reward;
        }

        if learn_enabled && self.history_size > params.min_steps {
            for _ in 0..params.history_iters {
                let t = (global_rand() as usize % (self.history_size - params.min_steps))
                    + params.min_steps;

                for i in 0..num_hidden_columns {
                    let column_pos = Int2::new(
                        (i / self.hidden_size.y as usize) as i32,
                        (i % self.hidden_size.y as usize) as i32,
                    );
                    self.learn_column(column_pos, t, mimic, params);
                }
            }
        }
    }

    /// Reset hidden CIs, value estimates, and history to zero.
    pub fn clear_state(&mut self) {
        self.hidden_cis.fill(0);
        self.hidden_values.fill(0.0);
        self.history_size = 0;
    }

    /// Return the current selected action CIs (one per hidden column).
    pub fn get_hidden_cis(&self) -> &[i32] {
        &self.hidden_cis
    }

    /// Return the current policy activation probabilities (one per hidden cell).
    pub fn get_hidden_acts(&self) -> &[f32] {
        &self.hidden_policy_acts
    }

    /// Return the current value estimates (one per hidden column).
    pub fn get_hidden_values(&self) -> &[f32] {
        &self.hidden_values
    }

    /// Return the spatial size of the hidden (action) layer.
    pub fn get_hidden_size(&self) -> Int3 {
        self.hidden_size
    }

    /// Return the maximum number of history samples stored.
    pub fn get_history_capacity(&self) -> usize {
        self.history_samples.len()
    }

    /// Return the number of history samples currently populated.
    pub fn get_history_size(&self) -> usize {
        self.history_size
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

    /// Serialise the full actor (weights + state + history) to a [`StreamWriter`].
    pub fn write(&self, writer: &mut dyn StreamWriter) {
        writer.write_int3(self.hidden_size);
        writer.write_i32(self.value_size as i32);
        writer.write_i32(self.value_num_dendrites_per_cell as i32);
        writer.write_i32(self.policy_num_dendrites_per_cell as i32);
        writer.write_i32_slice(&self.hidden_cis);
        writer.write_f32_slice(&self.hidden_values);
        writer.write_f32_slice(&self.hidden_td_scales);
        writer.write_i32(self.visible_layers.len() as i32);

        for (vl, vld) in self.visible_layers.iter().zip(self.visible_layer_descs.iter()) {
            writer.write_int3(vld.size);
            writer.write_i32(vld.radius);
            writer.write_f32_slice(&vl.value_weights);
            writer.write_f32_slice(&vl.policy_weights);
        }

        writer.write_i32(self.history_size as i32);
        writer.write_i32(self.history_samples.len() as i32);
        writer.write_i32(self.history_samples.start as i32);

        for t in 0..self.history_samples.len() {
            let s = self.history_samples.get(t);
            for vli in 0..self.visible_layers.len() {
                writer.write_i32_slice(&s.input_cis[vli]);
            }
            writer.write_i32_slice(&s.hidden_target_cis_prev);
            writer.write_f32_slice(&s.hidden_values);
            writer.write_f32(s.reward);
        }
    }

    /// Deserialise the actor from a [`StreamReader`].
    pub fn read(&mut self, reader: &mut dyn StreamReader) {
        self.hidden_size = reader.read_int3();
        self.value_size = reader.read_i32() as usize;
        self.value_num_dendrites_per_cell = reader.read_i32() as usize;
        self.policy_num_dendrites_per_cell = reader.read_i32() as usize;

        let num_hidden_columns = (self.hidden_size.x * self.hidden_size.y) as usize;
        let num_hidden_cells = num_hidden_columns * self.hidden_size.z as usize;
        let num_value_cells = num_hidden_columns * self.value_size;
        let value_num_dendrites = num_value_cells * self.value_num_dendrites_per_cell;
        let policy_num_dendrites = num_hidden_cells * self.policy_num_dendrites_per_cell;

        self.hidden_cis = vec![0i32; num_hidden_columns];
        reader.read_i32_slice(&mut self.hidden_cis);

        self.hidden_values = vec![0.0f32; num_hidden_columns];
        reader.read_f32_slice(&mut self.hidden_values);

        self.hidden_td_scales = vec![0.0f32; num_hidden_columns];
        reader.read_f32_slice(&mut self.hidden_td_scales);

        self.value_dendrite_acts = vec![0.0f32; value_num_dendrites];
        self.policy_dendrite_acts = vec![0.0f32; policy_num_dendrites];
        self.hidden_value_acts = vec![0.0f32; num_value_cells];
        self.hidden_policy_acts = vec![0.0f32; num_hidden_cells];

        let num_visible_layers = reader.read_i32() as usize;
        self.visible_layers = Vec::with_capacity(num_visible_layers);
        self.visible_layer_descs = Vec::with_capacity(num_visible_layers);

        for _ in 0..num_visible_layers {
            let size = reader.read_int3();
            let radius = reader.read_i32();
            let vld = VisibleLayerDesc { size, radius };

            let diam = vld.radius * 2 + 1;
            let area = (diam * diam) as usize;

            let mut value_weights = vec![0.0f32; value_num_dendrites * area * vld.size.z as usize];
            reader.read_f32_slice(&mut value_weights);

            let mut policy_weights =
                vec![0.0f32; policy_num_dendrites * area * vld.size.z as usize];
            reader.read_f32_slice(&mut policy_weights);

            self.visible_layers.push(VisibleLayer {
                value_weights,
                policy_weights,
            });
            self.visible_layer_descs.push(vld);
        }

        self.history_size = reader.read_i32() as usize;
        let num_history_samples = reader.read_i32() as usize;
        let history_start = reader.read_i32() as usize;

        self.history_samples.resize(num_history_samples);
        self.history_samples.start = history_start;

        for t in 0..num_history_samples {
            let s = self.history_samples.get_mut(t);
            s.input_cis = Vec::with_capacity(num_visible_layers);
            for vli in 0..num_visible_layers {
                let num_visible_columns =
                    (self.visible_layer_descs[vli].size.x * self.visible_layer_descs[vli].size.y)
                        as usize;
                let mut v = vec![0i32; num_visible_columns];
                reader.read_i32_slice(&mut v);
                s.input_cis.push(v);
            }
            s.hidden_target_cis_prev = vec![0i32; num_hidden_columns];
            reader.read_i32_slice(&mut s.hidden_target_cis_prev);
            s.hidden_values = vec![0.0f32; num_hidden_columns];
            reader.read_f32_slice(&mut s.hidden_values);
            s.reward = reader.read_f32();
        }
    }

    /// Serialise only the actor's transient state (CIs, values, history).
    pub fn write_state(&self, writer: &mut dyn StreamWriter) {
        writer.write_i32_slice(&self.hidden_cis);
        writer.write_f32_slice(&self.hidden_values);
        writer.write_i32(self.history_size as i32);
        writer.write_i32(self.history_samples.start as i32);

        for t in 0..self.history_samples.len() {
            let s = self.history_samples.get(t);
            for vli in 0..self.visible_layers.len() {
                writer.write_i32_slice(&s.input_cis[vli]);
            }
            writer.write_i32_slice(&s.hidden_target_cis_prev);
            writer.write_f32_slice(&s.hidden_values);
            writer.write_f32(s.reward);
        }
    }

    /// Deserialise only the actor's transient state.
    pub fn read_state(&mut self, reader: &mut dyn StreamReader) {
        reader.read_i32_slice(&mut self.hidden_cis);
        reader.read_f32_slice(&mut self.hidden_values);
        self.history_size = reader.read_i32() as usize;
        self.history_samples.start = reader.read_i32() as usize;

        for t in 0..self.history_samples.len() {
            let s = self.history_samples.get_mut(t);
            for vli in 0..s.input_cis.len() {
                reader.read_i32_slice(&mut s.input_cis[vli]);
            }
            reader.read_i32_slice(&mut s.hidden_target_cis_prev);
            reader.read_f32_slice(&mut s.hidden_values);
            s.reward = reader.read_f32();
        }
    }

    /// Serialise only the synaptic weights.
    pub fn write_weights(&self, writer: &mut dyn StreamWriter) {
        for vl in &self.visible_layers {
            writer.write_f32_slice(&vl.value_weights);
            writer.write_f32_slice(&vl.policy_weights);
        }
    }

    /// Deserialise only the synaptic weights.
    pub fn read_weights(&mut self, reader: &mut dyn StreamReader) {
        for vl in &mut self.visible_layers {
            reader.read_f32_slice(&mut vl.value_weights);
            reader.read_f32_slice(&mut vl.policy_weights);
        }
    }
}
