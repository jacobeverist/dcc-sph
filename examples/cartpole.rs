// CartPole RL example using AOgmaNeo Sparse Predictive Hierarchies (Rust port)
//
// Demonstrates how to use the Hierarchy with an Actor IO type to learn
// a control policy for the classic CartPole balancing task.
//
// Run with: cargo run --release --example cartpole

use dcc_sph::helpers::Int3;
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};

// --- CartPole environment ---

struct CartPole {
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,
}

impl CartPole {
    const GRAVITY: f32 = 9.8;
    const CART_MASS: f32 = 1.0;
    const POLE_MASS: f32 = 0.1;
    const TOTAL_MASS: f32 = Self::CART_MASS + Self::POLE_MASS;
    const POLE_HALF_LEN: f32 = 0.5;
    const FORCE_MAG: f32 = 10.0;
    const DT: f32 = 0.02;
    const X_LIMIT: f32 = 2.4;
    const THETA_LIMIT: f32 = 12.0 * std::f32::consts::PI / 180.0; // ~0.2094 rad
    const MAX_STEPS: usize = 500;

    fn new() -> Self {
        Self { x: 0.0, x_dot: 0.0, theta: 0.0, theta_dot: 0.0 }
    }

    fn reset(&mut self) {
        self.x = 0.0;
        self.x_dot = 0.0;
        self.theta = 0.0;
        self.theta_dot = 0.0;
    }

    /// Returns true if the episode is still alive.
    fn step(&mut self, action: i32) -> bool {
        let force = if action == 1 { Self::FORCE_MAG } else { -Self::FORCE_MAG };

        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();

        let temp = (force
            + Self::POLE_MASS * Self::POLE_HALF_LEN * self.theta_dot * self.theta_dot * sin_t)
            / Self::TOTAL_MASS;

        let theta_acc = (Self::GRAVITY * sin_t - cos_t * temp)
            / (Self::POLE_HALF_LEN
                * (4.0 / 3.0 - Self::POLE_MASS * cos_t * cos_t / Self::TOTAL_MASS));

        let x_acc =
            temp - Self::POLE_MASS * Self::POLE_HALF_LEN * theta_acc * cos_t / Self::TOTAL_MASS;

        // Euler integration
        self.x += self.x_dot * Self::DT;
        self.x_dot += x_acc * Self::DT;
        self.theta += self.theta_dot * Self::DT;
        self.theta_dot += theta_acc * Self::DT;

        self.x.abs() <= Self::X_LIMIT && self.theta.abs() <= Self::THETA_LIMIT
    }
}

// --- State encoding ---

/// Discretize a continuous value into a bin index in [0, num_bins-1].
fn encode_bin(value: f32, low: f32, high: f32, num_bins: i32) -> i32 {
    let normalized = ((value - low) / (high - low)).clamp(0.0, 1.0);
    (normalized * (num_bins - 1) as f32 + 0.5) as i32
}

fn main() {
    let num_bins: i32 = 16;
    let num_actions: i32 = 2;
    let num_episodes = 500;

    // --- Set up hierarchy ---

    // IO 0: observation (2×2 = 4 columns, 16 bins each) — input only, no prediction
    // IO 1: action     (1×1 = 1 column,  2 possible actions) — RL action output
    let io_descs = vec![
        IoDesc {
            size: Int3::new(2, 2, num_bins),
            io_type: IoType::None,
            num_dendrites_per_cell: 4,
            up_radius: 2,
            down_radius: 2,
            value_size: 64,
            value_num_dendrites_per_cell: 4,
            history_capacity: 256,
        },
        IoDesc {
            size: Int3::new(1, 1, num_actions),
            io_type: IoType::Action,
            num_dendrites_per_cell: 4,
            up_radius: 2,
            down_radius: 2,
            value_size: 64,
            value_num_dendrites_per_cell: 4,
            history_capacity: 256,
        },
    ];

    // Two hidden layers with recurrence
    let layer_descs = vec![
        LayerDesc {
            hidden_size: Int3::new(4, 4, num_bins),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: 2,
            down_radius: 2,
            ticks_per_update: 1,
        },
        LayerDesc {
            hidden_size: Int3::new(4, 4, num_bins),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: 2,
            down_radius: 2,
            ticks_per_update: 2,
        },
    ];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    // Tune RL parameters
    h.params.ios[1].actor.discount = 0.99;
    h.params.ios[1].actor.plr = 0.01;
    h.params.ios[1].actor.vlr = 0.1;

    // --- Training loop ---

    let mut env = CartPole::new();

    // Encoding ranges for each state variable
    let ranges = [
        (-CartPole::X_LIMIT,     CartPole::X_LIMIT),     // x
        (-3.0f32,                3.0f32),                 // x_dot
        (-CartPole::THETA_LIMIT, CartPole::THETA_LIMIT),  // theta
        (-3.0f32,                3.0f32),                 // theta_dot
    ];

    let mut obs_cis = vec![0i32; 4]; // 2×2 = 4 columns
    let mut act_cis = vec![0i32; 1]; // 1×1 = 1 column

    println!("Training CartPole with AOgmaNeo SPH ({num_episodes} episodes)...\n");
    println!("{:>7} | {:>5} | {:>12}", "Episode", "Steps", "Avg(last 10)");
    println!("{}", "-".repeat(30));

    let mut avg_window = [0.0f32; 10];
    let mut avg_idx = 0usize;

    for ep in 0..num_episodes {
        env.reset();
        h.clear_state();

        let mut steps = 0usize;

        for _ in 0..CartPole::MAX_STEPS {
            // Encode observation into sparse column indices
            let state = [env.x, env.x_dot, env.theta, env.theta_dot];
            for i in 0..4 {
                obs_cis[i] = encode_bin(state[i], ranges[i].0, ranges[i].1, num_bins);
            }

            // Feed back the previous step's action prediction as the action input
            act_cis[0] = h.get_prediction_cis(1)[0];

            // Step the hierarchy (reward = 1.0 per step alive)
            h.step(&[&obs_cis, &act_cis], true, 1.0, 0.0);

            // Read the chosen action and apply it to the environment
            let chosen_action = h.get_prediction_cis(1)[0];
            let alive = env.step(chosen_action);
            steps += 1;

            if !alive {
                break;
            }
        }

        // Track 10-episode rolling average
        avg_window[avg_idx % 10] = steps as f32;
        avg_idx += 1;

        let count = avg_idx.min(10);
        let avg: f32 = avg_window[..count].iter().sum::<f32>() / count as f32;

        if (ep + 1) % 10 == 0 || ep == 0 {
            println!("{:>7} | {:>5} | {:>12.1}", ep + 1, steps, avg);
        }
    }

    println!("\nDone.");
}
