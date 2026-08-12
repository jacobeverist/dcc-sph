// LunarLander gymnasium example using AOgmaNeo Sparse Predictive Hierarchies (Rust port)
//
// Port of python_ref/lunarlander_env_runner.py
//
// Demonstrates SPH actor-critic RL on LunarLander-v3 using Python's gymnasium
// library via PyO3.  Requires Python with `pip install gymnasium` installed.
// Note: LunarLander also needs Box2D: `pip install gymnasium[box2d]`
//
// Run with:
//   cargo run --release --example lunarlander --features gymnasium-examples

use dcc_sph::helpers::Int3;
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Add the venv's site-packages to sys.path so gymnasium is importable when
/// running via `cargo run` without manually activating the venv first.
/// Checks $VIRTUAL_ENV (set by `source .venv/bin/activate`) then falls back
/// to a `.venv` directory in the current working directory.
fn activate_venv(py: Python<'_>) -> PyResult<()> {
    let venv = std::env::var("VIRTUAL_ENV").ok().or_else(|| {
        let p = std::path::Path::new(".venv");
        if p.is_dir() {
            Some(".venv".to_string())
        } else {
            None
        }
    });
    if let Some(venv_path) = venv {
        let v = py.version_info();
        let site_pkgs = format!(
            "{}/lib/python{}.{}/site-packages",
            venv_path, v.major, v.minor
        );
        // pyo3 0.23+ takes the source as a &CStr rather than a &str.
        let code = std::ffi::CString::new(format!(
            "import sys; sys.path.insert(0, r'{site_pkgs}') if r'{site_pkgs}' not in sys.path else None"
        ))
        .expect("generated source contains no interior NUL");
        py.run(code.as_c_str(), None, None)?;
    }
    Ok(())
}

/// Sigmoid squashing function: maps any real to (0, 1).
fn sigmoid(x: f32) -> f32 {
    (x * 0.5).tanh() * 0.5 + 0.5
}

/// Squash an observation value and bin it into [0, num_bins-1].
/// Uses sensitivity=2.0, matching EnvRunner's default inf_sensitivity.
fn encode(val: f32, num_bins: i32) -> i32 {
    (sigmoid(val * 2.0) * (num_bins - 1) as f32 + 0.5) as i32
}

fn main() -> PyResult<()> {
    let obs_bins: i32 = 16; // bins per observation column
    let num_obs: usize = 8; // LunarLander has 8 observation floats
    let num_actions: i32 = 4; // 4 discrete actions
    let num_episodes = 5000;
    let max_steps = 1000;

    // --- Hierarchy ---

    // IO 0: 8 obs floats → (1, 8, 16) = 8 columns in a 1×8 grid, input-only
    // IO 1: action → (1, 1, 4), RL actor
    let io_descs = vec![
        IoDesc {
            size: Int3::new(1, num_obs as i32, obs_bins),
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

    // Two layers with exponential temporal memory
    let layer_descs = vec![
        LayerDesc {
            hidden_size: Int3::new(5, 5, 64),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: -1,
            down_radius: 2,
            ticks_per_update: 1,
        },
        LayerDesc {
            hidden_size: Int3::new(5, 5, 64),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: -1,
            down_radius: 2,
            ticks_per_update: 2,
        },
    ];

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    let mut obs_cis = vec![0i32; num_obs]; // 1×8 = 8 columns
    let mut act_cis = vec![0i32; 1]; // 1×1 = 1 column

    // 100-episode EMA average reward (alpha=0.01)
    let mut average_reward = 0.0f64;
    let mut initialized = false;

    println!("LunarLander-v3 with AOgmaNeo SPH ({num_episodes} episodes)");
    println!("Requires: pip install 'gymnasium[box2d]'");
    println!("{}", "-".repeat(65));

    Python::attach(|py| -> PyResult<()> {
        activate_venv(py)?;

        let gym = py.import("gymnasium")?;
        // let env = gym.call_method1("make", ("LunarLander-v3",))?;

        let kwargs = PyDict::new(py);
        // kwargs.set_item("render_mode", "human")?;
        kwargs.set_item("render_mode", "rgb_array")?;
        let env = gym.call_method("make", ("LunarLander-v3",), Some(&kwargs))?;

        // 3. Define positional arguments (as a tuple)
        // let args = ("Hello", "World");

        // 4. Call the method with both positional and keyword arguments
        // In Python, this is equivalent to: print("Hello", "World", sep="-", end="!\n")

        // env.call_method1("", (action,))?;

        for episode in 0..num_episodes {
            // Reset environment (discard info dict)
            env.call_method0("reset")?;

            let mut total_reward = 0.0f64;
            let mut steps = 0usize;
            let mut reward = 0.0f64;

            for _ in 0..max_steps {
                // Feed back previous action prediction
                act_cis[0] = h.get_prediction_cis(1).first().copied().unwrap_or(0);

                // Step gymnasium with the current action
                let action = act_cis[0];
                let step_result = env.call_method1("step", (action,))?;
                let (obs_arr, r, term, trunc, _): (Vec<f64>, f64, bool, bool, Py<PyAny>) =
                    step_result.extract()?;

                // Encode current observation
                for i in 0..num_obs {
                    obs_cis[i] = encode(obs_arr[i] as f32, obs_bins);
                }

                h.step(&[&obs_cis, &act_cis], true, reward as f32, 0.0);

                reward = r;
                total_reward += r;
                steps += 1;

                if term || trunc {
                    break;
                }
            }

            // 100-episode exponential moving average (alpha=0.01)
            if !initialized {
                average_reward = total_reward;
                initialized = true;
            } else {
                average_reward = 0.99 * average_reward + 0.01 * total_reward;
            }

            println!(
                "Episode {:>5} | steps: {:>4} | reward: {:>8.2} | avg(100): {:>8.2}",
                episode + 1,
                steps,
                total_reward,
                average_reward
            );
        }

        env.call_method0("close")?;
        Ok(())
    })?;

    println!("Done.");
    Ok(())
}
