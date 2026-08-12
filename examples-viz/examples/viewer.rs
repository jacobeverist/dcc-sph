// A window onto three of the demos, for when a number is not convincing enough.
//
// This is the whole of the `macroquad-demos` feature. It trains live and draws
// what is happening, the way the upstream SFML demos do — it is a way to *look* at
// a demo, not instrumentation, and it reports nothing the headless demos do not.
// Serious visualisation belongs in dcc-dashboard.
//
// The hierarchy configurations come from the same `build_hierarchy` functions the
// headless demos use, so the two cannot drift apart.
//
//   cargo run --release --example viewer --features macroquad-demos
//   cargo run --release --example viewer --features macroquad-demos -- --demo cat_mouse
//
// `--demo` takes `ball_physics`, `cat_mouse` or `car_racing`. Space toggles
// fast-forward (many simulation steps per frame, drawn once); Escape quits.
//
// The other six demos are text-only by design: a scrolling plot, an ASCII frame or
// a scatter says everything a window would, and they already print it.

use macroquad::prelude::*;

#[path = "../../examples/support/mod.rs"]
mod support;

use support::args::Args;
use support::encode::bin_unit;
use support::env::ball::{BallWorld, EPISODE_FRAMES, FRAME_H, FRAME_W};
use support::env::catmouse::{
    build_hierarchy as build_catmouse, CatMouseEnv, Map, ACTION_RES, ACTION_SIZE, OBS_RES, OBS_SIZE,
};
use support::env::racing::{
    build_hierarchy as build_racing, random_steer, Racing, Track, SENSOR_GRID, SENSOR_RES,
    STEER_RES,
};
use support::rng::seed_everything;
use support::viz::{blit_gray, hud, plot_series, scatter, View};

use dcc_sph::helpers::{Int3, VisibleLayerDesc};
use dcc_sph::hierarchy::{Hierarchy, IoDesc, IoType, LayerDesc};
use dcc_sph::image_encoder::ImageEncoder;

fn window_conf() -> Conf {
    Conf {
        window_title: "dcc_sph demos".to_owned(),
        window_width: 1000,
        window_height: 760,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let args = Args::parse();
    let demo = args.str("demo").unwrap_or("car_racing").to_string();
    let seed: u64 = args.get("seed", 12345);

    match demo.as_str() {
        "ball_physics" => run_ball(seed).await,
        "cat_mouse" => run_cat_mouse(&args, seed).await,
        "car_racing" => run_car_racing(&args, seed).await,
        other => {
            eprintln!("Unknown --demo {other:?}; expected ball_physics, cat_mouse or car_racing.");
            std::process::exit(1);
        }
    }
}

/// Steps to run per drawn frame while fast-forwarding.
const FAST_STEPS: usize = 200;

fn fast_forward() -> bool {
    is_key_down(KeyCode::Space)
}

// --- Ball physics ---

async fn run_ball(seed: u64) {
    let mut rng = seed_everything(seed);

    let enc_hidden = Int3::new(20, 20, 16);
    let mut enc = ImageEncoder::default();
    enc.init_random(
        enc_hidden,
        vec![VisibleLayerDesc { size: Int3::new(FRAME_W as i32, FRAME_H as i32, 1), radius: 6 }],
    );

    let io_descs = vec![IoDesc {
        size: enc_hidden,
        io_type: IoType::Prediction,
        up_radius: 4,
        ..Default::default()
    }];
    let layer_descs: Vec<LayerDesc> = (0..2)
        .map(|_| LayerDesc {
            hidden_size: Int3::new(10, 10, 32),
            num_dendrites_per_cell: 4,
            up_radius: 2,
            recurrent_radius: 0,
            down_radius: 2,
            ticks_per_update: 1,
        })
        .collect();

    let mut h = Hierarchy::new();
    h.init_random(&io_descs, &layer_descs);

    let mut world = BallWorld::new();
    world.reset(&mut rng);
    let mut frame = vec![0u8; FRAME_W * FRAME_H];
    let mut predicted = vec![0u8; FRAME_W * FRAME_H];
    let mut episodes_trained = 0u32;
    // Close the loop once there is something worth watching.
    let mut generating = false;

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        if is_key_pressed(KeyCode::G) {
            generating = !generating;
        }

        let steps = if fast_forward() { FAST_STEPS } else { 1 };
        for _ in 0..steps {
            world.rasterise(&mut frame);

            let closed = generating && world.frame > 5;
            if closed {
                let fed = h.get_prediction_cis(0).to_vec();
                h.step(&[&fed], false, 0.0, 0.0);
            } else {
                enc.step(&[&frame], true, true);
                let hidden = enc.get_hidden_cis().to_vec();
                h.step(&[&hidden], true, 0.0, 0.0);
            }

            enc.reconstruct(h.get_prediction_cis(0));
            predicted.copy_from_slice(enc.get_reconstruction(0));

            world.step();
            if world.frame >= EPISODE_FRAMES {
                world.reset(&mut rng);
                h.clear_state();
                episodes_trained += 1;
            }
        }

        clear_background(BLACK);
        let size = (screen_height() - 140.0).min(screen_width() * 0.45);
        blit_gray(&frame, FRAME_W, FRAME_H, 30.0, 110.0, size);
        blit_gray(&predicted, FRAME_W, FRAME_H, size + 70.0, 110.0, size);
        draw_text("actual", 30.0, 100.0, 22.0, LIGHTGRAY);
        draw_text("predicted", size + 70.0, 100.0, 22.0, LIGHTGRAY);

        hud(&[
            format!(
                "Ball Physics — episode {episodes_trained}, frame {}/{EPISODE_FRAMES}",
                world.frame
            ),
            format!(
                "G: {} | Space: fast-forward | Esc: quit",
                if generating { "generating from its own predictions" } else { "training" }
            ),
        ]);

        next_frame().await;
    }
}

// --- Cat and mouse ---

async fn run_cat_mouse(args: &Args, seed: u64) {
    let cells: usize = args.get("cells", 5);
    let braid: f32 = args.get("braid", 0.15);
    let timeout: usize = args.get("timeout", 600);

    let mut rng = seed_everything(seed);
    let map = Map::generate(cells, cells, braid, &mut rng);
    let (map_w, map_h) = (map.w as f32, map.h as f32);

    // Walls, as world-space points, built once.
    let mut walls: Vec<(f32, f32)> = Vec::new();
    for x in 0..map.w {
        for y in 0..map.h {
            if map.is_solid(x as i32, y as i32) {
                walls.push((x as f32 + 0.5, y as f32 + 0.5));
            }
        }
    }

    let mut env = CatMouseEnv::new(map, &mut rng);
    let mut cat_h = build_catmouse();
    let mut mouse_h = build_catmouse();

    let mut cat_obs_cis = vec![0i32; OBS_SIZE];
    let mut mouse_obs_cis = vec![0i32; OBS_SIZE];
    let mut cat_action_cis = vec![0i32; ACTION_SIZE];
    let mut mouse_action_cis = vec![0i32; ACTION_SIZE];
    let mut cat_actions = vec![0.5f32; ACTION_SIZE];
    let mut mouse_actions = vec![0.5f32; ACTION_SIZE];
    let mut cat_reward = 0.0f32;
    let mut mouse_reward = 0.0f32;

    let mut captures = 0u64;
    let mut episode_steps = 0usize;
    let mut separation_trace: Vec<f32> = Vec::new();

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        let steps = if fast_forward() { FAST_STEPS } else { 1 };
        for _ in 0..steps {
            let (cat_obs, mouse_obs) = env.observations();
            for i in 0..OBS_SIZE {
                cat_obs_cis[i] = bin_unit(cat_obs[i], OBS_RES);
                mouse_obs_cis[i] = bin_unit(mouse_obs[i], OBS_RES);
            }

            cat_h.step(&[&cat_obs_cis, &cat_action_cis], true, cat_reward, 0.0);
            mouse_h.step(&[&mouse_obs_cis, &mouse_action_cis], true, mouse_reward, 0.0);
            cat_reward = 0.0;
            mouse_reward = 0.0;

            for i in 0..ACTION_SIZE {
                cat_action_cis[i] = cat_h.get_prediction_cis(1)[i];
                mouse_action_cis[i] = mouse_h.get_prediction_cis(1)[i];
                cat_actions[i] = cat_action_cis[i] as f32 / (ACTION_RES - 1) as f32;
                mouse_actions[i] = mouse_action_cis[i] as f32 / (ACTION_RES - 1) as f32;
            }

            for _ in 0..4 {
                env.step(&cat_actions, &mouse_actions, 1.0 / 120.0);
                if env.done() {
                    cat_reward += 100.0;
                    mouse_reward -= 100.0;
                    captures += 1;
                    episode_steps = 0;
                    env.reset(&mut rng);
                }
            }

            episode_steps += 1;
            if timeout > 0 && episode_steps >= timeout {
                episode_steps = 0;
                env.reset(&mut rng);
            }
        }

        separation_trace.push(env.distance);
        if separation_trace.len() > 240 {
            separation_trace.remove(0);
        }

        clear_background(BLACK);
        let view = View::fit(map_w, map_h, false);
        let cell = view.scale;
        scatter(&walls, view, cell * 0.5, Color::from_rgba(50, 50, 60, 255));
        scatter(&[env.cat.pos], view, cell * 0.4, MAGENTA);
        scatter(&[env.mouse.pos], view, cell * 0.4, SKYBLUE);

        plot_series(
            &separation_trace,
            screen_width() - 260.0,
            screen_height() - 110.0,
            240.0,
            90.0,
            GREEN,
        );

        hud(&[
            format!("Cat and Mouse — {captures} captures, separation {:.1}", env.distance),
            "magenta: cat | blue: mouse | green: separation | Space: fast-forward".to_string(),
        ]);

        next_frame().await;
    }
}

// --- Car racing ---

async fn run_car_racing(args: &Args, seed: u64) {
    let assets: String = args.get(
        "assets",
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .to_string_lossy()
            .into_owned(),
    );
    let exploration: f32 = args.get("exploration", 0.02);

    let mut rng = seed_everything(seed);

    let track = match Track::load(std::path::Path::new(&assets)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Could not load the track from {assets}: {e}");
            std::process::exit(1);
        }
    };

    // The collision mask, as a texture, so the track is visible without shipping
    // upstream's 590 KB of artwork.
    let (tw, th) = (track.w, track.h);
    let mut mask = Image::gen_image_color(tw as u16, th as u16, BLACK);
    for x in 0..tw {
        for y in 0..th {
            if track.is_wall(x as f32 + 0.5, y as f32 + 0.5) {
                mask.set_pixel(x as u32, y as u32, Color::from_rgba(40, 40, 50, 255));
            }
        }
    }
    let mask_texture = Texture2D::from_image(&mask);
    mask_texture.set_filter(FilterMode::Nearest);

    let checkpoints: Vec<(f32, f32)> = track.checkpoints.clone();
    let mut env = Racing::new(track);
    let mut h = build_racing();

    let mut sensor_cis = vec![0i32; SENSOR_GRID * SENSOR_GRID];
    let mut action_cis = vec![0i32; 1];
    let mut crashes = 0u64;
    let mut frames = 0u64;
    let mut reward_trace: Vec<f32> = Vec::new();

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        let steps = if fast_forward() { FAST_STEPS } else { 1 };
        let mut last_reward = 0.0;
        for _ in 0..steps {
            let mut steer = if frames < 10 { STEER_RES / 2 } else { h.get_prediction_cis(1)[0] };
            if exploration > 0.0 && rng.chance(exploration) {
                steer = random_steer(&mut rng);
            }

            let (reward, crashed) = env.step(steer);
            if crashed {
                crashes += 1;
            }
            last_reward = reward;

            env.sensor_cis(SENSOR_RES, &mut sensor_cis);
            action_cis[0] = steer;
            h.step(&[&sensor_cis, &action_cis], true, reward, 0.0);
            frames += 1;
        }

        reward_trace.push(last_reward);
        if reward_trace.len() > 240 {
            reward_trace.remove(0);
        }

        clear_background(BLACK);
        let view = View::fit(tw as f32, th as f32, false);
        let (ox, oy) = view.to_screen(0.0, 0.0);
        draw_texture_ex(
            &mask_texture,
            ox,
            oy,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(tw as f32 * view.scale, th as f32 * view.scale)),
                ..Default::default()
            },
        );

        scatter(&checkpoints, view, 2.0, Color::from_rgba(0, 120, 0, 255));

        // The car, and its whiskers.
        let (cx, cy) = view.to_screen(env.car.position.0, env.car.position.1);
        for (s, &reading) in env.sensors.iter().enumerate() {
            let a = 0.16 * (s as f32 - env.sensors.len() as f32 * 0.5) + env.car.rotation;
            let len = reading * 70.0 * view.scale;
            draw_line(cx, cy, cx + a.cos() * len, cy + a.sin() * len, 1.0, DARKGREEN);
        }
        draw_circle(cx, cy, 4.0, RED);

        plot_series(
            &reward_trace,
            screen_width() - 260.0,
            screen_height() - 110.0,
            240.0,
            90.0,
            ORANGE,
        );

        hud(&[
            format!(
                "Car Racing — {} laps, {crashes} crashes, {frames} frames",
                env.laps
            ),
            "green: whiskers | orange: reward | Space: fast-forward | Esc: quit".to_string(),
        ]);

        next_frame().await;
    }
}
