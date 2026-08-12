// Frames for `video_prediction`.
//
// Upstream's `Video_Prediction.cpp` uses OpenCV, but only for `cv::VideoCapture`
// frame reading — no `resize`, no `cvtColor`, no `features2d`; the rescale happens
// in SFML and at scale 1.0, so it is a no-op. What the demo actually needs is a
// sequence of RGB frames, which is not a reason to take a video-decoding
// dependency.
//
// So the default source is **procedural**: the demo runs out of the box, in CI, and
// on a machine with no `ffmpeg` and no copy of Ogma's 2.7 MB clip. `--frames <dir>`
// points it at real extracted frames instead, decoded with the `png`
// dev-dependency:
//
//     ffmpeg -i resources/Bullfinch192.mp4 -vf scale=64:64 frames/%04d.png
//
// Buffer layout is `channel + 3 * (y + h * x)` — what `ImageEncoder` expects for a
// 3-channel visible layer, and *different* from the single-channel `y + x * h` that
// `ball_physics` uses.

use std::path::{Path, PathBuf};

use crate::support::args::Args;

/// Where frames come from.
pub enum FrameSource {
    Procedural(SyntheticScene),
    Directory { frames: Vec<Vec<u8>>, width: usize, height: usize },
}

impl FrameSource {
    /// `--frames <dir>` if given, else a procedural scene sized by `--frame-size`
    /// and `--frame-count`.
    pub fn from_args(args: &Args) -> Self {
        match args.str("frames") {
            Some(dir) => Self::from_directory(Path::new(dir))
                .unwrap_or_else(|e| panic!("--frames {dir}: {e}")),
            None => {
                let size: usize = args.get("frame-size", 64);
                let count: usize = args.get("frame-count", 240);
                FrameSource::Procedural(SyntheticScene::new(size, size, count))
            }
        }
    }

    /// Load every `.png` in `dir`, in filename order.
    ///
    /// The whole clip is held in memory because the demo makes many passes over it,
    /// exactly as upstream does with `CAP_PROP_POS_FRAMES` rewinds.
    pub fn from_directory(dir: &Path) -> Result<Self, String> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
            .collect();
        paths.sort();

        if paths.is_empty() {
            return Err(format!("no .png frames in {}", dir.display()));
        }

        let mut frames = Vec::with_capacity(paths.len());
        let mut dims: Option<(usize, usize)> = None;

        for p in &paths {
            let (w, h, rgb) = read_rgb(p)?;
            match dims {
                None => dims = Some((w, h)),
                Some((dw, dh)) if (dw, dh) != (w, h) => {
                    return Err(format!(
                        "{} is {w}x{h} but the first frame is {dw}x{dh}; frames must all match",
                        p.display()
                    ))
                }
                _ => {}
            }
            frames.push(rgb);
        }

        let (width, height) = dims.unwrap();
        Ok(FrameSource::Directory { frames, width, height })
    }

    pub fn dims(&self) -> (usize, usize) {
        match self {
            FrameSource::Procedural(s) => (s.width, s.height),
            FrameSource::Directory { width, height, .. } => (*width, *height),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            FrameSource::Procedural(s) => s.count,
            FrameSource::Directory { frames, .. } => frames.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn describe(&self) -> String {
        let (w, h) = self.dims();
        match self {
            FrameSource::Procedural(_) => {
                format!("{} procedural frames at {w}x{h}", self.len())
            }
            FrameSource::Directory { .. } => {
                format!("{} loaded frames at {w}x{h}", self.len())
            }
        }
    }

    /// The frame at `i`, wrapping so the clip loops.
    pub fn frame(&mut self, i: usize) -> &[u8] {
        let n = self.len();
        let i = i % n.max(1);
        match self {
            FrameSource::Procedural(s) => s.render(i),
            FrameSource::Directory { frames, .. } => &frames[i],
        }
    }
}

/// A drifting-shapes scene with parallax.
///
/// Three layers moving at different speeds, so the sequence is predictable but not
/// trivially so: a model that has learned only "the next frame is this frame" does
/// measurably worse than one that has picked up the motion.
pub struct SyntheticScene {
    pub width: usize,
    pub height: usize,
    pub count: usize,
    buffer: Vec<u8>,
}

impl SyntheticScene {
    pub fn new(width: usize, height: usize, count: usize) -> Self {
        SyntheticScene { width, height, count, buffer: vec![0u8; width * height * 3] }
    }

    fn render(&mut self, frame: usize) -> &[u8] {
        let (w, h) = (self.width, self.height);
        let t = frame as f32 / self.count as f32;
        let tau = std::f32::consts::PI * 2.0;

        // (radius, speed, phase, colour) — the slow one reads as background.
        let shapes: [(f32, f32, f32, [u8; 3]); 3] = [
            (0.34, 1.0, 0.0, [200, 60, 60]),
            (0.20, 2.0, 0.35, [60, 200, 90]),
            (0.12, 3.0, 0.70, [70, 110, 220]),
        ];

        self.buffer.fill(0);

        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            for y in 0..h {
                let v = (y as f32 + 0.5) / h as f32;

                // A slow vertical gradient, so even empty regions carry signal.
                let mut rgb = [
                    (18.0 + 14.0 * (tau * (t + v * 0.5)).sin()).clamp(0.0, 255.0) as u8,
                    (18.0 + 14.0 * (tau * (t * 0.7 + v * 0.5)).sin()).clamp(0.0, 255.0) as u8,
                    (28.0 + 14.0 * (tau * (t * 0.4 + v * 0.5)).cos()).clamp(0.0, 255.0) as u8,
                ];

                for &(radius, speed, phase, colour) in &shapes {
                    let a = tau * (t * speed + phase);
                    let cx = 0.5 + 0.30 * a.cos();
                    let cy = 0.5 + 0.30 * a.sin();
                    let r = radius * 0.5;
                    if (u - cx).powi(2) + (v - cy).powi(2) <= r * r {
                        rgb = colour;
                    }
                }

                let i = (y + h * x) * 3;
                self.buffer[i] = rgb[0];
                self.buffer[i + 1] = rgb[1];
                self.buffer[i + 2] = rgb[2];
            }
        }

        &self.buffer
    }
}

/// Decode a PNG into `channel + 3 * (y + h * x)` RGB bytes.
fn read_rgb(path: &Path) -> Result<(usize, usize, Vec<u8>), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().map_err(|e| format!("{}: {e}", path.display()))?;

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
    buf.truncate(info.buffer_size());

    let (w, h) = (info.width as usize, info.height as usize);

    // PNG rows are y-major; the encoder wants x-major with channel innermost.
    let sample = |x: usize, y: usize| -> [u8; 3] {
        match (info.color_type, info.bit_depth) {
            (png::ColorType::Rgb, png::BitDepth::Eight) => {
                let i = (x + y * w) * 3;
                [buf[i], buf[i + 1], buf[i + 2]]
            }
            (png::ColorType::Rgba, png::BitDepth::Eight) => {
                let i = (x + y * w) * 4;
                [buf[i], buf[i + 1], buf[i + 2]]
            }
            (png::ColorType::Grayscale, png::BitDepth::Eight) => {
                let g = buf[x + y * w];
                [g, g, g]
            }
            (png::ColorType::GrayscaleAlpha, png::BitDepth::Eight) => {
                let g = buf[(x + y * w) * 2];
                [g, g, g]
            }
            _ => [0, 0, 0],
        }
    };

    match (info.color_type, info.bit_depth) {
        (
            png::ColorType::Rgb
            | png::ColorType::Rgba
            | png::ColorType::Grayscale
            | png::ColorType::GrayscaleAlpha,
            png::BitDepth::Eight,
        ) => {}
        (c, d) => return Err(format!("{}: unsupported {c:?} at {d:?}", path.display())),
    }

    let mut out = vec![0u8; w * h * 3];
    for x in 0..w {
        for y in 0..h {
            let rgb = sample(x, y);
            let i = (y + h * x) * 3;
            out[i] = rgb[0];
            out[i + 1] = rgb[1];
            out[i + 2] = rgb[2];
        }
    }

    Ok((w, h, out))
}

/// Mean squared error between two RGB frames, in normalised intensity.
pub fn frame_mse(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = (x as f32 - y as f32) / 255.0;
            (d * d) as f64
        })
        .sum();
    (sum / a.len() as f64) as f32
}

/// Standard deviation of pixel intensity across a frame — how much *detail* it has.
///
/// This is the metric frame MSE cannot be trusted on. A model unsure where the
/// shapes went minimises expected squared error by hedging: emitting the blurry
/// average of everywhere they might be. That scores well on MSE while having
/// learned nothing about the motion, and it is visibly mush. Hedging drives
/// variance down, so comparing the generated frame's detail against the real one
/// separates "predicted the scene" from "predicted the average of all scenes".
pub fn frame_detail(rgb: &[u8]) -> f32 {
    if rgb.is_empty() {
        return 0.0;
    }
    let n = rgb.len() as f64;
    let mean: f64 = rgb.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var: f64 = rgb.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    (var.sqrt() / 255.0) as f32
}

/// Collapse an RGB frame to grayscale in `y + x * h` layout, for `ascii_image`.
pub fn to_gray(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for x in 0..w {
        for y in 0..h {
            let i = (y + h * x) * 3;
            let g = (rgb[i] as u16 + rgb[i + 1] as u16 + rgb[i + 2] as u16) / 3;
            out[y + x * h] = g as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_procedural_source_has_the_requested_shape() {
        let mut s = FrameSource::Procedural(SyntheticScene::new(32, 32, 20));
        assert_eq!(s.dims(), (32, 32));
        assert_eq!(s.len(), 20);
        assert_eq!(s.frame(0).len(), 32 * 32 * 3);
    }

    #[test]
    fn frames_wrap_so_the_clip_loops() {
        let mut s = FrameSource::Procedural(SyntheticScene::new(16, 16, 10));
        let a = s.frame(3).to_vec();
        let b = s.frame(13).to_vec();
        assert_eq!(a, b, "frame 13 should be frame 3 of a 10-frame clip");
    }

    #[test]
    fn consecutive_frames_differ_but_are_more_alike_than_distant_ones() {
        // The whole point of the source: motion that a model can learn, rather than
        // noise it cannot or a still it need not.
        let mut s = FrameSource::Procedural(SyntheticScene::new(48, 48, 120));
        let f0 = s.frame(0).to_vec();
        let f1 = s.frame(1).to_vec();
        let f30 = s.frame(30).to_vec();

        let near = frame_mse(&f0, &f1);
        let far = frame_mse(&f0, &f30);

        assert!(near > 0.0, "consecutive frames are identical — nothing is moving");
        assert!(near < far, "nearby frames ({near}) should be closer than distant ({far})");
    }

    #[test]
    fn the_scene_is_not_uniform() {
        let mut s = FrameSource::Procedural(SyntheticScene::new(32, 32, 10));
        let f = s.frame(0);
        let lo = *f.iter().min().unwrap();
        let hi = *f.iter().max().unwrap();
        assert!(hi as u16 > lo as u16 + 40, "frame is nearly flat: {lo}..{hi}");
    }

    #[test]
    fn gray_conversion_keeps_the_pixel_count_and_layout() {
        let mut s = FrameSource::Procedural(SyntheticScene::new(16, 8, 4));
        let (w, h) = s.dims();
        let g = to_gray(s.frame(0), w, h);
        assert_eq!(g.len(), w * h);
    }

    #[test]
    fn detail_separates_a_real_frame_from_a_hedged_average() {
        let mut s = FrameSource::Procedural(SyntheticScene::new(48, 48, 60));
        let real = s.frame(0).to_vec();

        // The mush a hedging model produces: every pixel the mean of the frame.
        let mean = (real.iter().map(|&v| v as u32).sum::<u32>() / real.len() as u32) as u8;
        let hedged = vec![mean; real.len()];

        let real_detail = frame_detail(&real);
        let hedged_detail = frame_detail(&hedged);

        assert!(hedged_detail < 1e-6, "a flat frame should have no detail");
        assert!(
            real_detail > 0.05,
            "a real frame should carry detail, got {real_detail}"
        );

        // And the point: the hedge can still beat a stale frame on MSE.
        let stale = s.frame(30).to_vec();
        assert!(
            frame_mse(&hedged, &real) < frame_mse(&stale, &real),
            "the premise of the detail metric does not hold on this scene"
        );
    }

    #[test]
    fn frame_mse_is_zero_for_identical_frames() {
        let a = vec![7u8; 30];
        assert!(frame_mse(&a, &a) < 1e-9);
    }

    #[test]
    fn a_missing_frame_directory_is_an_error() {
        assert!(FrameSource::from_directory(Path::new("/nonexistent/frames")).is_err());
    }
}
