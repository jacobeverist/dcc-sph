// A deliberately small drawing layer for the windowed demo viewer.
//
// This exists so a demo can be *eyeballed* — "is the car actually going round the
// track?" — and nothing more. Serious visualisation and instrumentation are
// dcc-dashboard's job, and if anything here starts growing state, options or
// layout logic of its own, that is the signal it belongs there instead.
//
// Five functions. Keep it that way.

use macroquad::prelude::*;

/// World-to-screen transform: `scale` pixels per world unit, centred on `(cx, cy)`.
#[derive(Clone, Copy)]
pub struct View {
    pub cx: f32,
    pub cy: f32,
    pub scale: f32,
    /// Set when world y increases upward and screen y increases downward.
    pub flip_y: bool,
}

impl View {
    /// A view that fits a `w` x `h` world region into the current window.
    pub fn fit(w: f32, h: f32, flip_y: bool) -> View {
        let scale = (screen_width() / w).min(screen_height() / h);
        View { cx: w * 0.5, cy: h * 0.5, scale, flip_y }
    }

    pub fn to_screen(self, x: f32, y: f32) -> (f32, f32) {
        let dy = if self.flip_y { self.cy - y } else { y - self.cy };
        (
            screen_width() * 0.5 + (x - self.cx) * self.scale,
            screen_height() * 0.5 + dy * self.scale,
        )
    }
}

/// Draw a grayscale buffer indexed `y + x * h`, scaled to fit `size` pixels square.
pub fn blit_gray(pixels: &[u8], w: usize, h: usize, x: f32, y: f32, size: f32) {
    let mut image = Image::gen_image_color(w as u16, h as u16, BLACK);
    for px in 0..w {
        for py in 0..h {
            let v = pixels[py + px * h];
            image.set_pixel(
                px as u32,
                py as u32,
                Color::from_rgba(v, v, v, 255),
            );
        }
    }
    let texture = Texture2D::from_image(&image);
    texture.set_filter(FilterMode::Nearest);
    draw_texture_ex(
        &texture,
        x,
        y,
        WHITE,
        DrawTextureParams { dest_size: Some(vec2(size, size)), ..Default::default() },
    );
}

/// Draw a series as a polyline inside the given rectangle, auto-scaled to its own range.
pub fn plot_series(values: &[f32], x: f32, y: f32, w: f32, h: f32, color: Color) {
    if values.len() < 2 {
        return;
    }
    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (hi - lo).max(1e-6);

    draw_rectangle_lines(x, y, w, h, 1.0, DARKGRAY);
    for i in 1..values.len() {
        let x0 = x + (i - 1) as f32 / (values.len() - 1) as f32 * w;
        let x1 = x + i as f32 / (values.len() - 1) as f32 * w;
        let y0 = y + h - (values[i - 1] - lo) / span * h;
        let y1 = y + h - (values[i] - lo) / span * h;
        draw_line(x0, y0, x1, y1, 1.5, color);
    }
}

/// Draw points in world space.
pub fn scatter(points: &[(f32, f32)], view: View, radius: f32, color: Color) {
    for &(x, y) in points {
        let (sx, sy) = view.to_screen(x, y);
        draw_circle(sx, sy, radius, color);
    }
}

/// Draw a stack of status lines in the top-left corner.
pub fn hud(lines: &[String]) {
    for (i, line) in lines.iter().enumerate() {
        draw_text(line, 10.0, 22.0 + i as f32 * 20.0, 20.0, LIGHTGRAY);
    }
}
