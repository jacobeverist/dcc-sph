// AOgmaNeo Rust port - helpers module

use std::sync::atomic::{AtomicUsize, Ordering};

// --- Constants ---

/// Practical negative infinity used for activation comparisons.
pub const LIMIT_MIN: f32 = -999999.0;

/// Practical positive infinity used for activation comparisons.
pub const LIMIT_MAX: f32 = 999999.0;

/// Small positive value used to avoid division by zero.
pub const LIMIT_SMALL: f32 = 0.000001;

/// Offset applied when deriving per-column RNG seeds to avoid seed collisions.
pub const RAND_SUBSEED_OFFSET: u64 = 12345;

/// Noise range (integer) for initial weight randomization. Weights are sampled
/// uniformly from `[0, INIT_WEIGHT_NOISEI)` (byte weights) or
/// `[-INIT_WEIGHT_NOISEF, INIT_WEIGHT_NOISEF)` (float weights).
pub const INIT_WEIGHT_NOISEI: u32 = 8;

/// Noise range (float) for initial weight randomization. See [`INIT_WEIGHT_NOISEI`].
pub const INIT_WEIGHT_NOISEF: f32 = 0.01;

/// Input threshold above which `softplusf` switches to linear pass-through to
/// avoid floating-point overflow.
pub const SOFTPLUS_LIMIT: f32 = 4.0;

const PCG_MULTIPLIER: u64 = 6364136223846793005;
const PCG_INCREMENT: u64 = 1442695040888963407;

/// Maximum value returned by [`rand_step`] (exclusive upper bound for modulo
/// operations that convert a raw RNG output to a uniform integer range).
pub const RAND_MAX: u32 = 0x00ffffff;

// --- Type aliases ---

/// Raw byte buffer (`Vec<u8>`). Used for encoder weights (unsigned, `[0,255]`).
pub type ByteBuffer = Vec<u8>;

/// Signed-byte buffer (`Vec<i8>`). Used for decoder / actor weights (`[-127,127]`).
pub type SByteBuffer = Vec<i8>;

/// Unsigned 16-bit integer buffer.
pub type UShortBuffer = Vec<u16>;

/// Signed 16-bit integer buffer.
pub type ShortBuffer = Vec<i16>;

/// Unsigned 32-bit integer buffer.
pub type UIntBuffer = Vec<u32>;

/// Signed 32-bit integer buffer. Used for column-index (`ci`) arrays.
pub type IntBuffer = Vec<i32>;

/// 32-bit float buffer. Used for activation / weight buffers.
pub type FloatBuffer = Vec<f32>;

// --- Vector types ---

/// 2-D integer coordinate used for spatial column positions.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Int2 {
    pub x: i32,
    pub y: i32,
}

impl Int2 {
    /// Construct a new [`Int2`].
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// 3-D integer size descriptor.
///
/// For layers: `x` and `y` are the spatial grid dimensions (number of columns
/// along each axis), and `z` is the column size (number of cells per column,
/// i.e., the vocabulary size of each discrete token).
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Int3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Int3 {
    /// Construct a new [`Int3`].
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// 4-D integer coordinate.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Int4 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
}

impl Int4 {
    /// Construct a new [`Int4`].
    pub fn new(x: i32, y: i32, z: i32, w: i32) -> Self {
        Self { x, y, z, w }
    }
}

/// 2-D float coordinate used for scale factors and projections.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Float2 {
    pub x: f32,
    pub y: f32,
}

impl Float2 {
    /// Construct a new [`Float2`].
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// 3-D float coordinate.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Float3 {
    /// Construct a new [`Float3`].
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// 4-D float coordinate.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Float4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Float4 {
    /// Construct a new [`Float4`].
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
}

// --- Shared descriptor ---

/// Describes one visible (input) connection to a compute layer.
///
/// Used identically by [`Encoder`](crate::encoder::Encoder),
/// [`Decoder`](crate::decoder::Decoder),
/// [`Actor`](crate::actor::Actor), and
/// [`ImageEncoder`](crate::image_encoder::ImageEncoder).
///
/// The receptive field of each hidden column covers a `(2*radius+1) × (2*radius+1)`
/// patch centred on the projected position in the visible layer.
#[derive(Clone, Debug)]
pub struct VisibleLayerDesc {
    /// Spatial size `(x, y, z)` of the visible layer.
    /// `x * y` is the number of columns; `z` is the column size.
    pub size: Int3,
    /// Half-width of the square receptive field (in visible-layer column units).
    pub radius: i32,
}

impl Default for VisibleLayerDesc {
    fn default() -> Self {
        Self {
            size: Int3::new(5, 5, 16),
            radius: 2,
        }
    }
}

// --- Math helpers ---

/// Ceiling integer division: `ceil(x / y)`.
pub fn ceil_divide(x: i32, y: i32) -> i32 {
    (x + y - 1) / y
}

/// Numerically stable sigmoid via `tanh`: `σ(x) = tanh(x)*0.5 + 0.5`.
pub fn sigmoidf(x: f32) -> f32 {
    x.tanh() * 0.5 + 0.5
}

/// Logit (inverse sigmoid): `logit(x) = ln(x / (1 − x))`.
pub fn logitf(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

/// Symmetric log: `sign(x) * ln(|x| + 1)`.
///
/// A bijective mapping ℝ → ℝ that compresses large magnitudes.
/// Inverse of [`symexpf`].
pub fn symlogf(x: f32) -> f32 {
    ((x > 0.0) as i32 as f32 * 2.0 - 1.0) * (x.abs() + 1.0).ln()
}

/// Symmetric exp (inverse of [`symlogf`]): `sign(x) * (exp(|x|) − 1)`.
pub fn symexpf(x: f32) -> f32 {
    ((x > 0.0) as i32 as f32 * 2.0 - 1.0) * (x.abs().exp() - 1.0)
}

/// Numerically stable softplus: `ln(1 + exp(x))`, clamped to linear above
/// [`SOFTPLUS_LIMIT`] to avoid overflow.
pub fn softplusf(x: f32) -> f32 {
    let in_range = if x < SOFTPLUS_LIMIT { 1.0f32 } else { 0.0f32 };
    (1.0 + (x * in_range).exp()).ln() * in_range + x * (1.0 - in_range)
}

/// Ceiling of `x` cast to `i32`.
pub fn ceilf_to_i32(x: f32) -> i32 {
    x.ceil() as i32
}

/// Round `x` to the nearest `i32` (away from zero on 0.5).
pub fn roundf2i(x: f32) -> i32 {
    (x + if x > 0.0 { 1.0 } else { 0.0 } - 0.5) as i32
}

/// Round `x` to the nearest `u8`.
pub fn roundf2b(x: f32) -> u8 {
    (x + 0.5) as u8
}

/// Round `x` to the nearest `i8` (away from zero on 0.5).
pub fn roundf2sb(x: f32) -> i8 {
    (x + if x > 0.0 { 1.0 } else { 0.0 } - 0.5) as i8
}

// --- Bounds checking ---

/// Returns `true` if `pos` is within `[0, upper_bound)` in both axes.
pub fn in_bounds0(pos: Int2, upper_bound: Int2) -> bool {
    pos.x >= 0 && pos.x < upper_bound.x && pos.y >= 0 && pos.y < upper_bound.y
}

/// Returns `true` if `pos` is within `[lower_bound, upper_bound)` in both axes.
pub fn in_bounds(pos: Int2, lower_bound: Int2, upper_bound: Int2) -> bool {
    pos.x >= lower_bound.x
        && pos.x < upper_bound.x
        && pos.y >= lower_bound.y
        && pos.y < upper_bound.y
}

// --- Projections ---

/// Project a hidden-layer column position into a visible-layer column position
/// using the given scale factors.
pub fn project(pos: Int2, to_scalars: Float2) -> Int2 {
    Int2::new(
        ((pos.x as f32 + 0.5) * to_scalars.x) as i32,
        ((pos.y as f32 + 0.5) * to_scalars.y) as i32,
    )
}

/// Project a float position into a visible-layer column position.
pub fn projectf(pos: Float2, to_scalars: Float2) -> Int2 {
    Int2::new(
        ((pos.x + 0.5) * to_scalars.x) as i32,
        ((pos.y + 0.5) * to_scalars.y) as i32,
    )
}

/// Clamp `pos` so that a receptive field of `radii` fits entirely within `size`.
pub fn min_overhang(pos: Int2, size: Int2, radii: Int2) -> Int2 {
    let mut new_pos = pos;

    let overhang_px = new_pos.x + radii.x >= size.x;
    let overhang_nx = new_pos.x - radii.x < 0;
    let overhang_py = new_pos.y + radii.y >= size.y;
    let overhang_ny = new_pos.y - radii.y < 0;

    if overhang_px && !overhang_nx {
        new_pos.x = size.x - 1 - radii.x;
    } else if overhang_nx && !overhang_px {
        new_pos.x = radii.x;
    }

    if overhang_py && !overhang_ny {
        new_pos.y = size.y - 1 - radii.y;
    } else if overhang_ny && !overhang_py {
        new_pos.y = radii.y;
    }

    new_pos
}

/// Like [`min_overhang`] but with a scalar (isotropic) radius.
pub fn min_overhang_scalar(pos: Int2, size: Int2, radius: i32) -> Int2 {
    min_overhang(pos, size, Int2::new(radius, radius))
}

// --- Addressing (row-major) ---

/// Compute a flat row-major index from a 2-D position and dimensions.
pub fn address2(pos: Int2, dims: Int2) -> usize {
    (pos.y + pos.x * dims.y) as usize
}

/// Compute a flat row-major index from a 3-D position and dimensions.
pub fn address3(pos: Int3, dims: Int3) -> usize {
    (pos.z + dims.z * (pos.y + dims.y * pos.x)) as usize
}

/// Compute a flat row-major index from a 4-D position and dimensions.
pub fn address4(pos: Int4, dims: Int4) -> usize {
    (pos.w + dims.w * (pos.z + dims.z * (pos.y + dims.y * pos.x))) as usize
}

// --- PCG32 RNG ---

/// Derive an initial PCG32 RNG state from an arbitrary `seed` integer.
///
/// Use this to create a deterministic per-column RNG: each column passes
/// `base_seed + column_index * RAND_SUBSEED_OFFSET` as the seed.
pub fn rand_get_state(seed: u64) -> u64 {
    let state = seed.wrapping_add(PCG_INCREMENT);
    state.wrapping_mul(PCG_MULTIPLIER).wrapping_add(PCG_INCREMENT)
}

#[inline]
fn rotr32(x: u32, r: u32) -> u32 {
    x >> r | x << (r.wrapping_neg() & 31)
}

/// Advance the PCG32 `state` by one step and return a 24-bit random integer.
pub fn rand_step(state: &mut u64) -> u32 {
    let x = *state;
    let count = (x >> 59) as u32;
    *state = x.wrapping_mul(PCG_MULTIPLIER).wrapping_add(PCG_INCREMENT);
    let x = x ^ (x >> 18);
    rotr32((x >> 27) as u32, count)
}

/// Advance `state` and return a uniform float in `[0, 1)`.
pub fn randf_step(state: &mut u64) -> f32 {
    (rand_step(state) % RAND_MAX) as f32 / RAND_MAX as f32
}

/// Advance `state` and return a uniform float in `[low, high)`.
pub fn randf_range_step(low: f32, high: f32, state: &mut u64) -> f32 {
    low + (high - low) * randf_step(state)
}

/// Advance `state` and return a standard-normal sample via Box-Muller.
pub fn rand_normalf_step(state: &mut u64) -> f32 {
    let u1 = randf_step(state);
    let u2 = randf_step(state);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Stochastically round `x` to an integer: always rounds the integer part
/// toward zero, and rounds the fractional part with probability equal to its
/// magnitude.
pub fn rand_roundf_step(x: f32, state: &mut u64) -> i32 {
    let i = x as i32;
    let abs_rem = (x - i as f32).abs();
    let s = if x > 0.0 { 1i32 } else { -1i32 };
    i + if randf_step(state) < abs_rem { s } else { 0 }
}

// Thread-local PCG32 state shared across all modules.
// This is the canonical global RNG used by global_rand(), global_randf(),
// and weight initialisation routines in all compute modules.
thread_local! {
    static GLOBAL_STATE: std::cell::Cell<u64> = std::cell::Cell::new(rand_get_state(12345));
}

/// Draw a 24-bit random integer from the thread-local global RNG.
pub fn global_rand() -> u32 {
    GLOBAL_STATE.with(|s| {
        let mut state = s.get();
        let r = rand_step(&mut state);
        s.set(state);
        r
    })
}

/// Draw a uniform float in `[0, 1)` from the thread-local global RNG.
pub fn global_randf() -> f32 {
    GLOBAL_STATE.with(|s| {
        let mut state = s.get();
        let r = randf_step(&mut state);
        s.set(state);
        r
    })
}

/// Overwrite the thread-local global RNG state. Mirrors AOgmaNeo's
/// `set_global_state` (C++ `helpers.h`): the default is `rand_get_state(12345)`, so
/// calling `set_global_state(rand_get_state(12345))` returns the RNG to its initial
/// stream. Needed to make routines that draw from the global RNG (notably
/// `init_random`) reproducible across in-process re-runs, and to align the seed with
/// the C++ reference for functional-fidelity testing.
pub fn set_global_state(state: u64) {
    GLOBAL_STATE.with(|s| s.set(state));
}

/// Read the current thread-local global RNG state (AOgmaNeo `get_global_state`).
pub fn get_global_state() -> u64 {
    GLOBAL_STATE.with(|s| s.get())
}

// --- CircleBuffer ---

/// A fixed-capacity ring buffer with O(1) push-to-front.
///
/// Elements are stored in a `Vec` and accessed via a rotating `start` index.
/// This is used by the [`Actor`](crate::actor::Actor) to maintain a history of
/// past samples for temporal-difference learning.
#[derive(Clone, Debug)]
pub struct CircleBuffer<T> {
    /// Raw storage. Elements may be in any rotated order; use [`get`](Self::get)
    /// and [`front`](Self::front) for correct access.
    pub data: Vec<T>,
    /// Index of the logical front element.
    pub start: usize,
}

impl<T: Default + Clone> CircleBuffer<T> {
    /// Create an empty buffer (capacity 0).
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            start: 0,
        }
    }

    /// Resize the buffer to `size` elements, filling new slots with `T::default()`.
    pub fn resize(&mut self, size: usize) {
        self.data.resize(size, T::default());
    }

    /// Rotate the front pointer backward (effectively prepending a slot that
    /// overwrites the oldest element).
    pub fn push_front(&mut self) {
        if self.data.is_empty() {
            return;
        }
        if self.start == 0 {
            self.start = self.data.len() - 1;
        } else {
            self.start -= 1;
        }
    }

    /// Return a reference to the logical front element.
    pub fn front(&self) -> &T {
        &self.data[self.start]
    }

    /// Return a mutable reference to the logical front element.
    pub fn front_mut(&mut self) -> &mut T {
        &mut self.data[self.start]
    }

    /// Return a reference to the logical back element (oldest element).
    pub fn back(&self) -> &T {
        let idx = (self.start + self.data.len() - 1) % self.data.len();
        &self.data[idx]
    }

    /// Return a mutable reference to the logical back element.
    pub fn back_mut(&mut self) -> &mut T {
        let idx = (self.start + self.data.len() - 1) % self.data.len();
        &mut self.data[idx]
    }

    /// Return a reference to the element at logical index `index`
    /// (0 = front / newest).
    pub fn get(&self, index: usize) -> &T {
        &self.data[(self.start + index) % self.data.len()]
    }

    /// Return a mutable reference to the element at logical index `index`.
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        let len = self.data.len();
        &mut self.data[(self.start + index) % len]
    }

    /// Return the capacity (number of slots) of the buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the buffer has zero capacity.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T: Default + Clone> Default for CircleBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

// --- Serialization traits ---

/// Trait for writing primitive values to a binary stream.
///
/// The default method implementations encode all values in little-endian byte
/// order. Implementors only need to provide [`write_bytes`](Self::write_bytes).
pub trait StreamWriter {
    /// Write raw bytes to the stream.
    fn write_bytes(&mut self, data: &[u8]);

    /// Write an `i32` in little-endian byte order.
    fn write_i32(&mut self, v: i32) {
        self.write_bytes(&v.to_le_bytes());
    }

    /// Write a `u32` in little-endian byte order.
    fn write_u32(&mut self, v: u32) {
        self.write_bytes(&v.to_le_bytes());
    }

    /// Write an `f32` in little-endian byte order.
    fn write_f32(&mut self, v: f32) {
        self.write_bytes(&v.to_le_bytes());
    }

    /// Write a single `u8`.
    fn write_u8(&mut self, v: u8) {
        self.write_bytes(&[v]);
    }

    /// Write a single `i8`.
    fn write_i8(&mut self, v: i8) {
        self.write_bytes(&[v as u8]);
    }

    /// Write a slice of `i32` values in little-endian byte order.
    fn write_i32_slice(&mut self, slice: &[i32]) {
        for &v in slice {
            self.write_i32(v);
        }
    }

    /// Write a slice of `f32` values in little-endian byte order.
    fn write_f32_slice(&mut self, slice: &[f32]) {
        for &v in slice {
            self.write_f32(v);
        }
    }

    /// Write a `u8` slice as raw bytes.
    fn write_u8_slice(&mut self, slice: &[u8]) {
        self.write_bytes(slice);
    }

    /// Write an `i8` slice as raw bytes.
    fn write_i8_slice(&mut self, slice: &[i8]) {
        // SAFETY: i8 and u8 have same representation
        let bytes = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len()) };
        self.write_bytes(bytes);
    }

    /// Write an [`Int3`] as three consecutive `i32` values.
    fn write_int3(&mut self, v: Int3) {
        self.write_i32(v.x);
        self.write_i32(v.y);
        self.write_i32(v.z);
    }
}

/// Trait for reading primitive values from a binary stream.
///
/// Symmetric counterpart to [`StreamWriter`]. Implementors only need to
/// provide [`read_bytes`](Self::read_bytes).
pub trait StreamReader {
    /// Read exactly `buf.len()` bytes from the stream.
    fn read_bytes(&mut self, buf: &mut [u8]);

    /// Read an `i32` in little-endian byte order.
    fn read_i32(&mut self) -> i32 {
        let mut buf = [0u8; 4];
        self.read_bytes(&mut buf);
        i32::from_le_bytes(buf)
    }

    /// Read a `u32` in little-endian byte order.
    fn read_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.read_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    /// Read an `f32` in little-endian byte order.
    fn read_f32(&mut self) -> f32 {
        let mut buf = [0u8; 4];
        self.read_bytes(&mut buf);
        f32::from_le_bytes(buf)
    }

    /// Read a single `u8`.
    fn read_u8(&mut self) -> u8 {
        let mut buf = [0u8; 1];
        self.read_bytes(&mut buf);
        buf[0]
    }

    /// Read a single `i8`.
    fn read_i8(&mut self) -> i8 {
        self.read_u8() as i8
    }

    /// Read a slice of `i32` values in little-endian byte order.
    fn read_i32_slice(&mut self, slice: &mut [i32]) {
        for v in slice.iter_mut() {
            *v = self.read_i32();
        }
    }

    /// Read a slice of `f32` values in little-endian byte order.
    fn read_f32_slice(&mut self, slice: &mut [f32]) {
        for v in slice.iter_mut() {
            *v = self.read_f32();
        }
    }

    /// Read a `u8` slice as raw bytes.
    fn read_u8_slice(&mut self, slice: &mut [u8]) {
        self.read_bytes(slice);
    }

    /// Read an `i8` slice as raw bytes.
    fn read_i8_slice(&mut self, slice: &mut [i8]) {
        // SAFETY: i8 and u8 have same representation
        let bytes = unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u8, slice.len()) };
        self.read_bytes(bytes);
    }

    /// Read an [`Int3`] as three consecutive `i32` values.
    fn read_int3(&mut self) -> Int3 {
        let x = self.read_i32();
        let y = self.read_i32();
        let z = self.read_i32();
        Int3::new(x, y, z)
    }
}

// --- Vec-based stream implementations ---

/// In-memory [`StreamWriter`] that appends to an owned `Vec<u8>`.
///
/// After writing, access the bytes via the `data` field.
///
/// # Example
/// ```rust
/// use dcc_sph::helpers::{VecWriter, StreamWriter};
/// let mut w = VecWriter::new();
/// w.write_i32(42);
/// assert_eq!(w.data, [42, 0, 0, 0]);
/// ```
pub struct VecWriter {
    /// The accumulated byte output.
    pub data: Vec<u8>,
}

impl VecWriter {
    /// Create a new, empty writer.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl Default for VecWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamWriter for VecWriter {
    fn write_bytes(&mut self, data: &[u8]) {
        self.data.extend_from_slice(data);
    }
}

/// In-memory [`StreamReader`] that reads sequentially from a borrowed byte slice.
///
/// # Example
/// ```rust
/// use dcc_sph::helpers::{VecWriter, SliceReader, StreamWriter, StreamReader};
/// let mut w = VecWriter::new();
/// w.write_i32(42);
/// let mut r = SliceReader::new(&w.data);
/// assert_eq!(r.read_i32(), 42);
/// ```
pub struct SliceReader<'a> {
    /// The underlying byte slice.
    pub data: &'a [u8],
    /// Current read position (byte offset).
    pub pos: usize,
}

impl<'a> SliceReader<'a> {
    /// Create a reader that reads from `data` starting at byte 0.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl<'a> StreamReader for SliceReader<'a> {
    fn read_bytes(&mut self, buf: &mut [u8]) {
        let end = self.pos + buf.len();
        buf.copy_from_slice(&self.data[self.pos..end]);
        self.pos = end;
    }
}

// --- File-based stream implementations ---

/// [`StreamWriter`] that writes to any `std::io::Write` sink (e.g., a file).
///
/// # Example
/// ```rust,no_run
/// use dcc_sph::helpers::{FileWriter, StreamWriter};
/// use std::fs::File;
/// let f = File::create("model.bin").unwrap();
/// let mut w = FileWriter::new(f);
/// w.write_i32(42);
/// ```
pub struct FileWriter<W: std::io::Write> {
    inner: W,
}

impl<W: std::io::Write> FileWriter<W> {
    /// Wrap a `Write` implementor.
    pub fn new(inner: W) -> Self {
        Self { inner }
    }
}

impl<W: std::io::Write> StreamWriter for FileWriter<W> {
    fn write_bytes(&mut self, data: &[u8]) {
        self.inner.write_all(data).expect("I/O write failed");
    }
}

/// [`StreamReader`] that reads from any `std::io::Read` source (e.g., a file).
///
/// # Example
/// ```rust,no_run
/// use dcc_sph::helpers::{FileReader, StreamReader};
/// use std::fs::File;
/// let f = File::open("model.bin").unwrap();
/// let mut r = FileReader::new(f);
/// let v = r.read_i32();
/// ```
pub struct FileReader<R: std::io::Read> {
    inner: R,
}

impl<R: std::io::Read> FileReader<R> {
    /// Wrap a `Read` implementor.
    pub fn new(inner: R) -> Self {
        Self { inner }
    }
}

impl<R: std::io::Read> StreamReader for FileReader<R> {
    fn read_bytes(&mut self, buf: &mut [u8]) {
        self.inner.read_exact(buf).expect("I/O read failed");
    }
}

// --- Thread pool size ---

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Set the number of threads in the global rayon thread pool.
///
/// **Must be called before any [`Hierarchy::step`](crate::hierarchy::Hierarchy::step)
/// or other rayon-parallel work.** If any rayon task has already run, the global
/// thread pool is already initialised and this call will have no effect on the pool
/// size (a warning is printed to stderr).
///
/// Passing `0` lets rayon choose the number of threads automatically.
pub fn set_num_threads(n: usize) {
    NUM_THREADS.store(n, Ordering::Relaxed);
    if rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .is_err()
    {
        eprintln!(
            "[dcc_sph] Warning: set_num_threads({n}) called after the rayon global thread pool \
             was already initialized. Thread count was not changed. \
             Call set_num_threads() before any parallel work."
        );
    }
}

/// Return the number of threads last passed to [`set_num_threads`], or `0` if
/// it was never called.
pub fn get_num_threads() -> usize {
    NUM_THREADS.load(Ordering::Relaxed)
}
