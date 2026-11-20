//! SC View: a Quest 3 test viewer. Loads an STL (default sea.stl in the app's files folder) and draws it in
//! stereo with OpenXR + Vulkan (multiview), logging GPU and frame times to logcat (tag SCVIEW).
//! Adapted from the openxr crate's vulkan example (MIT / Apache-2.0).

use std::{
    io::Cursor,
    time::{Duration, Instant},
};

#[cfg(target_os = "android")]
use android_activity::{AndroidApp as Host, InputStatus, MainEvent, PollEvent};
#[cfg(target_os = "android")]
use android_activity::input::Keycode;
#[cfg(not(target_os = "android"))]
mod host;
#[cfg(not(target_os = "android"))]
pub use host::Host;
#[cfg(not(target_os = "android"))]
use host::Keycode;
use ash::{
    util::read_spv,
    vk::{self, Handle},
};
use openxr as xr;

mod commands;
mod font;
mod icons;
mod panel;
mod raycast;
mod scene;
mod svr;
mod voice;

const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
const VIEW_COUNT: u32 = 2;
const MSAA: vk::SampleCountFlags = vk::SampleCountFlags::TYPE_4;
const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;
const PIPELINE_DEPTH: u32 = 2;
/// Push constants: two eye matrices, the tint, the section plane (see shaders/mesh.vert).
const PUSH_SIZE: u32 = 160;
/// "No section cut": every point is on the kept side.
const NO_PLANE: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const CUT_TINT: [f32; 4] = [1.0, 0.45, 0.1, 1.0];
const MEASURE_TINT: [f32; 4] = [1.0, 0.85, 0.2, 1.0];
/// Highlight tints (rgb, strength): body under the pointer, selected body.
// Light tints, so a part's own colour (e.g. one the AI just set) still shows; the selection also gets orange edges.
const HOVER_TINT: [f32; 4] = [0.35, 0.8, 1.0, 0.3];
const SELECT_TINT: [f32; 4] = [1.0, 0.55, 0.1, 0.25];
const SELECT_EDGE_TINT: [f32; 4] = [1.0, 0.5, 0.05, 1.0];
/// The whole model while it is a preview of a change not made yet: clearly green.
const PREVIEW_TINT: [f32; 4] = [0.2, 0.95, 0.4, 0.55];
/// Parts the interference check found overlapping.
const CLASH_TINT: [f32; 4] = [1.0, 0.12, 0.1, 0.6];
/// The overlap of two parts, shown as a body of its own.
const OVERLAP_TINT: [f32; 4] = [1.0, 0.05, 0.05, 0.9];

type M4 = [f32; 16];

fn identity() -> M4 {
    let mut m = [0.0; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m
}

/// Column-major product a * b.
fn mul(a: &M4, b: &M4) -> M4 {
    let mut o = [0.0; 16];
    for c in 0..4 {
        for r in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[k * 4 + r] * b[c * 4 + k];
            }
            o[c * 4 + r] = s;
        }
    }
    o
}

fn translate(x: f32, y: f32, z: f32) -> M4 {
    let mut m = identity();
    m[12] = x;
    m[13] = y;
    m[14] = z;
    m
}

fn scale(s: f32) -> M4 {
    let mut m = identity();
    m[0] = s;
    m[5] = s;
    m[10] = s;
    m
}

fn rot_y(a: f32) -> M4 {
    let (s, c) = a.sin_cos();
    let mut m = identity();
    m[0] = c;
    m[2] = -s;
    m[8] = s;
    m[10] = c;
    m
}

/// Rotation by angle a (radians) about a unit axis (Rodrigues).
fn rot_axis(k: [f32; 3], a: f32) -> M4 {
    let (s, c) = a.sin_cos();
    let t = 1.0 - c;
    let [x, y, z] = k;
    let mut m = identity();
    m[0] = t * x * x + c;
    m[1] = t * x * y + s * z;
    m[2] = t * x * z - s * y;
    m[4] = t * x * y - s * z;
    m[5] = t * y * y + c;
    m[6] = t * y * z + s * x;
    m[8] = t * x * z + s * y;
    m[9] = t * y * z - s * x;
    m[10] = t * z * z + c;
    m
}

/// Z-up model to Y-up world: (x, y, z) -> (x, z, -y).
fn z_up_to_y_up() -> M4 {
    let mut m = [0.0; 16];
    m[0] = 1.0; // x -> x
    m[6] = -1.0; // y -> -z   (column 1, row 2)
    m[9] = 1.0; // z -> y   (column 2, row 1)
    m[15] = 1.0;
    m
}

fn quat_to_mat(q: xr::Quaternionf) -> M4 {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    let mut m = identity();
    m[0] = 1.0 - 2.0 * (y * y + z * z);
    m[1] = 2.0 * (x * y + z * w);
    m[2] = 2.0 * (x * z - y * w);
    m[4] = 2.0 * (x * y - z * w);
    m[5] = 1.0 - 2.0 * (x * x + z * z);
    m[6] = 2.0 * (y * z + x * w);
    m[8] = 2.0 * (x * z + y * w);
    m[9] = 2.0 * (y * z - x * w);
    m[10] = 1.0 - 2.0 * (x * x + y * y);
    m
}

fn transpose3(m: &M4) -> M4 {
    let mut o = identity();
    for c in 0..3 {
        for r in 0..3 {
            o[c * 4 + r] = m[r * 4 + c];
        }
    }
    o
}

/// The view matrix for an eye pose (the inverse of the pose).
fn view_matrix(pose: &xr::Posef) -> M4 {
    let rt = transpose3(&quat_to_mat(pose.orientation));
    mul(&rt, &translate(-pose.position.x, -pose.position.y, -pose.position.z))
}

/// The matrix for a pose (rotation then translation).
fn pose_matrix(pose: &xr::Posef) -> M4 {
    mul(&translate(pose.position.x, pose.position.y, pose.position.z), &quat_to_mat(pose.orientation))
}

/// An OpenXR pose as the shared grab's (svr) pose.
fn svr_pose(p: &xr::Posef) -> svr::Pose {
    svr::Pose {
        position: [p.position.x as f64, p.position.y as f64, p.position.z as f64],
        orientation: [p.orientation.x as f64, p.orientation.y as f64, p.orientation.z as f64, p.orientation.w as f64],
    }
}

/// A model matrix (a turn, one scale and a move) as the shared grab's transform; `None` when it has no scale.
fn m4_to_transform(m: &M4) -> Option<svr::room::Transform> {
    let col = |c: usize| [m[c * 4] as f64, m[c * 4 + 1] as f64, m[c * 4 + 2] as f64];
    let scale = svr::vector::length(col(0));
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let axes = [0, 1, 2].map(|c| svr::vector::scale(col(c), 1.0 / svr::vector::length(col(c)).max(1e-30)));
    svr::room::Transform { axes, scale, translation: col(3), pivot: [0.0; 3] }.tidy()
}

/// The shared grab's transform as a model matrix.
fn transform_to_m4(t: &svr::room::Transform) -> M4 {
    let mut m = identity();
    for c in 0..3 {
        for r in 0..3 {
            m[c * 4 + r] = (t.axes[c][r] * t.scale) as f32;
        }
        m[12 + c] = t.translation[c] as f32;
    }
    m
}

/// Asymmetric projection from an OpenXR field of view, Vulkan clip space (y down, depth 0..1).
fn projection(fov: xr::Fovf, near: f32, far: f32) -> M4 {
    let tl = fov.angle_left.tan();
    let tr = fov.angle_right.tan();
    let tu = fov.angle_up.tan();
    let td = fov.angle_down.tan();
    let w = tr - tl;
    let h = td - tu;
    let mut m = [0.0; 16];
    m[0] = 2.0 / w;
    m[5] = 2.0 / h;
    m[8] = (tr + tl) / w;
    m[9] = (tu + td) / h;
    m[10] = -far / (far - near);
    m[11] = -1.0;
    m[14] = -(far * near) / (far - near);
    m
}

unsafe fn find_memory(inst: &ash::Instance, pd: vk::PhysicalDevice, bits: u32, want: vk::MemoryPropertyFlags) -> u32 {
    let props = inst.get_physical_device_memory_properties(pd);
    for i in 0..props.memory_type_count {
        if bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(want) {
            return i;
        }
    }
    panic!("no suitable memory type");
}

unsafe fn make_buffer(
    dev: &ash::Device,
    inst: &ash::Instance,
    pd: vk::PhysicalDevice,
    size: u64,
    usage: vk::BufferUsageFlags,
    props: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buf = dev.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None).unwrap();
    let req = dev.get_buffer_memory_requirements(buf);
    let mem = dev
        .allocate_memory(
            &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(find_memory(inst, pd, req.memory_type_bits, props)),
            None,
        )
        .unwrap();
    dev.bind_buffer_memory(buf, mem, 0).unwrap();
    (buf, mem)
}

/// The mesh the viewer draws: 28-byte vertices (position, normal, sRGB colour), triangle indices,
/// and the bounds (min xyz, max xyz) used to size and centre it.
struct Mesh {
    verts: Vec<u8>,
    indices: Vec<u32>,
    bounds: [f32; 6],
    bodies: Vec<Body>,
    /// An edit of the model already on show (the server's beacon said so): the view keeps its size and
    /// place, so what changed is seen to change. False for another model, which is fitted afresh.
    same_model: bool,
}

/// One body (an OBJ object from SolidCraft): a contiguous range of the index buffer.
struct Body {
    name: String,
    first: u32,
    count: u32,
    /// Bounds in model units (min xyz, max xyz), for fast ray rejection.
    aabb: [f32; 6],
}

/// What picking needs from a mesh, kept on the CPU after the upload.
struct Scene {
    pos: Vec<[f32; 3]>,
    indices: Vec<u32>,
    bodies: Vec<Body>,
    /// Edge lines (pairs of vertex indices): every body's feature edges, then every body's triangle edges.
    edge_idx: Vec<u32>,
    /// Per body, (first, count) in edge_idx: its feature edges, and all its triangle edges.
    feat: Vec<(u32, u32)>,
    all: Vec<(u32, u32)>,
    /// Vertices that are corners: where feature edges meet other than in a smooth chain.
    corner: Vec<bool>,
    /// Face patches: each triangle's patch, and each patch's triangles (indices into the triangle list).
    tri_patch: Vec<u32>,
    patch_tris: Vec<Vec<u32>>,
    patch_body: Vec<u32>,
}

/// One whole CAD edge as picked in the headset: the run of feature-edge segments through smooth joins.
#[derive(Clone)]
struct EdgeChain {
    body: usize,
    /// Positions of its segments in `edge_idx` (each the first of a pair).
    segs: Vec<u32>,
    /// Start, middle (by length) and end points, and the length (model units).
    start: [f32; 3],
    mid: [f32; 3],
    end: [f32; 3],
    len: f32,
}

impl EdgeChain {
    fn key(&self) -> u32 {
        self.segs.iter().copied().min().unwrap_or(0)
    }
}

/// Where a measure point went: a corner, a point on a feature edge, or the surface.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Snap {
    Corner,
    Edge,
    Surface,
}

/// Note pins: the head stands this far above the pinned point (metres, straight up in the room), and
/// the pointer picks it within this distance.
const NOTE_POST: f32 = 0.03;
const NOTE_REACH: f32 = 0.025;
fn note_head(w: [f32; 3]) -> [f32; 3] {
    [w[0], w[1] + NOTE_POST, w[2]]
}

/// The push/pull step (mm): the smallest of 0.5, 1, 2, 5, 10 that is at least 5 mm of hand movement
/// at the model's current scale (`unit`: metres per model mm). Not below 0.5 mm: SolidCraft 0.41
/// refuses moves of 0.2 mm or less on faces it moves fine by 0.5 mm.
fn pp_step(unit: f32) -> f32 {
    [0.5, 1.0, 2.0, 5.0, 10.0].into_iter().find(|s| s * unit >= 0.005).unwrap_or(10.0)
}

/// Edges sharper than this (between the two faces' normals) are feature edges.
const FEATURE_COS: f32 = 0.866; // 30 degrees

/// A body's edges. Vertices are shared by position (the mesh repeats them per normal), and an edge
/// is a feature edge where its faces meet at more than 30 degrees, or where it has only one face.
/// Face winding is not trusted, so normals are compared either way round.
/// Also returns each triangle's face patch (0..): triangles joined across smooth edges, so a patch is
/// one CAD face (or a smooth run of them), bounded by feature edges.
fn body_edges(pos: &[[f32; 3]], tris: &[u32], canon: &mut std::collections::HashMap<[i64; 3], u32>) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let key = |p: [f32; 3]| p.map(|x| (x * 1000.0).round() as i64);
    let mut edges: std::collections::HashMap<(u32, u32), ([f32; 3], u32, bool, u32)> = std::collections::HashMap::new();
    let mut order: Vec<(u32, u32)> = Vec::new();
    // Union-find over the body's triangles.
    let mut parent: Vec<u32> = (0..(tris.len() / 3) as u32).collect();
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    for (ti, t) in tris.chunks_exact(3).enumerate() {
        let ti = ti as u32;
        let v = [0, 1, 2].map(|k| *canon.entry(key(pos[t[k] as usize])).or_insert(t[k]));
        let n = cross(sub(pos[v[1] as usize], pos[v[0] as usize]), sub(pos[v[2] as usize], pos[v[0] as usize]));
        if dot(n, n) < 1e-20 {
            continue;
        }
        let n = normalize(n);
        for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            if a == b {
                continue;
            }
            let e = (a.min(b), a.max(b));
            match edges.get_mut(&e) {
                Some((n0, count, sharp, first)) => {
                    *count += 1;
                    if dot(*n0, n).abs() < FEATURE_COS {
                        *sharp = true;
                    } else if *count == 2 {
                        let (ra, rb) = (root(&mut parent, *first), root(&mut parent, ti));
                        parent[ra as usize] = rb;
                    }
                }
                None => {
                    edges.insert(e, (n, 1, false, ti));
                    order.push(e);
                }
            }
        }
    }
    let (mut feat, mut all) = (Vec::new(), Vec::new());
    for e in order {
        let (_, count, sharp, _) = edges[&e];
        all.extend_from_slice(&[e.0, e.1]);
        if sharp || count == 1 {
            feat.extend_from_slice(&[e.0, e.1]);
        }
    }
    // Number the patches 0.. in order of first triangle.
    let mut ids: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let patch = (0..parent.len() as u32)
        .map(|t| {
            let r = root(&mut parent, t);
            let next = ids.len() as u32;
            *ids.entry(r).or_insert(next)
        })
        .collect();
    (feat, all, patch)
}

impl Scene {
    fn new(m: Mesh) -> Scene {
        let pos: Vec<[f32; 3]> = m
            .verts
            .chunks_exact(VERTEX_SIZE)
            .map(|v| {
                let f = |o: usize| f32::from_le_bytes([v[o], v[o + 1], v[o + 2], v[o + 3]]);
                [f(0), f(4), f(8)]
            })
            .collect();
        let mut bodies = m.bodies;
        for b in &mut bodies {
            let mut bb = [f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN];
            for &i in &m.indices[b.first as usize..(b.first + b.count) as usize] {
                let p = pos[i as usize];
                for k in 0..3 {
                    bb[k] = bb[k].min(p[k]);
                    bb[k + 3] = bb[k + 3].max(p[k]);
                }
            }
            b.aabb = bb;
        }
        let t = Instant::now();
        let mut canon = std::collections::HashMap::new();
        let per_body: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> = bodies.iter().map(|b| body_edges(&pos, &m.indices[b.first as usize..(b.first + b.count) as usize], &mut canon)).collect();
        let mut tri_patch = vec![0u32; m.indices.len() / 3];
        let mut patch_tris: Vec<Vec<u32>> = Vec::new();
        let mut patch_body: Vec<u32> = Vec::new();
        for (bi, (b, (_, _, patch))) in bodies.iter().zip(&per_body).enumerate() {
            let base = patch_tris.len() as u32;
            for (i, &p) in patch.iter().enumerate() {
                let t = b.first / 3 + i as u32;
                tri_patch[t as usize] = base + p;
                while patch_tris.len() <= (base + p) as usize {
                    patch_tris.push(Vec::new());
                    patch_body.push(bi as u32);
                }
                patch_tris[(base + p) as usize].push(t);
            }
        }
        let mut edge_idx = Vec::new();
        let mut feat = Vec::new();
        for (f, _, _) in &per_body {
            feat.push((edge_idx.len() as u32, f.len() as u32));
            edge_idx.extend_from_slice(f);
        }
        let n_feat = edge_idx.len();
        let mut all = Vec::new();
        for (_, a, _) in &per_body {
            all.push((edge_idx.len() as u32, a.len() as u32));
            edge_idx.extend_from_slice(a);
        }
        // Corners: feature-edge vertices that are not the middle of a smooth chain (two edges, nearly straight on).
        let mut ends: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        for e in edge_idx[..n_feat].chunks_exact(2) {
            ends.entry(e[0]).or_default().push(e[1]);
            ends.entry(e[1]).or_default().push(e[0]);
        }
        let mut corner = vec![false; pos.len()];
        for (&v, others) in &ends {
            corner[v as usize] = match others[..] {
                [a, b] => {
                    let (da, db) = (normalize(sub(pos[a as usize], pos[v as usize])), normalize(sub(pos[b as usize], pos[v as usize])));
                    dot(da, db) > -FEATURE_COS
                }
                _ => true,
            };
        }
        log::info!("edges: {} feature, {} total, {} corners, {} faces in {} ms", n_feat / 2, (edge_idx.len() - n_feat) / 2, corner.iter().filter(|&&c| c).count(), patch_tris.len(), t.elapsed().as_millis());
        Scene { pos, indices: m.indices, bodies, edge_idx, feat, all, corner, tri_patch, patch_tris, patch_body }
    }

    /// A measure point near `p` on body `b`: a corner within `r`, else the nearest point on a
    /// feature edge within `r`, else `p` itself (model units).
    fn snap(&self, b: usize, p: [f32; 3], r: f32) -> ([f32; 3], Snap) {
        let (first, count) = self.feat[b];
        let mut best_corner: Option<(f32, [f32; 3])> = None;
        let mut best_edge: Option<(f32, [f32; 3])> = None;
        for e in self.edge_idx[first as usize..(first + count) as usize].chunks_exact(2) {
            let (a, q) = (self.pos[e[0] as usize], self.pos[e[1] as usize]);
            for (v, vp) in [(e[0], a), (e[1], q)] {
                let d = sub(vp, p);
                let d = dot(d, d).sqrt();
                if self.corner[v as usize] && d < r && best_corner.is_none_or(|(bd, _)| d < bd) {
                    best_corner = Some((d, vp));
                }
            }
            let ab = sub(q, a);
            let t = (dot(sub(p, a), ab) / dot(ab, ab).max(1e-20)).clamp(0.0, 1.0);
            let c = [0, 1, 2].map(|k| a[k] + t * ab[k]);
            let d = sub(c, p);
            let d = dot(d, d).sqrt();
            if d < r && best_edge.is_none_or(|(bd, _)| d < bd) {
                best_edge = Some((d, c));
            }
        }
        match (best_corner, best_edge) {
            (Some((_, c)), _) => (c, Snap::Corner),
            (None, Some((_, c))) => (c, Snap::Edge),
            _ => (p, Snap::Surface),
        }
    }

    /// The feature-edge segment of body `b` nearest `p` within `r` (model units): its position in `edge_idx`.
    fn nearest_segment(&self, b: usize, p: [f32; 3], r: f32) -> Option<u32> {
        let (first, count) = self.feat[b];
        let mut best: Option<(f32, u32)> = None;
        for (i, e) in self.edge_idx[first as usize..(first + count) as usize].chunks_exact(2).enumerate() {
            let (a, q) = (self.pos[e[0] as usize], self.pos[e[1] as usize]);
            let ab = sub(q, a);
            let t = (dot(sub(p, a), ab) / dot(ab, ab).max(1e-20)).clamp(0.0, 1.0);
            let c = [0, 1, 2].map(|k| a[k] + t * ab[k]);
            let d = sub(c, p);
            let d = dot(d, d).sqrt();
            if d < r && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, first + 2 * i as u32));
            }
        }
        best.map(|(_, k)| k)
    }

    /// The whole edge that segment `k` of body `b` belongs to: neighbouring segments are followed through
    /// vertices that are not corners (and have exactly two feature edges).
    fn edge_chain(&self, b: usize, k: u32) -> EdgeChain {
        let (first, count) = self.feat[b];
        let mut at: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        for (i, e) in self.edge_idx[first as usize..(first + count) as usize].chunks_exact(2).enumerate() {
            let seg = first + 2 * i as u32;
            at.entry(e[0]).or_default().push(seg);
            at.entry(e[1]).or_default().push(seg);
        }
        let ends = |seg: u32| (self.edge_idx[seg as usize], self.edge_idx[seg as usize + 1]);
        // Walks from vertex `v` (reached by segment `from`) while the join is smooth; returns the segments and vertices passed.
        let walk = |mut v: u32, mut from: u32, taken: &[u32]| -> (Vec<u32>, Vec<u32>) {
            let (mut segs, mut verts) = (Vec::new(), vec![v]);
            loop {
                let list = &at[&v];
                if self.corner[v as usize] || list.len() != 2 {
                    break;
                }
                let next = if list[0] == from { list[1] } else { list[0] };
                if next == k || taken.contains(&next) || segs.contains(&next) {
                    break; // round a loop, or into what the other direction already walked
                }
                let (a, c) = ends(next);
                v = if a == v { c } else { a };
                segs.push(next);
                verts.push(v);
                from = next;
            }
            (segs, verts)
        };
        let (a, c) = ends(k);
        let (fwd_s, fwd_v) = walk(c, k, &[k]);
        // (Round a closed loop the forward walk has taken every segment, so the backward one has nothing to add.)
        let mut taken = vec![k];
        taken.extend(&fwd_s);
        let (back_s, back_v) = walk(a, k, &taken);
        // Points in order: the far end of the back walk, back to a, then c and on through the forward walk.
        let mut order: Vec<u32> = back_v.iter().skip(1).rev().copied().collect();
        order.push(a);
        order.push(c);
        order.extend(fwd_v.iter().skip(1).copied());
        let pts: Vec<[f32; 3]> = order.iter().map(|&v| self.pos[v as usize]).collect();
        let mut segs = vec![k];
        segs.extend(fwd_s);
        segs.extend(back_s);
        let dist = |p: [f32; 3], q: [f32; 3]| {
            let d = sub(p, q);
            dot(d, d).sqrt()
        };
        let total: f32 = pts.windows(2).map(|w| dist(w[0], w[1])).sum();
        let mut run = 0.0;
        let mut mid = pts[0];
        for w in pts.windows(2) {
            let d = dist(w[0], w[1]);
            if run + d >= total / 2.0 {
                let t = (total / 2.0 - run) / d.max(1e-20);
                mid = [0, 1, 2].map(|j| w[0][j] + t * (w[1][j] - w[0][j]));
                break;
            }
            run += d;
        }
        EdgeChain { body: b, segs, start: pts[0], mid, end: *pts.last().unwrap(), len: total }
    }

    /// Bounds of all bodies (model units): min xyz, max xyz.
    fn bounds(&self) -> [f32; 6] {
        let mut bb = [f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN];
        for b in &self.bodies {
            for k in 0..3 {
                bb[k] = bb[k].min(b.aabb[k]);
                bb[k + 3] = bb[k + 3].max(b.aabb[k + 3]);
            }
        }
        bb
    }

    /// The nearest body hit by the ray o + t d (model units, t >= 0): (body, t). Hits on the
    /// cut-away side of the section plane do not count.
    /// `offsets` moves each body (exploded view); empty means none.
    /// Returns (body, t, triangle).
    fn pick(&self, o: [f32; 3], d: [f32; 3], hidden: &[bool], plane: [f32; 4], offsets: &[[f32; 3]]) -> Option<(usize, f32, usize)> {
        let mut best: Option<(usize, f32, usize)> = None;
        let o_world = o;
        for (bi, b) in self.bodies.iter().enumerate() {
            if hidden.get(bi).copied().unwrap_or(false) {
                continue;
            }
            // The ray in the body's own (unexploded) coordinates; the plane stays where it is in the model.
            let off = offsets.get(bi).copied().unwrap_or([0.0; 3]);
            let o = sub(o_world, off);
            let plane = [plane[0], plane[1], plane[2], plane[3] + dot([plane[0], plane[1], plane[2]], off)];
            let Some(t_box) = ray_box(o, d, &b.aabb) else { continue };
            if best.is_some_and(|(_, t, _)| t_box > t) {
                continue;
            }
            for (ti, tri) in self.indices[b.first as usize..(b.first + b.count) as usize].chunks_exact(3).enumerate() {
                if let Some(t) = ray_triangle(o, d, self.pos[tri[0] as usize], self.pos[tri[1] as usize], self.pos[tri[2] as usize]) {
                    let p = [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]];
                    if best.is_none_or(|(_, bt, _)| t < bt) && dot([plane[0], plane[1], plane[2]], p) + plane[3] >= 0.0 {
                        best = Some((bi, t, b.first as usize / 3 + ti));
                    }
                }
            }
        }
        best
    }
}

/// The names of the bodies whose flag is set.
fn names_where(scene: &Scene, flags: &[bool]) -> Vec<String> {
    scene.bodies.iter().zip(flags).filter(|(_, &f)| f).map(|(b, _)| b.name.clone()).collect()
}

/// Flags for the bodies named (names that no longer exist are ignored).
fn flags_for(scene: &Scene, names: &[String]) -> Vec<bool> {
    let set: std::collections::HashSet<&str> = names.iter().map(String::as_str).collect();
    scene.bodies.iter().map(|b| set.contains(b.name.as_str())).collect()
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Slab test: the entry distance of the ray into the box, if it hits.
fn ray_box(o: [f32; 3], d: [f32; 3], bb: &[f32; 6]) -> Option<f32> {
    let (mut t0, mut t1) = (0.0f32, f32::MAX);
    for k in 0..3 {
        let inv = 1.0 / d[k];
        let (mut a, mut b) = ((bb[k] - o[k]) * inv, (bb[k + 3] - o[k]) * inv);
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        t0 = t0.max(a);
        t1 = t1.min(b);
        if t0 > t1 {
            return None;
        }
    }
    Some(t0)
}

/// Möller-Trumbore, two-sided.
fn ray_triangle(o: [f32; 3], d: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<f32> {
    let e1 = sub(b, a);
    let e2 = sub(c, a);
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = sub(o, a);
    let u = dot(s, p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = dot(e2, q) * inv;
    (t >= 0.0).then_some(t)
}

/// The wrist panel's buttons.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Button {
    Hide,
    Isolate,
    ShowAll,
    Clear,
    Recentre,
    Undo,
    Room,
    Section,
    Measure,
    Edges,
    Explode,
    Note,
    DoNotes,
    DeleteNote,
    Redo,
    Faces,
    PushPull,
    Place,
    Size,
    Models,
    Keys,
    Guides,
    Keyboard,
    SaveFile,
    Snap,
    Repeat,
    SaveReq,
    Saved,
    Dims,
    Features,
    Handles,
    Move,
    Clash,
    Radial,
    ConfirmGo,
    ConfirmStop,
    CancelVoice,
    Reply,
    Tab(panel::Tab),
}

/// Axes (X red, Y green, Z blue, at the model origin) and a grid on the model's ground (XY) plane.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Guides {
    Off,
    Axes,
    Grid,
}

/// A long dimensions list is shown in pages: (dimensions on a page, pages). A list that fits has one page; a
/// longer one gives up its last row to "next page".
fn dims_paging(total: usize) -> (usize, usize) {
    if total <= MODELS_SHOWN {
        (total.max(1), 1)
    } else {
        let per = MODELS_SHOWN - 1;
        (per, total.div_ceil(per))
    }
}

/// Width of each of the two targets (- and +) at the right end of a dimension's row.
const DIM_ZONE: f32 = 0.022;

/// A dimension's value one step down (dir -1) or up (+1): the step is a round number about a tenth of the
/// value (60 goes by 5, 10 by 1, 6 by 0.5), and the result lands on a multiple of it.
fn dim_step_to(value: f32, dir: f32) -> f32 {
    let step = nice_step(value.abs() / 10.0).max(0.01);
    let k = value / step;
    let next = if dir > 0.0 { (k + 1e-3).floor() + 1.0 } else { (k - 1e-3).ceil() - 1.0 };
    (next * step * 1000.0).round() / 1000.0
}

/// A dimension's value for a row: no trailing zeros.
fn fmt_dim(v: f32) -> String {
    let s = format!("{v:.3}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// A round grid spacing (1, 2 or 5 times a power of ten) near x.
fn nice_step(x: f32) -> f32 {
    let p = 10f32.powf(x.max(1e-9).log10().floor());
    let m = x / p;
    p * if m < 1.5 { 1.0 } else if m < 3.5 { 2.0 } else if m < 7.5 { 5.0 } else { 10.0 }
}

/// A list shown above the panel: the keyboard and mouse help, or the Mac's models to open.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Overlay {
    Off,
    Keys,
    Models,
    Requests,
    Dims,
    Features,
    Reply,
}

/// Keyboard and mouse help (the Keys button, or F1).
/// Sizes the Size button steps through, as multiples of the size in the file (a 1:1000 design shown at x2 is 1:500).
const SIZE_STEPS: [f32; 4] = [0.5, 1.0, 2.0, 4.0];

const KEY_HELP: [&str; 18] = [
    "Type, then Enter: ask the AI",
    "Esc: clear text / close / free mouse",
    "Ctrl+Z undo, Ctrl+Y redo",
    "Ctrl+H hide, Ctrl+I isolate, Del hide",
    "Ctrl+A show all, Ctrl+D deselect",
    "Ctrl+R recentre, Ctrl+B room",
    "Ctrl+S section, Ctrl+M measure",
    "Ctrl+E edges, Ctrl+X explode",
    "Ctrl+N notes, Ctrl+F parts/faces/edges",
    "Ctrl+P push/pull, Ctrl+O models",
    "Ctrl+L surface, Ctrl+J size, Ctrl+U photo",
    "Ctrl+G axes/grid, Ctrl+K on-screen keys",
    "F1: this list, Ctrl+Shift+S save file",
    "Ctrl+T repeat, Ctrl+W save, Ctrl+Q saved",
    "Mouse: left = trigger (select)",
    "Right drag: turn / tip the model",
    "Wheel: zoom (push/pull: the face)",
    "Middle click: pointer straight ahead",
];
/// At most this many models are listed.
const MODELS_SHOWN: usize = 14;
/// Lines of the full reply shown at once.
const REPLY_ROWS: usize = 12;
/// Overlay rows: pitch, and the gap between it and the panel (metres).
/// Row spacing: the key help is a compact list; the lists you pick from (models, saved requests) have larger
/// text and a gap between the rows, so a row is easy to hit.
const OV_PITCH: f32 = 0.0145;
const OV_PICK_PITCH: f32 = 0.021;
/// Text size in those lists (metres per font pixel), and the height of a row's highlight (and so of what can be hit).
const OV_PICK_FP: f32 = 0.00039;
const OV_PICK_HIT: f32 = 0.018;

fn ov_pitch(ov: Overlay) -> f32 {
    if matches!(ov, Overlay::Keys | Overlay::Reply) {
        OV_PITCH
    } else {
        OV_PICK_PITCH
    }
}
const OV_GAP: f32 = 0.006;

/// Rows in the overlay (after its title line).
fn overlay_rows(ov: Overlay, models: usize) -> usize {
    match ov {
        Overlay::Off => 0,
        Overlay::Keys => KEY_HELP.len(),
        Overlay::Models | Overlay::Requests | Overlay::Dims | Overlay::Features => models.clamp(1, MODELS_SHOWN),
        Overlay::Reply => REPLY_ROWS,
    }
}

/// The overlay's height; it stands OV_GAP above the panel's top edge.
fn overlay_h(ov: Overlay, rows: usize) -> f32 {
    0.008 + (rows + 1) as f32 * ov_pitch(ov) + ov_strip(ov)
}

/// The saved-requests list has a strip at its bottom that shows the whole text of the request pointed at.
const STRIP_LINES: usize = 4;
fn ov_strip(ov: Overlay) -> f32 {
    if ov == Overlay::Requests {
        STRIP_LINES as f32 * 0.0138 + 0.008
    } else {
        0.0
    }
}

/// The overlay row at panel-local height y (None: the title, or outside).
fn overlay_row_at(ov: Overlay, y: f32, rows: usize, panel_h: f32) -> Option<usize> {
    let (pitch, top) = (ov_pitch(ov), panel_h / 2.0 + OV_GAP + overlay_h(ov, rows) - 0.004);
    let k = ((top - y) / pitch).floor();
    if k < 1.0 || (k as usize) > rows {
        return None;
    }
    // Only the highlighted part of a row counts: between two rows there is a gap that picks nothing.
    let row_y = top - k * pitch;
    ((row_y - y) <= OV_PICK_HIT).then(|| k as usize - 1)
}

/// The section cut: off; on, with the right trigger moving it; on and fixed (the trigger selects again).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Section {
    Off,
    Moving,
    Fixed,
}

/// Panel width in metres (panel-local: x right, y up, facing +z); its height and buttons come from `panel::layout`.
const PANEL_W: f32 = 0.24;
/// How far the mouse pointer turns per count of mouse movement (radians).
const MOUSE_RAD: f32 = 0.0012;
/// Mouse counts (moved up) for one push/pull step while dragging a face.
const PP_COUNTS: f32 = 40.0;

/// Size of one font pixel in metres (a 40-pixel text row is 13 mm).
const FONT_PX: f32 = 0.000325;

/// Which edge lines are drawn: none, feature edges (CAD-style outline), or every triangle edge.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Edges {
    Off,
    Feature,
    All,
}

/// What the panel shows that depends on the viewer's state.
#[derive(Clone, Copy)]
struct PanelState {
    hover: Option<Button>,
    has_selection: bool,
    can_undo: bool,
    room_off: bool,
    section: Section,
    measuring: bool,
    edges: Edges,
    exploded: bool,
    noting: bool,
    /// Notes the AI has not done yet.
    open_notes: usize,
    notes_selected: usize,
    can_redo: bool,
    /// The trigger picks faces rather than whole parts.
    faces: bool,
    pick_edges: bool,
    placing: bool,
    /// Which of SIZE_STEPS the model is at now (within 5%), if any.
    size_idx: Option<usize>,
    pushing: bool,
    guides: Guides,
    tab: panel::Tab,
    /// The floating on-screen keyboard is open.
    kb_open: bool,
    /// Which page of the full reply is shown.
    /// What Undo and Redo would take back or bring back: "" (nothing), "model" (the Mac says what, see
    /// voice::history), or the words to show.
    undo_what: &'static str,
    redo_what: &'static str,
    reply_page: usize,
    /// The page of the dimensions list, and of the feature tree, on view.
    dims_page: usize,
    feat_page: usize,
    /// The dimension handles are on show.
    handles: bool,
    /// The trigger moves parts.
    moving: bool,
    /// Overlapping parts are marked.
    clash: bool,
    overlay: Overlay,
    /// The overlay row the pointer is on.
    overlay_hover: Option<usize>,
}

/// A button's name: short (beside its icon) or long (the hint when it is pointed at). It says what
/// pressing the button does next, except for Edges, Guides and Faces, which show what is on now.
/// A choice's name on the radial menu: what it does, whatever state the tool is in.
fn radial_label(b: Button) -> &'static str {
    match b {
        Button::Hide => "Hide",
        Button::Isolate => "Isolate",
        Button::ShowAll => "Show all",
        Button::Clear => "Deselect",
        Button::Recentre => "Recentre",
        Button::Undo => "Undo",
        Button::Redo => "Redo",
        Button::Section => "Section",
        Button::Measure => "Measure",
        Button::Explode => "Explode",
        Button::Note => "Note",
        Button::Faces => "Parts/faces",
        Button::PushPull => "Push/pull",
        Button::Size => "Size",
        Button::Models => "Models",
        Button::Dims => "Dimensions",
        Button::Handles => "Handles",
        Button::Move => "Move",
        Button::Clash => "Overlaps",
        _ => "?",
    }
}

fn label(b: Button, st: &PanelState, short: bool) -> &'static str {
    let PanelState { room_off, section, measuring, edges, exploded, noting, faces, pick_edges, pushing, placing, size_idx, guides, overlay, .. } = *st;
    let sel = st.notes_selected > 0;
    match b {
        Button::Tab(t) => t.label(),
        Button::ConfirmGo => match (voice::preview(), short) {
            (true, true) => "Apply",
            (true, false) => "Apply the change on show",
            (false, true) => "Go on",
            (false, false) => "Go on with the big request",
        },
        Button::Reply => "Read the full reply (click: next page)",
        Button::CancelVoice => {
            if short {
                "Cancel"
            } else {
                "Cancel the recording: nothing is sent"
            }
        }
        Button::ConfirmStop => match (voice::preview(), short) {
            (true, true) => "Drop",
            (true, false) => "Drop it: change nothing",
            (false, true) => "Stop",
            (false, false) => "Stop: do not send the big request",
        },
        Button::Repeat => {
            if short {
                "Repeat"
            } else {
                "Repeat the last request"
            }
        }
        Button::SaveReq => {
            if short {
                "Keep"
            } else {
                "Keep (save) the last request to repeat"
            }
        }
        Button::Radial => if short { "Menu" } else { "Radial menu at the pointing hand" },
        Button::Clash => match (st.clash, short) {
            (true, true) => "No marks",
            (false, true) => "Overlaps",
            (true, false) => "Take the red marks off",
            (false, false) => "Check which parts overlap: they turn red and are selected",
        },
        Button::Move => match (st.moving, short) {
            (true, true) => "Stop move",
            (false, true) => "Move",
            (true, false) => "Stop moving parts",
            (false, false) => "Move a part: drag it; let go on another part's face to set it there",
        },
        Button::Handles => match (st.handles, short) {
            (true, true) => "No handles",
            (false, true) => "Handles",
            (true, false) => "Hide the dimension handles",
            (false, false) => "Dimension handles: drag one on the model to change its size",
        },
        Button::Features => match (overlay == Overlay::Features, short) {
            (true, true) => "Close",
            (false, true) => "Features",
            (true, false) => "Close the feature tree",
            (false, false) => "Feature tree: switch a feature off or on, or edit it",
        },
        Button::Dims => match (overlay == Overlay::Dims, short) {
            (true, true) => "Close",
            (false, true) => "Dims",
            (true, false) => "Close the dimensions",
            (false, false) => "Dimensions: change a size directly (no AI)",
        },
        Button::Saved => match (overlay == Overlay::Requests, short) {
            (true, true) => "Close",
            (false, true) => "Saved",
            (true, false) => "Close the saved requests",
            (false, false) => "Saved requests",
        },
        Button::SaveFile => {
            if short {
                "Save file"
            } else {
                "Save the design as a new file"
            }
        }
        Button::Snap => {
            if short {
                "Photo"
            } else {
                "Take a photo (kept on the Mac)"
            }
        }
        Button::Keyboard => match (st.kb_open, short) {
            (true, true) => "Close",
            (false, true) => "Keyboard",
            (true, false) => "Close the on-screen keyboard",
            (false, false) => "On-screen keyboard",
        },
        Button::Models => match (overlay == Overlay::Models, short) {
            (true, true) => "Close",
            (false, true) => "Models",
            (true, false) => "Close the model list",
            (false, false) => "Open another model",
        },
        Button::Keys => match (overlay == Overlay::Keys, short) {
            (true, true) => "Close",
            (false, true) => "Keys",
            (true, false) => "Close the key help",
            (false, false) => "Keyboard and mouse help",
        },
        Button::Guides => match (guides, short) {
            (Guides::Off, true) => "Guides",
            (Guides::Axes, true) => "Axes",
            (Guides::Grid, true) => "Grid",
            (Guides::Off, false) => "Axes and grid: off",
            (Guides::Axes, false) => "Axes on (next: grid)",
            (Guides::Grid, false) => "Axes and grid on (next: off)",
        },
        Button::DoNotes => match (sel, short) {
            (true, true) => "AI sel.",
            (false, true) => "AI notes",
            (true, false) => "AI: do the selected notes",
            (false, false) => "AI: do the open notes",
        },
        Button::DeleteNote => {
            if short {
                "Delete"
            } else {
                "Delete the selected notes"
            }
        }
        Button::Undo => "Undo",
        Button::Redo => "Redo",
        Button::Size => match (size_idx, short) {
            (Some(0), true) => "Size x0.5",
            (Some(1), true) => "Size x1",
            (Some(2), true) => "Size x2",
            (Some(3), true) => "Size x4",
            (None, true) => "Size",
            (Some(0), false) => "Size: half the file's size (next: x1)",
            (Some(1), false) => "Size: as in the file (next: x2)",
            (Some(2), false) => "Size: twice the file's (next: x4)",
            (Some(3), false) => "Size: four times (next: half)",
            _ => "Size: back to the file's size",
        },
        Button::Place => match (placing, short) {
            (true, true) => "Cancel",
            (false, true) => "Set down",
            (true, false) => "Cancel setting down",
            (false, false) => "Set the model down on a surface",
        },
        Button::PushPull => match (pushing, short) {
            (true, true) => "Stop",
            (false, true) => "Push/pull",
            (true, false) => "Stop push/pull",
            (false, false) => "Push or pull a face",
        },
        Button::Faces => match (faces, pick_edges, short) {
            (_, true, true) => "Edges",
            (_, true, false) => "Picking edges (next: parts)",
            (true, _, true) => "Faces",
            (false, _, true) => "Parts",
            (true, _, false) => "Picking faces (next: edges)",
            (false, _, false) => "Picking parts (next: faces)",
        },
        Button::Note => match (noting, short) {
            (true, true) => "Stop",
            (false, true) => "Note",
            (true, false) => "Stop adding notes",
            (false, false) => "Add voice notes",
        },
        Button::Explode => match (exploded, short) {
            (true, true) => "Collapse",
            (false, true) => "Explode",
            (true, false) => "Collapse the parts",
            (false, false) => "Explode the parts",
        },
        Button::Edges => match (edges, short) {
            (Edges::Off, true) => "No edges",
            (Edges::Feature, true) => "Edges",
            (Edges::All, true) => "Mesh",
            (Edges::Off, false) => "No edges (next: outline)",
            (Edges::Feature, false) => "Outline edges (next: mesh)",
            (Edges::All, false) => "Mesh lines (next: none)",
        },
        Button::Section => match (section, short) {
            (Section::Off, true) => "Section",
            (Section::Moving, true) => "Fix cut",
            (Section::Fixed, true) => "No cut",
            (Section::Off, false) => "Section cut",
            (Section::Moving, false) => "Fix the cut here",
            (Section::Fixed, false) => "Remove the cut",
        },
        Button::Measure => match (measuring, short) {
            (true, true) => "Stop",
            (false, true) => "Measure",
            (true, false) => "Stop measuring",
            (false, false) => "Measure a distance",
        },
        Button::Hide => {
            if short {
                "Hide"
            } else {
                "Hide the selection"
            }
        }
        Button::Isolate => {
            if short {
                "Isolate"
            } else {
                "Show only the selection"
            }
        }
        Button::ShowAll => {
            if short {
                "Show all"
            } else {
                "Show all parts"
            }
        }
        Button::Clear => {
            if short {
                "Deselect"
            } else {
                "Deselect everything"
            }
        }
        Button::Recentre => {
            if short {
                "Recentre"
            } else {
                "Recentre the model"
            }
        }
        Button::Room => match (room_off, short) {
            (true, true) => "Room on",
            (false, true) => "Room off",
            (true, false) => "Show the room",
            (false, false) => "Hide the room",
        },
    }
}

fn enabled(b: Button, st: &PanelState) -> bool {
    match b {
        Button::Hide | Button::Isolate | Button::Clear => st.has_selection,
        Button::Undo => st.can_undo,
        Button::DoNotes => st.open_notes > 0 || st.notes_selected > 0,
        Button::DeleteNote => st.notes_selected > 0,
        Button::Redo => st.can_redo,
        _ => true,
    }
}

/// A tool in use, a list open, or the current tab: shown in colour.
fn active(b: Button, st: &PanelState) -> bool {
    match b {
        Button::Tab(t) => t == st.tab,
        Button::Guides => st.guides != Guides::Off,
        Button::Models => st.overlay == Overlay::Models,
        Button::Keys => st.overlay == Overlay::Keys,
        Button::Saved => st.overlay == Overlay::Requests,
        Button::Dims => st.overlay == Overlay::Dims,
        Button::Features => st.overlay == Overlay::Features,
        Button::Handles => st.handles,
        Button::Move => st.moving,
        Button::Clash => st.clash,
        Button::Reply => st.overlay == Overlay::Reply,
        Button::Keyboard => st.kb_open,
        Button::Section => st.section == Section::Moving,
        Button::Measure => st.measuring,
        Button::Note => st.noting,
        Button::Faces => st.faces || st.pick_edges,
        Button::PushPull => st.pushing,
        Button::Place => st.placing,
        _ => false,
    }
}

/// Geometry for flat UI (quads and bitmap text), in the mesh vertex format.
#[derive(Default)]
struct Ui {
    verts: Vec<u8>,
    indices: Vec<u32>,
}

impl Ui {
    fn quad(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, z: f32, rgb: [u8; 3]) {
        let base = (self.verts.len() / VERTEX_SIZE) as u32;
        for (x, y) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] {
            for f in [x, y, z, 0.0, 0.0, 1.0] {
                self.verts.extend_from_slice(&f.to_le_bytes());
            }
            self.verts.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// The font has printable ASCII only: anything else is drawn as '?'.
    fn glyph(c: char) -> &'static (u8, [u64; font::FONT_H]) {
        let i = if (' '..='~').contains(&c) { c as usize - 32 } else { '?' as usize - 32 };
        &font::GLYPHS[i]
    }

    fn text_width(s: &str) -> f32 {
        s.chars().map(|c| Ui::glyph(c).0 as f32).sum::<f32>() * FONT_PX
    }

    /// Words wrapped into lines no wider than `width`, at most `max` lines (the last one shortened).
    fn wrap(s: &str, width: f32, max: usize) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        let mut cur = String::new();
        for w in s.split_whitespace() {
            let next = if cur.is_empty() { w.to_string() } else { format!("{cur} {w}") };
            if Ui::text_width(&next) <= width || cur.is_empty() {
                cur = next;
            } else {
                lines.push(std::mem::replace(&mut cur, w.to_string()));
            }
        }
        if !cur.is_empty() {
            lines.push(cur);
        }
        if lines.len() > max {
            let rest = lines[max - 1..].join(" ");
            lines.truncate(max - 1);
            lines.push(rest);
        }
        lines.into_iter().map(|l| Ui::fit(&l, width)).collect()
    }

    /// Draws text with its top-left corner at (x, top); each horizontal run of lit pixels is one quad.
    fn text(&mut self, s: &str, x: f32, top: f32, z: f32, rgb: [u8; 3]) {
        self.text_px(s, x, top, z, rgb, FONT_PX);
    }

    /// Width of text with each font pixel `fp` metres across.
    fn text_width_px(s: &str, fp: f32) -> f32 {
        s.chars().map(|c| Ui::glyph(c).0 as f32).sum::<f32>() * fp
    }

    /// Text with each font pixel `fp` metres across (FONT_PX is the normal size).
    fn text_px(&mut self, s: &str, mut x: f32, top: f32, z: f32, rgb: [u8; 3], fp: f32) {
        for c in s.chars() {
            let (w, rows) = Ui::glyph(c);
            let w = *w as u32;
            for (r, bits) in rows.iter().enumerate() {
                let y1 = top - r as f32 * fp;
                let mut px = 0;
                while px < w {
                    if bits >> (w - 1 - px) & 1 == 1 {
                        let start = px;
                        while px < w && bits >> (w - 1 - px) & 1 == 1 {
                            px += 1;
                        }
                        self.quad(x + start as f32 * fp, y1 - fp, x + px as f32 * fp, y1, z, rgb);
                    } else {
                        px += 1;
                    }
                }
            }
            x += w as f32 * fp;
        }
    }

    /// A 12x12 icon with its top-left corner at (x, top); `px` is the size of one icon pixel.
    fn icon(&mut self, bits: &icons::Icon, x: f32, top: f32, px: f32, z: f32, rgb: [u8; 3]) {
        for (r, &row) in bits.iter().enumerate() {
            let y1 = top - r as f32 * px;
            let mut c = 0u32;
            while c < 12 {
                if row >> (11 - c) & 1 == 1 {
                    let start = c;
                    while c < 12 && row >> (11 - c) & 1 == 1 {
                        c += 1;
                    }
                    self.quad(x + start as f32 * px, y1 - px, x + c as f32 * px, y1, z, rgb);
                } else {
                    c += 1;
                }
            }
        }
    }

    /// Text shortened with "..." to fit a width.
    fn fit(s: &str, width: f32) -> String {
        if Ui::text_width(s) <= width {
            return s.to_string();
        }
        let mut t: String = s.to_string();
        while !t.is_empty() && Ui::text_width(&t) + Ui::text_width("...") > width {
            t.pop();
        }
        t + "..."
    }
}

/// The wrist panel: a title line (the selected or pointed-at part) and the buttons.
/// `note`: a note's text, shown in place of the voice lines.
fn build_panel(title: &str, subtitle: &str, st: &PanelState, lay: &panel::Layout, note: Option<&str>) -> Ui {
    let PanelState { hover, overlay, overlay_hover, .. } = *st;
    let mut ui = Ui::default();
    let (hw, hh) = (PANEL_W / 2.0, lay.h / 2.0);
    ui.quad(-hw, -hh, hw, hh, 0.0, [28, 34, 46]);
    let line = font::FONT_H as f32 * FONT_PX;
    // The overlay above the panel: the on-screen keyboard; or a title, then the key help or the models (the open one in green).
    if overlay != Overlay::Off {
        let models = voice::MODELS.lock().ok().and_then(|m| m.clone());
        let requests = voice::REQUESTS.lock().ok().and_then(|m| m.clone());
        let dims = voice::DIMS.lock().ok().and_then(|m| m.clone());
        let feats = voice::FEATURES.lock().ok().and_then(|m| m.clone());
        let count = match overlay {
            Overlay::Requests => requests.as_ref().map_or(0, |m| m.len()),
            Overlay::Dims => dims.as_ref().map_or(0, |m| m.len()),
            Overlay::Features => feats.as_ref().map_or(0, |m| m.len()),
            _ => models.as_ref().map_or(0, |m| m.len()),
        };
        let rows = overlay_rows(overlay, count);
        let (y0, h) = (hh + OV_GAP, overlay_h(overlay, rows));
        ui.quad(-hw, y0, hw, y0 + h, 0.0, [28, 34, 46]);
        let top = y0 + h - 0.004;
        let reply_text = if overlay == Overlay::Reply { reply_lines() } else { Vec::new() };
        let pages = ((reply_text.len() + REPLY_ROWS - 1) / REPLY_ROWS).max(1);
        let page = st.reply_page.min(pages - 1);
        let title = match overlay {
            Overlay::Keys => "Keyboard and mouse".to_string(),
            Overlay::Requests => "Saved requests (click to run, x removes)".to_string(),
            Overlay::Dims => {
                let total = dims.as_ref().map_or(0, |d| d.len());
                let (per, pages) = dims_paging(total);
                if pages > 1 {
                    let first = st.dims_page.min(pages - 1) * per;
                    format!("Dimensions {}-{} of {} (- or + changes one, no AI)", first + 1, (first + per).min(total), total)
                } else if !voice::dims_of().is_empty() {
                    format!("Dimensions of {} (- or + changes one, no AI)", voice::dims_of())
                } else {
                    "Dimensions (click - or + to change, no AI)".to_string()
                }
            }
            Overlay::Features => {
                let total = feats.as_ref().map_or(0, |d| d.len());
                let (per, pages) = dims_paging(total);
                if pages > 1 {
                    let first = st.feat_page.min(pages - 1) * per;
                    format!("Features {}-{} of {} (click one for its sizes, on/off at the right)", first + 1, (first + per).min(total), total)
                } else {
                    "Features (click one for its sizes, on/off at the right)".to_string()
                }
            }
            Overlay::Reply => format!("Reply, page {}/{} (click: next page, Esc: close)", page + 1, pages),
            _ => "Open a model on the Mac (trigger / click)".to_string(),
        };
        ui.text(&Ui::fit(&title, PANEL_W - 0.02), -hw + 0.01, top, 0.001, [255, 255, 255]);
        let row_top = |i: usize| top - (i + 1) as f32 * ov_pitch(overlay);
        match overlay {
            Overlay::Keys => {
                for (i, s) in KEY_HELP.iter().enumerate() {
                    ui.text(&Ui::fit(s, PANEL_W - 0.02), -hw + 0.01, row_top(i) - 0.001, 0.001, [200, 215, 235]);
                }
            }
            Overlay::Reply => {
                if reply_text.is_empty() {
                    ui.text("Nothing yet: make a request first", -hw + 0.01, row_top(0) - 0.001, 0.001, [150, 170, 190]);
                }
                for (i, (l, c)) in reply_text.iter().skip(page * REPLY_ROWS).take(REPLY_ROWS).enumerate() {
                    ui.text(l, -hw + 0.01, row_top(i) - 0.001, 0.001, *c);
                }
            }
            Overlay::Requests => {
                // The strip at the bottom: the full words of the request pointed at (the rows cut them short).
                let strip = ov_strip(overlay);
                ui.quad(-hw + 0.004, y0 + strip - 0.0015, hw - 0.004, y0 + strip - 0.0005, 0.0004, [60, 72, 92]);
                let full = requests.as_ref().and_then(|l| overlay_hover.and_then(|i| l.get(i))).map(|r| r.text.clone());
                match full {
                    Some(t) => {
                        for (k, l) in Ui::wrap(&t, PANEL_W - 0.02, STRIP_LINES).into_iter().enumerate() {
                            ui.text(&l, -hw + 0.01, y0 + strip - 0.004 - k as f32 * 0.0138, 0.001, [255, 255, 255]);
                        }
                    }
                    None => ui.text("Point at a request to read it in full", -hw + 0.01, y0 + strip - 0.004, 0.001, [110, 125, 145]),
                }
                match &requests {
                None => ui.text("Asking the Mac...", -hw + 0.01, row_top(0) - 0.001, 0.001, [150, 170, 190]),
                Some(list) if list.is_empty() => ui.text("None yet: make a request, then press Keep", -hw + 0.01, row_top(0) - 0.001, 0.001, [150, 170, 190]),
                Some(list) => {
                    for (i, r) in list.iter().take(MODELS_SHOWN).enumerate() {
                        let y = row_top(i);
                        if overlay_hover == Some(i) {
                            ui.quad(-hw + 0.004, y - OV_PICK_HIT, hw - 0.004, y + 0.0005, 0.0005, [70, 150, 220]);
                            // The x at the right end removes it.
                            ui.quad(hw - 0.025, y - OV_PICK_HIT, hw - 0.004, y + 0.0005, 0.0006, [150, 60, 60]);
                        }
                        let colour = if r.quick { [140, 240, 150] } else { [235, 240, 250] };
                        ui.text_px(&Ui::fit(&r.text, (PANEL_W - 0.045) * FONT_PX / OV_PICK_FP), -hw + 0.01, y - 0.001, 0.001, colour, OV_PICK_FP);
                        ui.text_px("x", hw - 0.0165, y - 0.001, 0.001, [235, 240, 250], OV_PICK_FP);
                    }
                }
                }
            }
            Overlay::Features => match &feats {
                None => ui.text("Asking the Mac...", -hw + 0.01, row_top(0) - 0.001, 0.001, [150, 170, 190]),
                Some(list) if list.is_empty() => ui.text("This design has no features (an imported model)", -hw + 0.01, row_top(0) - 0.001, 0.001, [150, 170, 190]),
                Some(list) => {
                    let (per, pages) = dims_paging(list.len());
                    let page = st.feat_page.min(pages - 1);
                    if pages > 1 {
                        let y = row_top(per);
                        if overlay_hover == Some(per) {
                            ui.quad(-hw + 0.004, y - OV_PICK_HIT, hw - 0.004, y + 0.0005, 0.0005, [70, 150, 220]);
                        }
                        ui.text_px(&format!("Page {} of {}: click here for the next", page + 1, pages), -hw + 0.01, y - 0.001, 0.001, [150, 170, 190], OV_PICK_FP);
                    }
                    for (i, f) in list.iter().skip(page * per).take(per).enumerate() {
                        let y = row_top(i);
                        if overlay_hover == Some(i) {
                            ui.quad(-hw + 0.004, y - OV_PICK_HIT, hw - 0.004, y + 0.0005, 0.0005, [70, 150, 220]);
                        }
                        // The target at the right end switches the feature off or on; it says which it is now.
                        let (word, colour) = if f.off { ("off", [150, 60, 60]) } else { ("on", [40, 110, 70]) };
                        ui.quad(hw - DIM_ZONE * 2.0, y - OV_PICK_HIT, hw - 0.004, y + 0.0005, 0.0006, colour);
                        ui.text_px(word, hw - DIM_ZONE * 1.4, y - 0.001, 0.001, [235, 240, 250], OV_PICK_FP);
                        let kind = if f.error.is_empty() { f.kind.clone() } else { format!("{} (failing)", f.kind) };
                        let kw = Ui::text_width_px(&kind, OV_PICK_FP);
                        let right = hw - DIM_ZONE * 2.0 - 0.006;
                        let name_colour = if f.off { [110, 125, 145] } else if f.error.is_empty() { [235, 240, 250] } else { [240, 150, 80] };
                        ui.text_px(&Ui::fit(&f.name, (right + hw - 0.02 - kw) * FONT_PX / OV_PICK_FP), -hw + 0.01, y - 0.001, 0.001, name_colour, OV_PICK_FP);
                        ui.text_px(&kind, right - kw, y - 0.001, 0.001, [150, 170, 190], OV_PICK_FP);
                    }
                }
            },
            Overlay::Dims => match &dims {
                None => ui.text("Asking the Mac...", -hw + 0.01, row_top(0) - 0.001, 0.001, [150, 170, 190]),
                Some(list) if list.is_empty() => ui.text("Nothing to set: the design has no parameters or features", -hw + 0.01, row_top(0) - 0.001, 0.001, [150, 170, 190]),
                Some(list) => {
                    let (per, pages) = dims_paging(list.len());
                    let page = st.dims_page.min(pages - 1);
                    if pages > 1 {
                        // The last row turns the page.
                        let y = row_top(per);
                        if overlay_hover == Some(per) {
                            ui.quad(-hw + 0.004, y - OV_PICK_HIT, hw - 0.004, y + 0.0005, 0.0005, [70, 150, 220]);
                        }
                        ui.text_px(&format!("Page {} of {}: click here for the next", page + 1, pages), -hw + 0.01, y - 0.001, 0.001, [150, 170, 190], OV_PICK_FP);
                    }
                    for (i, d) in list.iter().skip(page * per).take(per).enumerate() {
                        let y = row_top(i);
                        if overlay_hover == Some(i) {
                            ui.quad(-hw + 0.004, y - OV_PICK_HIT, hw - 0.004, y + 0.0005, 0.0005, [70, 150, 220]);
                        }
                        // The two targets at the right end: one step down, one step up.
                        ui.quad(hw - DIM_ZONE * 2.0, y - OV_PICK_HIT, hw - DIM_ZONE - 0.002, y + 0.0005, 0.0006, [60, 72, 92]);
                        ui.quad(hw - DIM_ZONE, y - OV_PICK_HIT, hw - 0.004, y + 0.0005, 0.0006, [60, 72, 92]);
                        ui.text_px("-", hw - DIM_ZONE * 1.5 - 0.003, y - 0.001, 0.001, [235, 240, 250], OV_PICK_FP);
                        ui.text_px("+", hw - DIM_ZONE * 0.5 - 0.004, y - 0.001, 0.001, [235, 240, 250], OV_PICK_FP);
                        let value = format!("{} {}", fmt_dim(d.value), d.unit);
                        let vw = Ui::text_width_px(&value, OV_PICK_FP);
                        let right = hw - DIM_ZONE * 2.0 - 0.006;
                        ui.text_px(&Ui::fit(&d.label, (right + hw - 0.02 - vw) * FONT_PX / OV_PICK_FP), -hw + 0.01, y - 0.001, 0.001, [235, 240, 250], OV_PICK_FP);
                        ui.text_px(&value, right - vw, y - 0.001, 0.001, [140, 240, 150], OV_PICK_FP);
                    }
                }
            },
            _ => match &models {
                None => ui.text("Asking the Mac...", -hw + 0.01, row_top(0) - 0.001, 0.001, [150, 170, 190]),
                Some(list) if list.is_empty() => ui.text("No models found on the Mac", -hw + 0.01, row_top(0) - 0.001, 0.001, [150, 170, 190]),
                Some(list) => {
                    for (i, m) in list.iter().take(MODELS_SHOWN).enumerate() {
                        let y = row_top(i);
                        if overlay_hover == Some(i) {
                            ui.quad(-hw + 0.004, y - OV_PICK_HIT, hw - 0.004, y + 0.0005, 0.0005, [70, 150, 220]);
                        }
                        let size = format!("{:.1} MB", m.mb);
                        let sw = Ui::text_width_px(&size, OV_PICK_FP);
                        let colour = if m.current { [140, 240, 150] } else { [235, 240, 250] };
                        ui.text_px(&Ui::fit(&m.name, (PANEL_W - 0.03 - sw) * FONT_PX / OV_PICK_FP), -hw + 0.01, y - 0.001, 0.001, colour, OV_PICK_FP);
                        ui.text_px(&size, hw - 0.01 - sw, y - 0.001, 0.001, [150, 170, 190], OV_PICK_FP);
                    }
                }
            },
        }
    }
    // The title; the second line is the name and shortcut of the button pointed at, else the subtitle.
    let hint = hover.map(|b| {
        let key = panel::shortcut(b);
        // Undo and Redo say what they would do.
        let what = match b {
            Button::Undo => Some(("Undo", st.undo_what, voice::history().0)),
            Button::Redo => Some(("Redo", st.redo_what, voice::history().1)),
            _ => None,
        };
        if let Some((verb, what, from_mac)) = what.filter(|w| !w.1.is_empty()) {
            let what = if what == "model" && !from_mac.is_empty() { from_mac } else if what == "model" { "the last change".to_string() } else { what.to_string() };
            return format!("{verb}: {what}  [{key}]");
        }
        if key.is_empty() {
            label(b, st, false).to_string()
        } else {
            format!("{}  [{key}]", label(b, st, false))
        }
    });
    // The API balance left (an estimate from what this server has spent), at the right of the title.
    let balance = voice::spend();
    let bal_w = if balance.is_empty() { 0.0 } else { Ui::text_width(&format!("~${balance}")) + 0.006 };
    if !balance.is_empty() {
        ui.text(&format!("~${balance}"), hw - 0.01 - bal_w + 0.006, hh - 0.006, 0.001, [120, 210, 150]);
    }
    ui.text(&Ui::fit(title, PANEL_W - 0.02 - bal_w), -hw + 0.01, hh - 0.006, 0.001, [255, 255, 255]);
    match &hint {
        Some(h) => ui.text(&Ui::fit(h, PANEL_W - 0.02), -hw + 0.01, hh - 0.006 - line, 0.001, [190, 215, 245]),
        None => ui.text(&Ui::fit(subtitle, PANEL_W - 0.02), -hw + 0.01, hh - 0.006 - line, 0.001, [150, 170, 190]),
    }
    for s in &lay.slots {
        let b = s.b;
        if b == Button::Reply {
            // The voice lines: no look of their own, a faint tint when pointed at.
            if hover == Some(b) {
                ui.quad(s.x, s.top - s.h, s.x + s.w, s.top, 0.0004, [36, 44, 60]);
            }
            continue;
        }
        let (en, is_tab) = (enabled(b, st), matches!(b, Button::Tab(_)));
        // A tool in use (cut being moved, measuring) is shown in orange; the open tab in blue.
        let bg = match (hover == Some(b), en, active(b, st)) {
            (true, true, _) => [70, 150, 220],
            (_, true, true) if is_tab => [58, 100, 160],
            (_, true, true) => [170, 95, 35],
            (_, true, _) if b == Button::ConfirmGo => [40, 110, 70],
            (_, true, _) if b == Button::ConfirmStop || b == Button::CancelVoice => [120, 50, 50],
            (_, true, _) => [55, 66, 86],
            (_, false, _) => [40, 46, 58],
        };
        ui.quad(s.x, s.top - s.h, s.x + s.w, s.top, 0.0005, bg);
        let fg = if en { [235, 240, 250] } else { [110, 118, 130] };
        let (icon, isz) = (panel::icon_of(b), 12.0 * lay.icon_px);
        let name = label(b, st, true);
        if is_tab {
            // Icon on top, the name under it, both centred.
            ui.icon(icon, s.x + (s.w - isz) / 2.0, s.top - 0.003, lay.icon_px, 0.001, fg);
            ui.text(name, s.x + (s.w - Ui::text_width(name)) / 2.0, s.top - 0.0035 - isz, 0.001, fg);
        } else if lay.labels {
            ui.icon(icon, s.x + 0.004, s.top - (s.h - isz) / 2.0, lay.icon_px, 0.001, fg);
            ui.text(name, s.x + 0.008 + isz, s.top - (s.h - line) / 2.0, 0.001, fg);
        } else {
            ui.icon(icon, s.x + (s.w - isz) / 2.0, s.top - (s.h - isz) / 2.0, lay.icon_px, 0.001, fg);
        }
    }
    // Voice: the state (listening, thinking), else what was heard; then the AI's reply.
    let (state, heard, reply) = voice::STATUS.lock().map(|s| s.clone()).unwrap_or_default();
    let mut y = lay.voice_top;
    // Three lines: live text while speaking; the state and what was heard while the Mac works;
    // afterwards what was heard and the reply.
    let w = PANEL_W - 0.02;
    let (yellow, grey, white) = ([255, 200, 90], [150, 170, 190], [255, 255, 255]);
    let mut lines: Vec<(String, [u8; 3])> = Vec::new();
    let typed = voice::typed();
    if !typed.is_empty() {
        // Being typed: the end of it, with a cursor.
        let prefix = format!("{}> ", voice::typed_target());
        let mut tail = typed.clone();
        let mut t = format!("{prefix}{tail}_");
        while Ui::text_width(&t) > w * 3.0 - 0.01 && !tail.is_empty() {
            tail.remove(0);
            t = format!("{prefix}{tail}_");
        }
        lines.extend(Ui::wrap(&t, w, 3).into_iter().map(|l| (l, yellow)));
    } else if let Some(n) = note {
        lines.extend(Ui::wrap(n, w, 3).into_iter().map(|l| (l, [235, 215, 255])));
    } else if state.starts_with("Hearing: ") {
        lines.extend(Ui::wrap(&state, w, 3).into_iter().map(|l| (l, yellow)));
    } else if !state.is_empty() {
        lines.push((state, yellow));
        if !heard.is_empty() && !voice::confirm_pending() {
            lines.extend(Ui::wrap(&format!("You: {heard}"), w, 2).into_iter().map(|l| (l, grey)));
        }
    } else if !heard.is_empty() || !reply.is_empty() {
        if !heard.is_empty() {
            lines.push((Ui::fit(&format!("You: {heard}"), w), grey));
        }
        let room = 3 - lines.len();
        lines.extend(Ui::wrap(&reply, w, room).into_iter().map(|l| (l, white)));
    } else {
        lines.push(("Hold the left trigger and speak, or type, to change the model".to_string(), [110, 125, 145]));
    }
    for (l, c) in lines.into_iter().take(3) {
        ui.text(&Ui::fit(&l, w), -hw + 0.01, y, 0.001, c);
        y -= line;
    }
    ui
}

/// The on-screen keyboard: keys under a bar that shows what is typed (and is the handle to move it by).
/// The full text of the last exchange wrapped to the panel's width: what was heard, then the reply.
fn reply_lines() -> Vec<(String, [u8; 3])> {
    let (_, heard, reply) = voice::STATUS.lock().map(|s| s.clone()).unwrap_or_default();
    let w = PANEL_W - 0.02;
    let mut lines: Vec<(String, [u8; 3])> = Vec::new();
    if !heard.is_empty() {
        lines.extend(Ui::wrap(&format!("You: {heard}"), w, 100).into_iter().map(|l| (l, [150, 170, 190])));
    }
    if !reply.is_empty() {
        lines.extend(Ui::wrap(&reply, w, 100).into_iter().map(|l| (l, [255, 255, 255])));
    }
    lines
}

fn build_keyboard(lay: &panel::KbLayout, hover: Option<panel::KbHit>, shift: bool, typed: &str) -> Ui {
    let mut ui = Ui::default();
    let (hw, hh) = (lay.w / 2.0, lay.h / 2.0);
    ui.quad(-hw, -hh, hw, hh, 0.0, [28, 34, 46]);
    let bar_bg = if hover == Some(panel::KbHit::Bar) { [52, 70, 100] } else { [40, 50, 70] };
    ui.quad(-hw, hh - lay.bar_h, hw, hh, 0.0003, bar_bg);
    let bar_px = 0.0004;
    let room = lay.w - 0.07 - 0.016;
    let (shown, colour) = if typed.is_empty() {
        ("Keyboard: grab this bar to move it".to_string(), [150, 170, 190])
    } else {
        let prefix = format!("{}> ", voice::typed_target());
        let mut tail = typed.to_string();
        let mut t = format!("{prefix}{tail}_");
        while Ui::text_width_px(&t, bar_px) > room && !tail.is_empty() {
            tail.remove(0);
            t = format!("{prefix}{tail}_");
        }
        (t, [255, 200, 90])
    };
    ui.text_px(&shown, -hw + 0.008, hh - (lay.bar_h - font::FONT_H as f32 * bar_px) / 2.0, 0.001, colour, bar_px);
    for (i, k) in lay.keys.iter().enumerate() {
        let bg = match (hover == Some(panel::KbHit::Key(i)), k.act) {
            (true, _) => [70, 150, 220],
            (_, panel::KbAct::Shift) if shift => [170, 95, 35],
            (_, panel::KbAct::Send) => [40, 110, 70],
            (_, panel::KbAct::Close) => [120, 50, 50],
            _ => [55, 66, 86],
        };
        ui.quad(k.x, k.top - k.h, k.x + k.w, k.top, 0.0005, bg);
        let name = match k.act {
            panel::KbAct::Char(c) => (if shift { c.to_ascii_uppercase() } else { c }).to_string(),
            panel::KbAct::Back => "<-".to_string(),
            panel::KbAct::Shift => "^".to_string(),
            panel::KbAct::Clear => "Clear".to_string(),
            panel::KbAct::Space => "space".to_string(),
            panel::KbAct::Send => "Send".to_string(),
            panel::KbAct::Close => "Close".to_string(),
        };
        let fp = if matches!(k.act, panel::KbAct::Char(_) | panel::KbAct::Shift) { 0.0005 } else { 0.0004 };
        let (tw, th) = (Ui::text_width_px(&name, fp), font::FONT_H as f32 * fp);
        ui.text_px(&name, k.x + (k.w - tw) / 2.0, k.top - (k.h - th) / 2.0, 0.001, [235, 240, 250], fp);
    }
    ui
}

/// A flat panel centred at p, upright and turned to face the head: the columns are right, up, towards the head, p.
fn billboard(p: [f32; 3], head: [f32; 3]) -> M4 {
    let to_head = sub(head, p);
    let len = dot(to_head, to_head).sqrt().max(1e-4);
    let f = to_head.map(|x| x / len);
    let r0 = cross([0.0, 1.0, 0.0], f);
    let rl = dot(r0, r0).sqrt().max(1e-4);
    let r = r0.map(|x| x / rl);
    let u = cross(f, r);
    [r[0], r[1], r[2], 0.0, u[0], u[1], u[2], 0.0, f[0], f[1], f[2], 0.0, p[0], p[1], p[2], 1.0]
}

/// General 4x4 inverse (column-major); identity if singular.
fn invert(m: &M4) -> M4 {
    let mut inv = [0.0f32; 16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if det.abs() < 1e-20 {
        return identity();
    }
    inv.map(|x| x / det)
}

/// m * (x, y, z, w), xyz of the result.
fn transform(m: &M4, p: [f32; 3], w: f32) -> [f32; 3] {
    [0, 1, 2].map(|r| m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r] * w)
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = dot(v, v).sqrt().max(1e-12);
    v.map(|x| x / l)
}

/// Two unit vectors at right angles to `d` (unit) and to each other.
fn perpendiculars(d: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let other = if d[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let a = normalize(cross(d, other));
    (a, cross(d, a))
}

/// The matrix that turns the ray box (ray_box_mesh) into a bar from p to q, `thick` wide.
fn bar(p: [f32; 3], q: [f32; 3], thick: f32) -> M4 {
    let d = sub(q, p);
    let (a, b) = perpendiculars(normalize(d));
    // The box runs from z = 0 to z = -1, so -d as the z column puts its far end at q.
    [a[0] * thick, a[1] * thick, a[2] * thick, 0.0, b[0] * thick, b[1] * thick, b[2] * thick, 0.0, -d[0], -d[1], -d[2], 0.0, p[0], p[1], p[2], 1.0]
}

/// A unit box from (-0.5, -0.5, -1) to (0.5, 0.5, 0) as 28-byte vertices + indices: the pointer ray
/// (scaled to thickness and length, pointing down -Z like the controller's aim pose).
fn ray_box_mesh() -> (Vec<u8>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
    ];
    for (n, u, v) in faces {
        let base = (verts.len() / VERTEX_SIZE) as u32;
        for (su, sv) in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
            let p = [0, 1, 2].map(|k| 0.5 * n[k] + su * u[k] + sv * v[k]);
            // z from (-0.5..0.5) to (-1..0)
            for x in [p[0], p[1], p[2] - 0.5, n[0], n[1], n[2]] {
                verts.extend_from_slice(&x.to_le_bytes());
            }
            verts.extend_from_slice(&[90, 220, 255, 255]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, indices)
}

const VERTEX_SIZE: usize = 28;

/// The folders the model files are looked for in.
fn data_dirs(app: &Host) -> Vec<std::path::PathBuf> {
    [app.external_data_path(), app.internal_data_path()].into_iter().flatten().collect()
}

/// Parses the .scm format (made on the Mac by tools/obj_to_scm.py from a SolidCraft OBJ export, with colours).
fn parse_scm(b: &[u8]) -> Result<Mesh, String> {
    if b.len() < 36 || &b[0..4] != b"SCM1" {
        return Err("not an SCM1 file".into());
    }
    let u = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let nv = u(4) as usize;
    let ni = u(8) as usize;
    let mut bounds = [0.0f32; 6];
    for (k, v) in bounds.iter_mut().enumerate() {
        *v = f32::from_bits(u(12 + k * 4));
    }
    let vstart = 36;
    let istart = vstart + nv * VERTEX_SIZE;
    if ni % 3 != 0 || istart + ni * 4 > b.len() {
        return Err("truncated".into());
    }
    let indices: Vec<u32> = b[istart..istart + ni * 4].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    if indices.iter().any(|&i| i as usize >= nv) {
        return Err("index out of range".into());
    }
    // Optional body table after the indices.
    let mut bodies = Vec::new();
    let mut o = istart + ni * 4;
    if b.len() >= o + 8 && &b[o..o + 4] == b"BODY" {
        let n = u(o + 4) as usize;
        o += 8;
        for _ in 0..n {
            if b.len() < o + 10 {
                return Err("truncated body table".into());
            }
            let (first, count) = (u(o), u(o + 4));
            let len = u16::from_le_bytes([b[o + 8], b[o + 9]]) as usize;
            o += 10;
            if b.len() < o + len || first as usize + count as usize > ni {
                return Err("bad body table".into());
            }
            bodies.push(Body { name: String::from_utf8_lossy(&b[o..o + len]).into_owned(), first, count, aabb: [0.0; 6] });
            o += len;
        }
    }
    if bodies.is_empty() {
        bodies.push(Body { name: "model".into(), first: 0, count: ni as u32, aabb: [0.0; 6] });
    }
    Ok(Mesh { verts: b[vstart..istart].to_vec(), indices, bounds, bodies, same_model: false })
}

/// Reads `sea.scm` from the app's files folder (pushed by adb, or saved from the last Wi-Fi download).
fn load_scm(app: &Host) -> Option<Mesh> {
    for d in data_dirs(app) {
        let path = d.join("sea.scm");
        let Ok(b) = std::fs::read(&path) else { continue };
        match parse_scm(&b) {
            Ok(m) => {
                log::info!("loaded {} ({} vertices, {} triangles, {} MB)", path.display(), m.verts.len() / VERTEX_SIZE, m.indices.len() / 3, b.len() / 1_000_000);
                return Some(m);
            }
            Err(e) => log::error!("{}: {e}", path.display()),
        }
    }
    None
}

/// Model scale and centring: Z-up model units to Y-up metres, longest side 1.6 m, centred on the origin.
fn model_inner(bounds: &[f32; 6]) -> M4 {
    let size = longest_side(bounds);
    let c = |k: usize| (bounds[k] + bounds[k + 3]) * 0.5;
    mul(&mul(&scale(1.6 / size), &z_up_to_y_up()), &translate(-c(0), -c(1), -c(2)))
}

/// The longest side of a box of bounds, in model units (never zero).
fn longest_side(bounds: &[f32; 6]) -> f32 {
    (0..3).map(|k| bounds[k + 3] - bounds[k]).fold(1e-6f32, f32::max)
}

const BEACON_PORT: u16 = 47800;

/// Wi-Fi loading: listens for the Mac's beacon (`tools/serve.py` broadcasts "SCVIEW1 <port> <version>"
/// on UDP 47800 every second), and when the version changes downloads http://<mac>:<port>/model.scm,
/// keeps a copy as sea.scm for the next start, and hands the mesh to the render loop.
///
/// One thread for the whole process: reopening the app runs `android_main` again in the same process,
/// and a second thread could not bind the port. Each run hands the thread its channel in `NET_TX`.
static NET_TX: std::sync::Mutex<Option<std::sync::mpsc::Sender<Mesh>>> = std::sync::Mutex::new(None);
/// Set by a new run: download the current model again even if this thread already had it.
static NET_RESET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Download everything from the Mac again (after the access code was typed).
pub fn net_reset() {
    NET_RESET.store(true, std::sync::atomic::Ordering::SeqCst);
}

fn start_net(tx: std::sync::mpsc::Sender<Mesh>, save_dir: Option<std::path::PathBuf>) {
    voice::init_token(save_dir.as_ref().map(|d| d.join("token.txt")));
    if let Ok(mut t) = NET_TX.lock() {
        *t = Some(tx);
    }
    NET_RESET.store(true, std::sync::atomic::Ordering::SeqCst);
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::thread::spawn(move || net_thread(save_dir));
    });
}

/// GET a path from the Mac: the body of a 200 reply.
fn http_get(addr: std::net::SocketAddr, path: &str, timeout_s: u64) -> std::io::Result<Vec<u8>> {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(5))?;
    s.set_read_timeout(Some(Duration::from_secs(timeout_s)))?;
    s.write_all(format!("GET {path} HTTP/1.0\r\nX-Token: {}\r\n\r\n", voice::token()).as_bytes())?;
    let mut all = Vec::new();
    s.read_to_end(&mut all)?;
    let head = all.windows(4).position(|w| w == b"\r\n\r\n").ok_or_else(|| std::io::Error::other("no HTTP header"))?;
    if all.starts_with(b"HTTP/1.0 403") || all.starts_with(b"HTTP/1.1 403") {
        voice::token_refused();
    }
    if !all.starts_with(b"HTTP/1.0 200") && !all.starts_with(b"HTTP/1.1 200") {
        return Err(std::io::Error::other(String::from_utf8_lossy(&all[..head.min(80)]).to_string()));
    }
    Ok(all.split_off(head + 4))
}

fn net_thread(save_dir: Option<std::path::PathBuf>) {
    let sock = match std::net::UdpSocket::bind(("0.0.0.0", BEACON_PORT)) {
        Ok(s) => s,
        Err(e) => {
            log::error!("beacon socket: {e}");
            return;
        }
    };
    log::info!("listening for the Mac on UDP {BEACON_PORT}");
    let mut loaded: Option<(std::net::IpAddr, String)> = None;
    let mut shown_model: Option<(std::net::IpAddr, String)> = None;
    // The notes version (the beacon's 4th field) last fetched.
    let mut notes_loaded: Option<(std::net::IpAddr, String)> = None;
    let mut buf = [0u8; 256];
    loop {
        let Ok((n, from)) = sock.recv_from(&mut buf) else { continue };
        if NET_RESET.swap(false, std::sync::atomic::Ordering::SeqCst) {
            loaded = None;
            notes_loaded = None;
        }
        let msg = String::from_utf8_lossy(&buf[..n]).to_string();
        let mut parts = msg.split_whitespace();
        if parts.next() != Some("SCVIEW1") {
            continue;
        }
        let (Some(Ok(port)), Some(version)) = (parts.next().map(str::parse::<u16>), parts.next()) else { continue };
        if let Ok(mut m) = voice::MAC.lock() {
            *m = Some(std::net::SocketAddr::new(from.ip(), port));
        }
        if let Some(cver) = parts.next() {
            let key = (from.ip(), cver.to_string());
            if notes_loaded.as_ref() != Some(&key) {
                match http_get(std::net::SocketAddr::new(from.ip(), port), "/comments.txt", 10) {
                    Ok(b) => {
                        let list = voice::parse_comments(&String::from_utf8_lossy(&b));
                        log::info!("notes version {cver}: {} notes", list.len());
                        if let Ok(mut c) = voice::COMMENTS.lock() {
                            *c = Some(list);
                        }
                        notes_loaded = Some(key);
                    }
                    Err(e) => log::error!("notes download failed: {e}"),
                }
            }
        }
        if let Some(b) = parts.next() {
            voice::set_spend(b.strip_prefix('b').unwrap_or(""));
        }
        // The 6th field: p1 while the model on show is a preview of a change not made yet.
        if let Some(p) = parts.next() {
            voice::set_preview(p == "p1");
        }
        // The 7th field: m and a name for the model, the same through all its edits ("-" or absent: not said).
        let model_id = parts.next().and_then(|m| m.strip_prefix('m')).filter(|m| *m != "-").map(|m| (from.ip(), m.to_string()));
        let key = (from.ip(), version.to_string());
        if loaded.as_ref() == Some(&key) {
            continue;
        }
        log::info!("Mac at {}:{port} has model version {version}; downloading", from.ip());
        let t = Instant::now();
        let body = http_get(std::net::SocketAddr::new(from.ip(), port), "/model.scm", 30);
        match body.map_err(|e| e.to_string()).and_then(|b| parse_scm(&b).map(|m| (m, b))) {
            Ok((mut mesh, bytes)) => {
                mesh.same_model = model_id.is_some() && model_id == shown_model;
                log::info!("downloaded {} triangles ({} MB) in {} ms", mesh.indices.len() / 3, bytes.len() / 1_000_000, t.elapsed().as_millis());
                if let Some(d) = &save_dir {
                    let tmp = d.join("sea.scm.part");
                    if std::fs::write(&tmp, &bytes).and_then(|_| std::fs::rename(&tmp, d.join("sea.scm"))).is_err() {
                        log::error!("could not save the model in {}", d.display());
                    }
                }
                // Only count it as loaded when a render loop took it (none while the app is reopening).
                let sent = NET_TX.lock().ok().and_then(|t| t.as_ref().map(|tx| tx.send(mesh).is_ok())).unwrap_or(false);
                if sent {
                    shown_model = model_id;
                    loaded = Some(key);
                } else {
                    log::info!("no render loop to take the model yet; will fetch it again");
                }
            }
            Err(e) => {
                log::error!("download failed: {e}");
                std::thread::sleep(Duration::from_secs(2));
            }
        }
    }
}

/// An STL as a mesh: three vertices per triangle, the facet normal, light grey.
fn stl_mesh(stl: &[u8], tris: usize) -> Mesh {
    let mut verts = Vec::with_capacity(tris * 3 * VERTEX_SIZE);
    for i in 0..tris {
        let r = &stl[84 + i * 50..84 + i * 50 + 48];
        for k in 0..3 {
            verts.extend_from_slice(&r[12 + k * 12..24 + k * 12]);
            verts.extend_from_slice(&r[0..12]);
            verts.extend_from_slice(&[215, 220, 230, 255]);
        }
    }
    let mut bounds = [f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN];
    for v in verts.chunks_exact(VERTEX_SIZE) {
        for k in 0..3 {
            let x = f32::from_le_bytes([v[k * 4], v[k * 4 + 1], v[k * 4 + 2], v[k * 4 + 3]]);
            bounds[k] = bounds[k].min(x);
            bounds[k + 3] = bounds[k + 3].max(x);
        }
    }
    let bodies = vec![Body { name: "model".into(), first: 0, count: (tris * 3) as u32, aabb: [0.0; 6] }];
    Mesh { verts, indices: (0..(tris * 3) as u32).collect(), bounds, bodies, same_model: false }
}

/// Reads a binary STL: returns the file bytes (80-byte header, u32 count, 50-byte records) and the triangle count.
fn load_stl(app: &Host) -> Option<(Vec<u8>, usize)> {
    let mut dirs = Vec::new();
    if let Some(p) = app.external_data_path() {
        dirs.push(p);
    }
    if let Some(p) = app.internal_data_path() {
        dirs.push(p);
    }
    for d in dirs {
        let path = d.join("sea.stl");
        log::info!("looking for {}", path.display());
        if let Ok(bytes) = std::fs::read(&path) {
            if bytes.len() >= 84 {
                let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
                if 84 + n * 50 <= bytes.len() {
                    log::info!("loaded {} ({} triangles, {} MB)", path.display(), n, bytes.len() / 1_000_000);
                    return Some((bytes, n));
                }
            }
        }
    }
    None
}

/// Logs to logcat (tag SCVIEW) and also appends to scview.log in the app's files folder, because the
/// runtime fills logcat so fast that events from a few minutes ago are gone. Pull it with
/// `adb pull /sdcard/Android/data/org.scview.viewer/files/scview.log`.
struct FileLogger {
    #[cfg(target_os = "android")]
    android: android_logger::AndroidLogger,
    file: std::sync::Mutex<Option<std::fs::File>>,
}

impl log::Log for FileLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        #[cfg(target_os = "android")]
        return self.android.enabled(m);
        #[cfg(not(target_os = "android"))]
        return m.level() <= log::Level::Info;
    }
    fn log(&self, r: &log::Record) {
        #[cfg(target_os = "android")]
        self.android.log(r);
        #[cfg(not(target_os = "android"))]
        if self.enabled(r.metadata()) {
            eprintln!("[{}] {}", r.level(), r.args());
        }
        if let Ok(mut guard) = self.file.lock() {
            if let Some(f) = guard.as_mut() {
                use std::io::Write;
                let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
                let _ = writeln!(f, "{:02}:{:02}:{:02} {}", s / 3600 % 24, s / 60 % 60, s % 60, r.args());
            }
        }
    }
    fn flush(&self) {}
}

/// The Bluetooth mouse, gathered from this frame's input events.
#[derive(Default)]
struct Mouse {
    /// Movement this frame (counts) and wheel notches (up is positive).
    dx: f32,
    dy: f32,
    wheel: f32,
    left: bool,
    right: bool,
    middle: bool,
    /// Moves arrive as relative events: the app has the mouse.
    captured: bool,
    /// False after Esc gave the mouse back, until the next click.
    want_capture: bool,
    asked: Option<Instant>,
}

/// A key pressed on the keyboard: its code, the character it types, and Ctrl / Shift.
struct KeyPress {
    code: Keycode,
    ch: Option<char>,
    ctrl: bool,
    shift: bool,
}

/// Takes one input event into the mouse state or the key list.
#[cfg(target_os = "android")]
fn read_input(app: &Host, ev: &android_activity::input::InputEvent, mouse: &mut Mouse, keys: &mut Vec<KeyPress>) {
    use android_activity::input::{Axis, InputEvent, KeyAction, KeyMapChar, MotionAction, Source};
    match ev {
        InputEvent::KeyEvent(k) if k.action() == KeyAction::Down => {
            let ch = match app.device_key_character_map(k.device_id()).ok().and_then(|map| map.get(k.key_code(), k.meta_state()).ok()) {
                Some(KeyMapChar::Unicode(c)) => Some(c),
                _ => None,
            };
            keys.push(KeyPress { code: k.key_code(), ch, ctrl: k.meta_state().ctrl_on(), shift: k.meta_state().shift_on() });
        }
        InputEvent::MotionEvent(m) if matches!(m.source(), Source::Mouse | Source::MouseRelative) => {
            let rel = matches!(m.source(), Source::MouseRelative);
            let p = m.pointer_at_index(0);
            match m.action() {
                MotionAction::Scroll => mouse.wheel += p.axis_value(Axis::Vscroll),
                MotionAction::Move | MotionAction::HoverMove => {
                    // Captured, x and y are the move itself; otherwise the relative axes carry it.
                    let (dx, dy) = if rel { (p.x(), p.y()) } else { (p.axis_value(Axis::RelativeX), p.axis_value(Axis::RelativeY)) };
                    mouse.dx += dx;
                    mouse.dy += dy;
                    if rel {
                        mouse.wheel += p.axis_value(Axis::Vscroll);
                    }
                }
                _ => {}
            }
            let b = m.button_state();
            if b.primary() && !mouse.left {
                mouse.want_capture = true;
            }
            (mouse.left, mouse.right, mouse.middle) = (b.primary(), b.secondary(), b.teriary());
            if rel && !mouse.captured {
                log::info!("mouse captured");
            }
            mouse.captured = rel;
        }
        _ => {}
    }
}

/// Asks Android for the permissions the app needs (the microphone, and the room scan) that it does not have yet.
#[cfg(not(target_os = "android"))]
fn request_permissions(_app: &Host) {}

#[cfg(target_os = "android")]
fn request_permissions(app: &Host) {
    use jni_sys::{jobject, jvalue, JNIEnv, JavaVM, JNI_OK};
    // SAFETY: plain JNI calls on the activity object android-activity gives us; local references
    // are dropped before returning, and any Java exception is cleared.
    unsafe {
        let vm = app.vm_as_ptr() as *mut JavaVM;
        let activity = app.activity_as_ptr() as jobject;
        if vm.is_null() || activity.is_null() {
            return;
        }
        let mut env: *mut JNIEnv = std::ptr::null_mut();
        if ((**vm).AttachCurrentThread.unwrap())(vm, &mut env as *mut *mut JNIEnv as *mut *mut std::ffi::c_void, std::ptr::null_mut()) != JNI_OK {
            return;
        }
        let e = &**env;
        let cls = (e.GetObjectClass.unwrap())(env, activity);
        let (n, sg) = (std::ffi::CString::new("checkSelfPermission").unwrap(), std::ffi::CString::new("(Ljava/lang/String;)I").unwrap());
        let check = (e.GetMethodID.unwrap())(env, cls, n.as_ptr(), sg.as_ptr());
        let mut missing = Vec::new();
        for name in ["android.permission.RECORD_AUDIO", "com.oculus.permission.USE_SCENE"] {
            let c = std::ffi::CString::new(name).unwrap();
            let js = (e.NewStringUTF.unwrap())(env, c.as_ptr());
            let granted = if check.is_null() {
                (e.ExceptionClear.unwrap())(env);
                true
            } else {
                let args = [jvalue { l: js }];
                let r = (e.CallIntMethodA.unwrap())(env, activity, check, args.as_ptr());
                if (e.ExceptionCheck.unwrap())(env) != 0 {
                    (e.ExceptionClear.unwrap())(env);
                }
                r == 0
            };
            if granted {
                (e.DeleteLocalRef.unwrap())(env, js);
            } else {
                missing.push(js);
            }
        }
        if !missing.is_empty() {
            let string_cls_name = std::ffi::CString::new("java/lang/String").unwrap();
            let string_cls = (e.FindClass.unwrap())(env, string_cls_name.as_ptr());
            let arr = (e.NewObjectArray.unwrap())(env, missing.len() as i32, string_cls, std::ptr::null_mut());
            for (i, js) in missing.iter().enumerate() {
                (e.SetObjectArrayElement.unwrap())(env, arr, i as i32, *js);
            }
            let (n, sg) = (std::ffi::CString::new("requestPermissions").unwrap(), std::ffi::CString::new("([Ljava/lang/String;I)V").unwrap());
            let req = (e.GetMethodID.unwrap())(env, cls, n.as_ptr(), sg.as_ptr());
            if !req.is_null() {
                let args = [jvalue { l: arr }, jvalue { i: 1 }];
                (e.CallVoidMethodA.unwrap())(env, activity, req, args.as_ptr());
                log::info!("permissions requested: {}", missing.len());
            }
            if (e.ExceptionCheck.unwrap())(env) != 0 {
                (e.ExceptionClear.unwrap())(env);
            }
            for js in missing {
                (e.DeleteLocalRef.unwrap())(env, js);
            }
            (e.DeleteLocalRef.unwrap())(env, arr);
            (e.DeleteLocalRef.unwrap())(env, string_cls);
        }
        (e.DeleteLocalRef.unwrap())(env, cls);
    }
}

/// Asks Android for the mouse (pointer capture, `on`) or gives it back. Captured, it cannot leave the
/// app's window and every move arrives as a relative event. Only works while the app has focus.
#[cfg(not(target_os = "android"))]
fn pointer_capture(_app: &Host, _on: bool) {}

#[cfg(target_os = "android")]
fn pointer_capture(app: &Host, on: bool) {
    use jni_sys::{jobject, jvalue, JNIEnv, JavaVM, JNI_OK};
    // SAFETY: plain JNI calls on the activity object android-activity gives us; local references
    // are dropped before returning, and any Java exception is cleared.
    unsafe {
        let vm = app.vm_as_ptr() as *mut JavaVM;
        let activity = app.activity_as_ptr() as jobject;
        if vm.is_null() || activity.is_null() {
            return;
        }
        let mut env: *mut JNIEnv = std::ptr::null_mut();
        if ((**vm).AttachCurrentThread.unwrap())(vm, &mut env as *mut *mut JNIEnv as *mut *mut std::ffi::c_void, std::ptr::null_mut()) != JNI_OK {
            log::warn!("pointer capture: no JNI env");
            return;
        }
        let e = &**env;
        let call = |obj: jobject, name: &str, sig: &str, object: bool| -> jobject {
            let cls = (e.GetObjectClass.unwrap())(env, obj);
            let (n, s) = (std::ffi::CString::new(name).unwrap(), std::ffi::CString::new(sig).unwrap());
            let mid = (e.GetMethodID.unwrap())(env, cls, n.as_ptr(), s.as_ptr());
            (e.DeleteLocalRef.unwrap())(env, cls);
            if mid.is_null() {
                (e.ExceptionClear.unwrap())(env);
                return std::ptr::null_mut();
            }
            let args: [jvalue; 0] = [];
            let r = if object {
                (e.CallObjectMethodA.unwrap())(env, obj, mid, args.as_ptr())
            } else {
                (e.CallVoidMethodA.unwrap())(env, obj, mid, args.as_ptr());
                std::ptr::null_mut()
            };
            if (e.ExceptionCheck.unwrap())(env) != 0 {
                (e.ExceptionDescribe.unwrap())(env);
                (e.ExceptionClear.unwrap())(env);
                log::warn!("pointer capture: {name} threw");
                return std::ptr::null_mut();
            }
            r
        };
        let window = call(activity, "getWindow", "()Landroid/view/Window;", true);
        if window.is_null() {
            return;
        }
        let decor = call(window, "getDecorView", "()Landroid/view/View;", true);
        if !decor.is_null() {
            call(decor, if on { "requestPointerCapture" } else { "releasePointerCapture" }, "()V", false);
            (e.DeleteLocalRef.unwrap())(env, decor);
            // (asked again every couple of seconds while there is no mouse: log only when it changes)
            static LAST: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);
            if LAST.swap(on as i8, std::sync::atomic::Ordering::Relaxed) != on as i8 {
                log::info!("pointer capture {}", if on { "requested" } else { "released" });
            }
        }
        (e.DeleteLocalRef.unwrap())(env, window);
    }
}

fn init_logging(app: &Host) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let file = data_dirs(app).into_iter().next().and_then(|d| {
            let path = d.join("scview.log");
            // Keep one previous log once it passes 2 MB.
            if std::fs::metadata(&path).map(|m| m.len() > 2_000_000).unwrap_or(false) {
                let _ = std::fs::rename(&path, d.join("scview.log.1"));
            }
            std::fs::OpenOptions::new().create(true).append(true).open(path).ok()
        });
        let logger = FileLogger {
            #[cfg(target_os = "android")]
            android: android_logger::AndroidLogger::new(android_logger::Config::default().with_max_level(log::LevelFilter::Info).with_tag("SCVIEW")),
            file: std::sync::Mutex::new(file),
        };
        if log::set_boxed_logger(Box::new(logger)).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
    });
}

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: Host) {
    init_logging(&app);
    log::info!("--- start (UTC times) ---");
    std::panic::set_hook(Box::new(|info| log::error!("panic: {info}")));
    run(&app);
    log::info!("exiting");
}

/// The desktop entry point (`scview-desktop`): the same app on Linux, against the OpenXR runtime
/// chosen by `XR_RUNTIME_JSON` (Monado's simulated headset for development without a Quest).
#[cfg(not(target_os = "android"))]
pub fn desktop_main() {
    let host = Host::new();
    init_logging(&host);
    log::info!("--- start, desktop (UTC times); files in {:?} ---", host.external_data_path());
    std::panic::set_hook(Box::new(|info| log::error!("panic: {info}")));
    run(&host);
    log::info!("exiting");
}

fn run(app: &Host) {
    // The model: sea.scm (coloured), else a binary STL, else one triangle.
    let mesh = load_scm(app).unwrap_or_else(|| {
        let (stl, tris) = load_stl(app).unwrap_or_else(|| {
        log::error!("sea.stl not found; push it with: adb push sea.stl /sdcard/Android/data/org.scview.viewer/files/");
        let mut v = vec![0u8; 84 + 50];
        v[80] = 1;
        let tri: [f32; 9] = [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 50.0];
        for (i, f) in tri.iter().enumerate() {
            v[84 + 12 + i * 4..84 + 16 + i * 4].copy_from_slice(&f.to_le_bytes());
        }
        (v, 1)
        });
        stl_mesh(&stl, tris)
    });
    let mut tris = mesh.indices.len() / 3;
    let (mesh_tx, mesh_rx) = std::sync::mpsc::channel::<Mesh>();
    {
        let save_dir = data_dirs(app).into_iter().next();
        start_net(mesh_tx, save_dir);
    }

    let entry = unsafe { xr::Entry::load().expect("OpenXR loader (libopenxr_loader.so) not found") };
    #[cfg(target_os = "android")]
    entry.initialize_android_loader().unwrap();
    let available = entry.enumerate_extensions().unwrap();
    assert!(available.khr_vulkan_enable2);
    let mut enabled = xr::ExtensionSet::default();
    enabled.khr_vulkan_enable2 = true;
    #[cfg(target_os = "android")]
    {
        enabled.khr_android_create_instance = true;
    }
    // Passthrough (the room seen through the headset cameras) behind the model, where available.
    enabled.fb_passthrough = available.fb_passthrough;
    log::info!("XR_FB_passthrough available: {}", available.fb_passthrough);
    // Raycasts against the real room (to set the model down on a table or the floor), where available.
    let ray_ext = available.ext_future && available.other.iter().any(|e| e == raycast::EXTENSION_NAME);
    enabled.ext_future = ray_ext;
    log::info!("environment raycast available: {ray_ext}");
    // The instance is created by hand: the openxr crate passes ndk_context's context as the
    // Activity, but android-activity 0.6 stores the Application context there. Horizon OS then
    // cannot tie the app's frames to its window ("non-Activity Context", aperture from "random")
    // and shows its loading scene instead of the app.
    let xr_instance = unsafe {
        #[cfg(target_os = "android")]
        let mut names: Vec<&[u8]> = vec![b"XR_KHR_vulkan_enable2\0", b"XR_KHR_android_create_instance\0"];
        #[cfg(not(target_os = "android"))]
        let mut names: Vec<&[u8]> = vec![b"XR_KHR_vulkan_enable2\0"];
        if enabled.fb_passthrough {
            names.push(b"XR_FB_passthrough\0");
        }
        if ray_ext {
            names.push(b"XR_EXT_future\0");
            names.push(b"XR_META_environment_raycast\0");
        }
        let ptrs: Vec<*const std::os::raw::c_char> = names.iter().map(|n| n.as_ptr() as *const _).collect();
        #[cfg(target_os = "android")]
        let android_info = xr::sys::InstanceCreateInfoAndroidKHR {
            ty: xr::sys::InstanceCreateInfoAndroidKHR::TYPE,
            next: std::ptr::null(),
            application_vm: app.vm_as_ptr(),
            application_activity: app.activity_as_ptr(),
        };
        let mut info = xr::sys::InstanceCreateInfo {
            ty: xr::sys::InstanceCreateInfo::TYPE,
            #[cfg(target_os = "android")]
            next: &android_info as *const _ as *const _,
            #[cfg(not(target_os = "android"))]
            next: std::ptr::null(),
            create_flags: Default::default(),
            application_info: xr::sys::ApplicationInfo {
                application_name: [0; xr::sys::MAX_APPLICATION_NAME_SIZE],
                application_version: 0,
                engine_name: [0; xr::sys::MAX_ENGINE_NAME_SIZE],
                engine_version: 0,
                api_version: xr::Version::new(1, 0, 0),
            },
            enabled_api_layer_count: 0,
            enabled_api_layer_names: std::ptr::null(),
            enabled_extension_count: ptrs.len() as u32,
            enabled_extension_names: ptrs.as_ptr(),
        };
        for (d, s) in info.application_info.application_name.iter_mut().zip(b"SC View") {
            *d = *s as _;
        }
        for (d, s) in info.application_info.engine_name.iter_mut().zip(b"SC View") {
            *d = *s as _;
        }
        let mut handle = xr::sys::Instance::NULL;
        let r = (entry.fp().create_instance)(&info, &mut handle);
        assert!(r == xr::sys::Result::SUCCESS, "xrCreateInstance: {r:?}");
        let exts = xr::InstanceExtensions::load(&entry, handle, &enabled).unwrap();
        xr::Instance::from_raw(entry.clone(), handle, exts).unwrap()
    };
    let props = xr_instance.properties().unwrap();
    log::info!("OpenXR runtime: {} {}", props.runtime_name, props.runtime_version);
    let system = xr_instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY).unwrap();
    let blend_modes = xr_instance.enumerate_environment_blend_modes(system, VIEW_TYPE).unwrap();
    log::info!("blend modes {blend_modes:?}");
    let blend = if blend_modes.contains(&xr::EnvironmentBlendMode::OPAQUE) { xr::EnvironmentBlendMode::OPAQUE } else { blend_modes[0] };
    log::info!("using blend {blend:?}");

    let vk_version = vk::make_api_version(0, 1, 1, 0);
    let reqs = xr_instance.graphics_requirements::<xr::Vulkan>(system).unwrap();
    log::info!("Vulkan requirements: {} .. {}", reqs.min_api_version_supported, reqs.max_api_version_supported);

    unsafe {
        let vk_entry = ash::Entry::load().unwrap();
        let app_info = vk::ApplicationInfo::default().api_version(vk_version);
        let vk_instance = {
            let raw = xr_instance
                .create_vulkan_instance(
                    system,
                    std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                    &vk::InstanceCreateInfo::default().application_info(&app_info) as *const _ as *const _,
                )
                .expect("XR error creating Vulkan instance")
                .map_err(vk::Result::from_raw)
                .expect("Vulkan error creating Vulkan instance");
            ash::Instance::load(vk_entry.static_fn(), vk::Instance::from_raw(raw as _))
        };
        let pd = vk::PhysicalDevice::from_raw(xr_instance.vulkan_graphics_device(system, vk_instance.handle().as_raw() as _).unwrap() as _);
        let dev_props = vk_instance.get_physical_device_properties(pd);
        let name = std::ffi::CStr::from_ptr(dev_props.device_name.as_ptr()).to_string_lossy().into_owned();
        log::info!("GPU: {name}, Vulkan {}.{}, timestamp period {} ns", vk::api_version_major(dev_props.api_version), vk::api_version_minor(dev_props.api_version), dev_props.limits.timestamp_period);
        let ts_period = dev_props.limits.timestamp_period as f64;

        let qf = vk_instance
            .get_physical_device_queue_family_properties(pd)
            .into_iter()
            .enumerate()
            .find_map(|(i, p)| if p.queue_flags.contains(vk::QueueFlags::GRAPHICS) { Some(i as u32) } else { None })
            .expect("no graphics queue");
        let mut multiview = vk::PhysicalDeviceMultiviewFeatures { multiview: vk::TRUE, ..Default::default() };
        let vk_device = {
            let raw = xr_instance
                .create_vulkan_device(
                    system,
                    std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                    pd.as_raw() as _,
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(&[vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&[1.0])])
                        .push_next(&mut multiview) as *const _ as *const _,
                )
                .expect("XR error creating Vulkan device")
                .map_err(vk::Result::from_raw)
                .expect("Vulkan error creating Vulkan device");
            ash::Device::load(vk_instance.fp_v1_0(), vk::Device::from_raw(raw as _))
        };
        let queue = vk_device.get_device_queue(qf, 0);

        // Render pass: 4x MSAA colour + depth (transient, stay in tile memory),
        // resolved into the swapchain image; multiview over both eyes.
        let view_mask = !(!0 << VIEW_COUNT);
        let attachments = [
            vk::AttachmentDescription {
                format: COLOR_FORMAT,
                samples: MSAA,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ..Default::default()
            },
            vk::AttachmentDescription {
                format: DEPTH_FORMAT,
                samples: MSAA,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::DONT_CARE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ..Default::default()
            },
            vk::AttachmentDescription {
                format: COLOR_FORMAT,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::DONT_CARE,
                store_op: vk::AttachmentStoreOp::STORE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ..Default::default()
            },
        ];
        let color_ref = [vk::AttachmentReference { attachment: 0, layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL }];
        let resolve_ref = [vk::AttachmentReference { attachment: 2, layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL }];
        let depth_ref = vk::AttachmentReference { attachment: 1, layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
        let stages = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
        let render_pass = vk_device
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&[vk::SubpassDescription::default()
                        .color_attachments(&color_ref)
                        .resolve_attachments(&resolve_ref)
                        .depth_stencil_attachment(&depth_ref)
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)])
                    .dependencies(&[vk::SubpassDependency {
                        src_subpass: vk::SUBPASS_EXTERNAL,
                        dst_subpass: 0,
                        src_stage_mask: stages,
                        dst_stage_mask: stages,
                        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        ..Default::default()
                    }])
                    .push_next(&mut vk::RenderPassMultiviewCreateInfo::default().view_masks(&[view_mask]).correlation_masks(&[view_mask])),
                None,
            )
            .unwrap();

        let vert = read_spv(&mut Cursor::new(&include_bytes!("../shaders/mesh.vert.spv")[..])).unwrap();
        let frag = read_spv(&mut Cursor::new(&include_bytes!("../shaders/mesh.frag.spv")[..])).unwrap();
        let vert = vk_device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert), None).unwrap();
        let frag = vk_device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag), None).unwrap();
        // Two eye matrices (128 bytes) + a highlight tint (16 bytes).
        let max_push = vk_instance.get_physical_device_properties(pd).limits.max_push_constants_size;
        assert!(max_push >= PUSH_SIZE, "push constants limit {max_push} < {PUSH_SIZE}");
        let push_ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::VERTEX, offset: 0, size: PUSH_SIZE }];
        let pipeline_layout = vk_device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges), None).unwrap();
        let noop = vk::StencilOpState { fail_op: vk::StencilOp::KEEP, pass_op: vk::StencilOp::KEEP, depth_fail_op: vk::StencilOp::KEEP, compare_op: vk::CompareOp::ALWAYS, compare_mask: 0, write_mask: 0, reference: 0 };
        let binding = [vk::VertexInputBindingDescription { binding: 0, stride: VERTEX_SIZE as u32, input_rate: vk::VertexInputRate::VERTEX }];
        let attrs = [
            vk::VertexInputAttributeDescription { location: 0, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 0 },
            vk::VertexInputAttributeDescription { location: 1, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 12 },
            vk::VertexInputAttributeDescription { location: 2, binding: 0, format: vk::Format::R8G8B8A8_UNORM, offset: 24 },
        ];
        // Triangles for the model and UI; lines (with edge.vert) for the edge outline.
        let edge_vert = read_spv(&mut Cursor::new(&include_bytes!("../shaders/edge.vert.spv")[..])).unwrap();
        let edge_vert = vk_device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&edge_vert), None).unwrap();
        let make_pipeline = |vert: vk::ShaderModule, topology: vk::PrimitiveTopology| vk_device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .stages(&[
                        vk::PipelineShaderStageCreateInfo { stage: vk::ShaderStageFlags::VERTEX, module: vert, p_name: b"main\0".as_ptr() as _, ..Default::default() },
                        vk::PipelineShaderStageCreateInfo { stage: vk::ShaderStageFlags::FRAGMENT, module: frag, p_name: b"main\0".as_ptr() as _, ..Default::default() },
                    ])
                    .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&binding).vertex_attribute_descriptions(&attrs))
                    .input_assembly_state(&vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology))
                    .viewport_state(&vk::PipelineViewportStateCreateInfo::default().scissor_count(1).viewport_count(1))
                    .rasterization_state(&vk::PipelineRasterizationStateCreateInfo::default().cull_mode(vk::CullModeFlags::NONE).polygon_mode(vk::PolygonMode::FILL).line_width(1.0))
                    .multisample_state(&vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(MSAA))
                    .depth_stencil_state(
                        &vk::PipelineDepthStencilStateCreateInfo::default().depth_test_enable(true).depth_write_enable(true).depth_compare_op(vk::CompareOp::LESS).front(noop).back(noop),
                    )
                    .color_blend_state(&vk::PipelineColorBlendStateCreateInfo::default().attachments(&[vk::PipelineColorBlendAttachmentState {
                        blend_enable: vk::FALSE,
                        color_write_mask: vk::ColorComponentFlags::RGBA,
                        ..Default::default()
                    }]))
                    .dynamic_state(&vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]))
                    .layout(pipeline_layout)
                    .render_pass(render_pass)
                    .subpass(0)],
                None,
            )
            .unwrap()[0];
        let pipeline = make_pipeline(vert, vk::PrimitiveTopology::TRIANGLE_LIST);
        let edge_pipeline = make_pipeline(edge_vert, vk::PrimitiveTopology::LINE_LIST);
        // Face highlights: the face's own triangles again, pulled slightly towards the eye like the edges.
        let face_pipeline = make_pipeline(edge_vert, vk::PrimitiveTopology::TRIANGLE_LIST);
        vk_device.destroy_shader_module(edge_vert, None);
        vk_device.destroy_shader_module(vert, None);
        vk_device.destroy_shader_module(frag, None);

        let cmd_pool = vk_device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(qf).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER | vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )
            .unwrap();
        let cmds = vk_device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool).command_buffer_count(PIPELINE_DEPTH + 1)).unwrap();

        // The mesh: vertex and index buffers, device-local, uploaded through staging buffers.
        let upload = |bytes: &[u8], usage: vk::BufferUsageFlags| -> (vk::Buffer, vk::DeviceMemory) {
            let size = bytes.len() as u64;
            let (staging, staging_mem) = make_buffer(&vk_device, &vk_instance, pd, size, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);
            let ptr = vk_device.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty()).unwrap() as *mut u8;
            std::slice::from_raw_parts_mut(ptr, bytes.len()).copy_from_slice(bytes);
            vk_device.unmap_memory(staging_mem);
            let (buf, mem) = make_buffer(&vk_device, &vk_instance, pd, size, usage | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL);
            let c = cmds[PIPELINE_DEPTH as usize];
            vk_device.begin_command_buffer(c, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            vk_device.cmd_copy_buffer(c, staging, buf, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]);
            vk_device.end_command_buffer(c).unwrap();
            vk_device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[c])], vk::Fence::null()).unwrap();
            vk_device.queue_wait_idle(queue).unwrap();
            vk_device.destroy_buffer(staging, None);
            vk_device.free_memory(staging_mem, None);
            (buf, mem)
        };
        let t = Instant::now();
        let upload_mesh = |m: &Mesh| {
            let (vbuf, vmem) = upload(&m.verts, vk::BufferUsageFlags::VERTEX_BUFFER);
            let index_bytes: Vec<u8> = m.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
            let (ibuf, imem) = upload(&index_bytes, vk::BufferUsageFlags::INDEX_BUFFER);
            (vbuf, vmem, ibuf, imem, m.indices.len() as u32)
        };
        let (mut vbuf, mut vmem, mut ibuf, mut imem, mut index_count) = upload_mesh(&mesh);
        log::info!("uploaded {} triangles in {} ms", tris, t.elapsed().as_millis());
        let mut model_inner = model_inner(&mesh.bounds);
        // Metres per model unit in model_inner. It is kept through the edits of one model, so after an
        // edit the longest side may no longer be 1.6 m.
        let mut inner_scale = 1.6 / longest_side(&mesh.bounds);
        let mut scene = Scene::new(mesh);
        log::info!("{} bodies", scene.bodies.len());
        // Edge lines index the model's vertex buffer (never empty: Vulkan buffers need a size).
        let upload_edges = |scene: &Scene| {
            let mut bytes: Vec<u8> = scene.edge_idx.iter().flat_map(|i| i.to_le_bytes()).collect();
            if bytes.is_empty() {
                bytes.resize(8, 0);
            }
            upload(&bytes, vk::BufferUsageFlags::INDEX_BUFFER)
        };
        let (mut ebuf, mut emem) = upload_edges(&scene);
        let mut edges = Edges::Feature;
        // Per-body state for selection: hidden flags, the selected and hovered body.
        let mut hidden: Vec<bool> = vec![false; scene.bodies.len()];
        let mut selected: Vec<bool> = vec![false; scene.bodies.len()];
        let mut hovered: Option<usize>;
        // The pointer ray mesh, kept in its own small buffers.
        let (ray_verts, ray_indices) = ray_box_mesh();
        let (ray_vbuf, ray_vmem) = upload(&ray_verts, vk::BufferUsageFlags::VERTEX_BUFFER);
        let ray_index_bytes: Vec<u8> = ray_indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let (ray_ibuf, ray_imem) = upload(&ray_index_bytes, vk::BufferUsageFlags::INDEX_BUFFER);
        // The wrist panel is rebuilt every frame into host-visible buffers, one pair per frame in flight.
        const UI_MAX_VERTS: usize = 80_000;
        const UI_MAX_INDICES: usize = 120_000;
        let ui_bufs: Vec<(vk::Buffer, vk::DeviceMemory, *mut u8, vk::Buffer, vk::DeviceMemory, *mut u8)> = (0..PIPELINE_DEPTH * 2)
            .map(|_| {
                let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                let vsize = (UI_MAX_VERTS * VERTEX_SIZE) as u64;
                let isize = (UI_MAX_INDICES * 4) as u64;
                let (vb, vm) = make_buffer(&vk_device, &vk_instance, pd, vsize, vk::BufferUsageFlags::VERTEX_BUFFER, host);
                let (ib, im) = make_buffer(&vk_device, &vk_instance, pd, isize, vk::BufferUsageFlags::INDEX_BUFFER, host);
                let vp = vk_device.map_memory(vm, 0, vsize, vk::MemoryMapFlags::empty()).unwrap() as *mut u8;
                let ip = vk_device.map_memory(im, 0, isize, vk::MemoryMapFlags::empty()).unwrap() as *mut u8;
                (vb, vm, vp, ib, im, ip)
            })
            .collect();

        // Face highlight indices, rebuilt every frame (one buffer per frame in flight).
        const FACE_MAX_INDICES: usize = 600_000;
        let face_bufs: Vec<(vk::Buffer, vk::DeviceMemory, *mut u8)> = (0..PIPELINE_DEPTH)
            .map(|_| {
                let size = (FACE_MAX_INDICES * 4) as u64;
                let (b, m) = make_buffer(&vk_device, &vk_instance, pd, size, vk::BufferUsageFlags::INDEX_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);
                (b, m, vk_device.map_memory(m, 0, size, vk::MemoryMapFlags::empty()).unwrap() as *mut u8)
            })
            .collect();

        let (session, mut frame_wait, mut frame_stream) = xr_instance
            .create_session::<xr::Vulkan>(
                system,
                &xr::vulkan::SessionCreateInfo {
                    instance: vk_instance.handle().as_raw() as _,
                    physical_device: pd.as_raw() as _,
                    device: vk_device.handle().as_raw() as _,
                    queue_family_index: qf,
                    queue_index: 0,
                },
            )
            .unwrap();
        let space = session.create_reference_space(xr::ReferenceSpaceType::LOCAL, xr::Posef::IDENTITY).unwrap();
        // The stage space is fixed to the room's floor origin: a model set down is kept relative to it.
        let stage = session.create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY).ok();
        log::info!("stage space: {}", stage.is_some());
        let mut env_ray = if ray_ext { raycast::EnvRaycast::load(&entry, &xr_instance, session.as_raw()) } else { None };
        let mut ray_note = Instant::now() - Duration::from_secs(60);

        // Passthrough: created running (the openxr crate's `Passthrough::start` calls pause).
        // B or Y switches it on and off; off gives the dark background.
        let passthrough = if enabled.fb_passthrough {
            match session.create_passthrough(xr::PassthroughFlagsFB::IS_RUNNING_AT_CREATION).and_then(|pt| {
                let layer = session.create_passthrough_layer(&pt, xr::PassthroughFlagsFB::IS_RUNNING_AT_CREATION, xr::PassthroughLayerPurposeFB::RECONSTRUCTION)?;
                Ok((pt, layer))
            }) {
                Ok(p) => Some(p),
                Err(e) => {
                    log::info!("passthrough unavailable: {e:?}");
                    None
                }
            }
        } else {
            None
        };
        // The app is boundaryless (manifest BOUNDARYLESS_APP): Horizon shows no boundary, so the room is
        // normally always visible. "Room off" (for recordings and demos) hides it; moving more than
        // ROOM_OFF_WALK from where it was switched off brings it back, as there is no boundary warning.
        const ROOM_OFF_WALK: f32 = 0.5;
        let mut passthrough_on = passthrough.is_some();
        if !passthrough_on {
            log::error!("passthrough unavailable: the app is boundaryless, so take care, the room is not shown");
        }
        let mut room_off_at: Option<[f32; 3]> = None;
        // The panel's Undo goes back through hide/show changes and AI changes in the order they happened.
        enum UndoStep {
            /// The names of the bodies that were hidden.
            Visibility(Vec<String>),
            Model,
            /// A note added (undo deletes it) or deleted (undo brings it back), by id.
            NoteAdded(String),
            NoteDeleted(String),
            /// Moved: the id and where it was ("body\tx\ty\tz"); re-recorded: the id and its old words.
            NoteMoved(String, String),
            NoteText(String, String),
        }
        /// What a step is, for the Undo and Redo hints.
        fn step_what(s: &UndoStep) -> &'static str {
            match s {
                UndoStep::Visibility(_) => "show / hide",
                UndoStep::Model => "model",
                UndoStep::NoteAdded(_) | UndoStep::NoteDeleted(_) | UndoStep::NoteMoved(..) | UndoStep::NoteText(..) => "a note",
            }
        }
        let mut undo: Vec<UndoStep> = Vec::new();
        // Steps undone, for Redo (newest last); any new change clears it.
        let mut redo: Vec<UndoStep> = Vec::new();
        // Section cut: a plane in model units (it moves with the model). While it is being moved the
        // right trigger holds it: it follows the controller as if held in the hand.
        let mut section = Section::Off;
        let mut cut = NO_PLANE;
        // While the trigger holds the cut: the inverse of the aim pose then, and the plane's point and normal in the world then.
        let mut cut_drag: Option<(M4, [f32; 3], [f32; 3])> = None;
        // Measure: the right trigger puts down point A, then B (model units, millimetres from SolidCraft).
        let mut measuring = false;
        let mut measure: Vec<[f32; 3]> = Vec::new();
        // Exploded view: 0 = assembled, 1 = each body twice as far from the model's centre. The Explode
        // button animates to 0 or 1; the left stick (left/right) sets any amount in between.
        let mut explode = 0.0f32;
        // Voice notes: pins on parts with text, kept on the Mac. In note mode the right trigger pins
        // a note (then the left trigger records it) or deletes the pin pointed at.
        let mut notes: Vec<voice::Comment> = Vec::new();
        let mut noting = false;
        let mut pending_pin: Option<(usize, [f32; 3])> = None;
        // Selected notes (by id): B or "AI: selected" gives them to the AI, "Delete note" deletes them.
        let mut note_sel: std::collections::HashSet<String> = std::collections::HashSet::new();
        // In note mode the trigger on a pin drags it: the note's index, where it was dropped so far
        // (body, point), and whether it moved far enough to count as a move rather than a click.
        let mut note_drag: Option<(usize, Option<(usize, [f32; 3])>, bool)> = None;
        let mut explode_target: Option<f32> = None;

        // Controllers: one grip grabs the model (it follows the hand), both grips scale it, the
        // right thumbstick turns (left/right) and scales (up/down) it, A or X puts it back in front.
        let hand_paths = [xr_instance.string_to_path("/user/hand/left").unwrap(), xr_instance.string_to_path("/user/hand/right").unwrap()];
        let action_set = xr_instance.create_action_set("viewer", "Viewer", 0).unwrap();
        let grip_pose = action_set.create_action::<xr::Posef>("grip_pose", "Grip pose", &hand_paths).unwrap();
        let squeeze = action_set.create_action::<f32>("squeeze", "Squeeze", &hand_paths).unwrap();
        let stick = action_set.create_action::<xr::Vector2f>("stick", "Thumbstick", &hand_paths).unwrap();
        let recentre = action_set.create_action::<bool>("recentre", "Recentre", &hand_paths).unwrap();
        // Selection: right aim pose casts the ray, trigger selects, right stick click hides the
        // selected body, left stick click shows everything again.
        let aim_pose = action_set.create_action::<xr::Posef>("aim_pose", "Aim pose", &hand_paths).unwrap();
        let trigger = action_set.create_action::<f32>("trigger", "Trigger", &hand_paths).unwrap();
        let stick_click = action_set.create_action::<bool>("stick_click", "Thumbstick click", &hand_paths).unwrap();
        // B (right controller): the AI carries out the note pointed at.
        let b_button = action_set.create_action::<bool>("b_button", "B button", &hand_paths).unwrap();
        // A short buzz in the pointing hand: a dragged value has reached its limit.
        let haptic = action_set.create_action::<xr::Haptic>("haptic", "Buzz", &hand_paths).unwrap();
        {
            let p = |s: &str| xr_instance.string_to_path(s).unwrap();
            let bindings = [
                xr::Binding::new(&grip_pose, p("/user/hand/left/input/grip/pose")),
                xr::Binding::new(&grip_pose, p("/user/hand/right/input/grip/pose")),
                xr::Binding::new(&squeeze, p("/user/hand/left/input/squeeze/value")),
                xr::Binding::new(&squeeze, p("/user/hand/right/input/squeeze/value")),
                xr::Binding::new(&stick, p("/user/hand/left/input/thumbstick")),
                xr::Binding::new(&stick, p("/user/hand/right/input/thumbstick")),
                xr::Binding::new(&recentre, p("/user/hand/left/input/x/click")),
                xr::Binding::new(&recentre, p("/user/hand/right/input/a/click")),
                xr::Binding::new(&aim_pose, p("/user/hand/left/input/aim/pose")),
                xr::Binding::new(&aim_pose, p("/user/hand/right/input/aim/pose")),
                xr::Binding::new(&trigger, p("/user/hand/left/input/trigger/value")),
                xr::Binding::new(&trigger, p("/user/hand/right/input/trigger/value")),
                xr::Binding::new(&stick_click, p("/user/hand/left/input/thumbstick/click")),
                xr::Binding::new(&stick_click, p("/user/hand/right/input/thumbstick/click")),
                xr::Binding::new(&b_button, p("/user/hand/right/input/b/click")),
                xr::Binding::new(&haptic, p("/user/hand/right/output/haptic")),
            ];
            if let Err(e) = xr_instance.suggest_interaction_profile_bindings(p("/interaction_profiles/oculus/touch_controller"), &bindings) {
                log::info!("controller bindings: {e:?}");
            }
            // Desktop: Monado's keyboard-and-mouse controllers are Mixed Reality controllers (no A, B, X, Y):
            // the menu button recentres.
            #[cfg(not(target_os = "android"))]
            {
                let wmr = [
                    xr::Binding::new(&grip_pose, p("/user/hand/left/input/grip/pose")),
                    xr::Binding::new(&grip_pose, p("/user/hand/right/input/grip/pose")),
                    xr::Binding::new(&squeeze, p("/user/hand/left/input/squeeze/click")),
                    xr::Binding::new(&squeeze, p("/user/hand/right/input/squeeze/click")),
                    xr::Binding::new(&stick, p("/user/hand/left/input/thumbstick")),
                    xr::Binding::new(&stick, p("/user/hand/right/input/thumbstick")),
                    xr::Binding::new(&recentre, p("/user/hand/left/input/menu/click")),
                    xr::Binding::new(&recentre, p("/user/hand/right/input/menu/click")),
                    xr::Binding::new(&aim_pose, p("/user/hand/left/input/aim/pose")),
                    xr::Binding::new(&aim_pose, p("/user/hand/right/input/aim/pose")),
                    xr::Binding::new(&trigger, p("/user/hand/left/input/trigger/value")),
                    xr::Binding::new(&trigger, p("/user/hand/right/input/trigger/value")),
                    xr::Binding::new(&stick_click, p("/user/hand/left/input/thumbstick/click")),
                    xr::Binding::new(&stick_click, p("/user/hand/right/input/thumbstick/click")),
                    xr::Binding::new(&haptic, p("/user/hand/right/output/haptic")),
                ];
                if let Err(e) = xr_instance.suggest_interaction_profile_bindings(p("/interaction_profiles/microsoft/motion_controller"), &wmr) {
                    log::info!("desktop controller bindings: {e:?}");
                }
            }
        }
        session.attach_action_sets(&[&action_set]).unwrap();
        let hand_spaces: Vec<xr::Space> = hand_paths.iter().map(|&h| grip_pose.create_space(session.clone(), h, xr::Posef::IDENTITY).unwrap()).collect();
        let aim_space = aim_pose.create_space(session.clone(), hand_paths[1], xr::Posef::IDENTITY).unwrap();
        let mut trigger_was = false;
        let mut click_was = [false; 2];
        let mut b_was = false;
        // The spot pointed at when the wearer started speaking (body, point), shown until the answer.
        let mut talk_was = false;
        let mut talk_point: Option<(usize, [f32; 3])> = None;
        // Face picking: the trigger selects faces (patches between feature edges) instead of parts. Each
        // selected face: its patch, and a point on it and the normal there (model units, unexploded).
        let mut face_mode = false;
        let mut edge_mode = false;
        let mut token_kb_shown = false;
        let mut selected_edges: Vec<EdgeChain> = Vec::new();
        let mut edge_cache: Option<(u32, EdgeChain)> = None;
        let mut selected_faces: Vec<(u32, [f32; 3], [f32; 3])> = Vec::new();
        // Push/pull: in this mode the trigger grabs the face pointed at, and moving the controller along the
        // face's normal sets the distance (whole mm). While held: the face, its body, a point on it (model,
        // unexploded), its outward normal in the world and in the model, the controller position at the
        // grab, and the distance so far.
        let mut pushing = false;
        let mut overlay = Overlay::Off;
        let mut reply_page = 0usize;
        let mut dims_page = 0usize;
        let mut feat_page = 0usize;
        // Dimension handles: arrows on the model, one for each dimension that has a place there. The trigger
        // on one takes hold of it; the hand's move along the arrow changes the value (slowly with the grip,
        // see svr::precise), and letting go sends it to the Mac. The drag: the handle, the drag so far, the
        // hand's place and the arrow's direction in the room when it began, and the value now.
        // Moving a part: in this mode the trigger takes hold of the part pointed at and it follows the hand
        // (slowly with the grip, see svr::precise). Let go over another part and it is set on the face
        // pointed at there; let go over nothing and it stays where it was moved to. The drag: the part, the
        // point it was taken by (model units), the hand's place in the room then, and the move so far.
        let mut moving = false;
        let mut move_drag: Option<(usize, [f32; 3], [f32; 3], [svr::precise::Drag; 3], [f32; 3])> = None;
        // The radial menu: opened at the pointing hand by pressing the right thumbstick (or Ctrl+Shift+R, or
        // saying "menu"). Its choices depend on what is pointed at. Pushing the stick, or moving the hand,
        // towards a choice marks it; letting the stick go (or the trigger) runs it as that button. The
        // menu: the hand's pose when it opened, its choices, the one marked, whether the stick opened it,
        // and what it is a menu for.
        let mut radial: Option<(M4, Vec<Button>, Option<usize>, bool, &'static str, [f32; 3])> = None;
        let mut radial_ran = 0u32;
        // After the menu closes the stick is still pushed for a moment (it was let go of by its click): it
        // does not turn or size the model until it has been back in the middle.
        let mut stick_rest = false;
        let mut handles_on = false;
        let mut handle_hover: Option<usize>;
        let mut handle_drag: Option<(usize, svr::precise::Drag, [f32; 3], [f32; 3], f32)> = None;
        // The id of the handle being dragged. The list is fetched again whenever the model changes (someone
        // else's edit, an AI change): if the handle at that place in the list is then another one, the drag
        // is dropped rather than its value sent to the wrong dimension.
        let mut handle_drag_id = String::new();
        // The dragged value is at its limit (the hand buzzes once when it gets there).
        let mut handle_at_limit = false;
        let mut buzzes = 0u32;
        let mut guides = Guides::Off;
        let mut tab = panel::Tab::Select;
        let mut kb_shift = false;
        let mut kb_open = false;
        // Where the keyboard's centre is in the room, and a grab of its bar: (the grabbed point on it, distance along the ray).
        let mut kb_pos: Option<[f32; 3]> = None;
        let mut kb_drag: Option<([f32; 2], f32)> = None;
        let mut pp_drag: Option<(u32, usize, [f32; 3], [f32; 3], [f32; 3], [f32; 3], f32)> = None;
        let mut pick_ms_sum = 0.0f64;
        // Holding the model: SolidCraft's own world grab (svr::grab), so it handles here as in the desktop
        // app's VR mode. One grip carries it, both grips scale it about the point between the hands, and
        // a scale brought near real size snaps to it.
        let debug_ray = std::env::var_os("SCVIEW_DEBUG_RAY").is_some();
        let mut debug_ray_n = 0u32;
        let mut debug_handle_n = 0u32;
        let mut debug_state_n = 0u32;
        let mut hidden_sent: Vec<bool> = Vec::new();
        let mut grab = svr::grab::WorldGrab::default();
        let mut at_real_size = false;
        // Once the model has been handled it stays where it was put (no more auto-turning).
        let mut world: Option<M4> = None;
        let mut recentre_was = false;
        let mut grab_mode = String::new();
        let mut out_of_view_since: Option<Instant> = None;
        let mut last_input = Instant::now();

        let fences: Vec<_> = (0..PIPELINE_DEPTH).map(|_| vk_device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).unwrap()).collect();
        let qpool = vk_device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(PIPELINE_DEPTH * 2), None).unwrap();
        let mut ts_valid = [false; PIPELINE_DEPTH as usize];

        // Model placement: longest side 1.6 m (model_inner), rotated about its centre, 1.8 m ahead, a little below eye level.
        let mut place = translate(0.0, -0.30, -1.8);
        let mut need_anchor = true;
        let mut unfocused_since: Option<Instant> = None;

        struct Buf {
            fb: vk::Framebuffer,
            color: vk::ImageView,
            depth: vk::ImageView,
            depth_img: vk::Image,
            depth_mem: vk::DeviceMemory,
            msaa: vk::ImageView,
            msaa_img: vk::Image,
            msaa_mem: vk::DeviceMemory,
        }
        struct Sc {
            handle: xr::Swapchain<xr::Vulkan>,
            res: vk::Extent2D,
            bufs: Vec<Buf>,
        }
        let mut swapchain: Option<Sc> = None;
        let mut events = xr::EventDataBuffer::new();
        let mut running = false;
        let mut frame = 0usize;
        let mut destroy = false;
        let (mut n, mut wall_sum, mut gpu_sum, mut gpu_max, mut gpu_n) = (0u32, 0.0f64, 0.0f64, 0.0f64, 0u32);
        let mut last = Instant::now();
        let mut exit_requested = false;
        let mut no_render = 0u32;
        let mut total: u64 = 0;
        // The desktop has no Android lifecycle: always resumed.
        #[cfg_attr(not(target_os = "android"), allow(unused_mut))]
        let mut resumed = cfg!(not(target_os = "android"));
        let mut ready = false;

        request_permissions(app);
        let placement_file = data_dirs(app).into_iter().next().map(|d| d.join("placement.txt"));
        let mut anchor: Option<M4> = placement_file.as_deref().and_then(scene::load_placement);
        let mut placing = false;
        // While setting down: how far along the controller ray the model goes (metres), moved with the right stick.
        let mut place_dist = 0.8f32;
        let mut last_anchor_world: Option<M4> = None;
        let mut anchor_saved: Option<M4> = anchor;
        let mut anchor_save_at = Instant::now();
        let mut mouse = Mouse { want_capture: true, ..Default::default() };
        let mut keys: Vec<KeyPress> = Vec::new();
        // The mouse pointer: a ray from the head at a world yaw and pitch (radians), used in place of
        // the right controller's ray once the mouse moves, until the controller moves again.
        let mut mouse_dir: Option<(f32, f32)> = None;
        let mut use_mouse = false;
        let mut mouse_ctrl_ref: Option<[f32; 3]> = None;
        let mut middle_was = false;
        // Where the panel stands when the left controller is not tracked (put down to use the keyboard).
        let mut kb_panel_at: Option<[f32; 3]> = None;
        // Push/pull by mouse: steps so far, and when a wheel-only pull ends.
        let mut pp_acc = 0.0f32;
        // The drag of a face by hand, from the trigger's press to its release (see svr::precise).
        let mut pp_precise: Option<svr::precise::Drag> = None;
        let mut pp_wheel_until: Option<Instant> = None;
        'main: loop {
            #[cfg(not(target_os = "android"))]
            {
                if app.quit_requested() {
                    destroy = true;
                }
                (mouse.dx, mouse.dy, mouse.wheel) = (0.0, 0.0, 0.0);
                for (code, ch, ctrl, shift) in app.take_keys() {
                    keys.push(KeyPress { code, ch, ctrl, shift });
                }
            }
            #[cfg(target_os = "android")]
            app.poll_events(Some(Duration::ZERO), |e| {
                match &e {
                    PollEvent::Main(MainEvent::Destroy) => destroy = true,
                    PollEvent::Main(MainEvent::Resume { .. }) => resumed = true,
                    PollEvent::Main(MainEvent::Pause) => resumed = false,
                    PollEvent::Main(MainEvent::LostFocus) => {
                        mouse.captured = false;
                        mouse.asked = None;
                        log::info!("main event LostFocus");
                    }
                    PollEvent::Main(m) => log::info!("main event {m:?}"),
                    _ => {}
                }
            });
            // Input events: the keyboard and mouse (Bluetooth, paired with the headset); everything is
            // read, so the system does not decide the app has stopped responding.
            #[cfg(target_os = "android")]
            {
                (mouse.dx, mouse.dy, mouse.wheel) = (0.0, 0.0, 0.0);
            }
            #[cfg(target_os = "android")]
            let mut mouse_seen = false;
            #[cfg(target_os = "android")]
            if let Ok(mut it) = app.input_events_iter() {
                while it.next(|ev| {
                    if let android_activity::input::InputEvent::MotionEvent(_) = ev {
                        mouse_seen = true;
                    }
                    read_input(app, ev, &mut mouse, &mut keys);
                    InputStatus::Handled
                }) {}
            }
            // The mouse is in use but not ours: ask for it (at most every 2 s).
            // (Not only after a mouse event: none arrive until the app has the mouse.)
            #[cfg(target_os = "android")]
            let _ = mouse_seen;
            if !mouse.captured && mouse.want_capture && mouse.asked.map_or(true, |t| t.elapsed() > Duration::from_secs(2)) {
                mouse.asked = Some(Instant::now());
                pointer_capture(app, true);
            }
            if destroy && !exit_requested {
                exit_requested = true;
                match session.request_exit() {
                    Ok(()) => {}
                    Err(xr::sys::Result::ERROR_SESSION_NOT_RUNNING) => break,
                    Err(e) => panic!("{e}"),
                }
            }
            while let Some(event) = xr_instance.poll_event(&mut events).unwrap() {
                use xr::Event::*;
                match event {
                    SessionStateChanged(e) => {
                        log::info!("session state {:?}", e.state());
                        if e.state() == xr::SessionState::FOCUSED {
                            need_anchor = true;
                            // Back after a while away (headset off, system menu, sleep): bring the model back
                            // in front even if it had been moved by hand, so it is never left behind or on the head.
                            if unfocused_since.is_some_and(|t| t.elapsed() > Duration::from_secs(5)) {
                                log::info!("focus back after {:.0} s: re-placing the model", unfocused_since.map_or(0.0, |t| t.elapsed().as_secs_f32()));
                                world = None;
                                grab.release();
                            }
                            unfocused_since = None;
                        } else if unfocused_since.is_none() {
                            unfocused_since = Some(Instant::now());
                        }
                        match e.state() {
                            xr::SessionState::READY => ready = true,
                            xr::SessionState::STOPPING => {
                                session.end().unwrap();
                                running = false;
                                ready = false;
                            }
                            xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => break 'main,
                            _ => {}
                        }
                    }
                    InstanceLossPending(_) => break 'main,
                    ReferenceSpaceChangePending(_) => {
                        // The headset re-centred (e.g. put on after sleeping): place the model in front again,
                        // also when it had been moved by hand (that position is relative to the old centre).
                        log::info!("reference space changed: re-placing the model");
                        need_anchor = true;
                        world = None;
                        grab.release();
                    }
                    other => log::info!("xr event {}", match other { VisibilityMaskChangedKHR(_) => "visibility mask", ReferenceSpaceChangePending(_) => "ref space change", EventsLost(_) => "events lost", InteractionProfileChanged(_) => "interaction profile", _ => "other" }),
                }
            }
            if ready && resumed && !running {
                session.begin(VIEW_TYPE).unwrap();
                running = true;
                log::info!("session begun after Resume");
            }
            if !running {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }

            let fs = frame_wait.wait().unwrap();
            frame_stream.begin().unwrap();
            if !fs.should_render {
                no_render += 1;
                if no_render % 72 == 1 {
                    log::info!("should_render false ({no_render} frames so far)");
                }
                frame_stream.end(fs.predicted_display_time, blend, &[]).unwrap();
                continue;
            }

            let sc = swapchain.get_or_insert_with(|| {
                let views = xr_instance.enumerate_view_configuration_views(system, VIEW_TYPE).unwrap();
                let res = vk::Extent2D { width: views[0].recommended_image_rect_width, height: views[0].recommended_image_rect_height };
                log::info!("per-eye {}x{}, display period {:.2} ms", res.width, res.height, fs.predicted_display_period.as_nanos() as f64 / 1e6);
                let handle = session
                    .create_swapchain(&xr::SwapchainCreateInfo {
                        create_flags: xr::SwapchainCreateFlags::EMPTY,
                        usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT | xr::SwapchainUsageFlags::SAMPLED,
                        format: COLOR_FORMAT.as_raw() as _,
                        sample_count: 1,
                        width: res.width,
                        height: res.height,
                        face_count: 1,
                        array_size: VIEW_COUNT,
                        mip_count: 1,
                    })
                    .unwrap();
                let bufs = handle
                    .enumerate_images()
                    .unwrap()
                    .into_iter()
                    .map(|img| {
                        let range = |aspect| vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: VIEW_COUNT };
                        let color = vk_device
                            .create_image_view(
                                &vk::ImageViewCreateInfo::default().image(vk::Image::from_raw(img)).view_type(vk::ImageViewType::TYPE_2D_ARRAY).format(COLOR_FORMAT).subresource_range(range(vk::ImageAspectFlags::COLOR)),
                                None,
                            )
                            .unwrap();
                        // Multisampled transient image; lazily allocated memory keeps it in tile memory on the Adreno.
                        let transient = |format, usage, aspect| {
                            let image = vk_device
                                .create_image(
                                    &vk::ImageCreateInfo::default()
                                        .image_type(vk::ImageType::TYPE_2D)
                                        .format(format)
                                        .extent(vk::Extent3D { width: res.width, height: res.height, depth: 1 })
                                        .mip_levels(1)
                                        .array_layers(VIEW_COUNT)
                                        .samples(MSAA)
                                        .tiling(vk::ImageTiling::OPTIMAL)
                                        .usage(usage | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT)
                                        .initial_layout(vk::ImageLayout::UNDEFINED),
                                    None,
                                )
                                .unwrap();
                            let req = vk_device.get_image_memory_requirements(image);
                            let props = vk_instance.get_physical_device_memory_properties(pd);
                            let lazy = (0..props.memory_type_count).find(|&i| {
                                req.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED)
                            });
                            let mem_type = lazy.unwrap_or_else(|| find_memory(&vk_instance, pd, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL));
                            let mem = vk_device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mem_type), None).unwrap();
                            vk_device.bind_image_memory(image, mem, 0).unwrap();
                            let view = vk_device
                                .create_image_view(&vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D_ARRAY).format(format).subresource_range(range(aspect)), None)
                                .unwrap();
                            log::info!("4x MSAA {format:?}: {} MB, lazily allocated: {}", req.size / 1_000_000, lazy.is_some());
                            (image, mem, view)
                        };
                        let (msaa_img, msaa_mem, msaa) = transient(COLOR_FORMAT, vk::ImageUsageFlags::COLOR_ATTACHMENT, vk::ImageAspectFlags::COLOR);
                        let (depth_img, depth_mem, depth) = transient(DEPTH_FORMAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::ImageAspectFlags::DEPTH);
                        let fb = vk_device
                            .create_framebuffer(&vk::FramebufferCreateInfo::default().render_pass(render_pass).width(res.width).height(res.height).attachments(&[msaa, depth, color]).layers(1), None)
                            .unwrap();
                        Buf { fb, color, depth, depth_img, depth_mem, msaa, msaa_img, msaa_mem }
                    })
                    .collect();
                Sc { handle, res, bufs }
            });

            let idx = sc.handle.acquire_image().unwrap();
            total += 1;
            if total <= 12 || total % 360 == 0 {
            }
            vk_device.wait_for_fences(&[fences[frame]], true, u64::MAX).unwrap();
            vk_device.reset_fences(&[fences[frame]]).unwrap();
            if ts_valid[frame] {
                let mut t = [0u64; 2];
                if vk_device.get_query_pool_results(qpool, (frame * 2) as u32, &mut t, vk::QueryResultFlags::TYPE_64).is_ok() {
                    let ms = (t[1].wrapping_sub(t[0])) as f64 * ts_period / 1e6;
                    gpu_sum += ms;
                    gpu_n += 1;
                    if ms > gpu_max {
                        gpu_max = ms;
                    }
                }
            }

            let (view_flags, views) = session.locate_views(VIEW_TYPE, fs.predicted_display_time, &space).unwrap();
            // Wait for real head tracking before placing (off the head the pose can be all zeros).
            let head_tracked = view_flags.contains(xr::ViewStateFlags::POSITION_VALID | xr::ViewStateFlags::ORIENTATION_VALID);
            if need_anchor && head_tracked && views.len() >= 2 {
                // Put the model 1.8 m in front of where the wearer is looking (level), a little below eye height.
                let hp = [(views[0].pose.position.x + views[1].pose.position.x) / 2.0, (views[0].pose.position.y + views[1].pose.position.y) / 2.0, (views[0].pose.position.z + views[1].pose.position.z) / 2.0];
                let r = quat_to_mat(views[0].pose.orientation);
                let (mut fx, mut fz) = (-r[8], -r[10]);
                let len = (fx * fx + fz * fz).sqrt().max(1e-4);
                fx /= len;
                fz /= len;
                place = translate(hp[0] + 1.8 * fx, hp[1] - 0.30, hp[2] + 1.8 * fz);
                need_anchor = false;
                log::info!("model placed at head ({:.2}, {:.2}, {:.2}) facing ({:.2}, {:.2})", hp[0], hp[1], hp[2], fx, fz);
            }
            let angle = 0.0f32; // no auto-spin
            // The Mac wants its access code: the on-screen keyboard comes up once for it.
            if voice::need_token() && !token_kb_shown {
                token_kb_shown = true;
                kb_open = true;
            } else if !voice::need_token() {
                token_kb_shown = false;
            }
            let now_in = Instant::now();
            let dt = now_in.duration_since(last_input).as_secs_f32().min(0.1);
            last_input = now_in;
            let _ = session.sync_actions(&[xr::ActiveActionSet::new(&action_set)]);
            let cur = world.unwrap_or_else(|| mul(&place, &rot_y(angle)));
            let mut hand: [Option<xr::Posef>; 2] = [None, None];
            let mut held = [false; 2];
            for h in 0..2 {
                if let Ok(loc) = hand_spaces[h].locate(&space, fs.predicted_display_time) {
                    if loc.location_flags.contains(xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID) {
                        hand[h] = Some(loc.pose);
                    }
                }
                held[h] = hand[h].is_some() && squeeze.state(&session, hand_paths[h]).map(|st| st.current_state > 0.6).unwrap_or(false);
            }
            let pos = |p: &xr::Posef| [p.position.x, p.position.y, p.position.z];
            // A hand that is dragging something (a face) does not also hold the model: its grip means "slowly".
            let grips = [0, 1].map(|h| if held[h] && !(h == 1 && (pp_drag.is_some() || handle_drag.is_some() || move_drag.is_some())) { hand[h].map(|p| svr_pose(&p)) } else { None });
            if grips.iter().any(Option::is_some) {
                // The whole model matrix (model units to the room's metres) is what the hands hold; the
                // viewer keeps it as world * model_inner, so the result is taken apart again.
                if let Some(full) = m4_to_transform(&mul(&cur, &model_inner)) {
                    let moved = grab.update(&full, grips);
                    if moved != full {
                        world = Some(mul(&transform_to_m4(&moved), &invert(&model_inner)));
                    }
                    let real = moved.scale == svr::room::REAL_SIZE;
                    if real && !at_real_size && held[0] && held[1] {
                        log::info!("grab: snapped to real size");
                        voice::say("Real size (1:1).");
                    }
                    at_real_size = real;
                }
            } else {
                grab.release();
            }
            let mode = match (grab.holding(), held[0], held[1]) {
                (false, ..) => "none".to_string(),
                (true, true, true) => "two".to_string(),
                (true, l, _) => format!("one ({})", if l { "left" } else { "right" }),
            };
            if mode != grab_mode {
                log::info!("grab {mode}; held L {} R {}, tracked L {} R {}", held[0], held[1], hand[0].is_some(), hand[1].is_some());
                grab_mode = mode;
            }
            if stick_rest {
                let v = stick.state(&session, hand_paths[1]).map(|st| st.current_state).unwrap_or(xr::Vector2f { x: 0.0, y: 0.0 });
                stick_rest = v.x.abs() > 0.2 || v.y.abs() > 0.2;
            }
            if !grab.holding() && radial.is_none() && !stick_rest {
                if let Ok(st) = stick.state(&session, hand_paths[1]) {
                    let v = st.current_state;
                    if st.is_active && (v.x.abs() > 0.2 || v.y.abs() > 0.2) {
                        let base = world.unwrap_or(cur);
                        let turn = if v.x.abs() > 0.2 { rot_y(-v.x * 1.5 * dt) } else { identity() };
                        // Setting down: up/down moves the point along the ray instead of scaling the model.
                        if placing && v.y.abs() > 0.2 {
                            place_dist = (place_dist + v.y * 0.8 * dt).clamp(0.0, 6.0);
                        }
                        let grow = if v.y.abs() > 0.2 && !placing { scale((1.0 + v.y * dt).max(0.5)) } else { identity() };
                        world = Some(mul(&base, &mul(&turn, &grow)));
                    }
                }
                // Left stick up/down tilts the model about its centre, around the horizontal axis across
                // the view (so it tips towards or away from the wearer whichever way they face).
                if let Ok(st) = stick.state(&session, hand_paths[0]) {
                    let v = st.current_state;
                    if st.is_active && v.x.abs() > 0.3 && v.x.abs() > v.y.abs() {
                        explode = (explode + v.x * 0.8 * dt).clamp(0.0, 1.0);
                        explode_target = None;
                    }
                }
                if let (Ok(st), true) = (stick.state(&session, hand_paths[0]), !views.is_empty()) {
                    let v = st.current_state;
                    if st.is_active && v.y.abs() > 0.2 && v.y.abs() >= v.x.abs() {
                        let base = world.unwrap_or(cur);
                        let hr = quat_to_mat(views[0].pose.orientation);
                        let (rx, rz) = (hr[0], hr[2]);
                        let len = (rx * rx + rz * rz).sqrt().max(1e-4);
                        let tilt = rot_axis([rx / len, 0.0, rz / len], -v.y * 1.2 * dt);
                        let c = [base[12], base[13], base[14]];
                        world = Some(mul(&translate(c[0], c[1], c[2]), &mul(&tilt, &mul(&translate(-c[0], -c[1], -c[2]), &base))));
                    }
                }
            }
            // Mouse: right-button drag turns the model about its centre (left/right about the vertical,
            // up/down tips it towards the wearer); the wheel zooms, except in push/pull, where it pulls a face.
            if !grab.holding() && !views.is_empty() {
                let base = world.unwrap_or(cur);
                let c = [base[12], base[13], base[14]];
                let about_c = |m: &M4| mul(&translate(c[0], c[1], c[2]), &mul(m, &translate(-c[0], -c[1], -c[2])));
                if mouse.right && (mouse.dx != 0.0 || mouse.dy != 0.0) {
                    let hr = quat_to_mat(views[0].pose.orientation);
                    let len = (hr[0] * hr[0] + hr[2] * hr[2]).sqrt().max(1e-4);
                    let tilt = rot_axis([hr[0] / len, 0.0, hr[2] / len], mouse.dy * 0.006);
                    world = Some(mul(&about_c(&mul(&tilt, &rot_y(mouse.dx * 0.006))), &base));
                } else if mouse.wheel != 0.0 && !pushing {
                    world = Some(mul(&about_c(&scale(1.1f32.powf(mouse.wheel))), &base));
                }
            }
            let pressed = hand_paths.iter().any(|&h| recentre.state(&session, h).map(|st| st.current_state).unwrap_or(false));
            if pressed && !recentre_was {
                // Fitted afresh to the model as it is now (edits kept the view's size and place).
                model_inner = crate::model_inner(&scene.bounds());
                inner_scale = 1.6 / longest_side(&scene.bounds());
                world = None;
                grab.release();
                need_anchor = true;
                anchor = None;
                anchor_saved = None;
                if let Some(f) = &placement_file {
                    scene::forget_placement(f);
                }
                log::info!("recentre pressed");
            }
            recentre_was = pressed;
            // A model set down stays where it was put: where the user moves it the placement is updated,
            // otherwise it follows the stage (so it is also back in place after a re-centre or the next start).
            if let (Some(rel), Some(sp)) = (anchor.as_mut(), stage.as_ref().and_then(|st| scene::stage_pose(st, &space, fs.predicted_display_time))) {
                let pm = pose_matrix(&sp);
                if let (Some(w), true) = (world, last_anchor_world.is_some_and(|l| world != Some(l))) {
                    *rel = mul(&invert(&pm), &w);
                }
                let w = mul(&pm, rel);
                world = Some(w);
                last_anchor_world = Some(w);
                if !grab.holding() && anchor_saved != Some(*rel) && anchor_save_at.elapsed() > Duration::from_secs(2) {
                    if let Some(f) = &placement_file {
                        scene::save_placement(f, rel);
                    }
                    anchor_saved = Some(*rel);
                    anchor_save_at = Instant::now();
                }
            }
            // A new model from the Mac: swap the buffers (the GPU is idle first), keep where the user put it.
            if let Some(m) = mesh_rx.try_iter().last() {
                let t = Instant::now();
                vk_device.device_wait_idle().unwrap();
                vk_device.destroy_buffer(vbuf, None);
                vk_device.free_memory(vmem, None);
                vk_device.destroy_buffer(ibuf, None);
                vk_device.free_memory(imem, None);
                (vbuf, vmem, ibuf, imem, index_count) = upload_mesh(&m);
                // An edit keeps the view's size and place: a plate made twice as long is seen to grow, where
                // fitting it afresh would show it half as thick. Another model is fitted afresh.
                if !m.same_model {
                    model_inner = crate::model_inner(&m.bounds);
                    inner_scale = 1.6 / longest_side(&m.bounds);
                }
                tris = m.indices.len() / 3;
                // Keep what was hidden and selected, by body name (an AI change rebuilds the mesh).
                let (was_hidden, was_selected) = (names_where(&scene, &hidden), names_where(&scene, &selected));
                scene = Scene::new(m);
                vk_device.destroy_buffer(ebuf, None);
                vk_device.free_memory(emem, None);
                (ebuf, emem) = upload_edges(&scene);
                hidden = flags_for(&scene, &was_hidden);
                selected = flags_for(&scene, &was_selected);
                // A part or face held by its number in the old mesh may be another one in the new: a move or a
                // push/pull in progress is dropped, not sent for the wrong part.
                if move_drag.take().is_some() | pp_drag.take().is_some() {
                    log::info!("drag dropped: the model changed under it");
                    voice::say("The model changed, so that drag was dropped.");
                }
                // Face numbers change with the mesh: the face selection does not carry over.
                selected_faces.clear();
                selected_edges.clear();
                edge_cache = None;
                log::info!("new model: {tris} triangles uploaded in {} ms", t.elapsed().as_millis());
                voice::fetch_history();
                voice::fetch_explodes();
                voice::fetch_handles();
                // The mesh that follows a check carries the overlaps as bodies of their own: the marks stay
                // with it. Any other new mesh means the design changed, and the check no longer holds.
                if !scene.bodies.iter().any(|b| b.name.starts_with("Overlap ")) {
                    voice::clear_clash();
                }
            }
            let model = mul(&world.unwrap_or_else(|| mul(&place, &rot_y(angle))), &model_inner);
            if let Some(target) = explode_target {
                let step = 1.6 * dt;
                explode = if (target - explode).abs() <= step { target } else { explode + step * (target - explode).signum() };
                if explode == target {
                    explode_target = None;
                }
            }
            // Each body's move away from the model's centre (model units); empty when assembled.
            let clash_names: Vec<String> = voice::CLASH.lock().map(|c| c.clone()).unwrap_or_default();
            let offsets: Vec<[f32; 3]> = if explode > 0.0 {
                let bb = scene.bounds();
                let c = [0, 1, 2].map(|k| (bb[k] + bb[k + 3]) * 0.5);
                scene.bodies.iter().map(|b| [0, 1, 2].map(|k| ((b.aabb[k] + b.aabb[k + 3]) * 0.5 - c[k]) * explode)).collect()
            } else {
                Vec::new()
            };
            // A part being moved is drawn (and picked) where the hand has taken it.
            let offsets = match &move_drag {
                Some((b, _, _, _, by)) if *b < scene.bodies.len() => {
                    let mut o = if offsets.is_empty() { vec![[0.0f32; 3]; scene.bodies.len()] } else { offsets };
                    o[*b] = [0, 1, 2].map(|k| o[*b][k] + by[k]);
                    o
                }
                _ => offsets,
            };
            let off = |b: usize| offsets.get(b).copied().unwrap_or([0.0; 3]);

            // Room off: bring the room back if the wearer walks away from where they switched it off.
            if let (Some(at), Some((_, layer)), true) = (room_off_at, &passthrough, views.len() >= 2) {
                let head = [0, 1, 2].map(|k| (pos(&views[0].pose)[k] + pos(&views[1].pose)[k]) * 0.5);
                let d = sub(head, at);
                if dot(d, d).sqrt() > ROOM_OFF_WALK {
                    passthrough_on = true;
                    room_off_at = None;
                    log::info!("room back on: moved {:.2} m: {:?}", dot(d, d).sqrt(), layer.resume());
                }
            }
            // The wrist panel floats above the left hand, turned to face the head.
            // With the left controller put down while the keyboard or mouse is in use, it stands still,
            // down and to the left of where the wearer looked.
            let typing = !voice::typed().is_empty();
            if hand[0].is_some() {
                kb_panel_at = None;
            } else if (use_mouse || typing) && kb_panel_at.is_none() && views.len() >= 2 {
                let head = [0, 1, 2].map(|k| (pos(&views[0].pose)[k] + pos(&views[1].pose)[k]) * 0.5);
                let r = quat_to_mat(views[0].pose.orientation);
                let (fx, fz) = (-r[8], -r[10]);
                let len = (fx * fx + fz * fz).sqrt().max(1e-4);
                let (fx, fz) = (fx / len, fz / len);
                kb_panel_at = Some([head[0] + fx * 0.5 + fz * 0.18, head[1] - 0.25, head[2] + fz * 0.5 - fx * 0.18]);
            }
            let lay = panel::layout(tab, voice::confirm_pending(), voice::recording());
            // Its bottom edge about 9 cm above the hand.
            let panel_hand = hand[0].map(|lh| [lh.position.x, lh.position.y + 0.09 + lay.h / 2.0, lh.position.z]).or(kb_panel_at);
            let panel: Option<M4> = match (panel_hand, views.len() >= 2) {
                (Some(p), true) => {
                    let head = [0, 1, 2].map(|k| (pos(&views[0].pose)[k] + pos(&views[1].pose)[k]) * 0.5);
                    // Its bottom edge about 9 cm above the hand, so the hand stays in view.
                    let to_head = sub(head, p);
                    let len = dot(to_head, to_head).sqrt().max(1e-4);
                    let f = to_head.map(|x| x / len);
                    let r0 = cross([0.0, 1.0, 0.0], f);
                    let rl = dot(r0, r0).sqrt().max(1e-4);
                    let r = r0.map(|x| x / rl);
                    let u = cross(f, r);
                    Some([r[0], r[1], r[2], 0.0, u[0], u[1], u[2], 0.0, f[0], f[1], f[2], 0.0, p[0], p[1], p[2], 1.0])
                }
                _ => None,
            };

            // Pointing: a ray from the right controller (not while it is grabbing) finds the panel
            // button or the body under it; the panel is in front of the model.
            let mut ray: Option<(M4, f32)> = None;
            let mut hover_btn: Option<Button> = None;
            let mut on_panel = false;
            // The keyboard under the pointer: what is on it, and the point on it (its own space) and distance.
            let mut on_kb = false;
            let mut kb_hover: Option<panel::KbHit> = None;
            let mut kb_at: Option<([f32; 2], f32)> = None;
            // Where the ray meets the model (model units), for measuring.
            let mut hit_point: Option<[f32; 3]> = None;
            hovered = None;
            if let Some(list) = voice::take_comments() {
                notes = list;
                note_sel.retain(|id| notes.iter().any(|n| &n.id == id));
                note_drag = None;
            }
            if pending_pin.is_some() && !voice::pin_waiting() && !voice::busy() {
                pending_pin = None;
            }
            // Each note's body (None: gone or hidden) and its place in the world.
            let note_at: Vec<Option<(usize, [f32; 3])>> = notes
                .iter()
                .map(|n| {
                    let b = scene.bodies.iter().position(|b| b.name == n.body)?;
                    (!hidden[b]).then(|| (b, transform(&model, [0, 1, 2].map(|k| n.point[k] + off(b)[k]), 1.0)))
                })
                .collect();
            let mut hovered_note: Option<usize> = None;
            let mut hovered_face: Option<u32> = None;
            let mut hit_normal = [0.0f32; 3];
            let mut overlay_hover: Option<usize> = None;
            let mut overlay_del = false;
            let mut overlay_click = false;
            let mut overlay_x = 0.0f32;
            let n_models = if overlay == Overlay::Requests {
                voice::REQUESTS.lock().ok().and_then(|m| m.as_ref().map(|l| l.len())).unwrap_or(0)
            } else if overlay == Overlay::Features {
                voice::FEATURES.lock().ok().and_then(|m| m.as_ref().map(|l| l.len())).unwrap_or(0)
            } else if overlay == Overlay::Dims {
                voice::DIMS.lock().ok().and_then(|m| m.as_ref().map(|l| l.len())).unwrap_or(0)
            } else {
                voice::MODELS.lock().ok().and_then(|m| m.as_ref().map(|l| l.len())).unwrap_or(0)
            };
            let ov_rows = overlay_rows(overlay, n_models);
            // No pointer from a hand that holds the model; a hand that is dragging a face keeps it (its grip
            // then means "slowly").
            let ctrl_pm: Option<M4> = if held[1] && pp_drag.is_none() && handle_drag.is_none() && move_drag.is_none() {
                None
            } else {
                aim_space
                    .locate(&space, fs.predicted_display_time)
                    .ok()
                    .filter(|loc| loc.location_flags.contains(xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID))
                    .map(|loc| pose_matrix(&loc.pose))
            };
            // The mouse moves the pointer (not while the right button turns the model); the middle button
            // brings it back to straight ahead.
            let head_now = (views.len() >= 2).then(|| [0, 1, 2].map(|k| (pos(&views[0].pose)[k] + pos(&views[1].pose)[k]) * 0.5));
            if head_now.is_some() {
                let look = || {
                    let r = quat_to_mat(views[0].pose.orientation);
                    let f = [-r[8], -r[9], -r[10]];
                    ((-f[0]).atan2(-f[2]), f[1].clamp(-1.0, 1.0).asin())
                };
                if mouse.middle && !middle_was {
                    mouse_dir = Some(look());
                }
                // (The pointer stays put while the left button drags a face in push/pull.)
                if (mouse.dx != 0.0 || mouse.dy != 0.0) && !mouse.right && !(pp_drag.is_some() && mouse.left) {
                    let (yaw, pitch) = mouse_dir.unwrap_or_else(look);
                    mouse_dir = Some((yaw - mouse.dx * MOUSE_RAD, (pitch - mouse.dy * MOUSE_RAD).clamp(-1.45, 1.45)));
                    use_mouse = true;
                    mouse_ctrl_ref = None;
                }
                // Moving the right controller more than 3 cm hands the pointer back to it.
                if let (true, Some(pm)) = (use_mouse, ctrl_pm) {
                    let at = [pm[12], pm[13], pm[14]];
                    match mouse_ctrl_ref {
                        None => mouse_ctrl_ref = Some(at),
                        Some(r) if dot(sub(at, r), sub(at, r)).sqrt() > 0.03 => use_mouse = false,
                        _ => {}
                    }
                }
            }
            middle_was = mouse.middle;
            let mouse_pm: Option<M4> = match (use_mouse, mouse_dir, head_now) {
                (true, Some((yaw, pitch)), Some(o)) => {
                    let d = [-yaw.sin() * pitch.cos(), pitch.sin(), -yaw.cos() * pitch.cos()];
                    let r = normalize(cross(d, [0.0, 1.0, 0.0]));
                    let u = cross(r, d);
                    Some([r[0], r[1], r[2], 0.0, u[0], u[1], u[2], 0.0, -d[0], -d[1], -d[2], 0.0, o[0], o[1], o[2], 1.0])
                }
                _ => None,
            };
            let kb_lay = panel::kb_layout();
            let kb_m: Option<M4> = match (kb_open, kb_pos, head_now) {
                (true, Some(p), Some(h)) => Some(billboard(p, h)),
                _ => None,
            };
            {
                if let Some(pm) = if use_mouse { mouse_pm } else { ctrl_pm } {
                    {
                        let (o_w, d_w) = ([pm[12], pm[13], pm[14]], [-pm[8], -pm[9], -pm[10]]);
                        let mut len = 3.0f32;
                        // The floating keyboard, if the pointer is on it, comes before the panel.
                        if let Some(km) = &kb_m {
                            let inv = invert(km);
                            let (o, d) = (transform(&inv, o_w, 1.0), transform(&inv, d_w, 0.0));
                            if d[2] < -1e-4 {
                                let t = -o[2] / d[2];
                                let (x, y) = (o[0] + t * d[0], o[1] + t * d[1]);
                                if t > 0.0 && panel::kb_inside(&kb_lay, x, y) {
                                    on_kb = true;
                                    on_panel = true;
                                    len = t;
                                    kb_hover = panel::kb_hit(&kb_lay, x, y);
                                    kb_at = Some(([x, y], t));
                                }
                            }
                        }
                        if let (Some(pnl), false) = (&panel, on_kb) {
                            let inv = invert(pnl);
                            let (o, d) = (transform(&inv, o_w, 1.0), transform(&inv, d_w, 0.0));
                            if d[2] < -1e-4 {
                                let t = -o[2] / d[2];
                                let (x, y) = (o[0] + t * d[0], o[1] + t * d[1]);
                                // For tests without a headset (SCVIEW_DEBUG_RAY=1): where the pointer's ray meets the
                                // panel's plane, so a script can steer a simulated controller onto a button.
                                if debug_ray {
                                    debug_ray_n = debug_ray_n.wrapping_add(1);
                                    if debug_ray_n % 6 == 0 {
                                        log::info!("PANELRAY x {x:.4} y {y:.4} t {t:.3} h {:.4} row {:?} btn {:?}", lay.h, overlay_row_at(overlay, y, ov_rows, lay.h), panel::button_at(&lay, x, y));
                                    }
                                    // Now and then, where each button is (its middle), so a script can aim at one by name.
                                    if debug_ray_n % 90 == 1 {
                                        let slots: Vec<String> = lay.slots.iter().map(|s| format!("{:?}@{:.4},{:.4}", s.b, s.x + s.w / 2.0, s.top - s.h / 2.0)).collect();
                                        log::info!("PANELSLOTS {}", slots.join(";"));
                                    }
                                }
                                if t > 0.0 && x.abs() <= PANEL_W / 2.0 && y.abs() <= lay.h / 2.0 {
                                    on_panel = true;
                                    hover_btn = panel::button_at(&lay, x, y);
                                    len = t;
                                } else if t > 0.0 && overlay != Overlay::Off && x.abs() <= PANEL_W / 2.0 && y > lay.h / 2.0 && y <= lay.h / 2.0 + OV_GAP + overlay_h(overlay, ov_rows) {
                                    // The list or keyboard above the panel: a models row or a key can be picked.
                                    on_panel = true;
                                    overlay_click = true;
                                    len = t;
                                    if matches!(overlay, Overlay::Models | Overlay::Requests | Overlay::Dims | Overlay::Features) && n_models > 0 {
                                        overlay_hover = overlay_row_at(overlay, y, ov_rows, lay.h);
                                        overlay_x = x;
                                        overlay_del = overlay == Overlay::Requests && x > PANEL_W / 2.0 - 0.025;
                                    }
                                }
                            }
                        }
                        if !on_panel {
                            let inv = invert(&model);
                            let t0 = Instant::now();
                            // d is the world unit direction mapped into the model, so t is in metres.
                            let (o_m, d_m) = (transform(&inv, o_w, 1.0), transform(&inv, d_w, 0.0));
                            // The pointer looks past a part that is being moved, to what it would be set on.
                            let moving_hidden;
                            let pickable = match &move_drag {
                                Some((b, ..)) if *b < hidden.len() => {
                                    moving_hidden = {
                                        let mut h = hidden.clone();
                                        h[*b] = true;
                                        h
                                    };
                                    &moving_hidden
                                }
                                _ => &hidden,
                            };
                            let hit = scene.pick(o_m, d_m, pickable, cut, &offsets);
                            pick_ms_sum += t0.elapsed().as_secs_f64() * 1000.0;
                            // While a dimension handle is held the ray is the handle's: the parts it passes over
                            // are not lit up and the panel's title does not change.
                            let hit = if handle_drag.is_some() { None } else { hit };
                            hovered = hit.map(|(b, _, _)| b);
                            if let Some((_, _, tri)) = hit {
                                hovered_face = Some(scene.tri_patch[tri]);
                                let v = |k: usize| scene.pos[scene.indices[tri * 3 + k] as usize];
                                hit_normal = normalize(cross(sub(v(1), v(0)), sub(v(2), v(0))));
                            }
                            if let Some((_, t, _)) = hit {
                                len = t.min(3.0);
                                hit_point = Some([0, 1, 2].map(|k| o_m[k] + t * d_m[k]));
                            }
                            // A note pin's head (it stands NOTE_POST above the surface) within NOTE_REACH of the ray
                            // wins over the part: the head is easy to hit even from the side.
                            let mut best = f32::MAX;
                            for (i, at) in note_at.iter().enumerate() {
                                let Some((_, w)) = at else { continue };
                                if note_drag.is_some() {
                                    break; // while dragging a pin the ray finds the surface under it
                                }
                                let v = sub(note_head(*w), o_w);
                                let t = dot(v, d_w);
                                let miss = sub(v, d_w.map(|x| x * t));
                                let dist = dot(miss, miss).sqrt();
                                if t > 0.0 && dist < NOTE_REACH && dist < best {
                                    best = dist;
                                    hovered_note = Some(i);
                                }
                            }
                            if let Some(i) = hovered_note {
                                let v = sub(note_head(note_at[i].unwrap().1), o_w);
                                len = dot(v, d_w);
                            }
                        }
                        ray = Some((pm, len));
                    }
                }
            }
            // Measure points snap to a corner, else a feature edge, within 1.5 cm (in the world) of the pointer.
            let unit = dot([model[0], model[1], model[2]], [model[0], model[1], model[2]]).sqrt().max(1e-9); // metres per model unit
            let snap_now: Option<([f32; 3], Snap)> = match (measuring, hovered, hit_point) {
                (true, Some(b), Some(p)) => {
                    let (q, kind) = scene.snap(b, sub(p, off(b)), 0.015 / unit);
                    Some(([0, 1, 2].map(|k| q[k] + off(b)[k]), kind))
                }
                _ => None,
            };
            // Setting the model down: a point on the right controller's ray (place_dist along it, 0 = the tip).
            placing = placing && !pushing;
            // The room is raycast when the headset can; else (or while nothing is hit) the point is place_dist along the ray.
            if let Some(r) = env_ray.as_mut() {
                r.update(&xr_instance); // (creation takes a few frames: start it early)
            }
            let mut place_on_surface = false;
            let place_hit: Option<[f32; 3]> = if placing {
                if let Some(r) = env_ray.as_ref() {
                    if let (Some(msg), true) = (&r.failed, ray_note.elapsed() > Duration::from_secs(30)) {
                        ray_note = Instant::now();
                        log::warn!("environment raycast: {msg}");
                        voice::say("Room raycast not available (allow spatial data for the app): use the right stick to move the point.");
                    }
                }
                ctrl_pm.map(|pm| {
                    let (o, d) = ([pm[12], pm[13], pm[14]], [-pm[8], -pm[9], -pm[10]]);
                    match env_ray.as_ref().and_then(|r| r.cast(space.as_raw(), fs.predicted_display_time, o, d)) {
                        Some(raycast::Cast::Hit(p, _)) => {
                            place_on_surface = true;
                            p
                        }
                        _ => [0, 1, 2].map(|k| o[k] + place_dist * d[k]),
                    }
                })
            } else {
                None
            };
            // Edge picking: the whole edge under the pointer (within 1.5 cm), found again only when the segment changes.
            edge_mode = edge_mode && !pushing;
            let hover_edge: Option<EdgeChain> = match (edge_mode, hovered, hit_point) {
                (true, Some(b), Some(p)) => match scene.nearest_segment(b, sub(p, off(b)), 0.015 / unit) {
                    Some(k) => {
                        if edge_cache.as_ref().is_none_or(|(ck, c)| *ck != k || c.body != b) {
                            edge_cache = Some((k, scene.edge_chain(b, k)));
                        }
                        edge_cache.as_ref().map(|(_, c)| c.clone())
                    }
                    None => None,
                },
                _ => None,
            };
            voice::set_edges(
                selected_edges
                    .iter()
                    .map(|c| {
                        let f = |p: [f32; 3]| format!("{}\t{}\t{}", p[0], p[1], p[2]);
                        format!("{}\t{}\t{}\t{}\t{}", scene.bodies[c.body].name, f(c.start), f(c.mid), f(c.end), c.len)
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
            // Push-to-talk: hold the left trigger to speak a request to the Mac's AI, with the selection.
            for _ in 0..voice::take_changes() {
                undo.push(UndoStep::Model);
                redo.clear();
            }
            for step in voice::take_note_steps() {
                redo.clear();
                undo.push(match step {
                    voice::NoteStep::Added(id) => UndoStep::NoteAdded(id),
                    voice::NoteStep::Deleted(id) => UndoStep::NoteDeleted(id),
                    voice::NoteStep::Text(id, old) => UndoStep::NoteText(id, old),
                });
            }
            voice::set_selected(names_where(&scene, &selected));
            voice::set_faces(
                selected_faces
                    .iter()
                    .map(|(f, p, n)| format!("{}\t{}\t{}", voice::pin_text(&scene.bodies[scene.patch_body[*f as usize] as usize].name, *p), [n[0], n[1]].map(|x| x.to_string()).join("\t"), n[2]))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
            // In note mode, speaking needs a pin first (else nothing is recorded).
            let talk = trigger.state(&session, hand_paths[0]).map(|st| st.current_state > 0.6).unwrap_or(false);
            // Starting to speak to the AI while pointing at a part: that spot goes with the request ("a hole here").
            if talk && !talk_was && !voice::busy() && !voice::pin_waiting() {
                talk_point = match (hovered, hit_point) {
                    (Some(b), Some(p)) if !noting => Some((b, sub(p, off(b)))),
                    _ => None,
                };
                voice::set_point(talk_point.map(|(b, p)| format!("{}\t{}\t{}\t{}", voice::pin_text(&scene.bodies[b].name, p), hit_normal[0], hit_normal[1], hit_normal[2])));
                if let Some((b, p)) = talk_point {
                    log::info!("speaking while pointing at {} {p:?}", scene.bodies[b].name);
                }
            }
            talk_was = talk;
            if talk_point.is_some() && !talk && !voice::busy() {
                talk_point = None;
            }
            voice::update(talk && !(noting && !voice::pin_waiting() && !voice::busy()));
            // Keyboard: typing makes an AI request (Enter sends it, Esc drops it, Backspace deletes);
            // Ctrl with a letter works a panel button; Delete hides the selection; Esc with nothing
            // typed gives the mouse back to the system.
            let mut key_btn: Option<Button> = None;
            for k in keys.drain(..) {
                use crate::Keycode as K;
                let Ok(mut typed) = voice::TYPED.lock() else { continue };
                if k.ctrl {
                    // Ctrl with a key: the command bound to it (src/commands.rs).
                    key_btn = commands::by_key(k.code, k.shift).or(key_btn);
                    continue;
                }
                match k.code {
                    K::Enter | K::NumpadEnter => {
                        let text = typed.trim().to_string();
                        if text.is_empty() {
                            voice::confirm_ai();
                            continue;
                        }
                        if voice::busy() {
                            voice::say("Still busy with the last request: press Enter again in a moment.");
                            continue;
                        }
                        typed.clear();
                        // In push/pull, a number alone moves the face pointed at by exactly that many mm.
                        if let (true, Some(b), Some(p), Ok(mm)) = (pushing, hovered, hit_point, text.trim_end_matches("mm").trim().parse::<f32>()) {
                            if mm.abs() >= 0.5 {
                                log::info!("push/pull {} by {mm} mm (typed)", scene.bodies[b].name);
                                voice::push_pull(voice::pin_text(&scene.bodies[b].name, sub(p, off(b))), mm);
                            } else {
                                voice::say("SolidCraft moves a face by half a millimetre or more.");
                            }
                            continue;
                        }
                        // Like speaking: the spot under the pointer goes with it ("make this red").
                        talk_point = match (hovered, hit_point) {
                            (Some(b), Some(p)) => Some((b, sub(p, off(b)))),
                            _ => None,
                        };
                        voice::set_point(talk_point.map(|(b, p)| format!("{}\t{}\t{}\t{}", voice::pin_text(&scene.bodies[b].name, p), hit_normal[0], hit_normal[1], hit_normal[2])));
                        log::info!("typed request: {} characters", text.chars().count());
                        voice::ask_text(text);
                    }
                    K::F1 => key_btn = Some(Button::Keys),
                    K::Escape => {
                        if voice::recording() {
                            voice::cancel_recording();
                        } else if voice::confirm_pending() {
                            voice::cancel_confirm();
                        } else if typed.is_empty() && overlay != Overlay::Off {
                            overlay = Overlay::Off;
                        } else if typed.is_empty() && kb_open {
                            kb_open = false;
                        } else if typed.is_empty() {
                            mouse.want_capture = false;
                            pointer_capture(app, false);
                        }
                        typed.clear();
                    }
                    K::Del => {
                        typed.pop();
                    }
                    K::ForwardDel if typed.is_empty() => key_btn = Some(Button::Hide),
                    _ => {
                        if let Some(c) = k.ch.filter(|c| !c.is_control()) {
                            if typed.len() < 300 {
                                typed.push(c);
                            }
                        }
                    }
                }
            }
            if key_btn.is_some() {
                log::info!("key for button {key_btn:?}");
            }
            // Parts the server found for "select the posts": they become the selection.
            if let Some(names) = voice::take_select() {
                for (sel, b) in selected.iter_mut().zip(scene.bodies.iter()) {
                    *sel = names.iter().any(|n| n == &b.name);
                }
                log::info!("selected by name: {} part(s)", selected.iter().filter(|&&s| s).count());
            }
            // The Mac is told what is hidden whenever that changes.
            if hidden != hidden_sent && voice::MAC.lock().ok().is_some_and(|m| m.is_some()) {
                hidden_sent = hidden.clone();
                voice::sync_hidden(names_where(&scene, &hidden));
            }
            // A spoken or typed command the server recognised: the same button, one a frame.
            if key_btn.is_none() {
                if let Some(id) = voice::take_command() {
                    key_btn = commands::by_id(&id);
                    log::info!("command {id} for button {key_btn:?}");
                }
            }
            let ctrl_trig = trigger.state(&session, hand_paths[1]).map(|st| st.current_state > 0.6).unwrap_or(false);
            if ctrl_trig {
                use_mouse = false;
            }
            // The left mouse button works like the right trigger.
            let trig = ctrl_trig || mouse.left;
            // The full reply: a click anywhere on it goes to the next page.
            if overlay == Overlay::Reply && overlay_click && trig && !trigger_was {
                let pages = (reply_lines().len() + REPLY_ROWS - 1) / REPLY_ROWS;
                reply_page = if pages == 0 { 0 } else { (reply_page + 1) % pages };
            }
            // The radial menu.
            let radial_was_open = radial.is_some();
            let stick_down = stick_click.state(&session, hand_paths[1]).map(|st| st.current_state).unwrap_or(false);
            let mut radial_took_trigger = false;
            if radial.is_none() && (key_btn == Some(Button::Radial) || (stick_down && !click_was[1])) {
                if let Some((pm, _)) = ray {
                    let by_stick = key_btn != Some(Button::Radial);
                    let any_selected = selected.iter().any(|s| *s);
                    let (what, items): (&'static str, Vec<Button>) = if face_mode && hovered_face.is_some() {
                        ("face", vec![Button::PushPull, Button::Measure, Button::Section, Button::Note, Button::Faces, Button::Clear, Button::Undo, Button::Redo])
                    } else if hovered.is_some() || any_selected {
                        ("part", vec![Button::Hide, Button::Isolate, Button::Move, Button::Measure, Button::Handles, Button::Dims, Button::Undo, Button::Redo])
                    } else {
                        ("nothing", vec![Button::ShowAll, Button::Recentre, Button::Explode, Button::Size, Button::Models, Button::Clash, Button::Undo, Button::Redo])
                    };
                    // A menu for the part pointed at acts on it: it is selected if nothing is.
                    if let (Some(b), false, "part") = (hovered, any_selected, what) {
                        selected[b] = true;
                    }
                    let line: Vec<String> = items.iter().enumerate().map(|(i, b)| format!("{} {}", svr::radial::way(i, items.len()), radial_label(*b))).collect();
                    log::info!("radial: open for {what}: {}", line.join(" | "));
                    let names: Vec<&str> = items.iter().map(|b| radial_label(*b)).collect();
                    voice::say(&format!("Menu, from up, clockwise: {}", names.join(", ")));
                    // The ring stands a forearm's length along the ray, upright and facing the head, so it is
                    // seen as a ring from wherever the hand is, and "up" on it is up in the room. (Square to the
                    // ray, a hand held low and to the side showed it nearly edge on.)
                    let hand0 = [pm[12], pm[13], pm[14]];
                    let o = [0, 1, 2].map(|k| pm[12 + k] - pm[8 + k] * 0.3);
                    let to_head = head_now.map_or([pm[8], pm[9], pm[10]], |h| sub(h, o));
                    let flat = [to_head[0], 0.0, to_head[2]];
                    let n = if dot(flat, flat) > 1e-6 { normalize(flat) } else { [0.0, 0.0, 1.0] };
                    let (up, right) = ([0.0, 1.0, 0.0], normalize(cross([0.0, 1.0, 0.0], n)));
                    let mut frame = pm;
                    for k in 0..3 {
                        (frame[k], frame[4 + k], frame[8 + k], frame[12 + k]) = (right[k], up[k], n[k], o[k]);
                    }
                    radial = Some((frame, items, None, by_stick, what, hand0));
                }
                key_btn = None;
            } else if key_btn == Some(Button::Radial) {
                log::info!("radial: closed, nothing run");
                voice::say("");
                radial = None;
                key_btn = None;
            } else if let Some((pm0, items, pick, by_stick, _, hand0)) = radial.as_mut() {
                // The choice: the way the stick is pushed, else the way the hand has moved since it opened.
                let pushed = stick.state(&session, hand_paths[1]).ok().map(|st| st.current_state).filter(|v| v.x.abs() > 0.3 || v.y.abs() > 0.3);
                let now = match (pushed, ray) {
                    (Some(v), _) => svr::radial::sector(v.x, v.y, items.len(), 0.5),
                    (None, Some((pm, _))) => {
                        let d = sub([pm[12], pm[13], pm[14]], *hand0);
                        svr::radial::sector(dot(d, [pm0[0], pm0[1], pm0[2]]), dot(d, [pm0[4], pm0[5], pm0[6]]), items.len(), 0.04)
                    }
                    _ => *pick,
                };
                if now != *pick {
                    *pick = now;
                    voice::say(&now.map_or("Menu: let go here for nothing".to_string(), |i| format!("Menu: {} ({})", radial_label(items[i]), svr::radial::way(i, items.len()))));
                }
                let take = if *by_stick { !stick_down } else { trig && !trigger_was };
                if take {
                    radial_took_trigger = !*by_stick;
                    match *pick {
                        Some(i) => {
                            radial_ran += 1;
                            log::info!("radial: {:?} run ({})", items[i], svr::radial::way(i, items.len()));
                            key_btn = Some(items[i]);
                        }
                        // The stick pressed and let go without a push: what that press has always done.
                        None if *by_stick => key_btn = Some(Button::Hide),
                        None => log::info!("radial: closed, nothing run"),
                    }
                    radial = None;
                }
            }
            if radial_was_open && radial.is_none() {
                stick_rest = true;
            }
            // Dimension handles: which one the pointer is on, the drag of one, and the press that starts it.
            let handle_list = if handles_on { voice::HANDLES.lock().map(|h| h.clone()).unwrap_or_default() } else { Vec::new() };
            let handle_len = 0.05 / unit; // the arrow's length, in model units (5 cm in the room)
            handle_hover = None;
            if let (Some((pm, _)), None, false, false) = (ray, &handle_drag, on_panel, on_kb) {
                let (o, d) = ([pm[12], pm[13], pm[14]], normalize([-pm[8], -pm[9], -pm[10]]));
                let mut best = f32::MAX; // the pointer must be within 3 cm of the arrow's head
                // For test scripts: now and then, how far each arrow's head is from the ray, to the pointer's
                // right and upward (metres), so a script can steer onto one by name.
                let mut where_log: Vec<String> = Vec::new();
                debug_handle_n = debug_handle_n.wrapping_add(1);
                for (i, h) in handle_list.iter().enumerate() {
                    let tip = transform(&model, [0, 1, 2].map(|k| h.at[k] + h.dir[k] * handle_len), 1.0);
                    let v = sub(tip, o);
                    if debug_ray && debug_handle_n % 20 == 0 {
                        where_log.push(format!("{}@{:.4},{:.4}", h.label.replace(' ', "_"), dot(v, [pm[0], pm[1], pm[2]]), dot(v, [pm[4], pm[5], pm[6]])));
                    }
                    let t = dot(v, d);
                    let off_ray = sub(v, d.map(|x| x * t));
                    let dist = dot(off_ray, off_ray).sqrt();
                    // Of two arrows that line up from here, the nearer one is taken.
                    let score = dist + 0.02 * t;
                    if t > 0.0 && dist < 0.03 && score < best {
                        best = score;
                        handle_hover = Some(i);
                    }
                }
                if !where_log.is_empty() {
                    log::info!("HANDLERAY {}", where_log.join(";"));
                }
            }
            if let Some((i, ref mut drag, hand0, dir_w, ref mut value)) = handle_drag {
                match (handle_list.get(i).filter(|h| h.id == handle_drag_id), ray) {
                    (None, _) => {
                        log::info!("handle drag dropped: the model changed under it");
                        voice::say("The model changed, so that drag was dropped.");
                        handle_drag = None;
                    }
                    (Some(h), Some((pm, _))) => {
                        let along = (dot(sub([pm[12], pm[13], pm[14]], hand0), dir_w) / unit) as f64;
                        let fine = held[1];
                        let step = pp_step(unit) as f64;
                        let moved = svr::precise::snap(drag.update(along, fine), if fine { svr::precise::fine_step(step, 0.1) } else { step });
                        let mut now = h.value + h.gain * moved as f32;
                        if h.id.starts_with("joint:") {
                            // Whole degrees or millimetres; tenths with the grip.
                            now = if fine { (now * 10.0).round() / 10.0 } else { now.round() };
                        }
                        let stopped = now < h.min || now > h.max;
                        let now = now.clamp(h.min, h.max);
                        if stopped && !handle_at_limit {
                            buzzes += 1;
                            let buzz = xr::HapticVibration::new().amplitude(0.8).duration(xr::Duration::from_nanos(70_000_000)).frequency(xr::FREQUENCY_UNSPECIFIED);
                            let sent = haptic.apply_feedback(&session, hand_paths[1], &buzz);
                            log::info!("haptic: buzz {buzzes}, {} at its limit of {} {} ({:?})", h.label, now, h.unit, sent.map(|_| "sent"));
                        }
                        handle_at_limit = stopped;
                        if (now - *value).abs() > 1e-4 {
                            *value = now;
                            voice::say(&format!("{}: {} {}", h.label, fmt_dim(now), h.unit));
                        }
                        if !trig {
                            if (now - h.value).abs() > 1e-4 {
                                log::info!("handle {} ({}): {} -> {}", h.label, h.id, h.value, now);
                                voice::set_dim(&h.id, now);
                            }
                            handle_drag = None;
                        }
                    }
                    _ => {
                        if !trig {
                            handle_drag = None;
                        }
                    }
                }
            }
            // Moving a part: the move so far, and the release.
            if let Some((b, grab, hand0, ref mut drags, ref mut by)) = move_drag {
                if let Some((pm, _)) = ray {
                    let inv = invert(&model);
                    let d_m = transform(&inv, sub([pm[12], pm[13], pm[14]], hand0), 0.0);
                    let fine = held[1];
                    let step = pp_step(unit) as f64;
                    let step = if fine { svr::precise::fine_step(step, 0.1) } else { step };
                    let mut now = [0, 1, 2].map(|k| svr::precise::snap(drags[k].update(d_m[k] as f64, fine), step) as f32);
                    // Nearly along one of the model's axes: exactly along it.
                    let big = now.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    if big > 0.0 {
                        for v in now.iter_mut() {
                            if v.abs() < 0.2 * big {
                                *v = 0.0;
                            }
                        }
                    }
                    *by = now;
                }
                if !trig {
                    let by = *by;
                    let name = scene.bodies.get(b).map(|x| x.name.clone()).unwrap_or_default();
                    match (hovered.filter(|&t| t != b), hit_point) {
                        (Some(t), Some(p)) => {
                            log::info!("move {name}: set on {} at {:?}", scene.bodies[t].name, sub(p, off(t)));
                            voice::move_part(name, by, Some((grab, sub(p, off(t)), hit_normal)));
                        }
                        _ if by.iter().any(|v| v.abs() > 1e-4) => {
                            log::info!("move {name}: by {by:?} mm");
                            voice::move_part(name, by, None);
                        }
                        _ => {}
                    }
                    move_drag = None;
                }
            }
            let move_click = trig && !trigger_was && key_btn.is_none() && radial.is_none() && !radial_took_trigger && moving && move_drag.is_none() && !on_panel && !on_kb && hovered.is_some() && hit_point.is_some();
            if let (true, Some(b), Some(p), Some((pm, _))) = (move_click, hovered, hit_point, ray) {
                move_drag = Some((b, sub(p, off(b)), [pm[12], pm[13], pm[14]], [svr::precise::Drag::start(0.0, 0.0); 3], [0.0; 3]));
                log::info!("move {}: taken hold of", scene.bodies[b].name);
            }
            let handle_click = trig && !trigger_was && key_btn.is_none() && radial.is_none() && handle_drag.is_none() && handle_hover.is_some() && !move_click;
            if let (true, Some(i), Some((pm, _))) = (handle_click, handle_hover, ray) {
                if let Some(h) = handle_list.get(i) {
                    let dir_w = normalize(transform(&model, h.dir, 0.0));
                    handle_drag = Some((i, svr::precise::Drag::start(0.0, 0.0), [pm[12], pm[13], pm[14]], dir_w, h.value));
                    handle_at_limit = false;
                    handle_drag_id = h.id.clone();
                    log::info!("handle {}: taken hold of at {}", h.label, h.value);
                }
            }
            if ((trig && !trigger_was) && radial.is_none() && !radial_took_trigger && !handle_click && handle_drag.is_none() && !move_click && move_drag.is_none()) || key_btn.is_some() {
                if let (true, None) = (on_kb, key_btn) {
                    // The on-screen keyboard: a key types into the request line; the bar is taken hold of to move it.
                    match kb_hover {
                        Some(panel::KbHit::Key(i)) => {
                            if let (Some(k), Ok(mut typed)) = (kb_lay.keys.get(i), voice::TYPED.lock()) {
                                match k.act {
                                    panel::KbAct::Char(c) => {
                                        if typed.len() < 300 {
                                            typed.push(if kb_shift { c.to_ascii_uppercase() } else { c });
                                        }
                                        kb_shift = false;
                                    }
                                    panel::KbAct::Space => {
                                        if typed.len() < 300 {
                                            typed.push(' ');
                                        }
                                    }
                                    panel::KbAct::Back => {
                                        typed.pop();
                                    }
                                    panel::KbAct::Clear => typed.clear(),
                                    panel::KbAct::Shift => kb_shift = !kb_shift,
                                    panel::KbAct::Close => kb_open = false,
                                    panel::KbAct::Send => {
                                        let text = typed.trim().to_string();
                                        if text.is_empty() {
                                            voice::confirm_ai();
                                        } else if voice::busy() {
                                            voice::say("Still busy with the last request: press Send again in a moment.");
                                        } else {
                                            typed.clear();
                                            talk_point = None;
                                            voice::set_point(None);
                                            log::info!("typed request (on-screen keyboard): {} characters", text.chars().count());
                                            voice::ask_text(text);
                                        }
                                    }
                                }
                            }
                        }
                        Some(panel::KbHit::Bar) => kb_drag = kb_at,
                        None => {}
                    }
                } else if on_panel || key_btn.is_some() {
                    let model_pick = if key_btn.is_none() { overlay_hover } else { None };
                    if let (Some(i), Overlay::Requests) = (model_pick, overlay) {
                        // A saved request: run it on what is selected now, or (the x) remove it.
                        let r = voice::REQUESTS.lock().ok().and_then(|l| l.as_ref().and_then(|l| l.get(i).cloned()));
                        if let Some(r) = r {
                            if overlay_del {
                                voice::delete_saved(&r.id);
                            } else {
                                log::info!("saved request: {}", r.text);
                                voice::run_saved(r.text);
                                overlay = Overlay::Off;
                            }
                        }
                    } else if let (Some(i), Overlay::Features) = (model_pick, overlay) {
                        // A feature: the target at the right switches it off or on; the rest of the row opens its
                        // dimensions.
                        let total = voice::FEATURES.lock().ok().and_then(|l| l.as_ref().map(|l| l.len())).unwrap_or(0);
                        let (per, pages) = dims_paging(total);
                        feat_page = feat_page.min(pages - 1);
                        if pages > 1 && i == per {
                            feat_page = (feat_page + 1) % pages;
                            log::info!("features: page {} of {pages}", feat_page + 1);
                        } else if let Some(f) = voice::FEATURES.lock().ok().and_then(|l| l.as_ref().and_then(|l| l.get(feat_page * per + i).filter(|_| i < per).cloned())) {
                            if PANEL_W / 2.0 - overlay_x < DIM_ZONE * 2.0 {
                                log::info!("feature {} ({}): switch {}", f.name, f.id, if f.off { "on" } else { "off" });
                                voice::toggle_feature(&f.id);
                            } else {
                                log::info!("feature {} ({}): its dimensions", f.name, f.id);
                                voice::set_dims_of(&f.name);
                                overlay = Overlay::Dims;
                                dims_page = 0;
                                if let Ok(mut d) = voice::DIMS.lock() {
                                    *d = None;
                                }
                                voice::fetch_dims();
                            }
                        }
                    } else if let (Some(i), Overlay::Dims) = (model_pick, overlay) {
                        // A dimension: the - and + at the right end change it by one step, on the Mac, with no AI.
                        let total = voice::DIMS.lock().ok().and_then(|l| l.as_ref().map(|l| l.len())).unwrap_or(0);
                        let (per, pages) = dims_paging(total);
                        dims_page = dims_page.min(pages - 1);
                        let d = if pages > 1 && i == per {
                            // The last row of a long list: the next page.
                            dims_page = (dims_page + 1) % pages;
                            log::info!("dimensions: page {} of {pages}", dims_page + 1);
                            None
                        } else {
                            voice::DIMS.lock().ok().and_then(|l| l.as_ref().and_then(|l| l.get(dims_page * per + i).filter(|_| i < per).cloned()))
                        };
                        if let Some(d) = d {
                            let side = PANEL_W / 2.0 - overlay_x;
                            let dir = if side < DIM_ZONE { 1.0 } else if side < DIM_ZONE * 2.0 { -1.0 } else { 0.0 };
                            if dir != 0.0 {
                                // A joint moves 5 (degrees or mm) a press, from wherever it is, zero included.
                                let value = if d.id.starts_with("joint:") { ((d.value / 5.0 + dir * 1.001).round()) * 5.0 } else { dim_step_to(d.value, dir) };
                                log::info!("dimension {} ({}): {} -> {}", d.label, d.id, d.value, value);
                                voice::set_dim(&d.id, value);
                            }
                        }
                    } else if let Some(i) = model_pick {
                        let m = voice::MODELS.lock().ok().and_then(|l| l.as_ref().and_then(|l| l.get(i).cloned()));
                        if let Some(m) = m {
                            log::info!("opening model {}", m.name);
                            voice::open_model(&m.id);
                            // The steps belonged to the old model.
                            undo.clear();
                            redo.clear();
                            note_sel.clear();
                            overlay = Overlay::Off;
                        }
                    } else if let Some(b) = key_btn.or(hover_btn) {
                        log::info!("button {b:?}");
                        if matches!(b, Button::Hide | Button::Isolate | Button::ShowAll) {
                            undo.push(UndoStep::Visibility(names_where(&scene, &hidden)));
                            redo.clear();
                            if undo.len() > 50 {
                                undo.remove(0);
                            }
                        }
                        match b {
                            Button::Hide => {
                                for (h, s) in hidden.iter_mut().zip(selected.iter_mut()) {
                                    *h |= *s;
                                    *s = false;
                                }
                            }
                            Button::Isolate => {
                                if selected.iter().any(|&s| s) {
                                    for (h, &s) in hidden.iter_mut().zip(selected.iter()) {
                                        *h = !s;
                                    }
                                    selected.iter_mut().for_each(|s| *s = false);
                                }
                            }
                            Button::ShowAll => hidden.iter_mut().for_each(|h| *h = false),
                            Button::Clear => {
                                selected.iter_mut().for_each(|s| *s = false);
                                selected_faces.clear();
                                selected_edges.clear();
                            }
                            // Parts, then faces, then edges, then parts again.
                            Button::Faces => {
                                (face_mode, edge_mode) = match (face_mode, edge_mode) {
                                    (false, false) => (true, false),
                                    (true, _) => (false, true),
                                    (false, true) => (false, false),
                                };
                                if edge_mode {
                                    pushing = false;
                                }
                            }
                            Button::Models => {
                                overlay = if overlay == Overlay::Models { Overlay::Off } else { Overlay::Models };
                                if overlay == Overlay::Models {
                                    if let Ok(mut m) = voice::MODELS.lock() {
                                        *m = None;
                                    }
                                    voice::fetch_models();
                                }
                            }
                            Button::CancelVoice => voice::cancel_recording(),
                            Button::Reply => {
                                overlay = if overlay == Overlay::Reply { Overlay::Off } else { Overlay::Reply };
                                reply_page = 0;
                            }
                            Button::ConfirmGo => voice::confirm_ai(),
                            Button::ConfirmStop => voice::cancel_confirm(),
                            Button::Repeat => voice::repeat_last(),
                            Button::SaveReq => voice::save_last(),
                            Button::Radial => {} // opened and closed above
                            Button::Clash => {
                                if clash_names.is_empty() {
                                    voice::check_clash();
                                } else {
                                    voice::drop_clash();
                                }
                            }
                            Button::Move => {
                                moving = !moving;
                                if moving {
                                    voice::say("Move: hold the trigger on a part and move your hand. Let go on another part to set it there.");
                                }
                            }
                            Button::Handles => {
                                handles_on = !handles_on;
                                if handles_on {
                                    voice::fetch_handles();
                                }
                            }
                            Button::Features => {
                                overlay = if overlay == Overlay::Features { Overlay::Off } else { Overlay::Features };
                                if overlay == Overlay::Features {
                                    feat_page = 0;
                                    if let Ok(mut d) = voice::FEATURES.lock() {
                                        *d = None;
                                    }
                                    voice::fetch_features();
                                }
                            }
                            Button::Dims => {
                                overlay = if overlay == Overlay::Dims { Overlay::Off } else { Overlay::Dims };
                                if overlay == Overlay::Dims {
                                    voice::set_dims_of("");
                                    dims_page = 0;
                                    if let Ok(mut d) = voice::DIMS.lock() {
                                        *d = None;
                                    }
                                    voice::fetch_dims();
                                }
                            }
                            Button::Saved => {
                                overlay = if overlay == Overlay::Requests { Overlay::Off } else { Overlay::Requests };
                                if overlay == Overlay::Requests {
                                    if let Ok(mut r) = voice::REQUESTS.lock() {
                                        *r = None;
                                    }
                                    voice::fetch_requests();
                                }
                            }
                            Button::Guides => {
                                guides = match guides {
                                    Guides::Off => Guides::Axes,
                                    Guides::Axes => Guides::Grid,
                                    Guides::Grid => Guides::Off,
                                };
                            }
                            Button::Tab(t) => tab = t,
                            Button::SaveFile => voice::save_file(),
                            Button::Snap => voice::snapshot(),
                            Button::Keyboard => {
                                kb_open = !kb_open;
                                kb_drag = None;
                                // Opened: in front of the wearer, a little below eye level, unless it is already near.
                                if let (true, Some(head)) = (kb_open, head_now) {
                                    let far = kb_pos.map_or(true, |p| dot(sub(p, head), sub(p, head)).sqrt() > 1.5);
                                    if far {
                                        let r = quat_to_mat(views[0].pose.orientation);
                                        let l = (r[8] * r[8] + r[10] * r[10]).sqrt().max(1e-4);
                                        kb_pos = Some([head[0] - r[8] / l * 0.55, head[1] - 0.30, head[2] - r[10] / l * 0.55]);
                                    }
                                }
                            }
                            Button::Keys => overlay = if overlay == Overlay::Keys { Overlay::Off } else { Overlay::Keys },
                            Button::PushPull => {
                                pushing = !pushing;
                                pp_drag = None;
                                if pushing {
                                    measuring = false;
                                    noting = false;
                                    voice::clear_pin();
                                    if section == Section::Moving {
                                        section = Section::Fixed;
                                    }
                                }
                            }
                            Button::Recentre => {
                                model_inner = crate::model_inner(&scene.bounds());
                                inner_scale = 1.6 / longest_side(&scene.bounds());
                                world = None;
                                grab.release();
                                need_anchor = true;
                                anchor = None;
                                anchor_saved = None;
                                if let Some(f) = &placement_file {
                                    scene::forget_placement(f);
                                }
                            }
                            Button::Size => {
                                // Step the model to the next of SIZE_STEPS times its size in the file, about the middle of its base.
                                let w0 = world.unwrap_or(cur);
                                let sc0 = (w0[0] * w0[0] + w0[1] * w0[1] + w0[2] * w0[2]).sqrt().max(1e-6);
                                let bb = scene.bounds();
                                let now = inner_scale * 1000.0 * sc0;
                                let next = SIZE_STEPS.iter().copied().find(|&x| x > now * 1.05).unwrap_or(SIZE_STEPS[1]);
                                let k = next / now;
                                let m0 = mul(&w0, &model_inner);
                                let corners: Vec<[f32; 3]> = (0..8).map(|c| transform(&m0, [if c & 1 == 0 { bb[0] } else { bb[3] }, if c & 2 == 0 { bb[1] } else { bb[4] }, if c & 4 == 0 { bb[2] } else { bb[5] }], 1.0)).collect();
                                let low = corners.iter().map(|c| c[1]).fold(f32::MAX, f32::min);
                                let p = [corners.iter().map(|c| c[0]).sum::<f32>() / 8.0, low, corners.iter().map(|c| c[2]).sum::<f32>() / 8.0];
                                world = Some(mul(&translate(p[0], p[1], p[2]), &mul(&scale(k), &mul(&translate(-p[0], -p[1], -p[2]), &w0))));
                                voice::say(&format!("Size x{next} of the file's: {:.0} cm long.", next * longest_side(&bb) / 10.0));
                            }
                            Button::Place => {
                                placing = !placing;
                                if placing {
                                    pushing = false;
                                    edge_mode = false;
                                    face_mode = false;
                                    place_dist = 0.8;
                                    voice::say("Point at a table or the floor: a green square shows the surface. Trigger sets the model down there. (Yellow: no surface found, right stick up/down moves the point.)");
                                }
                            }
                            Button::Undo | Button::Redo => {
                                // Undo takes a step off one list, applies it, and puts the step that reverses it on the other; Redo the same the other way.
                                let undoing = b == Button::Undo;
                                let (from, to) = if undoing { (&mut undo, &mut redo) } else { (&mut redo, &mut undo) };
                                if let Some(step) = from.pop() {
                                    let back = match step {
                                        UndoStep::Visibility(names) => {
                                            let now = names_where(&scene, &hidden);
                                            hidden = flags_for(&scene, &names);
                                            UndoStep::Visibility(now)
                                        }
                                        UndoStep::Model => {
                                            if undoing {
                                                voice::undo_model()
                                            } else {
                                                voice::redo_model()
                                            }
                                            UndoStep::Model
                                        }
                                        // Undoing an add deletes the note, redoing it brings it back; a delete the other way round.
                                        UndoStep::NoteAdded(id) => {
                                            voice::note_delete(&id, !undoing, false);
                                            UndoStep::NoteAdded(id)
                                        }
                                        UndoStep::NoteDeleted(id) => {
                                            voice::note_delete(&id, undoing, false);
                                            UndoStep::NoteDeleted(id)
                                        }
                                        UndoStep::NoteMoved(id, pin) => {
                                            let now = notes.iter().find(|n| n.id == id).map(|n| voice::pin_text(&n.body, n.point)).unwrap_or_else(|| pin.clone());
                                            voice::note_edit(&id, Some(pin), None);
                                            UndoStep::NoteMoved(id, now)
                                        }
                                        UndoStep::NoteText(id, old) => {
                                            let now = notes.iter().find(|n| n.id == id).map(|n| n.text.clone()).unwrap_or_else(|| old.clone());
                                            voice::note_edit(&id, None, Some(old));
                                            UndoStep::NoteText(id, now)
                                        }
                                    };
                                    to.push(back);
                                }
                            }
                            Button::Room => {
                                if let Some((_, layer)) = &passthrough {
                                    passthrough_on = !passthrough_on;
                                    let r = if passthrough_on { layer.resume() } else { layer.pause() };
                                    log::info!("room {}: {r:?}", if passthrough_on { "on" } else { "off" });
                                    room_off_at = if passthrough_on {
                                        None
                                    } else {
                                        Some([0, 1, 2].map(|k| (pos(&views[0].pose)[k] + pos(&views[1].pose)[k]) * 0.5))
                                    };
                                }
                            }
                            Button::Section => {
                                section = match section {
                                    Section::Off => {
                                        // Start through the middle of the model, facing the wearer: the near half is cut away.
                                        let bb = scene.bounds();
                                        let c = [0, 1, 2].map(|k| (bb[k] + bb[k + 3]) * 0.5);
                                        let head = [0, 1, 2].map(|k| (pos(&views[0].pose)[k] + pos(&views[1].pose)[k]) * 0.5);
                                        let n = normalize(sub(c, transform(&invert(&model), head, 1.0)));
                                        cut = [n[0], n[1], n[2], -dot(n, c)];
                                        measuring = false;
                                        noting = false;
                                        voice::clear_pin();
                                        Section::Moving
                                    }
                                    Section::Moving => Section::Fixed,
                                    Section::Fixed => {
                                        cut = NO_PLANE;
                                        Section::Off
                                    }
                                };
                                cut_drag = None;
                                log::info!("section {section:?}");
                            }
                            Button::DoNotes => {
                                let ids: Vec<String> = notes.iter().filter(|n| note_sel.contains(&n.id) && !n.done).map(|n| n.id.clone()).collect();
                                if !note_sel.is_empty() {
                                    if ids.is_empty() {
                                        voice::say("Already done by the AI: move or re-record the note to do it again.");
                                    } else {
                                        voice::note_do(&ids.join(","));
                                    }
                                    note_sel.clear();
                                } else {
                                    log::info!("all notes to the AI");
                                    voice::note_do("all");
                                }
                            }
                            Button::DeleteNote => {
                                for id in note_sel.drain() {
                                    log::info!("deleting note {id}");
                                    voice::note_delete(&id, false, true);
                                }
                            }
                            Button::Note => {
                                noting = !noting;
                                pushing = false;
                                voice::clear_pin();
                                pending_pin = None;
                                if noting {
                                    measuring = false;
                                    if section == Section::Moving {
                                        section = Section::Fixed;
                                    }
                                }
                            }
                            Button::Explode => {
                                if voice::explode_views() > 0 {
                                    // An assembly with exploded views of its own: SolidCraft's, one after another.
                                    voice::explode_next();
                                } else {
                                    explode_target = Some(if explode > 0.01 { 0.0 } else { 1.0 });
                                }
                            }
                            Button::Edges => {
                                edges = match edges {
                                    Edges::Feature => Edges::All,
                                    Edges::All => Edges::Off,
                                    Edges::Off => Edges::Feature,
                                };
                                log::info!("edges {edges:?}");
                            }
                            Button::Measure => {
                                measuring = !measuring;
                                pushing = false;
                                noting = false;
                                voice::clear_pin();
                                measure.clear();
                                if section == Section::Moving {
                                    section = Section::Fixed;
                                }
                            }
                        }
                    }
                } else if let (Some(i), false, false) = (hovered_note, measuring, section == Section::Moving) {
                    if noting {
                        note_drag = Some((i, None, false));
                    } else if !note_sel.remove(&notes[i].id) {
                        note_sel.insert(notes[i].id.clone());
                    }
                } else if section == Section::Moving {
                    // Take hold of the cut: from now on it moves and turns with the controller.
                    if let Some((pm, _)) = ray {
                        let n = [cut[0], cut[1], cut[2]];
                        let p0 = n.map(|x| -cut[3] * x);
                        cut_drag = Some((invert(&pm), transform(&model, p0, 1.0), normalize(transform(&model, n, 0.0))));
                    }
                } else if noting {
                    if let (Some(b), Some(p)) = (hovered, hit_point) {
                        let pin = sub(p, off(b));
                        voice::set_pin(&scene.bodies[b].name, pin);
                        pending_pin = Some((b, pin));
                        log::info!("note pin on {} at {pin:?}", scene.bodies[b].name);
                    }
                } else if measuring {
                    if let Some((p, kind)) = snap_now {
                        if measure.len() >= 2 {
                            measure.clear();
                        }
                        measure.push(p);
                        log::info!("measure point {} ({kind:?}): {p:?}", measure.len());
                    }
                } else if let (true, Some(f), Some(b), Some(p), Some((pm, _))) = (pushing, hovered_face, hovered, hit_point, ray) {
                    // Grab the face: its normal turned to face the controller is "outward".
                    let ray_d = [-pm[8], -pm[9], -pm[10]];
                    let mut n_w = normalize(transform(&model, hit_normal, 0.0));
                    let mut n_m = hit_normal;
                    if dot(n_w, ray_d) > 0.0 {
                        n_w = n_w.map(|x| -x);
                        n_m = n_m.map(|x| -x);
                    }
                    pp_drag = Some((f, b, sub(p, off(b)), n_w, n_m, [pm[12], pm[13], pm[14]], 0.0));
                    pp_acc = 0.0;
                    log::info!("push/pull: grabbed a face of {}", scene.bodies[b].name);
                } else if placing {
                    // The model's lowest point, centred, goes to the touched point; it is then kept relative to the stage.
                    if let (Some(hit), Some(stage_pose)) = (place_hit, stage.as_ref().and_then(|st| scene::stage_pose(st, &space, fs.predicted_display_time))) {
                        // Level it: keep only the turn about the vertical and the size, so it stands flat on the surface.
                        // Real size where it fits on a desk (the design is in mm); a big design (a ship, a building) is
                        // scaled down to a tabletop model. (model_inner scales the longest side to 1.6 m.)
                        let bb0 = scene.bounds();
                        let longest_mm = (0..3).map(|k| bb0[k + 3] - bb0[k]).fold(0.0f32, f32::max).max(1e-3);
                        let real_m = longest_mm / 1000.0;
                        let (target_m, one_to_one) = if real_m <= 0.45 { (real_m, true) } else { (0.3, false) };
                        let (sc, yaw) = (target_m / (inner_scale * longest_mm), (-cur[2]).atan2(cur[0]));
                        let level = mul(&translate(cur[12], cur[13], cur[14]), &mul(&rot_y(yaw), &scale(sc)));
                        let model_l = mul(&level, &model_inner);
                        let bb = scene.bounds();
                        let corners: Vec<[f32; 3]> = (0..8).map(|c| transform(&model_l, [if c & 1 == 0 { bb[0] } else { bb[3] }, if c & 2 == 0 { bb[1] } else { bb[4] }, if c & 4 == 0 { bb[2] } else { bb[5] }], 1.0)).collect();
                        let low = corners.iter().map(|c| c[1]).fold(f32::MAX, f32::min);
                        let (cx, cz) = (corners.iter().map(|c| c[0]).sum::<f32>() / 8.0, corners.iter().map(|c| c[2]).sum::<f32>() / 8.0);
                        let w = mul(&translate(hit[0] - cx, hit[1] - low, hit[2] - cz), &level);
                        let rel = mul(&invert(&pose_matrix(&stage_pose)), &w);
                        anchor = Some(rel);
                        world = Some(w);
                        last_anchor_world = None;
                        if let Some(f) = &placement_file {
                            scene::save_placement(f, &rel);
                            anchor_saved = Some(rel);
                        }
                        voice::say(&if one_to_one {
                            format!("Set down at its real size ({:.0} mm long). Stick up/down changes the size.", longest_mm)
                        } else {
                            format!("Set down, scaled to {:.0} cm (the real size is {:.1} m). Stick up/down changes the size.", target_m * 100.0, real_m)
                        });
                        log::info!("model set down at {hit:?}");
                        placing = false;
                    } else if stage.is_none() {
                        voice::say("This headset gave no stage space, so the placement could not be kept.");
                        placing = false;
                    }
                } else if edge_mode {
                    // Edge picking: the trigger adds the edge under the pointer, or takes it out again.
                    if let Some(c) = &hover_edge {
                        if let Some(i) = selected_edges.iter().position(|e| e.body == c.body && e.key() == c.key()) {
                            selected_edges.remove(i);
                        } else {
                            selected_edges.push(c.clone());
                        }
                        log::info!("edges selected: {}", selected_edges.len());
                    }
                } else if let (true, Some(f), Some(b), Some(p)) = (face_mode, hovered_face, hovered, hit_point) {
                    // Face picking: the trigger adds the face to the selection, or takes it out again.
                    if let Some(i) = selected_faces.iter().position(|sf| sf.0 == f) {
                        selected_faces.remove(i);
                    } else {
                        selected_faces.push((f, sub(p, off(b)), hit_normal));
                    }
                    log::info!("faces selected: {}", selected_faces.len());
                } else if let Some(b) = hovered {
                    // The trigger adds a part to the selection, or takes it out again.
                    selected[b] = !selected[b];
                    log::info!("{} body {b}: {}", if selected[b] { "selected" } else { "deselected" }, scene.bodies[b].name);
                }
            }
            if !trig {
                cut_drag = None;
                kb_drag = None;
            }
            // Moving the keyboard: the grabbed point stays on the pointer's ray, at the distance it was grabbed.
            if let (Some((off, t0)), Some((pm, _)), Some(km)) = (kb_drag, ray, &kb_m) {
                let grab = [pm[12] - pm[8] * t0, pm[13] - pm[9] * t0, pm[14] - pm[10] * t0];
                kb_pos = Some([0, 1, 2].map(|k| grab[k] - km[k] * off[0] - km[4 + k] * off[1]));
            }
            // Dragging a pin: it follows the surface under the ray; on release a move (more than 5 mm in
            // the room) goes to the Mac, a click selects or deselects the note.
            if let Some((i, ref mut to, ref mut moved)) = note_drag {
                if let (Some(b), Some(p)) = (hovered, hit_point) {
                    let pin = sub(p, off(b));
                    if let Some((_, w0)) = note_at.get(i).copied().flatten() {
                        let w = transform(&model, p, 1.0);
                        if dot(sub(w, w0), sub(w, w0)).sqrt() > 0.005 {
                            *moved = true;
                        }
                    }
                    *to = Some((b, pin));
                }
                if !trig {
                    let n = &notes[i];
                    match (*moved, *to) {
                        (true, Some((b, pin))) => {
                            log::info!("note {} moved to {} {pin:?}", n.id, scene.bodies[b].name);
                            undo.push(UndoStep::NoteMoved(n.id.clone(), voice::pin_text(&n.body, n.point)));
                            redo.clear();
                            voice::note_edit(&n.id, Some(voice::pin_text(&scene.bodies[b].name, pin)), None);
                            // Show it at the new place until the Mac's list arrives.
                            notes[i].body = scene.bodies[b].name.clone();
                            notes[i].point = pin;
                            notes[i].done = false;
                        }
                        _ => {
                            if !note_sel.remove(&n.id) {
                                note_sel.insert(n.id.clone());
                            }
                        }
                    }
                    note_drag = None;
                }
            }
            // Push/pull drag: the controller's move along the face normal, in mm, whole numbers; on release it goes to the Mac.
            // With the mouse: the wheel pulls (up) or pushes a face one step a notch, held left button and moving
            // up or down the same; a wheel-only change goes to the Mac 0.8 s after the last notch.
            if let (true, true, None, Some(f), Some(b), Some(p), Some((pm, _))) = (pushing, mouse.wheel != 0.0, pp_drag, hovered_face, hovered, hit_point, ray) {
                let ray_d = [-pm[8], -pm[9], -pm[10]];
                let (mut n_w, mut n_m) = (normalize(transform(&model, hit_normal, 0.0)), hit_normal);
                if dot(n_w, ray_d) > 0.0 {
                    n_w = n_w.map(|x| -x);
                    n_m = n_m.map(|x| -x);
                }
                pp_drag = Some((f, b, sub(p, off(b)), n_w, n_m, [pm[12], pm[13], pm[14]], 0.0));
                pp_acc = 0.0;
            }
            if let Some((_, b, p, n_w, _, hand0, ref mut mm)) = pp_drag {
                let step = pp_step(unit);
                if mouse.wheel != 0.0 {
                    pp_wheel_until = Some(Instant::now() + Duration::from_millis(800));
                }
                if use_mouse {
                    if mouse.left {
                        pp_acc -= mouse.dy / PP_COUNTS;
                    }
                    pp_acc += mouse.wheel;
                    *mm = pp_acc.round() * step;
                } else if let Some((pm, _)) = ray {
                    // The hand's move along the face's normal, in mm. With the same hand's grip held it counts
                    // a tenth and lands on a finer step (svr::precise).
                    let along = (dot(sub([pm[12], pm[13], pm[14]], hand0), n_w) / unit) as f64;
                    let fine = held[1];
                    let drag = pp_precise.get_or_insert_with(|| svr::precise::Drag::start(along, 0.0));
                    let value = drag.update(along, fine);
                    let s = if fine { svr::precise::fine_step(step as f64, 0.5) } else { step as f64 };
                    *mm = svr::precise::snap(value, s) as f32;
                }
                if !trig && pp_wheel_until.map_or(true, |t| Instant::now() >= t) {
                    pp_wheel_until = None;
                    let d = *mm;
                    if d.abs() > 1e-4 {
                        log::info!("push/pull {} by {d} mm", scene.bodies[b].name);
                        voice::push_pull(voice::pin_text(&scene.bodies[b].name, p), d);
                    }
                    pp_drag = None;
                    pp_precise = None;
                }
            }
            // Re-recording: in note mode, with one note selected and no new pin, the left trigger records new words for it.
            let one_sel: Vec<&voice::Comment> = notes.iter().filter(|n| note_sel.contains(&n.id)).collect();
            if !voice::busy() {
                voice::set_redo(if noting && pending_pin.is_none() && one_sel.len() == 1 { Some(one_sel[0].id.as_str()) } else { None });
            }
            if let (Some((inv_aim0, pw0, nw0)), Some((pm, _))) = (&cut_drag, ray) {
                let r = mul(&pm, inv_aim0);
                let inv = invert(&model);
                let p = transform(&inv, transform(&r, *pw0, 1.0), 1.0);
                let n = normalize(transform(&inv, transform(&r, *nw0, 0.0), 0.0));
                cut = [n[0], n[1], n[2], -dot(n, p)];
            }
            trigger_was = trig;
            let b_down = b_button.state(&session, hand_paths[1]).map(|st| st.current_state).unwrap_or(false);
            if b_down && !b_was && voice::confirm_pending() {
                voice::confirm_ai();
            } else if b_down && !b_was {
                // B: the AI does the selected notes, else the one pointed at. Done notes are not done twice.
                let ids: Vec<String> = if !note_sel.is_empty() {
                    notes.iter().filter(|n| note_sel.contains(&n.id) && !n.done).map(|n| n.id.clone()).collect()
                } else {
                    hovered_note.filter(|&i| !notes[i].done).map(|i| notes[i].id.clone()).into_iter().collect()
                };
                if !ids.is_empty() {
                    log::info!("notes {ids:?} to the AI");
                    voice::note_do(&ids.join(","));
                    note_sel.clear();
                } else if !note_sel.is_empty() || hovered_note.is_some() {
                    voice::say("Already done by the AI: move or re-record the note to do it again.");
                }
            }
            b_was = b_down;
            for h in 0..2 {
                let click = stick_click.state(&session, hand_paths[h]).map(|st| st.current_state).unwrap_or(false);
                if click && !click_was[h] {
                    if h == 1 {
                        // The right stick opens the radial menu (above); pressed and let go, it hides.
                    } else {
                        hidden.iter_mut().for_each(|x| *x = false);
                        log::info!("showing all bodies");
                    }
                }
                click_was[h] = click;
            }
            // While the model has not been handled, bring it back in front if it stays out of view
            // (e.g. the headset was put on facing another way).
            if world.is_none() && !need_anchor && views.len() >= 2 {
                let c = mul(&projection(views[0].fov, 0.05, 100.0), &mul(&view_matrix(&views[0].pose), &mul(&place, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])));
                let (x, y, w) = (c[12], c[13], c[15]);
                let visible = w > 0.0 && (x / w).abs() < 1.2 && (y / w).abs() < 1.2;
                if visible {
                    out_of_view_since = None;
                } else if out_of_view_since.get_or_insert_with(Instant::now).elapsed() > Duration::from_millis(1500) {
                    log::info!("model out of view: re-placing it in front");
                    need_anchor = true;
                    out_of_view_since = None;
                }
            }
            // Push constants: two eye matrices for a model matrix, then the tint and the section plane.
            let push_for_cut = |m: &M4, tint: [f32; 4], plane: [f32; 4]| {
                let mut out = [0.0f32; 40];
                for (i, v) in views.iter().enumerate().take(2) {
                    out[i * 16..i * 16 + 16].copy_from_slice(&mul(&projection(v.fov, 0.05, 100.0), &mul(&view_matrix(&v.pose), m)));
                }
                out[32..36].copy_from_slice(&tint);
                out[36..40].copy_from_slice(&plane);
                out
            };
            let push_for = |m: &M4, tint: [f32; 4]| push_for_cut(m, tint, NO_PLANE);
            // Extra bars drawn with the ray box (world matrices): the outline of the cut, measure points and line.
            let mut bars: Vec<(M4, [f32; 4])> = Vec::new();
            // The radial menu: a ring of blocks where the hand was when it opened, the marked one bigger and
            // light blue, and a line from the middle to the hand.
            if let Some((pm0, items, pick, _, _, hand0)) = &radial {
                // Its frame was set when it opened: upright, facing the head, a forearm's length along the ray.
                let (right, up) = ([pm0[0], pm0[1], pm0[2]], [pm0[4], pm0[5], pm0[6]]);
                let o = [pm0[12], pm0[13], pm0[14]];
                let at = |x: f32, y: f32| [0, 1, 2].map(|k| o[k] + right[k] * x + up[k] * y);
                for i in 0..items.len() {
                    let (x, y) = svr::radial::centre(i, items.len(), 0.08);
                    let (size, tint) = if *pick == Some(i) { (0.03, [0.2, 0.75, 1.0, 1.0]) } else { (0.016, [1.0, 0.8, 0.15, 1.0]) };
                    bars.push((bar(at(x, y - size / 2.0), at(x, y + size / 2.0), size), tint));
                }
                bars.push((bar(at(0.0, -0.003), at(0.0, 0.003), 0.006), [1.0, 0.8, 0.15, 1.0]));
                // The hand's move since it opened, as a line from the middle (the ring is 8 cm out; 4 cm picks).
                if let Some((pm, _)) = ray {
                    let d = sub([pm[12], pm[13], pm[14]], *hand0);
                    let (x, y) = (dot(d, right).clamp(-0.08, 0.08), dot(d, up).clamp(-0.08, 0.08));
                    if x.abs() > 0.002 || y.abs() > 0.002 {
                        bars.push((bar(o, at(x, y), 0.003), [1.0, 0.8, 0.15, 1.0]));
                    }
                }
            }
            // Dimension handles: a thin shaft out of the model and a block at its end. The one pointed at is
            // light blue; the one being dragged is green and drawn where the value now puts it.
            for (i, h) in handle_list.iter().enumerate() {
                let dragged = handle_drag.as_ref().filter(|d| d.0 == i).map(|d| (d.4 - h.value) / h.gain.max(1e-6));
                let tint = if dragged.is_some() { [0.3, 1.0, 0.45, 1.0] } else if handle_hover == Some(i) { [0.35, 0.8, 1.0, 1.0] } else { [1.0, 0.8, 0.15, 1.0] };
                let shift = dragged.unwrap_or(0.0);
                let base = [0, 1, 2].map(|k| h.at[k] + h.dir[k] * shift.min(0.0));
                let tip = [0, 1, 2].map(|k| h.at[k] + h.dir[k] * (shift + handle_len));
                let head = 0.009 / unit;
                bars.push((mul(&model, &bar(base, tip, 0.004 / unit)), tint));
                bars.push((mul(&model, &bar([0, 1, 2].map(|k| tip[k] - h.dir[k] * head), [0, 1, 2].map(|k| tip[k] + h.dir[k] * head), 0.018 / unit)), tint));
            }
            if section != Section::Off {
                // A square around the model's middle, on the plane.
                let bb = scene.bounds();
                let c = [0, 1, 2].map(|k| (bb[k] + bb[k + 3]) * 0.5);
                let half = 0.55 * dot(sub([bb[3], bb[4], bb[5]], [bb[0], bb[1], bb[2]]), sub([bb[3], bb[4], bb[5]], [bb[0], bb[1], bb[2]])).sqrt();
                let n = [cut[0], cut[1], cut[2]];
                let on = sub(c, n.map(|x| x * (dot(n, c) + cut[3])));
                let (a, b) = perpendiculars(n);
                let corner = |sa: f32, sb: f32| [0, 1, 2].map(|k| on[k] + half * (sa * a[k] + sb * b[k]));
                let pts = [corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)];
                let thick = if section == Section::Moving { 0.004 } else { 0.002 } / unit;
                for i in 0..4 {
                    bars.push((mul(&model, &bar(pts[i], pts[(i + 1) % 4], thick)), CUT_TINT));
                }
            }
            // Note pins (in the world, a fixed size whatever the model's scale): a thin post up from the
            // point and a head on top. Purple; the one pointed at brighter and bigger; the new pin waiting
            // for its words in yellow.
            let pin_bars = |bars: &mut Vec<(M4, [f32; 4])>, w: [f32; 3], head: f32, tint: [f32; 4]| {
                let top = note_head(w);
                bars.push((bar(w, top, 0.003), tint));
                bars.push((bar(sub(top, [0.0, head / 2.0, 0.0]), [top[0], top[1] + head / 2.0, top[2]], head), tint));
            };
            for (i, at) in note_at.iter().enumerate() {
                let Some((_, w)) = at else { continue };
                if let Some((di, Some((b, pin)), true)) = note_drag {
                    if di == i {
                        let w = transform(&model, [0, 1, 2].map(|k| pin[k] + off(b)[k]), 1.0);
                        pin_bars(&mut bars, w, 0.022, [1.0, 0.9, 0.2, 1.0]);
                        continue;
                    }
                }
                let (head, tint) = match (hovered_note == Some(i) || note_sel.contains(&notes[i].id), notes[i].done) {
                    _ if note_sel.contains(&notes[i].id) => (0.024, [1.0, 0.55, 0.1, 1.0]),
                    (true, false) => (0.026, [1.0, 0.75, 1.0, 1.0]),
                    (true, true) => (0.026, [0.7, 1.0, 0.75, 1.0]),
                    (false, false) => (0.018, [0.7, 0.3, 1.0, 1.0]),
                    (false, true) => (0.018, [0.25, 0.8, 0.35, 1.0]),
                };
                pin_bars(&mut bars, *w, head, tint);
            }
            if let Some((b, p)) = talk_point {
                // A green cross-hair cube where "here" is.
                let w = transform(&model, [0, 1, 2].map(|k| p[k] + off(b)[k]), 1.0);
                bars.push((bar(sub(w, [0.0, 0.006, 0.0]), [w[0], w[1] + 0.006, w[2]], 0.012), [0.3, 1.0, 0.4, 1.0]));
            }
            if let (true, Some((pm, len))) = (use_mouse, ray) {
                // The mouse pointer: a small white cube where its ray ends (no beam: it starts at the eyes),
                // the same size to the eye at any distance.
                let w = [pm[12] - pm[8] * len, pm[13] - pm[9] * len, pm[14] - pm[10] * len];
                let s = 0.007 * len.max(0.2);
                bars.push((bar(sub(w, [0.0, s / 2.0, 0.0]), [w[0], w[1] + s / 2.0, w[2]], s), [1.0, 1.0, 1.0, 1.0]));
            }
            if let Some((b, pin)) = pending_pin {
                let w = transform(&model, [0, 1, 2].map(|k| pin[k] + off(b)[k]), 1.0);
                pin_bars(&mut bars, w, 0.02, [1.0, 0.9, 0.2, 1.0]);
            }
            if guides != Guides::Off {
                // Axes from the model origin (X red, Y green, Z blue), 60% of the model's longest side long.
                let bb = scene.bounds();
                let ext = (0..3).map(|k| bb[k + 3] - bb[k]).fold(0.0f32, f32::max).max(1e-3);
                for (k, tint) in [[1.0, 0.2, 0.2, 1.0], [0.2, 0.85, 0.2, 1.0], [0.3, 0.5, 1.0, 1.0]].into_iter().enumerate() {
                    let mut q = [0.0f32; 3];
                    q[k] = ext * 0.6;
                    bars.push((mul(&model, &bar([0.0; 3], q, 0.004 / unit)), tint));
                }
                if guides == Guides::Grid {
                    // Lines every nice_step(ext / 10) across the model's footprint, on its lowest level.
                    let step = nice_step(ext / 10.0);
                    let (x0, x1) = ((bb[0] / step).floor() * step, (bb[3] / step).ceil() * step);
                    let (y0, y1) = ((bb[1] / step).floor() * step, (bb[4] / step).ceil() * step);
                    let z = bb[2];
                    let grey = [0.55, 0.6, 0.65, 1.0];
                    let t = 0.0012 / unit;
                    let (nx, ny) = (((x1 - x0) / step).round() as usize, ((y1 - y0) / step).round() as usize);
                    for i in 0..=nx.min(40) {
                        let x = x0 + i as f32 * step;
                        bars.push((mul(&model, &bar([x, y0, z], [x, y1, z], t)), grey));
                    }
                    for j in 0..=ny.min(40) {
                        let y = y0 + j as f32 * step;
                        bars.push((mul(&model, &bar([x0, y, z], [x1, y, z], t)), grey));
                    }
                }
            }
            if measuring {
                let s = 0.006 / unit;
                for p in &measure {
                    bars.push((mul(&model, &bar(sub(*p, [0.0, 0.0, s / 2.0]), [p[0], p[1], p[2] + s / 2.0], s)), MEASURE_TINT));
                }
                if measure.len() == 2 {
                    bars.push((mul(&model, &bar(measure[0], measure[1], 0.0015 / unit)), MEASURE_TINT));
                }
                // Where the next point would go: white on a surface, cyan on an edge, magenta on a corner.
                if let Some((p, kind)) = snap_now {
                    let s = 0.004 / unit;
                    let tint = match kind {
                        Snap::Corner => [1.0, 0.2, 0.9, 1.0],
                        Snap::Edge => [0.2, 0.9, 1.0, 1.0],
                        Snap::Surface => [1.0, 1.0, 1.0, 1.0],
                    };
                    bars.push((mul(&model, &bar(sub(p, [0.0, 0.0, s / 2.0]), [p[0], p[1], p[2] + s / 2.0], s)), tint));
                }
            }
            // Draw runs: consecutive bodies with the same state (and, when exploded, the same move) share
            // one draw call; hidden ones are skipped.
            let previewing = voice::preview();
            // A joint being dragged by its handle: the bodies it carries are drawn where the value now puts
            // them (turned about the joint's axis, or slid along it), until the Mac's new model arrives.
            let overlaps_shown = !clash_names.is_empty() && scene.bodies.iter().any(|b| b.name.starts_with("Overlap "));
            const LIVE: [f32; 3] = [1.0e30, 0.0, 0.0];
            let live: Option<(Vec<bool>, M4)> = handle_drag.as_ref().and_then(|d| {
                let h = handle_list.get(d.0)?;
                let (turns, o, z, names) = h.joint.as_ref()?;
                let by = d.4 - h.value;
                let m = if *turns {
                    mul(&mul(&translate(o[0], o[1], o[2]), &rot_axis(normalize(*z), by.to_radians())), &translate(-o[0], -o[1], -o[2]))
                } else {
                    translate(z[0] * by, z[1] * by, z[2] * by)
                };
                Some((scene.bodies.iter().map(|b| names.iter().any(|n| *n == b.name)).collect(), m))
            });
            let off_draw = |i: usize| if live.as_ref().is_some_and(|l| l.0[i]) { LIVE } else { off(i) };
            let mut runs: Vec<(u32, u32, Option<[f32; 4]>, [f32; 3])> = Vec::new();
            for (i, b) in scene.bodies.iter().enumerate() {
                // An overlap stays on show whatever is hidden: Isolate on the parts in it must not take it away.
                let tint = if b.name.starts_with("Overlap ") {
                    Some(OVERLAP_TINT)
                } else if hidden[i] {
                    None
                } else if clash_names.iter().any(|n| *n == b.name) {
                    // With the overlaps there as bodies, the parts are drawn as red edges only, so the
                    // overlap inside them shows; without, the parts themselves turn red.
                    if overlaps_shown { None } else { Some(CLASH_TINT) }
                } else if selected[i] {
                    Some(SELECT_TINT)
                } else if Some(i) == hovered && !edge_mode && !face_mode && !pushing {
                    Some(HOVER_TINT)
                } else if previewing {
                    Some(PREVIEW_TINT)
                } else {
                    Some([0.0; 4])
                };
                match runs.last_mut() {
                    Some(last) if last.2 == tint && last.3 == off_draw(i) && last.0 + last.1 == b.first => last.1 += b.count,
                    _ => runs.push((b.first, b.count, tint, off_draw(i))),
                }
            }

            let cmd = cmds[frame];
            vk_device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            vk_device.cmd_reset_query_pool(cmd, qpool, (frame * 2) as u32, 2);
            vk_device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qpool, (frame * 2) as u32);
            vk_device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(render_pass)
                    .framebuffer(sc.bufs[idx as usize].fb)
                    .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: sc.res })
                    .clear_values(&[
                        // Transparent where there is no model, so passthrough shows through.
                        vk::ClearValue { color: vk::ClearColorValue { float32: if passthrough_on { [0.0; 4] } else { [0.02, 0.03, 0.05, 1.0] } } },
                        vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
                    ]),
                vk::SubpassContents::INLINE,
            );
            vk_device.cmd_set_viewport(cmd, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: sc.res.width as f32, height: sc.res.height as f32, min_depth: 0.0, max_depth: 1.0 }]);
            vk_device.cmd_set_scissor(cmd, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: sc.res }]);
            vk_device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            let push = |cmd, data: &[f32; 40]| {
                let bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, PUSH_SIZE as usize);
                vk_device.cmd_push_constants(cmd, pipeline_layout, vk::ShaderStageFlags::VERTEX, 0, bytes);
            };
            vk_device.cmd_bind_vertex_buffers(cmd, 0, &[vbuf], &[0]);
            vk_device.cmd_bind_index_buffer(cmd, ibuf, 0, vk::IndexType::UINT32);
            // A body moved by the exploded view: its matrix, and the cut moved back so it stays put in the model.
            let moved = |o: [f32; 3]| match &live {
                Some((_, m)) if o == LIVE => (mul(&model, m), NO_PLANE),
                _ => (mul(&model, &translate(o[0], o[1], o[2])), [cut[0], cut[1], cut[2], cut[3] + dot([cut[0], cut[1], cut[2]], o)]),
            };
            let mut last_state = None;
            for &(first, count, tint, o) in &runs {
                let Some(tint) = tint else { continue };
                if last_state != Some((tint, o)) {
                    let (m, plane) = moved(o);
                    push(cmd, &push_for_cut(&m, tint, plane));
                    last_state = Some((tint, o));
                }
                vk_device.cmd_draw_indexed(cmd, count, 1, first, 0, 0);
            }
            let _ = index_count;
            // Edge lines for the visible bodies in dark grey (feature edges or every triangle edge), and
            // selected bodies' feature edges in orange (also when edges are off). Consecutive bodies with the
            // same colour and move share one draw.
            let mut draws: Vec<(u32, u32, [f32; 4], [f32; 3])> = Vec::new();
            for i in 0..scene.bodies.len() {
                if hidden[i] {
                    continue;
                }
                let (range, tint) = if overlaps_shown && clash_names.iter().any(|n| *n == scene.bodies[i].name) {
                    // A round part has no edges along its sides (a pin is two rings, seen from the side two
                    // dashes): its triangle edges go under the outline, dimmer, so its shape is still there.
                    if scene.feat[i].1 * 2 < scene.all[i].1 {
                        draws.push((scene.all[i].0, scene.all[i].1, [0.5, 0.1, 0.08, 1.0], off_draw(i)));
                    }
                    (scene.feat[i], [1.0, 0.15, 0.1, 1.0])
                } else if selected[i] {
                    (scene.feat[i], SELECT_EDGE_TINT)
                } else {
                    match edges {
                        Edges::Off => continue,
                        Edges::Feature => (scene.feat[i], [0.06, 0.06, 0.08, 1.0]),
                        Edges::All => (scene.all[i], [0.06, 0.06, 0.08, 1.0]),
                    }
                };
                if range.1 == 0 {
                    continue;
                }
                match draws.last_mut() {
                    Some(last) if last.2 == tint && last.3 == off_draw(i) && last.0 + last.1 == range.0 => last.1 += range.1,
                    _ => draws.push((range.0, range.1, tint, off_draw(i))),
                }
            }
            if !draws.is_empty() {
                vk_device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, edge_pipeline);
                vk_device.cmd_bind_index_buffer(cmd, ebuf, 0, vk::IndexType::UINT32);
                let mut last_state = None;
                for (first, count, tint, o) in draws {
                    if last_state != Some((tint, o)) {
                        let (m, plane) = moved(o);
                        push(cmd, &push_for_cut(&m, tint, plane));
                        last_state = Some((tint, o));
                    }
                    vk_device.cmd_draw_indexed(cmd, count, 1, first, 0, 0);
                }
                vk_device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            }
            // Picked edges orange, the edge pointed at (in edge mode) light blue.
            {
                let mut list: Vec<(&EdgeChain, [f32; 4])> = selected_edges.iter().map(|c| (c, [1.0, 0.55, 0.1, 1.0])).collect();
                if let Some(h) = &hover_edge {
                    if !selected_edges.iter().any(|e| e.body == h.body && e.key() == h.key()) {
                        list.push((h, [0.35, 0.8, 1.0, 1.0]));
                    }
                }
                if !list.is_empty() {
                    vk_device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, edge_pipeline);
                    vk_device.cmd_bind_index_buffer(cmd, ebuf, 0, vk::IndexType::UINT32);
                    for (c, tint) in list {
                        if hidden[c.body] {
                            continue;
                        }
                        // Lines are one pixel wide, so the edge is drawn seven times, a millimetre apart, to show up.
                        let d = 0.001 / unit;
                        for sh in [[0.0, 0.0, 0.0], [d, 0.0, 0.0], [-d, 0.0, 0.0], [0.0, d, 0.0], [0.0, -d, 0.0], [0.0, 0.0, d], [0.0, 0.0, -d]] {
                            let o = off(c.body);
                            let (m, plane) = moved([o[0] + sh[0], o[1] + sh[1], o[2] + sh[2]]);
                            push(cmd, &push_for_cut(&m, tint, plane));
                            for &k in &c.segs {
                                vk_device.cmd_draw_indexed(cmd, 2, 1, k, 0, 0);
                            }
                        }
                    }
                    vk_device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                }
            }
            // Face highlights: selected faces orange, the face pointed at (in face mode) light blue.
            {
                let mut list: Vec<(u32, [f32; 4])> = selected_faces.iter().map(|sf| (sf.0, [1.0, 0.55, 0.1, 0.55])).collect();
                if let (true, Some(f), None) = (face_mode || pushing, hovered_face, pp_drag) {
                    if !selected_faces.iter().any(|sf| sf.0 == f) {
                        list.push((f, [0.35, 0.8, 1.0, 0.5]));
                    }
                }
                // The face being pushed or pulled, drawn where it would go (green).
                let pp_preview = pp_drag.map(|(f, _, _, _, n_m, _, mm)| (f, n_m.map(|x| x * mm)));
                let (fb, _, fp) = face_bufs[frame];
                let mut written = 0usize;
                let mut draws: Vec<(u32, u32, [f32; 4], [f32; 3])> = Vec::new();
                for (f, tint, shift) in list.into_iter().map(|(f, t)| (f, t, [0.0f32; 3])).chain(pp_preview.map(|(f, s)| (f, [0.3, 1.0, 0.45, 0.6], s))) {
                    let b = scene.patch_body[f as usize] as usize;
                    if hidden[b] {
                        continue;
                    }
                    let start = written;
                    for &t in &scene.patch_tris[f as usize] {
                        if written + 3 > FACE_MAX_INDICES {
                            break;
                        }
                        let tri = &scene.indices[t as usize * 3..t as usize * 3 + 3];
                        std::ptr::copy_nonoverlapping(tri.as_ptr() as *const u8, fp.add(written * 4), 12);
                        written += 3;
                    }
                    if written > start {
                        draws.push((start as u32, (written - start) as u32, tint, [0, 1, 2].map(|k| off(b)[k] + shift[k])));
                    }
                }
                if !draws.is_empty() {
                    vk_device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, face_pipeline);
                    vk_device.cmd_bind_vertex_buffers(cmd, 0, &[vbuf], &[0]);
                    vk_device.cmd_bind_index_buffer(cmd, fb, 0, vk::IndexType::UINT32);
                    for (first, count, tint, o) in draws {
                        let (m, plane) = moved(o);
                        push(cmd, &push_for_cut(&m, tint, plane));
                        vk_device.cmd_draw_indexed(cmd, count, 1, first, 0, 0);
                    }
                    vk_device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                }
            }
            if let Some(pnl) = &panel {
                let n_hidden = hidden.iter().filter(|&&h| h).count();
                let n_sel = selected.iter().filter(|&&s| s).count();
                let first_sel = selected.iter().position(|&s| s);
                // Title: the part under the pointer, else the selection; subtitle: counts and a hint.
                // Size of a part's bounding box (model units are millimetres).
                let size = |b: usize| {
                    let bb = scene.bodies[b].aabb;
                    let f = |v: f32| if v >= 100.0 { format!("{v:.0}") } else { format!("{v:.1}") };
                    format!("{} x {} x {} mm", f(bb[3] - bb[0]), f(bb[4] - bb[1]), f(bb[5] - bb[2]))
                };
                let snap_text = match snap_now {
                    Some((_, Snap::Corner)) => " on the corner",
                    Some((_, Snap::Edge)) => " on the edge",
                    Some((_, Snap::Surface)) => " on the surface",
                    None => "",
                };
                let (title, subtitle) = if measuring {
                    match measure.len() {
                        2 => {
                            let d = sub(measure[1], measure[0]);
                            (format!("{:.1} mm", dot(d, d).sqrt()), format!("x {:.1}  y {:.1}  z {:.1}  -  trigger: new A", d[0].abs(), d[1].abs(), d[2].abs()))
                        }
                        1 => ("Measure: point at B".to_string(), format!("Trigger puts down B{}", snap_text)),
                        _ => ("Measure: point at A".to_string(), format!("Trigger puts down A{}", snap_text)),
                    }
                } else if section == Section::Moving {
                    ("Moving the cut".to_string(), "Hold the trigger and move your hand".to_string())
                } else if let Some((_, b, _, _, _, _, mm)) = pp_drag {
                    (format!("Push/pull: {mm:+.1} mm"), format!("{}  -  {} mm steps, release to apply", scene.bodies[b].name, pp_step(unit)))
                } else if pushing {
                    ("Push/pull: point at a face".to_string(), "Hold the trigger, move along the face, let go".to_string())
                } else if let Some(i) = hovered_note {
                    (
                        format!("{} on {}", if notes[i].done { "Done note" } else { "Note" }, notes[i].body),
                        if noting { "Drag to move, click to select  -  B: AI".to_string() } else { "Trigger selects  -  B: the AI does it".to_string() },
                    )
                } else if noting {
                    if pending_pin.is_some() {
                        ("Pinned: now say the note".to_string(), "Hold the left trigger and speak".to_string())
                    } else if one_sel.len() == 1 {
                        ("1 note selected".to_string(), "Left trigger re-records it  -  B: AI does it".to_string())
                    } else {
                        ("Notes: point at a part".to_string(), format!("Right trigger pins a note  -  {} notes", notes.len()))
                    }
                } else {
                    let title = match (hovered, n_sel, first_sel) {
                        (Some(h), _, _) if edge_mode => format!("Edge of {}", scene.bodies[h].name),
                        (None, _, _) if edge_mode && !selected_edges.is_empty() => format!("{} edges selected", selected_edges.len()),
                        (Some(h), _, _) if face_mode => format!("Face of {}", scene.bodies[h].name),
                        (None, _, _) if face_mode && !selected_faces.is_empty() => format!("{} faces selected", selected_faces.len()),
                        (Some(h), _, _) => scene.bodies[h].name.clone(),
                        (None, 1, Some(s)) => scene.bodies[s].name.clone(),
                        (None, 0, _) => "Point at a part".to_string(),
                        (None, n, _) => format!("{n} parts selected"),
                    };
                    let subtitle = match (hovered, n_sel, first_sel) {
                        _ if edge_mode => format!("Pick: edges  -  {} selected  -  speak to change them", selected_edges.len()),
                        _ if face_mode => format!("Pick: faces  -  {} selected  -  speak to change them", selected_faces.len()),
                        (Some(h), _, _) => format!("{}  -  {n_sel} selected", size(h)),
                        (None, 1, Some(s)) => size(s),
                        (None, 0, _) if explode > 0.0 => format!("{} parts, exploded {:.0}% (left stick)", scene.bodies.len(), explode * 100.0),
                        (None, 0, _) => format!("{} parts, {n_hidden} hidden", scene.bodies.len()),
                        (None, _, _) => format!("{n_sel} selected, {n_hidden} hidden"),
                    };
                    (title, subtitle)
                };
                let size_idx = {
                    let w = world.unwrap_or(cur);
                    let sc = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
                    let now = inner_scale * 1000.0 * sc;
                    SIZE_STEPS.iter().position(|&x| (now / x - 1.0).abs() < 0.05)
                };
                // For tests without a headset (SCVIEW_DEBUG_RAY=1): what the viewer holds, ten times a second.
                if debug_ray {
                    debug_state_n = debug_state_n.wrapping_add(1);
                    if debug_state_n % 6 == 0 {
                        let names = |v: &Vec<bool>| scene.bodies.iter().zip(v).filter(|(_, &on)| on).map(|(b, _)| b.name.as_str()).collect::<Vec<_>>().join(",");
                        log::info!(
                            "VIEWSTATE hover={} sel=[{}] faces={} edges={} hidden=[{}] section={:?} cut={:?} measuring={} measure={:?} explode={:.2} mode={} pushing={} pull={:?} handles={} handle={} dragging={} moving={} move={} clash=[{}] live={} buzzes={} radial={} ran={}",
                            hovered.map_or("-", |i| scene.bodies[i].name.as_str()),
                            names(&selected),
                            selected_faces.len(),
                            selected_edges.len(),
                            names(&hidden),
                            section,
                            cut,
                            measuring,
                            measure,
                            explode,
                            if edge_mode { "edges" } else if face_mode { "faces" } else { "parts" },
                            pushing,
                            pp_drag.as_ref().map(|d| d.6),
                            handle_list.len(),
                            handle_hover.and_then(|i| handle_list.get(i)).map_or("-".to_string(), |h| h.label.replace(' ', "_")),
                            handle_drag.as_ref().map_or("-".to_string(), |d| format!("{}", d.4)),
                            moving,
                            move_drag.as_ref().map_or("-".to_string(), |d| format!("{}:{},{},{}", scene.bodies[d.0].name, d.4[0], d.4[1], d.4[2])),
                            clash_names.join(","),
                            handle_drag.as_ref().and_then(|d| handle_list.get(d.0)).and_then(|h| h.joint.as_ref()).map_or(0, |j| j.3.len()),
                            buzzes,
                            radial.as_ref().map_or("-".to_string(), |r| format!("{}:{}", r.4, r.2.map_or("-".to_string(), |i| format!("{:?}", r.1[i])))),
                            radial_ran
                        );
                    }
                }
                let st = PanelState { hover: hover_btn, has_selection: n_sel > 0 || !selected_faces.is_empty() || !selected_edges.is_empty(), can_undo: !undo.is_empty(), room_off: !passthrough_on, section, measuring, edges, exploded: explode > 0.01, noting, open_notes: notes.iter().filter(|n| !n.done).count(), notes_selected: notes.iter().filter(|n| note_sel.contains(&n.id)).count(), can_redo: !redo.is_empty(), faces: face_mode, pick_edges: edge_mode, pushing, placing, size_idx, guides, tab, kb_open, undo_what: undo.last().map_or("", step_what), redo_what: redo.last().map_or("", step_what), reply_page, dims_page, feat_page, handles: handles_on, moving, clash: !clash_names.is_empty(), overlay, overlay_hover };
                let ui = build_panel(&title, &subtitle, &st, &lay, hovered_note.map(|i| notes[i].text.as_str()));
                let nv = (ui.verts.len() / VERTEX_SIZE).min(UI_MAX_VERTS);
                let ni = ui.indices.len().min(UI_MAX_INDICES) / 6 * 6;
                let ni = ui.indices[..ni].iter().take_while(|&&i| (i as usize) < nv).count() / 6 * 6;
                let (vb, _, vp, ib, _, ip) = ui_bufs[frame];
                std::ptr::copy_nonoverlapping(ui.verts.as_ptr(), vp, nv * VERTEX_SIZE);
                std::ptr::copy_nonoverlapping(ui.indices.as_ptr() as *const u8, ip, ni * 4);
                push(cmd, &push_for(pnl, [0.0; 4]));
                vk_device.cmd_bind_vertex_buffers(cmd, 0, &[vb], &[0]);
                vk_device.cmd_bind_index_buffer(cmd, ib, 0, vk::IndexType::UINT32);
                vk_device.cmd_draw_indexed(cmd, ni as u32, 1, 0, 0, 0);
            }
            if let Some(km) = &kb_m {
                // The floating keyboard, drawn from the second set of UI buffers.
                let kui = build_keyboard(&kb_lay, kb_hover, kb_shift, &voice::typed());
                let nv = (kui.verts.len() / VERTEX_SIZE).min(UI_MAX_VERTS);
                let ni = kui.indices.len().min(UI_MAX_INDICES) / 6 * 6;
                let ni = kui.indices[..ni].iter().take_while(|&&i| (i as usize) < nv).count() / 6 * 6;
                let (vb, _, vp, ib, _, ip) = ui_bufs[PIPELINE_DEPTH as usize + frame];
                std::ptr::copy_nonoverlapping(kui.verts.as_ptr(), vp, nv * VERTEX_SIZE);
                std::ptr::copy_nonoverlapping(kui.indices.as_ptr() as *const u8, ip, ni * 4);
                push(cmd, &push_for(km, [0.0; 4]));
                vk_device.cmd_bind_vertex_buffers(cmd, 0, &[vb], &[0]);
                vk_device.cmd_bind_index_buffer(cmd, ib, 0, vk::IndexType::UINT32);
                vk_device.cmd_draw_indexed(cmd, ni as u32, 1, 0, 0, 0);
            }
            // The radial menu's names, each beside its block, on the ring's own upright plane. They use the
            // keyboard's buffers, so they are left out while the keyboard is open (the panel still lists them).
            if let (None, Some((pm0, items, pick, _, _, _))) = (&kb_m, &radial) {
                let mut rui = Ui { verts: Vec::new(), indices: Vec::new() };
                let fp = FONT_PX * 1.7;
                let h = font::FONT_H as f32 * fp;
                for (i, b) in items.iter().enumerate() {
                    let (x, y) = svr::radial::centre(i, items.len(), 0.108);
                    let name = radial_label(*b);
                    let w = Ui::text_width_px(name, fp);
                    // Outward of the block: to its right on the right half, to its left on the left, centred above and below.
                    let x0 = if x > 0.02 { x } else if x < -0.02 { x - w } else { x - w / 2.0 };
                    let top = if y > 0.02 { y + h } else if y < -0.02 { y } else { y + h / 2.0 };
                    let picked = *pick == Some(i);
                    rui.quad(x0 - 0.004, top - h - 0.003, x0 + w + 0.004, top + 0.003, 0.0, if picked { [30, 110, 180] } else { [18, 24, 34] });
                    rui.text_px(name, x0, top, 0.001, if picked { [255, 255, 255] } else { [215, 225, 240] }, fp);
                }
                let nv = (rui.verts.len() / VERTEX_SIZE).min(UI_MAX_VERTS);
                let ni = rui.indices.len().min(UI_MAX_INDICES) / 6 * 6;
                let ni = rui.indices[..ni].iter().take_while(|&&i| (i as usize) < nv).count() / 6 * 6;
                let (vb, _, vp, ib, _, ip) = ui_bufs[PIPELINE_DEPTH as usize + frame];
                std::ptr::copy_nonoverlapping(rui.verts.as_ptr(), vp, nv * VERTEX_SIZE);
                std::ptr::copy_nonoverlapping(rui.indices.as_ptr() as *const u8, ip, ni * 4);
                push(cmd, &push_for(pm0, [0.0; 4]));
                vk_device.cmd_bind_vertex_buffers(cmd, 0, &[vb], &[0]);
                vk_device.cmd_bind_index_buffer(cmd, ib, 0, vk::IndexType::UINT32);
                vk_device.cmd_draw_indexed(cmd, ni as u32, 1, 0, 0, 0);
            }
            if let (Some((pm, len)), false) = (ray, use_mouse) {
                let mut s = identity();
                s[0] = 0.004;
                s[5] = 0.004;
                s[10] = len;
                push(cmd, &push_for(&mul(&pm, &s), [0.0; 4]));
                vk_device.cmd_bind_vertex_buffers(cmd, 0, &[ray_vbuf], &[0]);
                vk_device.cmd_bind_index_buffer(cmd, ray_ibuf, 0, vk::IndexType::UINT32);
                vk_device.cmd_draw_indexed(cmd, ray_indices.len() as u32, 1, 0, 0, 0);
            }
            // Setting down: a square on the horizontal plane through the controller tip, with a post up from it.
            if let Some(hit) = place_hit {
                let h = 0.10;
                let c = [[-h, -h], [h, -h], [h, h], [-h, h]].map(|[x, z]| [hit[0] + x, hit[1], hit[2] + z]);
                // Green: on a surface the room scan found; yellow: only a point along the ray (right stick moves it).
                let tint = if place_on_surface { [0.3, 1.0, 0.4, 1.0] } else { [1.0, 0.85, 0.2, 1.0] };
                for i in 0..4 {
                    bars.push((bar(c[i], c[(i + 1) % 4], 0.004), tint));
                }
                bars.push((bar(hit, [hit[0], hit[1] + 0.15, hit[2]], 0.004), tint));
            }
            if !bars.is_empty() {
                vk_device.cmd_bind_vertex_buffers(cmd, 0, &[ray_vbuf], &[0]);
                vk_device.cmd_bind_index_buffer(cmd, ray_ibuf, 0, vk::IndexType::UINT32);
                for (m, tint) in &bars {
                    push(cmd, &push_for(m, *tint));
                    vk_device.cmd_draw_indexed(cmd, ray_indices.len() as u32, 1, 0, 0, 0);
                }
            }
            vk_device.cmd_end_render_pass(cmd);
            vk_device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qpool, (frame * 2 + 1) as u32);
            vk_device.end_command_buffer(cmd).unwrap();
            ts_valid[frame] = true;

            sc.handle.wait_image(xr::Duration::INFINITE).unwrap();
            vk_device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fences[frame]).unwrap();
            sc.handle.release_image().unwrap();

            let rect = xr::Rect2Di { offset: xr::Offset2Di { x: 0, y: 0 }, extent: xr::Extent2Di { width: sc.res.width as _, height: sc.res.height as _ } };
            let layer_view = |i: usize| {
                xr::CompositionLayerProjectionView::new()
                    .pose(views[i].pose)
                    .fov(views[i].fov)
                    .sub_image(xr::SwapchainSubImage::new().swapchain(&sc.handle).image_array_index(i as u32).image_rect(rect))
            };
            let views_arr = [layer_view(0), layer_view(1)];
            let projection = xr::CompositionLayerProjection::new()
                .space(&space)
                .layer_flags(if passthrough_on { xr::CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA } else { xr::CompositionLayerFlags::EMPTY })
                .views(&views_arr);
            // Just after a resume (e.g. back from Bluetooth settings) the head pose can be invalid, and the
            // runtime refuses a projection layer with ERROR_POSE_INVALID: submit an empty frame instead.
            let pose_ok = view_flags.contains(xr::ViewStateFlags::ORIENTATION_VALID);
            let ended = match (&passthrough, passthrough_on && pose_ok) {
                _ if !pose_ok => frame_stream.end(fs.predicted_display_time, blend, &[]),
                (Some((_, layer)), true) => {
                    let pt_layer = xr::sys::CompositionLayerPassthroughFB {
                        ty: xr::sys::CompositionLayerPassthroughFB::TYPE,
                        next: std::ptr::null(),
                        flags: xr::CompositionLayerFlags::EMPTY,
                        space: xr::sys::Space::NULL,
                        layer_handle: *layer.inner(),
                    };
                    // Same header layout as XrCompositionLayerBaseHeader, which the runtime reads first.
                    let base = &*(&pt_layer as *const xr::sys::CompositionLayerPassthroughFB as *const xr::CompositionLayerBase<'_, xr::Vulkan>);
                    frame_stream.end(fs.predicted_display_time, blend, &[base, &projection])
                }
                _ => frame_stream.end(fs.predicted_display_time, blend, &[&projection]),
            };
            if let Err(e) = ended {
                log::warn!("end frame failed: {e:?}");
            }
            frame = (frame + 1) % PIPELINE_DEPTH as usize;

            let now = Instant::now();
            wall_sum += now.duration_since(last).as_secs_f64() * 1000.0;
            last = now;
            n += 1;
            if n == 120 {
                log::info!(
                    "{} tris: frame {:.2} ms ({:.1} fps), GPU {:.2} ms avg / {:.2} ms max, pick {:.2} ms avg, {} draws",
                    tris,
                    wall_sum / n as f64,
                    1000.0 * n as f64 / wall_sum,
                    if gpu_n > 0 { gpu_sum / gpu_n as f64 } else { 0.0 },
                    gpu_max,
                    pick_ms_sum / n as f64,
                    runs.iter().filter(|r| r.2.is_some()).count()
                );
                pick_ms_sum = 0.0;
                n = 0;
                wall_sum = 0.0;
                gpu_sum = 0.0;
                gpu_max = 0.0;
                gpu_n = 0;
            }
        }

        drop((session, frame_wait, frame_stream, space));
        vk_device.wait_for_fences(&fences, true, !0).unwrap();
        for f in fences {
            vk_device.destroy_fence(f, None);
        }
        vk_device.destroy_query_pool(qpool, None);
        if let Some(sc) = swapchain {
            for b in sc.bufs {
                vk_device.destroy_framebuffer(b.fb, None);
                vk_device.destroy_image_view(b.color, None);
                vk_device.destroy_image_view(b.depth, None);
                vk_device.destroy_image(b.depth_img, None);
                vk_device.free_memory(b.depth_mem, None);
                vk_device.destroy_image_view(b.msaa, None);
                vk_device.destroy_image(b.msaa_img, None);
                vk_device.free_memory(b.msaa_mem, None);
            }
        }
        vk_device.destroy_buffer(vbuf, None);
        vk_device.free_memory(vmem, None);
        vk_device.destroy_buffer(ibuf, None);
        vk_device.free_memory(imem, None);
        for &(vb, vm, _, ib, im, _) in &ui_bufs {
            vk_device.destroy_buffer(vb, None);
            vk_device.free_memory(vm, None);
            vk_device.destroy_buffer(ib, None);
            vk_device.free_memory(im, None);
        }
        vk_device.destroy_buffer(ray_vbuf, None);
        vk_device.free_memory(ray_vmem, None);
        vk_device.destroy_buffer(ray_ibuf, None);
        vk_device.free_memory(ray_imem, None);
        vk_device.destroy_pipeline(pipeline, None);
        vk_device.destroy_pipeline(edge_pipeline, None);
        vk_device.destroy_pipeline(face_pipeline, None);
        for &(b, m, _) in &face_bufs {
            vk_device.destroy_buffer(b, None);
            vk_device.free_memory(m, None);
        }
        vk_device.destroy_buffer(ebuf, None);
        vk_device.free_memory(emem, None);
        vk_device.destroy_pipeline_layout(pipeline_layout, None);
        vk_device.destroy_command_pool(cmd_pool, None);
        vk_device.destroy_render_pass(render_pass, None);
        vk_device.destroy_device(None);
        vk_instance.destroy_instance(None);
    }
}

#[cfg(test)]
mod dim_tests {
    use super::{dim_step_to, fmt_dim};

    #[test]
    fn a_dimension_steps_by_a_round_tenth_and_lands_on_multiples() {
        assert_eq!(dim_step_to(60.0, 1.0), 65.0);
        assert_eq!(dim_step_to(60.0, -1.0), 55.0);
        assert_eq!(dim_step_to(10.0, 1.0), 11.0);
        assert_eq!(dim_step_to(6.0, -1.0), 5.5);
        // A value between steps goes to the next multiple, not past it.
        assert_eq!(dim_step_to(62.0, 1.0), 65.0);
        assert_eq!(dim_step_to(62.0, -1.0), 60.0);
        assert_eq!(dim_step_to(0.3, 1.0), 0.32);
    }

    #[test]
    fn a_long_list_is_paged_and_a_short_one_is_not() {
        use super::{dims_paging, MODELS_SHOWN};
        assert_eq!(dims_paging(0), (1, 1));
        assert_eq!(dims_paging(6), (6, 1));
        assert_eq!(dims_paging(MODELS_SHOWN), (MODELS_SHOWN, 1));
        // One more than fits: the last row becomes "next page", so 13 to a page.
        assert_eq!(dims_paging(MODELS_SHOWN + 1), (MODELS_SHOWN - 1, 2));
        assert_eq!(dims_paging(23), (13, 2));
        assert_eq!(dims_paging(26), (13, 2));
        assert_eq!(dims_paging(27), (13, 3));
    }

    #[test]
    fn a_value_is_shown_without_trailing_zeros() {
        assert_eq!(fmt_dim(60.0), "60");
        assert_eq!(fmt_dim(5.5), "5.5");
        assert_eq!(fmt_dim(0.125), "0.125");
    }
}
