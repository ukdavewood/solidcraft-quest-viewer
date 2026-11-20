//! The model in the room and the world grab, shared with SolidCraft's own VR viewer.
//!
//! `room.rs` and `grab.rs` are copies of `crates/vr/src/room.rs` and `crates/vr/src/grab.rs`, and
//! [`Pose`] and [`vector`] of parts of `crates/xr/src/view.rs` and `crates/vr/src/lib.rs`, from
//! SolidCraft pull request #18712 (https://github.com/tangency-labs/solidcraft/pull/18712, by
//! Sebastian Höpfl, at commit 0219f09c3), copyright SolidCraft contributors, licensed MIT OR
//! Apache-2.0 like this repository. They are kept as close to the originals as possible, so a later
//! version can be copied over them: only the `use` lines differ, and `Transform::placement` (the
//! SolidCraft renderer's matrix type) and its test are left out. Changes to the behaviour belong upstream.
//!
//! What they give this viewer: one tested implementation of holding the model with one hand and
//! scaling it with two, with the snap to real size, so a design handles the same way here and in
//! the desktop app's VR mode.
//!
//! `precise.rs` is this viewer's own (not from the pull request): the slow, snapped drag every tool shares.
//! So is `radial.rs`: the sector arithmetic of the radial menu.
#![allow(dead_code)]

pub mod grab;
pub mod precise;
pub mod radial;
pub mod room;

use vector::finite;

/// A position and an orientation in the room: what a headset reports for an eye, and what a flat
/// layer is placed with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Metres.
    pub position: [f64; 3],
    /// A unit quaternion (x, y, z, w).
    pub orientation: [f64; 4],
}

impl Pose {
    /// At the room's origin, looking down −Z.
    pub const IDENTITY: Pose = Pose { position: [0.0; 3], orientation: [0.0, 0.0, 0.0, 1.0] };

    /// The pose's own right, up and back directions in the room, or `None` when the orientation is
    /// not a rotation (a headset that lost tracking may report zeros).
    pub fn axes(&self) -> Option<[[f64; 3]; 3]> {
        let [x, y, z, w] = self.orientation;
        let length = (x * x + y * y + z * z + w * w).sqrt();
        if !(length.is_finite() && length > 1e-6) || !finite(self.position) {
            return None;
        }
        let (x, y, z, w) = (x / length, y / length, z / length, w / length);
        Some([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)],
            [2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)],
            [2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)],
        ])
    }
}

/// Small vector helpers for the room's metres.
pub mod vector {
    /// The dot product.
    pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// The cross product, right-handed.
    pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    }

    /// `a` + `b`.
    pub fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }

    /// `a` − `b`.
    pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    /// `a` made `s` times as long.
    pub fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
        [a[0] * s, a[1] * s, a[2] * s]
    }

    /// The length of `a`.
    pub fn length(a: [f64; 3]) -> f64 {
        dot(a, a).sqrt()
    }

    /// Whether every component of `a` is a finite number.
    pub fn finite(a: [f64; 3]) -> bool {
        a.iter().all(|x| x.is_finite())
    }

    /// `a` at length 1, or `None` when it has no direction.
    pub fn unit(a: [f64; 3]) -> Option<[f64; 3]> {
        let l = length(a);
        (l.is_finite() && l > 1e-9).then(|| scale(a, 1.0 / l))
    }

    /// The 3 × 3 matrix with these `columns` applied to `v`.
    pub fn turn(columns: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
        add(add(scale(columns[0], v[0]), scale(columns[1], v[1])), scale(columns[2], v[2]))
    }
}
