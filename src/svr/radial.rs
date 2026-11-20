//! A radial menu's arithmetic: which sector a push of the thumbstick, or a move of the hand, points
//! at, and where each sector sits. This viewer's own (not from the pull request).
//!
//! Sectors are numbered clockwise from the one straight up. The middle is dead: a small push or a
//! hand that has barely moved picks nothing, so letting go there cancels.

use std::f32::consts::TAU;

/// The most sectors a menu may have (the ergonomics rule: seven or eight choices a level).
pub const MOST: usize = 8;

/// The sector that (x to the right, y up) points at, of `n`, or None inside the dead middle.
pub fn sector(x: f32, y: f32, n: usize, dead: f32) -> Option<usize> {
    if n == 0 || !x.is_finite() || !y.is_finite() || (x * x + y * y).sqrt() < dead {
        return None;
    }
    let step = TAU / n as f32;
    // The angle clockwise from up, shifted by half a sector so each sector is centred on its direction.
    let a = (x.atan2(y) + step / 2.0).rem_euclid(TAU);
    Some(((a / step) as usize).min(n - 1))
}

/// Where sector `i` of `n` sits on a ring of radius `r`: (to the right, up).
pub fn centre(i: usize, n: usize, r: f32) -> (f32, f32) {
    let a = TAU * i as f32 / n.max(1) as f32;
    (r * a.sin(), r * a.cos())
}

/// A word for the way sector `i` of `n` lies, for a line of text ("up", "up right", ...).
pub fn way(i: usize, n: usize) -> &'static str {
    const WAYS: [&str; 8] = ["up", "up right", "right", "down right", "down", "down left", "left", "up left"];
    WAYS[((i * 8 + n / 2) / n.max(1)) % 8]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_middle_picks_nothing() {
        assert_eq!(sector(0.0, 0.0, 8, 0.5), None);
        assert_eq!(sector(0.3, 0.3, 8, 0.5), None);
        assert_eq!(sector(f32::NAN, 1.0, 8, 0.5), None);
        assert_eq!(sector(1.0, 0.0, 0, 0.5), None);
    }

    #[test]
    fn eight_sectors_go_clockwise_from_up() {
        let at = |x, y| sector(x, y, 8, 0.5);
        assert_eq!(at(0.0, 1.0), Some(0));
        assert_eq!(at(0.7, 0.7), Some(1));
        assert_eq!(at(1.0, 0.0), Some(2));
        assert_eq!(at(0.7, -0.7), Some(3));
        assert_eq!(at(0.0, -1.0), Some(4));
        assert_eq!(at(-0.7, -0.7), Some(5));
        assert_eq!(at(-1.0, 0.0), Some(6));
        assert_eq!(at(-0.7, 0.7), Some(7));
    }

    #[test]
    fn a_sector_is_centred_on_its_direction() {
        // Just either side of up is still up; a little under halfway to the next sector too.
        assert_eq!(sector(0.2, 1.0, 8, 0.5), Some(0));
        assert_eq!(sector(-0.2, 1.0, 8, 0.5), Some(0));
        assert_eq!(sector(0.5, 1.0, 8, 0.5), Some(1));
        // Six sectors: straight right falls between the second and third, and goes to the third.
        assert_eq!(sector(0.0, 1.0, 6, 0.5), Some(0));
        assert_eq!(sector(0.0, -1.0, 6, 0.5), Some(3));
    }

    #[test]
    fn each_centre_is_in_its_own_sector() {
        for n in 1..=MOST {
            for i in 0..n {
                let (x, y) = centre(i, n, 1.0);
                assert_eq!(sector(x, y, n, 0.5), Some(i), "sector {i} of {n}");
            }
        }
    }

    #[test]
    fn the_ways_have_names() {
        assert_eq!((0..8).map(|i| way(i, 8)).collect::<Vec<_>>(), ["up", "up right", "right", "down right", "down", "down left", "left", "up left"]);
        assert_eq!((0..4).map(|i| way(i, 4)).collect::<Vec<_>>(), ["up", "right", "down", "left"]);
    }
}
