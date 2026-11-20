//! Precision for anything dragged by hand: a face being pulled, a dimension's handle, a part being moved.
//!
//! A drag follows the hand one to one. While the same hand's grip is held it follows at a tenth
//! ([`FINE`]) and lands on a finer step, so a value can be set to the half millimetre with a hand that
//! cannot hold still to the millimetre. The grip can be pressed and let go during one drag: the value
//! carries on from where it is and never jumps. Plain numbers, no headset types: tested here, and written
//! like the rest of this module so it can be offered to SolidCraft's own VR viewer.

/// How much of the hand's movement a fine drag takes.
pub const FINE: f64 = 0.1;

/// One drag, from the press to the release.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drag {
    /// The value so far, before snapping (the unit the positions are given in).
    value: f64,
    /// Where the hand was along the drag's direction at the last update.
    last: f64,
}

impl Drag {
    /// A drag that starts now, with the hand at `at` along the drag's direction and the value at `value`.
    pub fn start(at: f64, value: f64) -> Drag {
        Drag { value, last: at }
    }

    /// The hand is now at `at`: the value moves by what the hand moved since the last update, a tenth of
    /// it while `fine`. Returns the value, not yet snapped. A position that is no number changes nothing.
    pub fn update(&mut self, at: f64, fine: bool) -> f64 {
        if at.is_finite() {
            self.value += (at - self.last) * if fine { FINE } else { 1.0 };
            self.last = at;
        }
        self.value
    }

    /// The value so far, not snapped.
    pub fn value(&self) -> f64 {
        self.value
    }
}

/// `value` on the nearest multiple of `step` (as it is when the step is no positive number).
pub fn snap(value: f64, step: f64) -> f64 {
    if step.is_finite() && step > 0.0 { (value / step).round() * step } else { value }
}

/// The step a fine drag lands on, for the step a normal drag uses: a fifth of it, but not below `least`
/// (what the thing being dragged can take: SolidCraft refuses a face moved by less than half a millimetre).
pub fn fine_step(step: f64, least: f64) -> f64 {
    (step / 5.0).max(least)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_drag_follows_the_hand_and_a_fine_drag_a_tenth_of_it() {
        let mut d = Drag::start(0.0, 0.0);
        assert_eq!(d.update(100.0, false), 100.0);
        let mut f = Drag::start(0.0, 0.0);
        assert!((f.update(100.0, true) - 10.0).abs() < 1e-12);
    }

    /// Pressing or letting go of the grip mid-drag carries on from the value so far: no jump.
    #[test]
    fn switching_to_fine_and_back_never_jumps() {
        let mut d = Drag::start(5.0, 20.0);
        assert_eq!(d.update(15.0, false), 30.0);
        // The grip goes down with the hand still: nothing moves.
        assert_eq!(d.update(15.0, true), 30.0);
        assert!((d.update(25.0, true) - 31.0).abs() < 1e-12);
        // And up again: on from 31, one to one.
        assert!((d.update(25.0, false) - 31.0).abs() < 1e-12);
        assert!((d.update(20.0, false) - 26.0).abs() < 1e-12);
        assert_eq!(d.value(), d.update(f64::NAN, false));
    }

    #[test]
    fn values_land_on_steps() {
        assert_eq!(snap(13.4, 1.0), 13.0);
        assert_eq!(snap(-2.6, 0.5), -2.5);
        assert_eq!(snap(7.3, 0.0), 7.3);
        assert_eq!(snap(7.3, f64::NAN), 7.3);
        assert_eq!(fine_step(5.0, 0.5), 1.0);
        assert_eq!(fine_step(1.0, 0.5), 0.5);
        assert!((fine_step(1.0, 0.0) - 0.2).abs() < 1e-12);
    }
}
