//! One list of the viewer's commands. The panel's buttons, the keyboard and spoken or typed phrases all
//! point at it, so an input can be swapped or added (a radial menu, gestures) without touching what a
//! command does, and the three can never drift apart.
//!
//! `docs/commands.md` (for people) and `tools/commands.tsv` (the phrases, read by `serve.py`) are written
//! from this table: run `SCVIEW_WRITE_COMMANDS=1 cargo test --lib commands` after changing it. A test
//! fails while they differ.

#![allow(dead_code)] // the labels and the two tables are for the written files and the tests

use crate::{Button, Keycode};
use Keycode as K;

pub struct Command {
    /// A stable name: what the server sends back when a request is one of `say`.
    pub id: &'static str,
    /// What it does, in a few words.
    pub label: &'static str,
    /// The panel button that does it.
    pub button: Button,
    /// Ctrl + this key (and Shift where true). The first is the one shown on the panel.
    pub keys: &'static [(Keycode, bool)],
    /// Another way on the keyboard, for the panel's hint (not looked up).
    pub also: &'static str,
    /// Exact phrases, spoken or typed, lower case without punctuation. The server answers these itself,
    /// without the AI.
    pub say: &'static [&'static str],
}

#[rustfmt::skip]
pub const COMMANDS: &[Command] = &[
    Command { id: "edit.undo", label: "Undo the last change", button: Button::Undo, keys: &[(K::Z, false)], also: "", say: &["undo", "undo that", "undo it", "undo the last change", "go back"] },
    Command { id: "edit.redo", label: "Redo", button: Button::Redo, keys: &[(K::Y, false), (K::Z, true)], also: "", say: &["redo", "redo that", "redo it"] },
    Command { id: "view.hide", label: "Hide the selected parts", button: Button::Hide, keys: &[(K::H, false)], also: "Del", say: &["hide", "hide this", "hide these", "hide it", "hide them", "hide the selection"] },
    Command { id: "view.isolate", label: "Show only the selected parts", button: Button::Isolate, keys: &[(K::I, false)], also: "", say: &["isolate", "isolate this", "isolate these", "isolate it", "show only this", "show only these"] },
    Command { id: "view.show_all", label: "Show every part", button: Button::ShowAll, keys: &[(K::A, false)], also: "", say: &["show all", "show everything", "unhide all", "unhide everything"] },
    Command { id: "select.clear", label: "Deselect", button: Button::Clear, keys: &[(K::D, false)], also: "", say: &["deselect", "deselect all", "clear selection", "clear the selection", "select nothing"] },
    Command { id: "view.recentre", label: "Bring the model back in front", button: Button::Recentre, keys: &[(K::R, false)], also: "", say: &["recentre", "recenter", "bring it back", "reset the view"] },
    Command { id: "select.mode", label: "Pick parts, faces or edges", button: Button::Faces, keys: &[(K::F, false)], also: "", say: &["pick faces", "pick edges", "pick parts"] },
    Command { id: "tool.push_pull", label: "Push or pull a face", button: Button::PushPull, keys: &[(K::P, false)], also: "", say: &["push pull", "pull"] },
    Command { id: "tool.place", label: "Set the model down on a surface", button: Button::Place, keys: &[(K::L, false)], also: "", say: &["set down", "set it down"] },
    Command { id: "view.size", label: "Step the model's size", button: Button::Size, keys: &[(K::J, false)], also: "", say: &["size", "next size"] },
    Command { id: "view.snap", label: "Stand the model upright", button: Button::Snap, keys: &[(K::U, false)], also: "", say: &["upright", "stand it up"] },
    Command { id: "file.save", label: "Save as a new file", button: Button::SaveFile, keys: &[(K::S, true)], also: "", say: &["save the file", "save the design"] },
    Command { id: "tool.section", label: "Section cut", button: Button::Section, keys: &[(K::S, false)], also: "", say: &["section", "section cut", "cut view"] },
    Command { id: "tool.measure", label: "Measure", button: Button::Measure, keys: &[(K::M, false)], also: "", say: &["measure"] },
    Command { id: "view.edges", label: "Edges: outline, all, none", button: Button::Edges, keys: &[(K::E, false)], also: "", say: &["edges", "toggle edges"] },
    Command { id: "view.explode", label: "Exploded view", button: Button::Explode, keys: &[(K::X, false)], also: "", say: &["explode", "exploded view", "explode it"] },
    Command { id: "note.add", label: "Pin a voice note", button: Button::Note, keys: &[(K::N, false)], also: "", say: &["note", "add a note"] },
    Command { id: "view.room", label: "Room on or off", button: Button::Room, keys: &[(K::B, false)], also: "", say: &["room", "toggle the room"] },
    Command { id: "file.models", label: "Open another model", button: Button::Models, keys: &[(K::O, false)], also: "", say: &["models", "open a model", "open another model"] },
    Command { id: "view.guides", label: "Axes and grid", button: Button::Guides, keys: &[(K::G, false)], also: "", say: &["guides", "axes", "grid"] },
    Command { id: "ui.keyboard", label: "On-screen keyboard", button: Button::Keyboard, keys: &[(K::K, false)], also: "", say: &["keyboard"] },
    Command { id: "ai.repeat", label: "Repeat the last request", button: Button::Repeat, keys: &[(K::T, false)], also: "", say: &["repeat", "repeat that", "again", "do it again"] },
    Command { id: "ai.keep", label: "Keep the last request", button: Button::SaveReq, keys: &[(K::W, false)], also: "", say: &["keep that", "save that request"] },
    Command { id: "ai.saved", label: "Saved requests", button: Button::Saved, keys: &[(K::Q, false)], also: "", say: &["saved requests", "show saved requests"] },
    Command { id: "ui.radial", label: "Radial menu at the pointing hand: move towards a choice, trigger to take it", button: Button::Radial, keys: &[(K::R, true)], also: "Right thumbstick: press, push towards a choice, let go", say: &["menu", "radial menu", "open the menu"] },
    Command { id: "tool.clash", label: "Check which parts overlap, and mark them red", button: Button::Clash, keys: &[(K::C, true)], also: "", say: &["check interference", "interference", "interference check", "check for clashes", "do any parts overlap", "check overlaps"] },
    Command { id: "tool.move", label: "Move a part: drag it, let go on another part's face to set it there", button: Button::Move, keys: &[(K::M, true)], also: "", say: &["move", "move parts", "move a part", "move this"] },
    Command { id: "edit.handles", label: "Dimension handles on the model: drag one to change its size", button: Button::Handles, keys: &[(K::H, true)], also: "", say: &["handles", "dimension handles", "show the handles", "hide the handles"] },
    Command { id: "edit.features", label: "Feature tree: switch a feature off or on, or edit it", button: Button::Features, keys: &[(K::C, false)], also: "", say: &["features", "feature tree", "show the features", "timeline"] },
    Command { id: "edit.dimensions", label: "Dimensions: change a size directly", button: Button::Dims, keys: &[(K::V, false)], also: "", say: &["dimensions", "show dimensions", "show the dimensions"] },
];

/// The command a key does with Ctrl held: the binding with that Shift, else the one without.
pub fn by_key(code: Keycode, shift: bool) -> Option<Button> {
    let find = |s: bool| COMMANDS.iter().find(|c| c.keys.contains(&(code, s))).map(|c| c.button);
    find(shift).or_else(|| find(false))
}

/// The command with this id (what the server sends for a spoken or typed command).
pub fn by_id(id: &str) -> Option<Button> {
    COMMANDS.iter().find(|c| c.id == id).map(|c| c.button)
}

fn key_name(k: Keycode) -> String {
    format!("{k:?}")
}

/// The keyboard shortcut of a button, for the panel's hint ("" when it has none).
pub fn shortcut(b: Button) -> String {
    let Some(c) = COMMANDS.iter().find(|c| c.button == b) else { return String::new() };
    let Some(&(k, shift)) = c.keys.first() else { return String::new() };
    let key = format!("Ctrl+{}{}", if shift { "Shift+" } else { "" }, key_name(k));
    if c.also.is_empty() { key } else { format!("{key} or {}", c.also) }
}

/// `docs/commands.md`.
pub fn table_md() -> String {
    let mut s = String::from(
        "# Commands\n\nWritten from `src/commands.rs`; do not edit by hand (`SCVIEW_WRITE_COMMANDS=1 cargo test --lib commands`).\n\
         Every command can be reached from the panel, from the keyboard (a Bluetooth keyboard on the headset,\nor the terminal of the desktop build as `^h`), and by saying or typing one of its phrases.\n\n\
         | Command | What it does | Keyboard | Say or type |\n|---|---|---|---|\n",
    );
    for c in COMMANDS {
        let keys: Vec<String> = c.keys.iter().map(|&(k, sh)| format!("Ctrl+{}{}", if sh { "Shift+" } else { "" }, key_name(k))).collect();
        let keys = if c.also.is_empty() { keys.join(", ") } else { format!("{}, {}", keys.join(", "), c.also) };
        let say = c.say.join(", ");
        s.push_str(&format!("| `{}` | {} | {} | {} |\n", c.id, c.label, keys, say));
    }
    s
}

/// `tools/commands.tsv`: id, label, phrases separated by `|`. Commands with no phrase are left out.
pub fn phrases_tsv() -> String {
    COMMANDS.iter().filter(|c| !c.say.is_empty()).map(|c| format!("{}\t{}\t{}\n", c.id, c.label, c.say.join("|"))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_keys_and_phrases_are_each_bound_once() {
        let mut ids = HashSet::new();
        let mut keys = HashSet::new();
        let mut said = HashSet::new();
        for c in COMMANDS {
            assert!(ids.insert(c.id), "two commands are called {}", c.id);
            for &(k, shift) in c.keys {
                assert!(keys.insert((key_name(k), shift)), "{} takes a key another command has", c.id);
            }
            for p in c.say {
                assert!(said.insert(*p), "\"{p}\" is said for two commands");
                // The server compares against its normalised text: lower case, letters, digits and single spaces.
                assert!(p.chars().all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == ' ') && !p.contains("  ") && p.trim() == *p, "\"{p}\" is not normalised");
            }
        }
    }

    #[test]
    fn a_key_finds_its_command_and_shift_falls_back() {
        assert_eq!(by_key(K::Z, false), Some(Button::Undo));
        assert_eq!(by_key(K::Z, true), Some(Button::Redo));
        assert_eq!(by_key(K::S, false), Some(Button::Section));
        assert_eq!(by_key(K::S, true), Some(Button::SaveFile));
        // Shift with a key that has no shifted binding does the plain one.
        assert_eq!(by_key(K::I, true), Some(Button::Isolate));
        assert_eq!(by_key(K::H, true), Some(Button::Handles));
        assert_eq!(by_id("view.hide"), Some(Button::Hide));
        assert_eq!(by_id("nothing.like.this"), None);
        assert_eq!(shortcut(Button::Hide), "Ctrl+H or Del");
        assert_eq!(shortcut(Button::SaveFile), "Ctrl+Shift+S");
    }

    /// The written tables are the ones this file gives.
    #[test]
    fn the_written_tables_are_up_to_date() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        for (path, text) in [("docs/commands.md", table_md()), ("tools/commands.tsv", phrases_tsv())] {
            let path = root.join(path);
            if std::env::var_os("SCVIEW_WRITE_COMMANDS").is_some() {
                std::fs::write(&path, &text).unwrap();
            }
            assert_eq!(std::fs::read_to_string(&path).unwrap_or_default(), text, "{} is out of date: SCVIEW_WRITE_COMMANDS=1 cargo test --lib commands", path.display());
        }
    }
}
