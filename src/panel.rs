//! The wrist panel's layouts (tabs with icons, or a compact grid of icons), the icon for each button,
//! the shortcut keys shown as hints, and the keys of the on-screen keyboard. All sizes are in metres,
//! in panel-local space: x to the right, y up, the panel centred on the origin.

use crate::icons::{self, Icon};
use crate::{Button, PANEL_W};

/// The pages of the Tabs layout.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tab {
    Select,
    View,
    Tools,
    Notes,
    More,
}

pub const TABS: [Tab; 5] = [Tab::Select, Tab::View, Tab::Tools, Tab::Notes, Tab::More];

impl Tab {
    pub fn label(self) -> &'static str {
        match self {
            Tab::Select => "Select",
            Tab::View => "View",
            Tab::Tools => "Tools",
            Tab::Notes => "AI",
            Tab::More => "More",
        }
    }

    fn icon(self) -> &'static Icon {
        match self {
            Tab::Select => &icons::SELECT,
            Tab::View => &icons::SHOW_ALL,
            Tab::Tools => &icons::TOOLS,
            Tab::Notes => &icons::DO_NOTES,
            Tab::More => &icons::MORE,
        }
    }

    /// The buttons on the page.
    fn buttons(self) -> &'static [Button] {
        match self {
            Tab::Select => &[Button::Faces, Button::Hide, Button::Isolate, Button::ShowAll],
            Tab::View => &[Button::Room, Button::Edges, Button::Explode, Button::Guides, Button::Recentre],
            Tab::Tools => &[Button::Section, Button::Measure, Button::PushPull, Button::Place, Button::Size, Button::Dims, Button::Features, Button::Handles, Button::Move, Button::Clash],
            Tab::Notes => &[Button::Note, Button::DoNotes, Button::DeleteNote, Button::Repeat, Button::SaveReq, Button::Saved],
            Tab::More => &[Button::Models, Button::Keys, Button::Keyboard, Button::SaveFile, Button::Snap],
        }
    }
}

/// One button's rectangle: left x, top y, width, height.
pub struct Slot {
    pub b: Button,
    pub x: f32,
    pub top: f32,
    pub w: f32,
    pub h: f32,
}

pub struct Layout {
    pub slots: Vec<Slot>,
    /// Total height of the panel.
    pub h: f32,
    /// Top of the first of the three voice lines.
    pub voice_top: f32,
    /// Pixel size for the icons (metres), and whether a slot also shows its name beside the icon.
    pub icon_px: f32,
    pub labels: bool,
}

/// Height of the three voice lines and the space under them.
const VOICE_H: f32 = 0.039 + 0.006;
const GAP: f32 = 0.0025;

pub fn layout(tab: Tab, confirm: bool, recording: bool) -> Layout {
    // Positions are first worked out as distances down from the panel's top edge.
    let mut at: Vec<(Button, f32, f32, f32, f32)> = Vec::new(); // button, x, from_top, w, h
    let (tab_w, tab_h) = (0.0455, 0.031);
    let tx0 = -(5.0 * tab_w + 4.0 * GAP) / 2.0;
    for (i, t) in TABS.iter().enumerate() {
        at.push((Button::Tab(*t), tx0 + i as f32 * (tab_w + GAP), 0.036, tab_w, tab_h));
    }
    let (bw, bh, pitch) = (0.0755, 0.026, 0.029);
    let bx0 = -(3.0 * bw + 2.0 * GAP) / 2.0;
    let page_top = 0.036 + tab_h + 0.004;
    for (i, b) in tab.buttons().iter().enumerate() {
        at.push((*b, bx0 + (i % 3) as f32 * (bw + GAP), page_top + (i / 3) as f32 * pitch, bw, bh));
    }
    let quick_top = page_top + 2.0 * pitch;
    // While recording, the Deselect slot is the Cancel button.
    for (i, b) in [Button::Undo, Button::Redo, if recording { Button::CancelVoice } else { Button::Clear }].iter().enumerate() {
        at.push((*b, bx0 + i as f32 * (bw + GAP), quick_top, bw, 0.024));
    }
    let voice_from_top = quick_top + 0.024 + 0.004;
    // A big request waiting for an OK: two buttons under the first voice line.
    if confirm {
        at.push((Button::ConfirmGo, -bw - GAP / 2.0, voice_from_top + 0.0145, bw, 0.026));
        at.push((Button::ConfirmStop, GAP / 2.0, voice_from_top + 0.0145, bw, 0.026));
    }
    // The voice lines are a button too: click to read the whole reply (later slots lose to the buttons above).
    at.push((Button::Reply, -PANEL_W / 2.0 + 0.005, voice_from_top, PANEL_W - 0.01, 0.039));
    let h = voice_from_top + VOICE_H;
    let slots = at.into_iter().map(|(b, x, ft, w, sh)| Slot { b, x, top: h / 2.0 - ft, w, h: sh }).collect();
    Layout { slots, h, voice_top: h / 2.0 - voice_from_top, icon_px: 0.0008, labels: true }
}

/// The button under the panel-local point (x, y).
pub fn button_at(lay: &Layout, x: f32, y: f32) -> Option<Button> {
    lay.slots.iter().find(|s| x >= s.x && x <= s.x + s.w && y <= s.top && y >= s.top - s.h).map(|s| s.b)
}

/// The icon drawn on a button.
pub fn icon_of(b: Button) -> &'static Icon {
    match b {
        Button::Hide => &icons::HIDE,
        Button::Isolate => &icons::ISOLATE,
        Button::ShowAll => &icons::SHOW_ALL,
        Button::Clear => &icons::CLEAR,
        Button::Recentre => &icons::RECENTRE,
        Button::Undo => &icons::UNDO,
        Button::Redo => &icons::REDO,
        Button::Room => &icons::ROOM,
        Button::Section => &icons::SECTION,
        Button::Measure => &icons::MEASURE,
        Button::Edges => &icons::EDGES,
        Button::Explode => &icons::EXPLODE,
        Button::Note => &icons::NOTE,
        Button::DoNotes => &icons::DO_NOTES,
        Button::DeleteNote => &icons::DELETE_NOTE,
        Button::Faces => &icons::FACES,
        Button::PushPull => &icons::PUSH_PULL,
        Button::Place => &icons::ROOM,
        Button::Size => &icons::GUIDES,
        Button::Models => &icons::MODELS,
        Button::Keys => &icons::KEYS,
        Button::Guides => &icons::GUIDES,
        Button::Keyboard => &icons::KEYBOARD,
        Button::SaveFile => &icons::FLOPPY,
        Button::Snap => &icons::RECENTRE,
        Button::Repeat => &icons::REPEAT,
        Button::SaveReq => &icons::SAVE,
        Button::Saved => &icons::LIST,
        Button::Dims => &icons::LIST,
        Button::Features => &icons::LIST,
        Button::Handles => &icons::PUSH_PULL,
        Button::Move => &icons::RECENTRE,
        Button::Clash => &icons::SECTION,
        Button::Radial => &icons::RECENTRE,
        Button::ConfirmGo => &icons::CHECK,
        Button::ConfirmStop => &icons::CLEAR,
        Button::CancelVoice => &icons::CLEAR,
        Button::Reply => &icons::LIST,
        Button::Tab(t) => t.icon(),
    }
}

/// The keyboard shortcut for a button, shown with its name when it is pointed at: the command table's
/// (src/commands.rs), else one of the few keys that are not Ctrl with a letter.
pub fn shortcut(b: Button) -> String {
    let key = crate::commands::shortcut(b);
    if !key.is_empty() {
        return key;
    }
    match b {
        Button::DoNotes => "B",
        Button::Keys => "F1",
        Button::ConfirmGo => "Enter or B",
        Button::ConfirmStop => "Esc",
        Button::CancelVoice => "Esc, or say cancel",
        _ => "",
    }
    .to_string()
}

// --- the on-screen keyboard: a floating panel of its own, with a bar to grab it by ---

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum KbAct {
    Char(char),
    Back,
    Shift,
    Clear,
    Space,
    Send,
    Close,
}

/// A key's rectangle in the keyboard's own space (origin at its centre, y up): left x, top y, width, height.
pub struct KbKey {
    pub act: KbAct,
    pub x: f32,
    pub top: f32,
    pub w: f32,
    pub h: f32,
}

pub struct KbLayout {
    pub w: f32,
    pub h: f32,
    /// The bar along the top: it shows what is typed, and dragging it moves the keyboard.
    pub bar_h: f32,
    pub keys: Vec<KbKey>,
}

/// What a point on the keyboard is on.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum KbHit {
    Key(usize),
    Bar,
}

/// How much bigger than the wrist panel's buttons the keyboard is.
const KB_SCALE: f32 = 1.5;

/// The keys: digits, three letter rows, and a bottom row of Clear, Space and Send, under the bar.
pub fn kb_layout() -> KbLayout {
    let s = KB_SCALE;
    let (px, py, kh, gap) = (0.0235 * s, 0.0215 * s, 0.019 * s, 0.0025 * s);
    let (bar_h, pad) = (0.024, 0.006);
    let (w, h) = (10.0 * px + 2.0 * pad, bar_h + pad + 5.0 * py + pad);
    let x0 = -5.0 * px;
    let mut keys = Vec::new();
    let mut row = |r: usize, items: &[(KbAct, f32)]| {
        let mut col = 0.0;
        for &(act, units) in items {
            keys.push(KbKey { act, x: x0 + col * px + gap / 2.0, top: h / 2.0 - bar_h - pad - r as f32 * py, w: units * px - gap, h: kh });
            col += units;
        }
    };
    let chars = |s: &str| s.chars().map(|c| (KbAct::Char(c), 1.0)).collect::<Vec<_>>();
    row(0, &chars("1234567890"));
    row(1, &chars("qwertyuiop"));
    let mut r2 = chars("asdfghjkl");
    r2.push((KbAct::Back, 1.0));
    row(2, &r2);
    let mut r3 = vec![(KbAct::Shift, 1.0)];
    r3.extend(chars("zxcvbnm.-"));
    row(3, &r3);
    row(4, &[(KbAct::Clear, 2.0), (KbAct::Space, 5.0), (KbAct::Send, 3.0)]);
    // A Close key at the right end of the bar.
    keys.push(KbKey { act: KbAct::Close, x: w / 2.0 - pad - 0.05, top: h / 2.0 - 0.003, w: 0.05, h: bar_h - 0.006 });
    KbLayout { w, h, bar_h, keys }
}

/// Is the keyboard-space point (x, y) on the keyboard?
pub fn kb_inside(lay: &KbLayout, x: f32, y: f32) -> bool {
    x.abs() <= lay.w / 2.0 && y.abs() <= lay.h / 2.0
}

/// What is at (x, y): a key, or the bar (not over the Close key), else nothing.
pub fn kb_hit(lay: &KbLayout, x: f32, y: f32) -> Option<KbHit> {
    if let Some(i) = lay.keys.iter().position(|k| x >= k.x && x <= k.x + k.w && y <= k.top && y >= k.top - k.h) {
        return Some(KbHit::Key(i));
    }
    (kb_inside(lay, x, y) && y > lay.h / 2.0 - lay.bar_h).then_some(KbHit::Bar)
}
