//! Desktop stand-in for the Android activity, so the viewer can run on Linux against any OpenXR
//! runtime, such as Monado with its keyboard-and-mouse simulated headset. For development without a
//! headset: there is no passthrough and no room raycast, and the "keyboard" is the terminal the viewer
//! was started from (see `Host::take_keys`).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};

/// The keys the viewer reacts to (the same names as android-activity's `Keycode`).
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keycode {
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Del, Enter, Escape, F1, ForwardDel, NumpadEnter,
    Unknown,
}

/// One key as the terminal line asked for it: code, character, Ctrl, Shift.
pub type TermKey = (Keycode, Option<char>, bool, bool);

pub struct Host {
    files: PathBuf,
    lines: Mutex<Receiver<String>>,
    quit: Arc<AtomicBool>,
}

impl Host {
    /// Files go in `$SCVIEW_FILES`, else `~/.scview-desktop/files` (the headset's app folder).
    pub fn new() -> Host {
        let files = std::env::var_os("SCVIEW_FILES").map(PathBuf::from).unwrap_or_else(|| {
            let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
            home.join(".scview-desktop").join("files")
        });
        let _ = std::fs::create_dir_all(&files);
        let (tx, rx) = channel();
        let quit = Arc::new(AtomicBool::new(false));
        let q = quit.clone();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                match stdin.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let l = line.trim_end_matches(['\r', '\n']).to_string();
                        if l.trim() == "/quit" {
                            q.store(true, Ordering::SeqCst);
                        } else if tx.send(l).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        Host { files, lines: Mutex::new(rx), quit }
    }

    pub fn external_data_path(&self) -> Option<PathBuf> {
        Some(self.files.clone())
    }

    pub fn internal_data_path(&self) -> Option<PathBuf> {
        None
    }

    /// True once `/quit` was typed in the terminal.
    pub fn quit_requested(&self) -> bool {
        self.quit.load(Ordering::SeqCst)
    }

    /// Lines typed in the terminal, as key presses. Plain text is typed and sent with Enter (an AI
    /// request, like typing on the headset's keyboard). `^x` is Ctrl+x (a panel button, see F1 in the
    /// app); `^X` is Ctrl+Shift+x. `/f1` shows the keys, `/esc` is Escape, `/del` deletes the selection,
    /// `/enter` presses Enter alone (confirms a waiting AI request).
    pub fn take_keys(&self) -> Vec<TermKey> {
        let mut out = Vec::new();
        let Ok(rx) = self.lines.lock() else { return out };
        while let Ok(line) = rx.try_recv() {
            let t = line.trim();
            match t {
                "" => continue,
                "/f1" => out.push((Keycode::F1, None, false, false)),
                "/esc" => out.push((Keycode::Escape, None, false, false)),
                "/del" => out.push((Keycode::ForwardDel, None, false, false)),
                "/enter" => out.push((Keycode::Enter, None, false, false)),
                _ if t.len() == 2 && t.starts_with('^') => {
                    let c = t.chars().nth(1).unwrap_or(' ');
                    let shift = c.is_ascii_uppercase();
                    out.push((letter(c), None, true, shift));
                }
                _ => {
                    out.extend(t.chars().map(|c| (Keycode::Unknown, Some(c), false, false)));
                    out.push((Keycode::Enter, None, false, false));
                }
            }
        }
        out
    }
}

fn letter(c: char) -> Keycode {
    use Keycode::*;
    match c.to_ascii_lowercase() {
        'a' => A, 'b' => B, 'c' => C, 'd' => D, 'e' => E, 'f' => F, 'g' => G, 'h' => H, 'i' => I, 'j' => J,
        'k' => K, 'l' => L, 'm' => M, 'n' => N, 'o' => O, 'p' => P, 'q' => Q, 'r' => R, 's' => S,
        't' => T, 'u' => U, 'v' => V, 'w' => W, 'x' => X, 'y' => Y, 'z' => Z,
        _ => Unknown,
    }
}
