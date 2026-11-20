//! Push-to-talk: records the headset microphone with AAudio while the left trigger is held, then
//! sends the recording (16 kHz mono WAV) to the Mac's `serve.py` (`POST /ask`), which turns it into
//! text, lets Claude change the design and replies with what it heard and did. The changed model then
//! arrives through the usual beacon download.
//!
//! The app asks for the RECORD_AUDIO permission itself; it can also be granted with
//! `adb shell pm grant org.scview.viewer android.permission.RECORD_AUDIO`.

use std::io::{Read, Write};
#[cfg(target_os = "android")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// The Mac's HTTP address, learned from its beacon.
pub static MAC: Mutex<Option<std::net::SocketAddr>> = Mutex::new(None);
/// True while the talk button is held.
static RECORDING: AtomicBool = AtomicBool::new(false);
/// True from the press until the Mac's answer (or an error) arrives.
static BUSY: AtomicBool = AtomicBool::new(false);
/// The recording so far (as recorded, at `rate`), shared with the live-text thread.
static PCM: Mutex<(Vec<i16>, u32)> = Mutex::new((Vec::new(), RATE));
/// Names of the bodies selected in the headset, sent with each request ("make the selected part green").
static SELECTED: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// Requests that changed the model since the render loop last asked (each is one Undo step).
static CHANGES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// A voice note pinned to a point on a body (model units, the body's unexploded position).
#[derive(Clone, Debug)]
pub struct Comment {
    pub id: String,
    pub body: String,
    pub point: [f32; 3],
    pub text: String,
    /// The AI has carried the note out.
    pub done: bool,
}

/// A new list of notes from the Mac, for the render loop to take.
pub static COMMENTS: Mutex<Option<Vec<Comment>>> = Mutex::new(None);
/// Where the next recording goes as a note ("body\tx\ty\tz"); None sends it to the AI.
static PIN: Mutex<Option<String>> = Mutex::new(None);
/// Where the right pointer was when the wearer started speaking ("body\tx\ty\tz"), sent with the request.
static POINT: Mutex<Option<String>> = Mutex::new(None);

pub fn set_point(p: Option<String>) {
    if let Ok(mut pt) = POINT.lock() {
        *pt = p;
    }
}

/// A note to re-record: the next recording replaces its words (when no new pin is waiting).
static REDO: Mutex<Option<String>> = Mutex::new(None);

/// Undo steps from note changes the Mac confirmed, for the render loop to take.
pub enum NoteStep {
    Added(String),
    Deleted(String),
    /// Re-recorded: the id and the words it had before.
    Text(String, String),
}
static NOTE_STEPS: Mutex<Vec<NoteStep>> = Mutex::new(Vec::new());

/// The next recording re-records this note (None: it does not).
pub fn set_redo(id: Option<&str>) {
    if let Ok(mut r) = REDO.lock() {
        *r = id.map(str::to_string);
    }
}

/// Moves a note (pin "body\tx\ty\tz") and/or sets its words, on the Mac.
pub fn note_edit(id: &str, pin: Option<String>, text: Option<String>) {
    let id = id.to_string();
    std::thread::spawn(move || {
        let pin_head = pin.map(|p| format!("X-Pin: {}\r\n", percent(&p))).unwrap_or_default();
        match post_with("/comment/edit", text.unwrap_or_default().as_bytes(), 20, &format!("X-Id: {}\r\n{pin_head}", percent(&id))) {
            Ok(json) => set_status("", Some(""), Some(&json_string(&json, "reply").unwrap_or_default())),
            Err(e) => set_status("", None, Some(&e)),
        }
    });
}

pub fn pin_text(body: &str, p: [f32; 3]) -> String {
    format!("{body}\t{}\t{}\t{}", p[0], p[1], p[2])
}

/// The Mac's /comments.txt: one note per line, "id \t body \t x \t y \t z \t done (0/1) \t text".
pub fn parse_comments(text: &str) -> Vec<Comment> {
    text.lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.splitn(7, '\t').collect();
            let [id, body, x, y, z, done, t] = f[..] else { return None };
            Some(Comment { id: id.into(), body: body.into(), point: [x.parse().ok()?, y.parse().ok()?, z.parse().ok()?], text: t.into(), done: done == "1" })
        })
        .collect()
}

/// Push/pull: moves a face ("body\tx\ty\tz", a point on it) by `mm` (positive = outward) on the Mac.
pub fn push_pull(face: String, mm: f32) {
    if !mm.is_finite() {
        set_status("", None, Some("That pull gave no usable distance; try again."));
        return;
    }
    if BUSY.swap(true, Ordering::SeqCst) {
        return;
    }
    set_status(&format!("Moving the face {mm:+.1} mm..."), Some(""), Some(""));
    std::thread::spawn(move || {
        match post_with("/pushpull", &[], 120, &format!("X-Face: {}\r\nX-Distance: {mm:.3}\r\n", percent(&face))) {
            Ok(json) => {
                if json.contains("\"changed\": true") {
                    CHANGES.fetch_add(1, Ordering::SeqCst);
                }
                set_status("", None, Some(&json_string(&json, "reply").unwrap_or_default()));
            }
            Err(e) => set_status("", None, Some(&e)),
        }
        BUSY.store(false, Ordering::SeqCst);
    });
}

/// Gives a note (its id, or "all" for every note not yet done) to the Mac's AI as a request.
pub fn note_do(id: &str) {
    if BUSY.swap(true, Ordering::SeqCst) {
        return;
    }
    set_status("AI is working on the note...", Some(""), Some(""));
    let id = id.to_string();
    std::thread::spawn(move || {
        match post_with("/comment/do", &[], 600, &format!("X-Id: {}\r\n", percent(&id))) {
            Ok(json) => {
                let n = json.split("\"changes\":").nth(1).and_then(|r| r.trim_start().split(|c: char| !c.is_ascii_digit()).next()?.parse::<u32>().ok()).unwrap_or(0);
                CHANGES.fetch_add(n, Ordering::SeqCst);
                set_status("", None, Some(&json_string(&json, "reply").unwrap_or_default()));
            }
            Err(e) => set_status("", None, Some(&e)),
        }
        BUSY.store(false, Ordering::SeqCst);
    });
}

pub fn take_comments() -> Option<Vec<Comment>> {
    COMMENTS.lock().ok().and_then(|mut c| c.take())
}

/// The next recording becomes a note at this point on this body.
pub fn set_pin(body: &str, p: [f32; 3]) {
    if let Ok(mut pin) = PIN.lock() {
        *pin = Some(pin_text(body, p));
    }
}

pub fn clear_pin() {
    if let Ok(mut pin) = PIN.lock() {
        *pin = None;
    }
}

/// A recording now would be a note: a new pin is waiting, or a note is set to be re-recorded.
pub fn pin_waiting() -> bool {
    PIN.lock().map(|p| p.is_some()).unwrap_or(false) || REDO.lock().map(|r| r.is_some()).unwrap_or(false)
}

/// Shows a message in the voice lines.
/// The access code the Mac's server asks for (shown in its log), kept in a file in the app's folder.
static TOKEN: Mutex<String> = Mutex::new(String::new());
static TOKEN_FILE: Mutex<Option<std::path::PathBuf>> = Mutex::new(None);
static NEED_TOKEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Where the code is kept; reads it if it is there.
pub fn init_token(path: Option<std::path::PathBuf>) {
    if let Some(p) = &path {
        if let Ok(t) = std::fs::read_to_string(p) {
            if let Ok(mut c) = TOKEN.lock() {
                *c = t.trim().to_string();
            }
        }
    }
    if let Ok(mut f) = TOKEN_FILE.lock() {
        *f = path;
    }
}

pub fn token() -> String {
    TOKEN.lock().map(|t| t.clone()).unwrap_or_default()
}

/// The server refused the request for want of the code: ask for it.
pub fn token_refused() {
    if !NEED_TOKEN.swap(true, Ordering::SeqCst) {
        say("The Mac wants an access code: type the code shown in its serve.py window and press Enter.");
    }
}

pub fn need_token() -> bool {
    NEED_TOKEN.load(Ordering::SeqCst)
}

/// The code was typed: keep it and let the downloads start again.
fn set_token(code: &str) {
    if let Ok(mut t) = TOKEN.lock() {
        *t = code.trim().to_string();
    }
    if let Some(Some(p)) = TOKEN_FILE.lock().ok().map(|f| f.clone()) {
        let _ = std::fs::write(p, code.trim());
    }
    NEED_TOKEN.store(false, Ordering::SeqCst);
    crate::net_reset();
    say("Access code saved: trying the Mac again.");
}

pub fn say(msg: &str) {
    set_status("", Some(""), Some(msg));
}

pub fn busy() -> bool {
    BUSY.load(Ordering::SeqCst)
}

pub fn take_note_steps() -> Vec<NoteStep> {
    NOTE_STEPS.lock().map(|mut s| std::mem::take(&mut *s)).unwrap_or_default()
}

/// Deletes a note on the Mac (`restore` = false), or brings a deleted one back (true).
pub fn note_delete(id: &str, restore: bool, record_undo: bool) {
    let id = id.to_string();
    std::thread::spawn(move || {
        let path = if restore { "/comment/restore" } else { "/comment/delete" };
        match post_with(path, &[], 20, &format!("X-Id: {}\r\n", percent(&id))) {
            Ok(json) => {
                set_status("", Some(""), Some(&json_string(&json, "reply").unwrap_or_default()));
                if record_undo && json.contains("\"ok\": true") {
                    if let Ok(mut s) = NOTE_STEPS.lock() {
                        s.push(NoteStep::Deleted(id));
                    }
                }
            }
            Err(e) => set_status("", None, Some(&e)),
        }
    });
}

/// Selected faces, one per line: "body\tx\ty\tz\tnx\tny\tnz" (a point on the face and the normal there).
static FACES: Mutex<String> = Mutex::new(String::new());

pub fn set_faces(f: String) {
    if let Ok(mut s) = FACES.lock() {
        *s = f;
    }
}

/// Selected edges, one per line: "body\tstart xyz\tmiddle xyz\tend xyz\tlength" (tab-separated numbers).
static EDGES: Mutex<String> = Mutex::new(String::new());

pub fn set_edges(e: String) {
    if let Ok(mut s) = EDGES.lock() {
        *s = e;
    }
}

pub fn set_selected(names: Vec<String>) {
    if let Ok(mut s) = SELECTED.lock() {
        *s = names;
    }
}

/// How many AI changes arrived since the last call.
pub fn take_changes() -> u32 {
    CHANGES.swap(0, Ordering::SeqCst)
}

/// Asks the Mac to undo the last AI change (all of its steps).
pub fn undo_model() {
    model_history("/undo", "Undoing...");
}

/// Asks the Mac to redo the last AI change it undid.
pub fn redo_model() {
    model_history("/redo", "Redoing...");
}

fn model_history(path: &'static str, msg: &str) {
    if BUSY.swap(true, Ordering::SeqCst) {
        return;
    }
    set_status(msg, Some(""), Some(""));
    std::thread::spawn(move || {
        match post(path, &[], 60) {
            Ok(json) => set_status("", None, Some(&json_string(&json, "reply").unwrap_or_default())),
            Err(e) => set_status("", None, Some(&e)),
        }
        BUSY.store(false, Ordering::SeqCst);
        fetch_history();
    });
}

/// What the Mac's Undo and Redo would do next, in words ("thickness = 12"); "" when there is nothing.
static HISTORY: Mutex<(String, String)> = Mutex::new((String::new(), String::new()));

pub fn history() -> (String, String) {
    HISTORY.lock().map(|h| h.clone()).unwrap_or_default()
}

/// Fetches what Undo and Redo would do next (GET /history.txt) in the background.
pub fn fetch_history() {
    std::thread::spawn(|| {
        let Some(mac) = MAC.lock().ok().and_then(|m| *m) else { return };
        if let Ok(b) = crate::http_get(mac, "/history.txt", 10) {
            let text = String::from_utf8_lossy(&b).to_string();
            let first = |kind: &str| text.lines().find_map(|l| l.strip_prefix(kind).and_then(|r| r.strip_prefix('\t'))).unwrap_or("").to_string();
            if let Ok(mut h) = HISTORY.lock() {
                *h = (first("u"), first("r"));
            }
        }
    });
}

/// JSON as a percent-encoded header value.
fn percent(s: &str) -> String {
    s.bytes()
        .map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") })
        .collect()
}

/// What the panel shows: (state line, what was heard, the reply).
pub static STATUS: Mutex<(String, String, String)> = Mutex::new((String::new(), String::new(), String::new()));

fn set_status(state: &str, heard: Option<&str>, reply: Option<&str>) {
    if let Ok(mut s) = STATUS.lock() {
        s.0 = state.to_string();
        if let Some(h) = heard {
            s.1 = h.to_string();
        }
        if let Some(r) = reply {
            s.2 = r.to_string();
        }
    }
}

#[cfg(target_os = "android")]
#[allow(non_camel_case_types)]
type AAudioStreamBuilder = c_void;
#[cfg(target_os = "android")]
#[allow(non_camel_case_types)]
type AAudioStream = c_void;

#[cfg(target_os = "android")]
#[link(name = "aaudio")]
extern "C" {
    fn AAudio_createStreamBuilder(builder: *mut *mut AAudioStreamBuilder) -> i32;
    fn AAudioStreamBuilder_setDirection(builder: *mut AAudioStreamBuilder, direction: i32);
    fn AAudioStreamBuilder_setSampleRate(builder: *mut AAudioStreamBuilder, rate: i32);
    fn AAudioStreamBuilder_setChannelCount(builder: *mut AAudioStreamBuilder, count: i32);
    fn AAudioStreamBuilder_setFormat(builder: *mut AAudioStreamBuilder, format: i32);
    fn AAudioStreamBuilder_openStream(builder: *mut AAudioStreamBuilder, stream: *mut *mut AAudioStream) -> i32;
    fn AAudioStreamBuilder_delete(builder: *mut AAudioStreamBuilder) -> i32;
    fn AAudioStream_requestStart(stream: *mut AAudioStream) -> i32;
    fn AAudioStream_requestStop(stream: *mut AAudioStream) -> i32;
    fn AAudioStream_close(stream: *mut AAudioStream) -> i32;
    fn AAudioStream_read(stream: *mut AAudioStream, buffer: *mut c_void, frames: i32, timeout_ns: i64) -> i32;
    fn AAudioStream_getSampleRate(stream: *mut AAudioStream) -> i32;
    fn AAudioStream_getChannelCount(stream: *mut AAudioStream) -> i32;
}

#[cfg(target_os = "android")]
const AAUDIO_DIRECTION_INPUT: i32 = 1;
#[cfg(target_os = "android")]
const AAUDIO_FORMAT_PCM_I16: i32 = 1;
const RATE: u32 = 16_000;

/// A recording the wearer cancelled (its audio is dropped), and: the talk button must be let go before a new one starts.
static DISCARD: AtomicBool = AtomicBool::new(false);
static NEED_RELEASE: AtomicBool = AtomicBool::new(false);

pub fn recording() -> bool {
    RECORDING.load(Ordering::SeqCst)
}

/// Drops the recording in progress: nothing is sent (the trigger must be released before the next one).
pub fn cancel_recording() {
    if RECORDING.load(Ordering::SeqCst) {
        DISCARD.store(true, Ordering::SeqCst);
        NEED_RELEASE.store(true, Ordering::SeqCst);
        RECORDING.store(false, Ordering::SeqCst);
    }
}

/// Called every frame with whether the talk button is held.
pub fn update(held: bool) {
    if !held {
        NEED_RELEASE.store(false, Ordering::SeqCst);
    }
    if held && !RECORDING.load(Ordering::SeqCst) && !BUSY.load(Ordering::SeqCst) && !NEED_RELEASE.load(Ordering::SeqCst) {
        RECORDING.store(true, Ordering::SeqCst);
        BUSY.store(true, Ordering::SeqCst);
        DISCARD.store(false, Ordering::SeqCst);
        set_status("Listening... say cancel to drop it", Some(""), Some(""));
        if let Ok(mut p) = PCM.lock() {
            p.0.clear();
        }
        // Live text: every second, the audio so far goes to the Mac and the words come back.
        std::thread::spawn(|| {
            while RECORDING.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(900));
                let (pcm, rate) = match PCM.lock() {
                    Ok(p) => (p.0.clone(), p.1),
                    Err(_) => break,
                };
                if pcm.len() < rate as usize / 2 || !RECORDING.load(Ordering::SeqCst) {
                    continue;
                }
                match post("/transcribe", &wav(&resample(&pcm, rate, RATE)), 10)
                    .map(|json| json_string(&json, "text").unwrap_or_default())
                {
                    Ok(text) if !text.is_empty() && RECORDING.load(Ordering::SeqCst) => set_status(&format!("Hearing: {text}"), None, None),
                    Ok(_) => {}
                    Err(e) => log::info!("voice: live text: {e}"),
                }
            }
        });
        std::thread::spawn(|| {
            let result = record().and_then(|pcm| {
                if DISCARD.swap(false, Ordering::SeqCst) {
                    set_status("", Some(""), Some("Cancelled: nothing was sent."));
                    Ok(())
                } else {
                    send(&pcm)
                }
            });
            if let Err(e) = result {
                log::error!("voice: {e}");
                set_status("", None, Some(&e));
            }
            BUSY.store(false, Ordering::SeqCst);
        });
    } else if !held {
        RECORDING.store(false, Ordering::SeqCst);
    }
}

/// Desktop: records the default microphone with PipeWire's `pw-record` until the button is released.
#[cfg(not(target_os = "android"))]
fn record() -> Result<Vec<i16>, String> {
    let mut child = std::process::Command::new("pw-record")
        .args(["--rate", &RATE.to_string(), "--channels", "1", "--format", "s16", "--raw", "-"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("microphone: pw-record did not start ({e})"))?;
    let Some(mut out) = child.stdout.take() else {
        let _ = child.kill();
        return Err("microphone: no pw-record output".into());
    };
    log::info!("voice: recording at {RATE} Hz with pw-record");
    if let Ok(mut p) = PCM.lock() {
        p.1 = RATE;
    }
    let mut pcm: Vec<i16> = Vec::new();
    let mut buf = [0u8; 3200];
    let mut odd: Option<u8> = None;
    let max = RATE as usize * 30;
    while RECORDING.load(Ordering::SeqCst) && pcm.len() < max {
        let n = match out.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        let start = pcm.len();
        let mut bytes: Vec<u8> = odd.take().into_iter().collect();
        bytes.extend_from_slice(&buf[..n]);
        if bytes.len() % 2 == 1 {
            odd = bytes.pop();
        }
        pcm.extend(bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])));
        if let Ok(mut p) = PCM.lock() {
            p.0.extend_from_slice(&pcm[start..]);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    log::info!("voice: {:.1} s recorded", pcm.len() as f32 / RATE as f32);
    if pcm.len() < RATE as usize / 4 {
        return Err("Too short: hold the left trigger while you speak.".into());
    }
    Ok(pcm)
}

/// Records until the button is released; 16 kHz mono samples.
#[cfg(target_os = "android")]
fn record() -> Result<Vec<i16>, String> {
    // SAFETY: plain AAudio C calls; the builder and stream pointers come from AAudio and are released here.
    unsafe {
        let mut builder: *mut AAudioStreamBuilder = std::ptr::null_mut();
        if AAudio_createStreamBuilder(&mut builder) != 0 {
            return Err("microphone: no stream builder".into());
        }
        AAudioStreamBuilder_setDirection(builder, AAUDIO_DIRECTION_INPUT);
        AAudioStreamBuilder_setSampleRate(builder, RATE as i32);
        AAudioStreamBuilder_setChannelCount(builder, 1);
        AAudioStreamBuilder_setFormat(builder, AAUDIO_FORMAT_PCM_I16);
        let mut stream: *mut AAudioStream = std::ptr::null_mut();
        let r = AAudioStreamBuilder_openStream(builder, &mut stream);
        AAudioStreamBuilder_delete(builder);
        if r != 0 || stream.is_null() {
            return Err(format!("microphone unavailable ({r}): allow the microphone permission for the app"));
        }
        let rate = AAudioStream_getSampleRate(stream).max(1) as u32;
        let channels = AAudioStream_getChannelCount(stream).max(1) as usize;
        log::info!("voice: recording at {rate} Hz, {channels} channel(s)");
        AAudioStream_requestStart(stream);
        if let Ok(mut p) = PCM.lock() {
            p.1 = rate;
        }
        let mut pcm: Vec<i16> = Vec::new();
        let mut buf = vec![0i16; 1024 * channels];
        let max = rate as usize * 30; // 30 s at most
        while RECORDING.load(Ordering::SeqCst) && pcm.len() < max {
            let n = AAudioStream_read(stream, buf.as_mut_ptr() as *mut c_void, 1024, 50_000_000);
            if n < 0 {
                break;
            }
            let start = pcm.len();
            for f in buf[..n as usize * channels].chunks_exact(channels) {
                pcm.push(f[0]);
            }
            if let Ok(mut p) = PCM.lock() {
                p.0.extend_from_slice(&pcm[start..]);
            }
        }
        AAudioStream_requestStop(stream);
        AAudioStream_close(stream);
        log::info!("voice: {:.1} s recorded", pcm.len() as f32 / rate as f32);
        if pcm.len() < rate as usize / 4 {
            return Err("Too short: hold the left trigger while you speak.".into());
        }
        Ok(resample(&pcm, rate, RATE))
    }
}

/// Linear resampling to the rate whisper expects.
fn resample(pcm: &[i16], from: u32, to: u32) -> Vec<i16> {
    if from == to {
        return pcm.to_vec();
    }
    let n = (pcm.len() as u64 * to as u64 / from as u64) as usize;
    (0..n)
        .map(|i| {
            let x = i as f64 * from as f64 / to as f64;
            let j = x as usize;
            let f = x - j as f64;
            let a = pcm[j.min(pcm.len() - 1)] as f64;
            let b = pcm[(j + 1).min(pcm.len() - 1)] as f64;
            (a + (b - a) * f) as i16
        })
        .collect()
}

fn wav(pcm: &[i16]) -> Vec<u8> {
    let data_len = (pcm.len() * 2) as u32;
    let mut w = Vec::with_capacity(44 + data_len as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_len).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); // PCM
    w.extend_from_slice(&1u16.to_le_bytes()); // mono
    w.extend_from_slice(&RATE.to_le_bytes());
    w.extend_from_slice(&(RATE * 2).to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        w.extend_from_slice(&s.to_le_bytes());
    }
    w
}

/// A model the Mac can open: (id = its path on the Mac, name, size in MB, open now).
#[derive(Clone)]
pub struct ModelFile {
    pub id: String,
    pub name: String,
    pub mb: f32,
    pub current: bool,
}

/// The Mac's model list, once fetched (None: not yet).
pub static MODELS: Mutex<Option<Vec<ModelFile>>> = Mutex::new(None);

/// Fetches the Mac's model list (GET /models.txt) in the background.
pub fn fetch_models() {
    std::thread::spawn(|| {
        let Some(mac) = MAC.lock().ok().and_then(|m| *m) else {
            say("The Mac isn't running serve.py (no beacon seen).");
            return;
        };
        match crate::http_get(mac, "/models.txt", 10) {
            Ok(b) => {
                let list: Vec<ModelFile> = String::from_utf8_lossy(&b)
                    .lines()
                    .filter_map(|l| {
                        let f: Vec<&str> = l.split('\t').collect();
                        (f.len() >= 4).then(|| ModelFile { id: f[0].into(), name: f[1].into(), mb: f[2].parse().unwrap_or(0.0), current: f[3] == "1" })
                    })
                    .collect();
                log::info!("models list: {} files", list.len());
                if let Ok(mut m) = MODELS.lock() {
                    *m = Some(list);
                }
            }
            Err(e) => say(&format!("Model list: {e}")),
        }
    });
}

/// Asks the Mac to open another model; the beacon's new version then brings it to the headset.
pub fn open_model(id: &str) {
    let id = id.to_string();
    std::thread::spawn(move || {
        let r = post_typed("/model/open", b"", "text/plain", 10, &format!("X-Id: {}\r\n", percent(&id)));
        say(&r.map(|json| json_string(&json, "reply").unwrap_or_default()).unwrap_or_else(|e| e));
    });
}

/// A saved request: (id, the words, whether the Mac can repeat it without the AI).
#[derive(Clone)]
pub struct SavedReq {
    pub id: String,
    pub text: String,
    pub quick: bool,
}

/// The saved requests, once fetched (None: not yet).
pub static REQUESTS: Mutex<Option<Vec<SavedReq>>> = Mutex::new(None);

/// The words of the latest request the AI answered, for Repeat and Save.
static LAST_REQUEST: Mutex<String> = Mutex::new(String::new());

/// The estimated API balance left, in dollars, from the Mac's beacon ("" when it does not say).
static SPEND: Mutex<String> = Mutex::new(String::new());

pub fn set_spend(b: &str) {
    if let Ok(mut s) = SPEND.lock() {
        *s = if b.is_empty() || b == "-" { String::new() } else { b.to_string() };
    }
}

pub fn spend() -> String {
    SPEND.lock().map(|s| s.clone()).unwrap_or_default()
}

/// Fetches the saved requests (GET /requests.txt) in the background.
pub fn fetch_requests() {
    std::thread::spawn(|| {
        let Some(mac) = MAC.lock().ok().and_then(|m| *m) else {
            say("The Mac isn't running serve.py (no beacon seen).");
            return;
        };
        match crate::http_get(mac, "/requests.txt", 10) {
            Ok(b) => {
                let list: Vec<SavedReq> = String::from_utf8_lossy(&b)
                    .lines()
                    .filter_map(|l| {
                        let f: Vec<&str> = l.split('\t').collect();
                        (f.len() >= 3).then(|| SavedReq { id: f[0].into(), text: f[1].into(), quick: f[2] == "r" })
                    })
                    .collect();
                if let Ok(mut r) = REQUESTS.lock() {
                    *r = Some(list);
                }
            }
            Err(e) => say(&format!("Saved requests: {e}")),
        }
    });
}

/// Saves the last request's words (and what the AI did for them, if the Mac still has it).
pub fn save_last() {
    let text = LAST_REQUEST.lock().map(|t| t.clone()).unwrap_or_default();
    if text.is_empty() {
        say("Nothing to save yet: make a request first.");
        return;
    }
    std::thread::spawn(move || {
        let r = post_typed("/requests/save", text.as_bytes(), "text/plain; charset=utf-8", 10, "");
        say(&r.map(|json| json_string(&json, "reply").unwrap_or_default()).unwrap_or_else(|e| e));
        fetch_requests();
    });
}

/// Asks the Mac to save the open design as a new file next to its source.
pub fn save_file() {
    std::thread::spawn(|| {
        set_status("Saving...", None, None);
        let r = post_typed("/save", b"", "text/plain", 120, "");
        set_status("", None, Some(&r.map(|json| json_string(&json, "reply").unwrap_or_default()).unwrap_or_else(|e| e)));
    });
}

/// Takes a photo of the headset's view (the Mac does it through adb and keeps it in Pictures/SC View).
pub fn snapshot() {
    std::thread::spawn(|| {
        set_status("Taking a photo...", None, None);
        let r = post_typed("/capture", b"", "text/plain", 60, "");
        set_status("", None, Some(&r.map(|json| json_string(&json, "reply").unwrap_or_default()).unwrap_or_else(|e| e)));
    });
}

/// Removes a saved request.
pub fn delete_saved(id: &str) {
    let id = id.to_string();
    std::thread::spawn(move || {
        let _ = post_typed("/requests/delete", b"", "text/plain", 10, &format!("X-Id: {}\r\n", percent(&id)));
        fetch_requests();
    });
}

/// A parameter or feature dimension the Mac can set directly (GET /dimensions.txt).
#[derive(Clone)]
pub struct Dim {
    pub id: String,
    pub label: String,
    pub value: f32,
    pub unit: String,
}

/// The dimensions, once fetched (None: not yet).
pub static DIMS: Mutex<Option<Vec<Dim>>> = Mutex::new(None);

/// Only this feature's dimensions are listed ("" for all): set when a feature is picked in the feature tree.
static DIMS_OF: Mutex<String> = Mutex::new(String::new());

pub fn dims_of() -> String {
    DIMS_OF.lock().map(|f| f.clone()).unwrap_or_default()
}

pub fn set_dims_of(feature: &str) {
    if let Ok(mut f) = DIMS_OF.lock() {
        *f = feature.to_string();
    }
}

/// Moves a part on the Mac (POST /move, no AI). `onto`: the point it was taken hold of and the point on
/// another part it was let go over, when it should be set on that face; else it moves by `by` (model units).
pub fn move_part(body: String, by: [f32; 3], onto: Option<([f32; 3], [f32; 3], [f32; 3])>) {
    let finite = |v: &[f32; 3]| v.iter().all(|x| x.is_finite());
    if !(finite(&by) && onto.as_ref().is_none_or(|(a, b, n)| finite(a) && finite(b) && finite(n))) {
        set_status("", None, Some("That move gave no usable position; try again."));
        return;
    }
    if BUSY.swap(true, Ordering::SeqCst) {
        return;
    }
    set_status("Moving the part...", Some(""), Some(""));
    std::thread::spawn(move || {
        let p = |v: [f32; 3]| format!("{}\t{}\t{}", v[0], v[1], v[2]);
        let extra = match onto {
            Some((from, to, n)) => {
                format!("X-Body: {}\r\nX-From: {}\r\nX-To: {}\r\nX-Normal: {}\r\n", percent(&body), percent(&p(from)), percent(&p(to)), percent(&p(n)))
            }
            None => format!("X-Body: {}\r\nX-Translate: {}\r\n", percent(&body), percent(&p(by))),
        };
        match post_typed("/move", b"", "text/plain", 120, &extra) {
            Ok(json) => {
                if json.contains("\"changed\": true") {
                    CHANGES.fetch_add(1, Ordering::SeqCst);
                }
                set_status("", None, Some(&json_string(&json, "reply").unwrap_or_default()));
            }
            Err(e) => set_status("", None, Some(&e)),
        }
        BUSY.store(false, Ordering::SeqCst);
    });
}

/// The bodies the last interference check found overlapping (empty: none, or not checked).
pub static CLASH: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Forget the last check (the model changed, or the wearer switched the marks off).
pub fn clear_clash() {
    if let Ok(mut c) = CLASH.lock() {
        c.clear();
    }
}

/// The wearer switched the marks off: forget them, and have the Mac show the design without the overlaps'
/// own red bodies again (POST /interference with X-Clear).
pub fn drop_clash() {
    clear_clash();
    std::thread::spawn(|| {
        let _ = post_typed("/interference", b"", "text/plain", 60, "X-Clear: 1\r\n");
    });
}

/// Asks the Mac which parts overlap (POST /interference). The bodies found are marked and selected, so
/// Isolate shows only those; the reply names the pairs and their overlap.
pub fn check_clash() {
    set_status("Checking for overlaps...", Some(""), Some(""));
    std::thread::spawn(|| match post_typed("/interference", b"", "text/plain", 120, "") {
        Ok(json) => {
            let names: Vec<String> = json_string(&json, "select").map(|s| s.split('\u{1f}').map(str::to_string).collect()).unwrap_or_default();
            log::info!("clash: {} bodies overlap: {}", names.len(), json_string(&json, "reply").unwrap_or_default());
            if let Ok(mut c) = CLASH.lock() {
                *c = names.clone();
            }
            if !names.is_empty() {
                if let Ok(mut q) = SELECT_IN.lock() {
                    *q = Some(names);
                }
            }
            set_status("", None, Some(&json_string(&json, "reply").unwrap_or_default()));
        }
        Err(e) => set_status("", None, Some(&e)),
    });
}

/// A dimension's handle on the model (GET /handles.txt): where it is, the way it moves, and what a move of
/// one model unit adds to the value (2 for a diameter taken hold of at its rim).
#[derive(Clone)]
pub struct Handle {
    pub id: String,
    pub label: String,
    pub value: f32,
    pub unit: String,
    pub at: [f32; 3],
    pub dir: [f32; 3],
    pub gain: f32,
    /// The least and most the value may be (a joint's limits; a size stays above 0.1).
    pub min: f32,
    pub max: f32,
    /// A joint's handle: whether it turns (else it slides), its origin and axis in the model, and the
    /// bodies it carries, so they can be shown moving while it is dragged.
    pub joint: Option<(bool, [f32; 3], [f32; 3], Vec<String>)>,
}

/// The handles of the model on show (empty until fetched, or when the design has none).
pub static HANDLES: Mutex<Vec<Handle>> = Mutex::new(Vec::new());

/// Fetches the dimension handles (GET /handles.txt) in the background.
pub fn fetch_handles() {
    std::thread::spawn(|| {
        let Some(mac) = MAC.lock().ok().and_then(|m| *m) else { return };
        if let Ok(b) = crate::http_get(mac, "/handles.txt", 20) {
            let list: Vec<Handle> = String::from_utf8_lossy(&b)
                .lines()
                .filter_map(|l| {
                    let f: Vec<&str> = l.split('\t').collect();
                    if f.len() < 11 {
                        return None;
                    }
                    let n = |i: usize| f[i].parse::<f32>().ok();
                    Some(Handle { id: f[0].into(), label: f[1].into(), value: n(2)?, unit: f[3].into(), at: [n(4)?, n(5)?, n(6)?], dir: [n(7)?, n(8)?, n(9)?], gain: n(10)?, min: f.get(12).and_then(|v| v.parse().ok()).unwrap_or(if f.len() > 12 { f32::NEG_INFINITY } else { 0.1 }), max: f.get(13).and_then(|v| v.parse().ok()).unwrap_or(f32::INFINITY), joint: if f.len() >= 22 { Some((f[14] == "revolute", [n(15)?, n(16)?, n(17)?], [n(18)?, n(19)?, n(20)?], f[21].split('\u{1f}').map(str::to_string).collect())) } else { None } })
                })
                .collect();
            log::info!("handles: {} on the model", list.len());
            if let Ok(mut h) = HANDLES.lock() {
                *h = list;
            }
        }
    });
}

/// A feature of the design's timeline (GET /features.txt).
#[derive(Clone)]
pub struct Feature {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub off: bool,
    pub error: String,
}

/// The feature tree, once fetched (None: not yet).
pub static FEATURES: Mutex<Option<Vec<Feature>>> = Mutex::new(None);

/// Fetches the feature tree (GET /features.txt) in the background.
pub fn fetch_features() {
    std::thread::spawn(|| {
        let Some(mac) = MAC.lock().ok().and_then(|m| *m) else {
            say("The Mac isn't running serve.py (no beacon seen).");
            return;
        };
        match crate::http_get(mac, "/features.txt", 10) {
            Ok(b) => {
                let list: Vec<Feature> = String::from_utf8_lossy(&b)
                    .lines()
                    .filter_map(|l| {
                        let f: Vec<&str> = l.split('\t').collect();
                        (f.len() >= 4).then(|| Feature { id: f[0].into(), name: f[1].into(), kind: f[2].into(), off: f[3] == "1", error: f.get(4).unwrap_or(&"").to_string() })
                    })
                    .collect();
                if let Ok(mut d) = FEATURES.lock() {
                    *d = Some(list);
                }
            }
            Err(e) => say(&format!("Feature tree: {e}")),
        }
    });
}

/// Switches a feature off (suppressed) or on again (POST /feature, no AI): an undo step like any change.
pub fn toggle_feature(id: &str) {
    let id = id.to_string();
    std::thread::spawn(move || {
        let r = post_typed("/feature", b"", "text/plain", 60, &format!("X-Id: {}\r\n", percent(&id)));
        if r.as_ref().is_ok_and(|json| json.contains("\"changed\": true")) {
            CHANGES.fetch_add(1, Ordering::SeqCst);
        }
        say(&r.map(|json| json_string(&json, "reply").unwrap_or_default()).unwrap_or_else(|e| e));
        fetch_features();
    });
}

/// Fetches the dimensions (GET /dimensions.txt) in the background.
pub fn fetch_dims() {
    std::thread::spawn(|| {
        let Some(mac) = MAC.lock().ok().and_then(|m| *m) else {
            say("The Mac isn't running serve.py (no beacon seen).");
            return;
        };
        match crate::http_get(mac, "/dimensions.txt", 10) {
            Ok(b) => {
                let only = dims_of();
                let list: Vec<Dim> = String::from_utf8_lossy(&b)
                    .lines()
                    .filter(|l| only.is_empty() || l.split('\t').nth(1).is_some_and(|label| label.strip_prefix(only.as_str()).is_some_and(|rest| rest.starts_with(' '))))
                    .filter_map(|l| {
                        let f: Vec<&str> = l.split('\t').collect();
                        let value = f.get(2)?.parse().ok()?;
                        Some(Dim { id: f[0].into(), label: f[1].into(), value, unit: f.get(3).unwrap_or(&"").to_string() })
                    })
                    .collect();
                if let Ok(mut d) = DIMS.lock() {
                    *d = Some(list);
                }
            }
            Err(e) => say(&format!("Dimensions: {e}")),
        }
    });
}

/// Sets one dimension on the Mac (POST /dimension, no AI); the beacon's new version brings the changed model.
pub fn set_dim(id: &str, value: f32) {
    if !value.is_finite() {
        say("That drag gave no usable value; try again.");
        return;
    }
    let id = id.to_string();
    std::thread::spawn(move || {
        let r = post_typed("/dimension", b"", "text/plain", 30, &format!("X-Name: {}\r\nX-Value: {}\r\n", percent(&id), value));
        // A change the panel's Undo can take back, like any other change to the model.
        if r.as_ref().is_ok_and(|json| json.contains("\"changed\": true")) {
            CHANGES.fetch_add(1, Ordering::SeqCst);
        }
        say(&r.map(|json| json_string(&json, "reply").unwrap_or_default()).unwrap_or_else(|e| e));
        fetch_dims();
    });
}

/// Asks the AI again with the last request's words (the Mac repeats what it did before, without the AI, when it can).
pub fn repeat_last() {
    let text = LAST_REQUEST.lock().map(|t| t.clone()).unwrap_or_default();
    if text.is_empty() {
        say("Nothing to repeat yet: make a request first.");
    } else {
        ask_replaying(text);
    }
}

/// Runs a saved request on what is selected now.
pub fn run_saved(text: String) {
    ask_replaying(text);
}

fn ask_replaying(text: String) {
    if BUSY.swap(true, Ordering::SeqCst) {
        say("Still busy with the last request.");
        return;
    }
    std::thread::spawn(move || {
        if let Err(e) = ask(text.as_bytes(), "text/plain; charset=utf-8", Some(&text), true) {
            log::error!("repeat: {e}");
            set_status("", None, Some(&e));
        }
        BUSY.store(false, Ordering::SeqCst);
    });
}

/// What is being typed on the keyboard (an AI request), shown on the panel until Enter or Esc.
pub static TYPED: Mutex<String> = Mutex::new(String::new());

pub fn typed() -> String {
    TYPED.lock().map(|t| t.clone()).unwrap_or_default()
}

/// Sends a typed request to the Mac's AI, like a spoken one (with the selection and any pointed-at spot).
pub fn ask_text(text: String) {
    if need_token() {
        set_token(&text);
        return;
    }
    if BUSY.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(move || {
        if let Err(e) = send_body(text.as_bytes(), "text/plain; charset=utf-8", Some(&text)) {
            log::error!("typed request: {e}");
            set_status("", None, Some(&e));
        }
        BUSY.store(false, Ordering::SeqCst);
    });
}

/// POSTs a WAV to the Mac; returns the JSON body of the reply.
fn post(path: &str, body: &[u8], timeout_s: u64) -> Result<String, String> {
    post_with(path, body, timeout_s, "")
}

/// POST with extra header lines (each ending in \r\n).
fn post_with(path: &str, body: &[u8], timeout_s: u64, extra: &str) -> Result<String, String> {
    post_typed(path, body, "audio/wav", timeout_s, extra)
}

fn post_typed(path: &str, body: &[u8], content_type: &str, timeout_s: u64, extra: &str) -> Result<String, String> {
    let mac = MAC.lock().ok().and_then(|m| *m).ok_or("The Mac isn't running serve.py (no beacon seen).")?;
    let mut s = std::net::TcpStream::connect_timeout(&mac, Duration::from_secs(5)).map_err(|e| format!("Mac: {e}"))?;
    s.set_read_timeout(Some(Duration::from_secs(timeout_s))).map_err(|e| e.to_string())?;
    let selected = SELECTED.lock().map(|s| s.clone()).unwrap_or_default();
    let quoted: Vec<String> = selected.iter().map(|n| format!("\"{}\"", n.replace('\\', "\\\\").replace('"', "\\\""))).collect();
    let sel = percent(&format!("[{}]", quoted.join(",")));
    let faces = FACES.lock().map(|f| f.clone()).unwrap_or_default();
    let extra = if faces.is_empty() { extra.to_string() } else { format!("{extra}X-Faces: {}\r\n", percent(&faces)) };
    let edges = EDGES.lock().map(|f| f.clone()).unwrap_or_default();
    let extra = if edges.is_empty() { extra } else { format!("{extra}X-Edges: {}\r\n", percent(&edges)) };
    let extra = format!("{extra}X-Token: {}\r\n", token());
    let head = format!("POST {path} HTTP/1.0\r\nContent-Type: {content_type}\r\nX-Selected: {sel}\r\n{extra}Content-Length: {}\r\n\r\n", body.len());
    s.write_all(head.as_bytes()).and_then(|_| s.write_all(body)).map_err(|e| format!("Mac: {e}"))?;
    let mut all = Vec::new();
    s.read_to_end(&mut all).map_err(|e| format!("Mac: {e}"))?;
    let text = String::from_utf8_lossy(&all);
    if text.starts_with("HTTP/1.0 403") || text.starts_with("HTTP/1.1 403") {
        token_refused();
        return Err("The Mac wants an access code: type the code shown in its serve.py window and press Enter.".into());
    }
    Ok(text.split("\r\n\r\n").nth(1).unwrap_or("").to_string())
}

/// Sends the recording to the Mac and shows its answer.
fn send(pcm: &[i16]) -> Result<(), String> {
    // Keep the live text on show while the Mac works (its final transcript replaces it).
    let partial = STATUS.lock().ok().and_then(|s| s.0.strip_prefix("Hearing: ").map(str::to_string));
    send_body(&wav(pcm), "audio/wav", partial.as_deref())
}

/// What the request line is for: a new note (a pin is waiting), new words for a note, or the AI.
pub fn typed_target() -> &'static str {
    if need_token() {
        "Code"
    } else if PIN.lock().map(|p| p.is_some()).unwrap_or(false) {
        "Note"
    } else if REDO.lock().map(|r| r.is_some()).unwrap_or(false) {
        "Edit note"
    } else {
        "AI"
    }
}

/// A recording or typed text for whatever is waiting for it: a pin (a new note), a note to re-word, or the AI.
fn send_body(body: &[u8], ctype: &str, partial: Option<&str>) -> Result<(), String> {
    // A note: the recording is saved as text at the pin, not sent to the AI.
    if let Some(pin) = PIN.lock().ok().and_then(|mut p| p.take()) {
        set_status("Saving the note...", partial, None);
        if let Ok(mut p) = POINT.lock() {
            *p = None;
        }
        let json = post_typed("/comment", body, ctype, 60, &format!("X-Pin: {}\r\n", percent(&pin)))?;
        let heard = json_string(&json, "heard").unwrap_or_default();
        set_status("", Some(&heard), Some(&json_string(&json, "reply").unwrap_or_else(|| "No answer from the Mac.".into())));
        if let (Some(id), true) = (json_string(&json, "id"), json.contains("\"ok\": true")) {
            if let Ok(mut s) = NOTE_STEPS.lock() {
                s.push(NoteStep::Added(id));
            }
        }
        return Ok(());
    }
    // Re-recording a note: the new words replace the old ones (kept for Undo).
    if let Some(id) = REDO.lock().ok().and_then(|mut r| r.take()) {
        set_status("Saving the new words...", partial, None);
        let json = post_typed("/comment", body, ctype, 60, &format!("X-Id: {}\r\n", percent(&id)))?;
        let heard = json_string(&json, "heard").unwrap_or_default();
        set_status("", Some(&heard), Some(&json_string(&json, "reply").unwrap_or_else(|| "No answer from the Mac.".into())));
        if let (Some(old), true) = (json_string(&json, "old"), json.contains("\"ok\": true")) {
            if let Ok(mut s) = NOTE_STEPS.lock() {
                s.push(NoteStep::Text(id, old));
            }
        }
        return Ok(());
    }
    ask(body, ctype, partial, false)
}

/// An AI request (a recording or typed text) to the Mac, and its answer on the panel.
fn ask(body: &[u8], content_type: &str, heard_so_far: Option<&str>, replay: bool) -> Result<(), String> {
    set_status("Thinking...", heard_so_far, None);
    CONFIRM.store(false, Ordering::SeqCst);
    let point = POINT.lock().ok().and_then(|mut p| p.take()).map(|p| format!("X-Point: {}\r\n", percent(&p))).unwrap_or_default();
    let point = if replay { format!("{point}X-Replay: 1\r\n") } else { point };
    let json = post_typed("/ask", body, content_type, 180, &point)?;
    show_answer(&json);
    Ok(())
}

/// The Mac's answer to an AI request: what it heard and its reply, or a big request waiting for an OK.
fn show_answer(json: &str) {
    let heard = json_string(json, "heard").unwrap_or_default();
    let reply = json_string(json, "reply").unwrap_or_else(|| "No answer from the Mac.".into());
    log::info!("voice: heard {heard:?}, reply {reply:?}, cost {:?}", json_string(json, "cost"));
    if !heard.is_empty() && json.contains("\"ok\": true") {
        if let Ok(mut l) = LAST_REQUEST.lock() {
            *l = heard.clone();
        }
    }
    if json.contains("\"changed\": true") || json.contains("\"changed\":true") {
        CHANGES.fetch_add(1, Ordering::SeqCst);
    }
    // The request named parts to select ("select the posts"): the render loop selects them.
    if let Some(names) = json_string(json, "select") {
        if let Ok(mut q) = SELECT_IN.lock() {
            *q = Some(names.split('\u{1f}').map(str::to_string).collect());
        }
    }
    // The request was a command's phrase ("hide these"): the render loop runs it as that button.
    if let Some(id) = json_string(json, "command") {
        if let Ok(mut q) = COMMANDS_IN.lock() {
            q.push(id);
        }
    }
    match json_string(json, "confirm_k") {
        Some(k) => {
            CONFIRM.store(true, Ordering::SeqCst);
            // The Mac's worst case for one more step (the whole request at the full input price), in cents.
            let cents = json_string(json, "confirm_cost").and_then(|c| c.parse::<f32>().ok()).map(|d| d * 100.0).unwrap_or(0.0);
            let cost = if cents < 0.1 { "<0.1c".to_string() } else if cents < 10.0 { format!("~{cents:.1}c") } else { format!("~{:.0}c", cents) };
            set_status(&format!("Big: ~{k}k tokens, {cost}. Go on?"), Some(&heard), None);
        }
        None if json.contains("\"preview\": true") => {
            // Shown, not made: the panel offers Apply and Drop.
            PREVIEW.store(true, Ordering::SeqCst);
            set_status("Preview. Apply it?", Some(&heard), Some(&reply));
        }
        None => set_status("", Some(&heard), Some(&reply)),
    }
}

/// Parts the server found for a "select ..." request, for the render loop to select (replacing the selection).
static SELECT_IN: Mutex<Option<Vec<String>>> = Mutex::new(None);

pub fn take_select() -> Option<Vec<String>> {
    SELECT_IN.lock().ok()?.take()
}

/// How many exploded views of its own the design has (SolidCraft's, for assemblies); 0 until known.
static EXPLODES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn explode_views() -> u32 {
    EXPLODES.load(Ordering::SeqCst)
}

/// Asks the Mac how many exploded views the design has (GET /explodes.txt), in the background.
pub fn fetch_explodes() {
    std::thread::spawn(|| {
        let Some(mac) = MAC.lock().ok().and_then(|m| *m) else { return };
        if let Ok(b) = crate::http_get(mac, "/explodes.txt", 10) {
            EXPLODES.store(String::from_utf8_lossy(&b).lines().filter(|l| !l.trim().is_empty()).count() as u32, Ordering::SeqCst);
        }
    });
}

/// Shows the design's next exploded view, then the assembled design again (POST /explode).
pub fn explode_next() {
    std::thread::spawn(|| {
        let r = post_typed("/explode", b"", "text/plain", 60, "");
        say(&r.map(|json| json_string(&json, "reply").unwrap_or_default()).unwrap_or_else(|e| e));
    });
}

/// Tells the Mac which parts are hidden in the headset (POST /visibility), so the AI knows what the wearer
/// cannot see. Sent when the set changes; nothing is shown for it.
pub fn sync_hidden(names: Vec<String>) {
    std::thread::spawn(move || {
        let list = names.iter().map(|n| format!("\"{}\"", n.replace('\\', "\\\\").replace('"', "\\\""))).collect::<Vec<_>>().join(",");
        let _ = post_typed("/visibility", b"", "text/plain", 10, &format!("X-Hidden: {}\r\n", percent(&format!("[{list}]"))));
    });
}

/// Commands the server recognised in a spoken or typed request, for the render loop to run, oldest first.
static COMMANDS_IN: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// The next recognised command, if any.
pub fn take_command() -> Option<String> {
    let mut q = COMMANDS_IN.lock().ok()?;
    if q.is_empty() { None } else { Some(q.remove(0)) }
}

/// A big request is waiting for the wearer's OK (Enter or B to go on, Esc to stop).
static CONFIRM: AtomicBool = AtomicBool::new(false);

pub fn confirm_pending() -> bool {
    CONFIRM.load(Ordering::SeqCst) || PREVIEW.load(Ordering::SeqCst)
}

/// The model on show is a preview: what the last request would give. The design itself is unchanged until
/// the wearer applies it. The Mac's beacon says so every second, and its answer to the request does at once.
static PREVIEW: AtomicBool = AtomicBool::new(false);

pub fn preview() -> bool {
    PREVIEW.load(Ordering::SeqCst)
}

pub fn set_preview(on: bool) {
    PREVIEW.store(on, Ordering::SeqCst);
}

/// Applies (`true`) or drops the change on show.
fn settle_preview(apply: bool) {
    if BUSY.swap(true, Ordering::SeqCst) {
        return;
    }
    PREVIEW.store(false, Ordering::SeqCst);
    set_status(if apply { "Applying..." } else { "Dropping..." }, None, None);
    std::thread::spawn(move || {
        match post_typed(if apply { "/preview/apply" } else { "/preview/cancel" }, b"", "text/plain", 120, "") {
            Ok(json) => {
                if json.contains("\"changed\": true") {
                    CHANGES.fetch_add(1, Ordering::SeqCst);
                }
                set_status("", None, Some(&json_string(&json, "reply").unwrap_or_default()));
            }
            Err(e) => set_status("", None, Some(&e)),
        }
        BUSY.store(false, Ordering::SeqCst);
    });
}

/// Esc: drop the waiting request, or the change on show.
pub fn cancel_confirm() {
    if PREVIEW.load(Ordering::SeqCst) {
        settle_preview(false);
    } else if CONFIRM.swap(false, Ordering::SeqCst) {
        set_status("", None, Some("Stopped: that request was not sent on."));
    }
}

/// The wearer said OK: the Mac carries on with the request that was waiting.
pub fn confirm_ai() {
    if PREVIEW.load(Ordering::SeqCst) {
        settle_preview(true);
        return;
    }
    if !CONFIRM.load(Ordering::SeqCst) || BUSY.swap(true, Ordering::SeqCst) {
        return;
    }
    CONFIRM.store(false, Ordering::SeqCst);
    set_status("Continuing...", None, None);
    std::thread::spawn(|| {
        match post_typed("/ask/confirm", b"", "text/plain", 180, "") {
            Ok(json) => show_answer(&json),
            Err(e) => {
                log::error!("confirm: {e}");
                set_status("", None, Some(&e));
            }
        }
        BUSY.store(false, Ordering::SeqCst);
    });
}

/// The string value of a top-level key in a small flat JSON object (enough for the /ask reply).
fn json_string(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let start = json.find(&pat)? + pat.len();
    let rest = json[start..].trim_start().strip_prefix(':')?.trim_start().strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push(' '),
                't' => out.push(' '),
                'u' => {
                    let hex: String = chars.by_ref().take(4).collect();
                    out.push(u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32).unwrap_or('?'));
                }
                other => out.push(other),
            },
            c => out.push(c),
        }
    }
    None
}
