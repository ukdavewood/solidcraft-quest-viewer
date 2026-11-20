//! `scview-desktop`: the viewer on Linux, against the OpenXR runtime in `XR_RUNTIME_JSON`.

#[cfg(not(target_os = "android"))]
fn main() {
    scview::desktop_main();
}

#[cfg(target_os = "android")]
fn main() {}
