# SC View: sample code for a Quest 3 viewer and AI remote for SolidCraft

**An experiment, shared as sample code to read and start from. Not a finished product, and not supported.**

SC View shows a SolidCraft model in 3D on a Meta Quest 3, over your real room (passthrough). You select parts, faces and edges with the controllers, move and size the model, and change it by voice: a server on your Mac keeps the design open in a headless SolidCraft session and passes your requests to Claude, which edits the design with SolidCraft's tools. The headset shows the result a few seconds later.

This was written with Claude Code. The guide *Building a Quest 3 viewer for SolidCraft with Claude Code* describes how, step by step, so you can ask Claude to build your own version. This code is what came out of it.

> Unofficial. Not affiliated with or endorsed by the SolidCraft project (`tangency-labs`), Meta or Anthropic. Tested on one Quest 3 and one Mac, with SolidCraft v0.41.0.

## What is here

- `src/` : the viewer, a native OpenXR app in Rust (Vulkan with multiview, passthrough, android-activity). `lib.rs` is the app and the render loop, `voice.rs` the speech and network requests, `panel.rs` and `icons.rs` and `font.rs` the wrist panel, `raycast.rs` hand-written bindings for `XR_META_environment_raycast` (to find tables and floors), `scene.rs` the kept placement.
- `shaders/` : the GLSL sources and the compiled SPIR-V the app includes.
- `tools/serve.py` : the server (Python 3, standard library only). It finds the headset by a UDP beacon, serves the model over HTTP, runs `whisper-cli` for speech to text, talks to SolidCraft over MCP and to the Claude API, and keeps count of the cost. `obj_to_scm.py` converts SolidCraft's OBJ export to the mesh format the headset reads. `deploy.sh` installs the app. `make_font.swift` generated the bitmap font.

## What you can do in the headset

- Point and select parts, faces or edges (trigger or mouse). Hide, isolate, show all.
- Grab the model with the grip to move, turn and size it. Section cut, measure, exploded view, axes and grid.
- Push/pull a face by distance.
- Hold the left trigger and speak: "make the selected parts red", "fillet these edges 1 mm", "put a 6 mm hole through here". What you point at and what is selected go with the request.
- Set the model down on a table or the floor (a raycast against the room), and step its size with a Size button (a 1:1000 model at x2 is 1:500).
- Undo and redo the AI's changes, repeat or save requests (a saved request can run again without calling the AI), and save the result as a new `.sc` file next to the original.
- A Bluetooth keyboard and mouse work too (F1 lists the keys), and there is an on-screen keyboard.

## What you need

- A Meta Quest 3 in developer mode, and `adb` to install the app.
- A computer on the same Wi-Fi network with the SolidCraft command-line program (`solidcraft-cli`), `whisper-cli` (whisper.cpp) with an English model, and an Anthropic API key.
- To build the viewer: Rust (the toolchain in `rust-toolchain.toml`), `cargo-apk`, the Android SDK and NDK, and a JDK.
- Meta's OpenXR loader for Android, `libopenxr_loader.so`. It is **not** included here: download it from Meta's OpenXR Mobile SDK and put it at `lib/arm64-v8a/libopenxr_loader.so`.

## Build the viewer

```sh
# Make a signing key once (the app is signed with it; keep it, or the headset will want the old app removed first):
keytool -genkeypair -keystore test.keystore -storepass scviewtest -alias scview -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=scview"

cargo apk build --release
adb install -r target/release/apk/scview.apk
```

`tools/deploy.sh` installs and starts the app, and copies the server's access code to the headset (see below). The first run asks for the microphone permission. Start the app from Library, Unknown Sources, with the headset on.

## Run the server

```sh
# once: keep your Anthropic API key in the macOS Keychain
security add-generic-password -a "$USER" -s scview-anthropic -w

python3 -I tools/serve.py /path/to/model.step --quest <headset ip>
```

The source can be a `.step`, `.stp` or `.sc` design (the AI can change these), or a view-only `.obj` / `.scm`. A folder means "the newest model in it". Point the server at the SolidCraft program with `SOLIDCRAFT_CLI=/path/to/solidcraft-cli` (or have it on your `PATH`), and at the whisper model with `SCVIEW_WHISPER_MODEL=/path/to/ggml-small.en.bin`.

The first run prints an **access code** and keeps it in `~/.scview/token`. The headset asks for the code once (type it with the keyboard that comes up); `tools/deploy.sh` can copy it for you. Every request without the code is refused. `--no-auth` turns the check off (tests only).

Useful options: `--balance <dollars>` shows your remaining API credit in the headset (an estimate), `--context-confirm` and `--context-max` stop a very large AI request until you say "go on", `--models <folder>` adds a folder to the headset's Models list.

## Without the Claude API, and without any AI

- `--ai ollama` sends voice requests to a local model behind an OpenAI-compatible endpoint (Ollama by default: `--ai-url`, `--ai-model`) instead of the Claude API. No key, no cost. See [docs/local-ai-tier.md](docs/local-ai-tier.md).
- A request that only sets one parameter or feature dimension ("set thickness to 12", "make the hole 8mm") is done by the server itself, with no AI. `GET /dimensions.txt` lists what can be set and `POST /dimension` sets one, for a slider or a handle in the headset. `--no-direct` turns this off. See [docs/direct-edits.md](docs/direct-edits.md).

Both were built and tested on Linux with text requests and a simulated headset, not yet with a Quest or with speech.

## Things to know

- The server listens on your local network. The access code stops other devices using it, but the traffic is not encrypted: use it on a network you trust.
- Voice requests go to the Anthropic API and cost a little each time (a typical request is a fraction of a cent with Claude Haiku). Your API key stays on the computer, in the Keychain, and is only sent to api.anthropic.com. Speech is turned into text on the computer.
- The headset's runtime did not offer the room-scan extensions (`XR_FB_scene` and friends), so surfaces are found with `XR_META_environment_raycast`. Keeping the model in place between sessions uses the stage space and may not be reliable; Meta recommends spatial anchors.
- Push/pull can fail on some faces (an error from SolidCraft's geometry kernel); the server reports the reason.
- The logs are in `~/.scview/serve.log` on the computer and `scview.log` in the app's folder on the headset.
- The app id is `org.scview.viewer` (`package` in `Cargo.toml`).

## Licence

Licensed under either of the [MIT licence](LICENSE-MIT) or the [Apache License, Version 2.0](LICENSE-APACHE), at your option. This is the same dual licence as SolidCraft's code. It does not use SolidCraft's UI kit, icons, theme or name, which have their own Brand and Interface Licence.
