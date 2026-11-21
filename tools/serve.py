#!/usr/bin/env python3
"""Serve a model to the SC View app on the Quest over Wi-Fi, and let the wearer change it by voice.

Usage: python3 -I serve.py <model or folder> [--port 47801] [--quest <ip>] [--no-ai]

<model> can be a .step/.stp or .sc design (opened in a headless SolidCraft session, so the AI can
change it), an .obj (with its .mtl) or a ready .scm (view only). A folder means "the newest model
file in it". The source is checked every second; when it changes it is opened again and the
headset downloads it.

The Quest finds this Mac from a UDP beacon ("SCVIEW1 <port> <version>") sent every second to the
broadcast address on port 47800, then fetches http://<mac>:<port>/model.scm.
--quest <ip> also sends the beacon straight to the headset, for networks that drop broadcasts.
macOS may ask once whether Python may accept incoming connections: allow it.
The headset must send an access code (printed at start, kept in ~/.scview/token) with every request.

Voice: the headset POSTs a WAV recording (or plain text) to /ask. whisper-cli turns speech into
text, Claude Haiku changes the design with SolidCraft's MCP tools, and the changed model is
exported and sent back. The Anthropic API key is read from the macOS Keychain item
"scview-anthropic" (store it once with: security add-generic-password -a "$USER" -s scview-anthropic -w).
The key never leaves this Mac except to api.anthropic.com.

Notes: the headset POSTs a recording to /comment with the pin (body name and point) in X-Pin; the
words are saved with it in "<model file>.comments.json" next to the model, so the desktop can read
them. /comment/delete and /comment/restore (X-Id) mark a note deleted or bring it back. /comment/do
(X-Id, or "all" for every note not yet done) gives notes to the AI as requests, with the pin as the
place "here" refers to; done notes are marked "done" with the AI's reply. The headset reads the live
notes from /comments.txt; the beacon's 4th field is the notes version.
"""
import argparse
import hmac
import secrets
import shutil
import http.server
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# The SolidCraft command-line program (SOLIDCRAFT_CLI, or solidcraft-cli on the PATH) and the whisper.cpp model
# (SCVIEW_WHISPER_MODEL, or ggml-small.en.bin in ~/.cache/whisper).
DEFAULT_CLI = os.environ.get("SOLIDCRAFT_CLI") or shutil.which("solidcraft-cli") or "solidcraft-cli"
DEFAULT_WHISPER_MODEL = os.environ.get("SCVIEW_WHISPER_MODEL") or os.path.expanduser("~/.cache/whisper/ggml-small.en.bin")
BEACON_PORT = 47800
MODEL_EXTS = (".step", ".stp", ".sc", ".obj", ".scm")
CLAUDE_MODEL = "claude-haiku-5-5"
# Which AI answers /ask (--ai, --ai-url, --ai-model, --ai-think; or SCVIEW_AI, SCVIEW_AI_URL, SCVIEW_AI_MODEL).
# "anthropic" is the Claude API; "ollama" is any server that speaks OpenAI's chat completions (Ollama, llama.cpp,
# LM Studio, vLLM) on this computer or the local network: no key, no cost, nothing leaves the network.
AI = {"backend": "anthropic", "url": "http://localhost:11434/v1", "model": "qwen3.8:27b", "think": "none", "ctx": 0}

state = {"data": None, "version": None, "name": None, "counter": 0, "source": None, "notes": [], "notes_version": 0}
lock = threading.Lock()
args = None
session = None  # the headless SolidCraft session (Mcp), when the source is a design
work = tempfile.mkdtemp(prefix="scview-")


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:  # also kept in ~/.scview/serve.log, which a restart does not wipe
        os.makedirs(SCV_DIR, exist_ok=True)
        with open(os.path.join(SCV_DIR, "serve.log"), "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# --- headless SolidCraft session over MCP (JSON-RPC on stdio) ---

class Mcp:
    def __init__(self, cli):
        self.p = subprocess.Popen([cli, "mcp", "--headless", "--allow-any-path", "--compact"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.n = 0
        self.lock = threading.Lock()
        self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                "clientInfo": {"name": "scview", "version": "1"}})
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}) + "\n")
        self.p.stdin.flush()
        self.tools = {t["name"]: t for t in self.rpc("tools/list")["tools"]}

    def rpc(self, method, params=None):
        with self.lock:
            self.n += 1
            msg = {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}
            self.p.stdin.write(json.dumps(msg) + "\n")
            self.p.stdin.flush()
            while True:
                line = self.p.stdout.readline()
                if not line:
                    raise RuntimeError("the SolidCraft session ended")
                r = json.loads(line)
                if r.get("id") == self.n:
                    if "error" in r:
                        raise RuntimeError(r["error"].get("message", str(r["error"])))
                    return r["result"]

    def call(self, name, arguments):
        """A tool call: (text, is_error)."""
        r = self.rpc("tools/call", {"name": name, "arguments": arguments})
        text = "\n".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")
        return text, bool(r.get("isError"))


def export_scm():
    """Export the session's design to OBJ, convert it, and publish it as a new version."""
    obj = os.path.join(work, "model.obj")
    out = os.path.join(work, "model.scm")
    text, err = session.call("export", {"path": obj, "format": "obj"})
    if err:
        raise RuntimeError(f"export failed: {text[:200]}")
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "obj_to_scm.py"), obj, out], check=True,
                   stdout=subprocess.DEVNULL)
    with open(out, "rb") as f:
        data = f.read()
    with lock:
        state["counter"] += 1
        state["data"] = data
        state["version"] = f"{int(time.time()):x}-{state['counter']}"
    return data


# --- Claude ---

AI_TOOLS = ["inspect_design", "list_commands", "describe_command", "execute", "batch", "set_parameter", "undo",
            "measure", "body_topology"]

SYSTEM = """You change the open design in the CAD program SolidCraft for someone wearing a VR headset, who \
sees the model in 3D and speaks the request (it reaches you as a speech transcript, so expect small \
mis-hearings and fix them sensibly). Lengths are millimetres; Z is up. The request comes with a summary of \
the design (bodies with their bounding boxes, parameters, timeline), so do not inspect again unless you \
need more. Be quick: aim for one or two tool calls, call independent tools in the same reply, and do not \
undo and retry unless a result is clearly wrong.

Recipes (use these exact commands):
- Hide bodies: execute browser.visibility {"items": ["Body name", ...], "visible": false}. Show them: \
"visible": true. Show everything: {"folder": "bodies", "visible": true}. Show only some: {"items": [...], \
"visible": true, "isolate": true}.
- A hole: execute solid.hole {"position": [x, y, z], "diameter": D} with the position a point ON a face of \
the body (for example the centre of a bounding-box face, which lies on a flat end face); it drills into \
the face, through all by default ("depth" for a blind hole). Pick the face from the bounding box: "through \
the middle" of a long part usually means along its longest axis from one end face. Always pass "targets": ["Body name", ...] with the body or bodies meant (the selected or pinned ones): without it a through hole also cuts every other body in its path.
- Round or bevel edges ("fillet" may be heard as "fill it"): solid.fillet / solid.chamfer with "edges": \
[{"body": "Body name", "index": i}, ...] (edge indices from body_topology; shorter than midpoints, so use them \
for many edges) and "radius" / "distance". Only sharp edges \
can be rounded: an error saying an edge "is smooth" (its faces meet tangentially) or that a round "is too \
small" means drop THAT edge (the error gives its position) and retry with the rest; "larger than the faces \
beside its edges allow" means a smaller radius. Default radius when none is said: about a third of the \
shortest edge you round, at most 1 mm. After three failed tries, stop and say which edges could not be done.
- A new shape: describe_command first if unsure, e.g. sketch.create {"plane": "XY", "offset": z} then \
sketch.circle.center / sketch.rectangle.two_point and solid.extrude {"distance": d, "operation": \
"new"|"join"|"cut"}; extrude "through_all": true for a cut through everything.
- Colour: execute appearance.assign {"bodies": ["Body name", ...], "color": "#rrggbb"} (opacity 0-1 \
optional); {"bodies": [...], "color": null} clears it. For a plain colour name use a mid-tone that \
stays recognisable when lit in VR, not a pure primary: blue #3070ff, red #e03030, green #30b040, \
yellow #f0d020, orange #f08020, purple #9050d0, black #303030, white #f0f0f0.
- Faces: the headset can send selected faces and the face pointed at, matched to SolidCraft's faces (name, \
index, a point on it). Commands that act on faces take that point or the face name; for the edges of a face, \
use body_topology and keep the edges whose names include that face's name.
- What a request refers to: when the wearer is pointing at a body (given with the request), "this", "this \
part", "it" and "here" mean the pointed body and point, even if other bodies are selected; "selected", "these" \
and "the selected parts" mean the selected bodies. When nothing is pointed at, "this", "these" and "it" mean \
the selected bodies. Never change a body the request does not refer to.
- The whole design: "the whole ship", "the whole model", "everything", "all the parts" mean every body, whatever \
is selected or pointed at. In execute params write "bodies": ["*"] (or "targets": ["*"]) and the server fills in every \
body name; never list hundreds of names yourself and never fall back to the selected bodies for such a request.
- Names: the SolidCraft tools cut long body lists short, so do not read names from inspect_design or measure. Call \
find_bodies with a pattern to get exact names. For "each one a different colour" (or any many-body recolouring) call \
colour_each once with the bodies or a pattern; never write one command per body (a long reply gets cut off and does nothing).
- Bodies by name pattern ("all the sails", "everything called flat_sail..."): write the pattern with * in the same \
list, e.g. "bodies": ["flat_sail*"] (case does not matter); the server expands it to every matching body.
- Move: solid.move {"bodies": [...], "translate": [x, y, z]} in millimetres; "right" is +X, "left" -X, "forward" +Y, \
"back" -Y, "up" +Z, "down" -Z unless the request says otherwise; say which one you used.
- Mis-hearings: the request is a speech transcript. A word that makes no sense where a colour or part is meant is \
probably a colour misheard ("right" for "red", "white" or "write"): choose the nearest colour, and say which you used.
- Tool argument values: inspect_design "timeline" is true, false or "summary" (booleans, not strings), \
"bodies" true, false or "names".
For anything else use list_commands with a filter and describe_command; never guess parameter names. \
Make only the changes asked for. Reply in one short sentence (it is shown on a small panel) saying what \
you did, or what you could not do."""


# A design with more bodies than this gets a short summary: bounding boxes only for the bodies the request is about.
SUMMARY_BODIES = 30
# Names of the other bodies are listed (without boxes) while they fit in this many characters.
SUMMARY_NAME_CHARS = 5000


def design_summary(focus=()):
    """Bodies with bounding boxes, parameters and the timeline, sent with each request (saves a round).
    `focus`: names of the selected or pointed-at bodies; in a big design only these get boxes."""
    parts = []
    try:
        text, err = session.call("measure", {"summary": True})
        if not err:
            m = json.loads(text)
            bodies = m.get("bodies", [])
            line = lambda b: (f"- {b.get('name')}: bbox {[round(v, 2) for v in (b.get('bbox') or {}).get('min', [])]} .. "
                              f"{[round(v, 2) for v in (b.get('bbox') or {}).get('max', [])]}")
            if len(bodies) <= SUMMARY_BODIES:
                parts += [line(b) for b in bodies]
            else:
                focus = set(focus)
                shown = [b for b in bodies if b.get("name") in focus][:SUMMARY_BODIES]
                parts.append(f"The design has {len(bodies)} bodies. Boxes are given only for the bodies this request is about:")
                parts += [line(b) for b in shown]
                others = [str(b.get("name")) for b in bodies if b.get("name") not in focus]
                names = ", ".join(others)
                if len(names) <= SUMMARY_NAME_CHARS:
                    parts.append(f"All other bodies: {names}")
                else:
                    parts.append(f"{len(others)} other bodies (too many to list): look names up with inspect_design "
                                 f"(bodies \"names\") or measure, only if the request needs them.")
        text, err = session.call("inspect_design", {"timeline": "summary", "bodies": False, "measure": False,
                                                    "sketches": False})
        if not err:
            d = json.loads(text)
            parts.append(f"Parameters: {json.dumps(d.get('params', []))[:800]}")
            parts.append(f"Timeline: {json.dumps(d.get('timeline', []))[:800]}")
            hidden = d.get("visibility", {}).get("hidden_bodies") if isinstance(d.get("visibility"), dict) else None
            if hidden:
                parts.append(f"Hidden bodies: {hidden}")
    except (RuntimeError, ValueError) as e:
        parts.append(f"(summary unavailable: {e})")
    return "\n".join(parts)


def api_key():
    """The Anthropic API key: the macOS Keychain, else the Linux keyring (secret-tool, service scview-anthropic),
    else ANTHROPIC_API_KEY."""
    for cmd in (["security", "find-generic-password", "-s", "scview-anthropic", "-w"],
                ["secret-tool", "lookup", "service", "scview-anthropic"]):
        if not shutil.which(cmd[0]):
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return os.environ.get("ANTHROPIC_API_KEY") or None


def claude(key, body):
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Claude API {e.code}: {e.read()[:300].decode(errors='replace')}")


def to_openai(body):
    """An Anthropic Messages request as an OpenAI chat completions request (text and tool calls only)."""
    msgs = [{"role": "system", "content": "\n".join(b["text"] for b in body.get("system", []))}]
    for m in body["messages"]:
        content = m["content"]
        if isinstance(content, str):
            msgs.append({"role": m["role"], "content": content})
            continue
        text = "\n".join(c["text"] for c in content if c.get("type") == "text")
        if m["role"] == "assistant":
            out = {"role": "assistant", "content": text}
            calls = [{"id": c["id"], "type": "function",
                      "function": {"name": c["name"], "arguments": json.dumps(c.get("input") or {})}}
                     for c in content if c.get("type") == "tool_use"]
            if calls:
                out["tool_calls"] = calls
            msgs.append(out)
            continue
        for c in content:  # a user turn: the tool results first, then anything said after them
            if c.get("type") == "tool_result":
                r = c.get("content")
                r = r if isinstance(r, str) else json.dumps(r)
                msgs.append({"role": "tool", "tool_call_id": c["tool_use_id"],
                             "content": ("ERROR: " + r) if c.get("is_error") else r})
        if text:
            msgs.append({"role": "user", "content": text})
    out = {"model": AI["model"], "messages": msgs, "max_tokens": body.get("max_tokens", 4000), "stream": False,
           "tools": [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                       "parameters": t.get("input_schema", {"type": "object"})}}
                     for t in body.get("tools", [])]}
    if (body.get("tool_choice") or {}).get("type") == "none":
        out["tool_choice"] = "none"
    if AI["think"]:
        out["reasoning_effort"] = AI["think"]
    return out


def from_openai(r):
    """An OpenAI chat completions reply as an Anthropic Messages reply, which is what run_ai() reads."""
    choice = (r.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = []
    if (msg.get("content") or "").strip():
        content.append({"type": "text", "text": msg["content"].strip()})
    for i, c in enumerate(msg.get("tool_calls") or []):
        f = c.get("function") or {}
        a = f.get("arguments")
        try:
            a = json.loads(a) if isinstance(a, str) else (a or {})
        except ValueError:
            a = {}
        content.append({"type": "tool_use", "id": c.get("id") or f"call_{uuid.uuid4().hex[:8]}_{i}",
                        "name": f.get("name", ""), "input": a if isinstance(a, dict) else {}})
    u = r.get("usage") or {}
    return {"content": content, "stop_reason": "max_tokens" if choice.get("finish_reason") == "length" else "end_turn",
            "usage": {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")}}


def local_chat(body):
    """One step with the local model, through the OpenAI-compatible endpoint at AI["url"]."""
    req = urllib.request.Request(AI["url"].rstrip("/") + "/chat/completions", data=json.dumps(to_openai(body)).encode(),
                                 headers={"content-type": "application/json",
                                          "authorization": "Bearer " + (os.environ.get("SCVIEW_AI_KEY") or "ollama")})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return from_openai(json.loads(r.read()))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"local AI {e.code}: {e.read()[:300].decode(errors='replace')}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"The local AI at {AI['url']} does not answer ({e.reason}). Is Ollama running?")


def ai_step(key, body):
    """One step of the AI loop with whichever backend is chosen. Both take and return the Anthropic shapes."""
    return local_chat(body) if AI["backend"] != "anthropic" else claude(key, body)


def tidy_args(name, a):
    """Small models often send booleans as strings ("false", "\\"summary\\""): fix that for the read-only tools."""
    if name not in ("inspect_design", "measure") or not isinstance(a, dict):
        return a
    out = {}
    for k, v in a.items():
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):  # a list sent as JSON text
                try:
                    v = json.loads(s)
                    out[k] = v
                    continue
                except ValueError:
                    pass
            s = s.strip('"')
            v = {"true": True, "false": False}.get(s.lower(), s)
        out[k] = v
    return out


def all_body_names():
    text, err = session.call("inspect_design", {"timeline": False, "bodies": "names", "measure": False, "sketches": False})
    try:
        return [b for b in json.loads(text).get("bodies", []) if isinstance(b, str)] if not err else []
    except ValueError:
        return []


def expand_all_bodies(name, a):
    """The AI writes "bodies": ["*"] for the whole design, or a name pattern like ["flat_sail*"] (* matches any
    characters, ignoring case); SolidCraft wants exact names, so fill them in."""
    if name not in ("execute", "batch") or not isinstance(a, dict):
        return a
    names = []

    def expand(items):
        out = []
        for x in items:
            if isinstance(x, str) and (x == "all" or "*" in x):
                if not names:
                    names.extend(all_body_names())
                pat = re.compile(re.escape("*" if x == "all" else x).replace(r"\*", ".*"), re.I)
                out += [n for n in names if pat.fullmatch(n)]
            else:
                out.append(x)
        return list(dict.fromkeys(out))  # no repeats, in order

    def fix(x):
        if isinstance(x, dict):
            res = {}
            for k, v in x.items():
                if k in ("bodies", "targets", "items"):
                    v = [v] if v in ("*", "all") else v
                    if isinstance(v, list) and any(isinstance(i, str) and (i == "all" or "*" in i) for i in v):
                        res[k] = expand(v)
                        continue
                res[k] = fix(v)
            return res
        if isinstance(x, list):
            return [fix(v) for v in x]
        return x
    return fix(a)


def find_bodies(inp):
    """The find_bodies tool: exact names matching a pattern, as JSON text."""
    pat = str((inp or {}).get("pattern") or "*")
    try:
        limit = int((inp or {}).get("limit") or 200)
    except (TypeError, ValueError):
        limit = 200
    rx = re.compile(re.escape(pat).replace(r"\*", ".*"), re.I)
    hits = [n for n in all_body_names() if rx.fullmatch(n)]
    return json.dumps({"count": len(hits), "bodies": hits[:max(1, min(limit, 400))]})


def distinct_colours(n):
    """n distinct mid-tone colours (#rrggbb): hues spread by the golden ratio, brightness and strength alternating."""
    import colorsys
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, (0.75, 0.55)[i % 2], (0.85, 0.7)[(i // 2) % 2])
        out.append(f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}")
    return out


def colour_each(inp):
    """The colour_each tool: one batch assigns every named body its own colour. Returns (text, is_error)."""
    names = expand_all_bodies("execute", {"bodies": list((inp or {}).get("bodies") or [])})["bodies"]
    if not names:
        return "No bodies matched.", True
    colors = [c for c in ((inp or {}).get("colors") or []) if isinstance(c, str)] or distinct_colours(len(names))
    cmds = [{"command": "appearance.assign", "params": {"bodies": [n], "color": colors[i % len(colors)]}} for i, n in enumerate(names)]
    text, err = session.call("batch", {"commands": cmds})
    return (f"Coloured {len(names)} bodies, each with its own colour." if not err else text[:300]), err


def undo_depth():
    try:
        text, err = session.call("inspect_design", {"timeline": False, "bodies": False, "measure": False,
                                                    "sketches": False})
        return int(json.loads(text).get("undo", 0)) if not err else None
    except (RuntimeError, ValueError):
        return None


# Undo depth before each AI change that changed the design, newest last: the headset's Undo goes back to it.
ai_undo = []


# AI changes undone from the headset, newest last: (steps undone, the request). A new AI change clears it.
ai_redo = []


def undo_last_ai():
    if not ai_undo:
        return "Nothing to undo."
    depth, heard = ai_undo.pop()
    steps = 0
    for _ in range(100):
        d = undo_depth()
        if d is None or d <= depth:
            break
        session.call("undo", {})
        steps += 1
    ai_redo.append((steps, heard))
    for r in reversed(recent):  # the latest request that changed something
        if r["changed"] and not r["undone"]:
            r["undone"] = True
            break
    export_scm()
    return f"Undid: {heard}"


def redo_last_ai():
    if not ai_redo:
        return "Nothing to redo."
    steps, heard = ai_redo.pop()
    before = undo_depth()
    for _ in range(steps):
        text, err = session.call("redo", {})
        if err:
            log(f"redo: {text[:200]}")
            break
    if before is not None:
        ai_undo.append((before, heard))
    for r in reversed(recent):
        if r["changed"] and r["undone"]:
            r["undone"] = False
            break
    export_scm()
    return f"Redid: {heard}"


# --- what the AI has cost, and requests to repeat ---

SCV_DIR = os.environ.get("SCVIEW_DIR") or os.path.expanduser("~/.scview")  # spend and saved requests (SCVIEW_DIR for tests)
ACCESS_CODE = None  # the code the headset must send (X-Token); None = no check (--no-auth)


def load_access_code():
    """The access code kept in SCV_DIR/token (made on first use): anything on the network could otherwise
    change the design and spend the API credit."""
    path = os.path.join(SCV_DIR, "token")
    try:
        with open(path) as f:
            code = f.read().strip()
        if code:
            return code
    except OSError:
        pass
    code = secrets.token_hex(4)
    os.makedirs(SCV_DIR, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(code + "\n")
    return code
# Haiku 5.5 prices per million tokens (prompts up to 100K): input, output; cache reads and writes are assumed at the
# usual 10% and 125% of the input price. Bigger prompts cost five times as much.
PRICE_IN, PRICE_OUT = 0.10, 0.50
spend = {"balance": None, "spent": 0.0}
spend_lock = threading.Lock()


def load_spend():
    try:
        with open(os.path.join(SCV_DIR, "spend.json")) as f:
            spend.update(json.load(f))
    except (OSError, ValueError):
        pass


def save_spend():
    os.makedirs(SCV_DIR, exist_ok=True)
    with open(os.path.join(SCV_DIR, "spend.json"), "w") as f:
        json.dump(spend, f)


def add_spend(usage):
    """Add one API reply's cost (an estimate from its token counts) to the running total; returns that cost in dollars."""
    if AI["backend"] != "anthropic":
        return 0.0  # a local model costs nothing per request
    plain, cread, cwrite = (usage.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    big = 5.0 if plain + cread + cwrite > 100_000 else 1.0
    cost = big * ((plain + 0.1 * cread + 1.25 * cwrite) * PRICE_IN + (usage.get("output_tokens") or 0) * PRICE_OUT) / 1e6
    with spend_lock:
        spend["spent"] = spend.get("spent", 0.0) + cost
        try:
            save_spend()
        except OSError:
            pass
    return cost


def balance_text():
    """The estimated balance left, for the headset ("-" when no balance was given)."""
    with spend_lock:
        return "-" if spend.get("balance") is None else f"{spend['balance'] - spend['spent']:.3f}"


def norm(text):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


last_ai = {"st": None}  # the conversation state of the latest AI run
learned = {}            # normalised request words -> {"text", "steps"}: what the AI did, kept until the server stops
saved = []              # requests the wearer saved: {"id", "text", "steps" (or None)}, kept in ~/.scview/requests.json
saved_lock = threading.Lock()


def load_saved():
    try:
        with open(os.path.join(SCV_DIR, "requests.json")) as f:
            saved[:] = json.load(f).get("requests", [])
    except (OSError, ValueError):
        pass


def save_saved():
    os.makedirs(SCV_DIR, exist_ok=True)
    with open(os.path.join(SCV_DIR, "requests.json"), "w") as f:
        json.dump({"requests": saved}, f, indent=1)


def generalize(step, selected):
    """A recorded tool call with the list of selected bodies replaced by a marker, so it can be run on another selection."""
    sel = set(selected)

    def fix(x):
        if isinstance(x, dict):
            return {k: (["$SELECTED"] if k in ("bodies", "targets", "items") and sel and isinstance(v, list)
                        and all(isinstance(i, str) for i in v) and set(v) == sel else fix(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [fix(v) for v in x]
        return x
    return {"name": step["name"], "input": fix(step["input"])}


def specialize(x, selected):
    if isinstance(x, dict):
        return {k: (list(selected) if v == ["$SELECTED"] else specialize(v, selected)) for k, v in x.items()}
    if isinstance(x, list):
        return [specialize(v, selected) for v in x]
    return x


def learn(heard, st, changed):
    """Remember what the AI did for these words (not for requests about a pointed spot or faces, or that used undo)."""
    if not (changed and st and st.get("steps")) or st.get("no_learn") or st.get("pointed") or st.get("faces"):
        return
    learned[norm(heard)] = {"text": heard, "steps": [generalize(x, st["selected"]) for x in st["steps"]]}


# The last few requests of this session, so a follow-up ("each one", "those", "again") can be understood.
recent = []


def body_lists(x):
    """Every list of body names or patterns in a recorded tool call's input."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ("bodies", "targets", "items") and isinstance(v, list):
                out.append([i for i in v if isinstance(i, str)])
            elif k == "body" and isinstance(v, str):
                out.append([v])
            else:
                out += body_lists(v)
    elif isinstance(x, list):
        for v in x:
            out += body_lists(v)
    return out


def remember(heard, reply, st, changed, selected):
    """Note a request, what the AI said, and the bodies it acted on (patterns written out as names)."""
    items = [i for stp in ((st or {}).get("steps") or []) for lst in body_lists(stp["input"]) for i in lst]
    patterns = list(dict.fromkeys(i for i in items if "*" in i or i == "all"))
    names = list(dict.fromkeys(i for i in items if i not in patterns))
    if patterns:
        try:
            names += expand_all_bodies("execute", {"bodies": patterns})["bodies"]
        except Exception:
            pass
    recent.append({"text": heard, "reply": (reply or "")[:160], "changed": changed, "undone": False,
                   "names": list(dict.fromkeys(names)), "patterns": patterns, "selected": list(selected)})
    del recent[:-6]


def history_text():
    """The recent requests for the AI's prompt (empty when there are none)."""
    if not recent:
        return ""
    lines = []
    for i, r in enumerate(recent, 1):
        names = r["names"]
        shown = ", ".join(names[:12]) + (f" (and {len(names) - 12} more)" if len(names) > 12 else "")
        acted = (f" It acted on {len(names)} bodies: {shown}" + (f"; pattern {', '.join(r['patterns'])}" if r["patterns"] else "") + ".") if names else ""
        sel = f" Selected then: {', '.join(r['selected'][:6])}." if r["selected"] else ""
        lines.append(f'{i}. "{r["text"]}"{" (since undone)" if r["undone"] else ""}.{acted}{sel} You replied: {r["reply"]}')
    return ("Recent requests this session, oldest first. Words like \"those\", \"them\", \"each one\", \"the same ones\" or \"again\" "
            "mean the bodies of the latest one unless something is selected or pointed at now:\n" + "\n".join(lines)[:2500] + "\n\n")


def find_entry(text):
    key = norm(text)
    with saved_lock:
        for r in saved:
            if norm(r["text"]) == key and r.get("steps"):
                return r
    return learned.get(key)


def replay(entry, selected):
    """Run an entry's recorded steps on the current selection, without the AI. (reply, True), or None if it cannot
    be done (the steps are rolled back first)."""
    steps = entry.get("steps")
    if not steps or ("$SELECTED" in json.dumps(steps) and not selected):
        return None
    before = undo_depth()
    try:
        for stp in steps:
            text, err = session.call(stp["name"], expand_all_bodies(stp["name"], specialize(stp["input"], selected)))
            if err:
                raise RuntimeError(text[:160])
    except Exception as e:
        log(f"  repeat failed ({e}): rolled back; asking the AI instead")
        for _ in range(len(steps) + 2):
            d = undo_depth()
            if before is None or d is None or d <= before:
                break
            session.call("undo", {})
        return None
    log(f"  repeated without the AI: {entry['text']!r} ({len(steps)} steps)")
    return f"Repeated without the AI: {entry['text']}", True


topology_cache = {}  # (model version, body) -> body_topology faces


def resolve_face(body_name, p, n):
    """The SolidCraft face of `body_name` that the headset's surface point p (with surface normal n, either
    way round) lies on: flat faces by distance to their plane, cylinders by distance to the surface, other
    faces by distance to their reference point. Returns a short description for the AI, or None."""
    key = (state["version"], body_name)
    if key not in topology_cache:
        text, err = session.call("body_topology", {"body": body_name})
        if err:
            return None
        topology_cache[key] = json.loads(text).get("faces", [])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    sub = lambda a, b: [x - y for x, y in zip(a, b)]
    best = None
    for f in topology_cache[key]:
        if f.get("plane_normal"):
            fn = f["plane_normal"]
            err = abs(dot(sub(p, f["point"]), fn)) + (0 if abs(dot(fn, n)) > 0.9 else 5)
        elif f.get("cylinder"):
            c = f["cylinder"]
            v = sub(p, c["origin"])
            along = dot(v, c["axis"])
            radial = sub(v, [along * a for a in c["axis"]])
            err = abs(dot(radial, radial) ** 0.5 - c["radius"]) + (0 if abs(dot(n, c["axis"])) < 0.3 else 5)
        else:
            d = sub(p, f.get("point") or f["centroid"])
            err = 2 + dot(d, d) ** 0.5
        # Several faces can share a plane or cylinder: prefer the one whose centre is nearer.
        d = sub(p, f["centroid"])
        err += 0.001 * dot(d, d) ** 0.5
        if best is None or err < best[0]:
            best = (err, f)
    if best is None or best[0] > 1.0:
        return None
    f = best[1]
    kind = ("flat, normal " + json.dumps([round(x, 3) for x in f["plane_normal"]]) if f.get("plane_normal") else
            f"cylindrical, radius {f['cylinder']['radius']:.3g} mm, axis {json.dumps([round(x, 3) for x in f['cylinder']['axis']])}"
            if f.get("cylinder") else "curved")
    return (f'face "{f["name"]}" (index {f["index"]}) of "{body_name}": {kind}, area {f["area"]:.3g} mm2, '
            f'a point on it {json.dumps([round(x, 3) for x in f["point"]])}')


edge_topo_cache = {}  # (model version, body) -> body_topology edges


def resolve_edge(body_name, start, mid, end, length):
    """The SolidCraft edge of `body_name` that the headset picked (its start, middle and end points and length,
    mm), as a short description for the AI with its index, or None."""
    key = (state["version"], body_name)
    if key not in edge_topo_cache:
        text, err = session.call("body_topology", {"body": body_name})
        if err:
            return None
        edge_topo_cache[key] = json.loads(text).get("edges", [])
    dist = lambda a, b: sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    closed = dist(start, end) < 0.02 * max(length, 1e-9)
    centre = [(a + b) / 2 for a, b in zip(start, mid)]
    best = None
    for e in edge_topo_cache[key]:
        if e.get("start") is None or e.get("end") is None or e.get("mid") is None:
            continue
        e_closed = dist(e["start"], e["end"]) < 0.02 * max(e.get("length", 1e-9), 1e-9)
        if e_closed != closed:
            continue
        if closed:
            err = dist(centre, [(a + b) / 2 for a, b in zip(e["start"], e["mid"])])
        else:
            err = min(dist(start, e["start"]) + dist(end, e["end"]), dist(start, e["end"]) + dist(end, e["start"])) + dist(mid, e["mid"])
        err += abs(length - e.get("length", length))
        if best is None or err < best[0]:
            best = (err, e)
    if best is None or best[0] > 0.1 * length + 1.0:
        return None
    e = best[1]
    r = lambda v: [round(x, 3) for x in v]
    return (f'edge index {e["index"]} of "{body_name}" ("{e.get("name", "")}"), length {e.get("length", length):.4g} mm, '
            f'from {r(e["start"])} to {r(e["end"])}')


# A request that grew past the confirm size and waits for the wearer's OK: its conversation so far.
pending = {"st": None}


# --- direct edits: parameters and feature dimensions set without the AI ---
# Every way in (a spoken or typed request, the panel's /dimension, later a slider or a grabbed handle) ends at
# set_dimension(), so they all behave the same, share one undo, and cost nothing.

def key_of(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def dimensions():
    """Everything that can be set directly: user parameters (by name) and feature inputs ("Hole1 Diameter"),
    as {"id", "label", "value", "unit", "expression", "kind", "feature", "input"}."""
    text, err = session.call("execute", {"command": "parameters.list", "params": {"filter": "all"}})
    if err:
        raise RuntimeError(text[:200])
    out = []
    for q in json.loads(text).get("parameters", []):
        feature, inp = q.get("feature"), q.get("input")
        label = q["name"] if q.get("source") == "user" or not (feature and inp) else f"{feature} {inp}"
        out.append({"id": q["name"], "label": label, "value": q.get("display_value", q.get("value")), "unit": q.get("unit") or "",
                    "expression": str(q.get("expression", "")), "kind": q.get("source", ""), "feature": feature, "input": inp})
    return out


FILLER = {"the", "of", "a", "an", "s", "my", "its", "value", "parameter", "dimension"}


def find_dimension(words, dims):
    """The one dimension the words name, else None (two candidates is None: the AI can ask or work it out).
    Tried in order: the exact name or label; the only input called that ("diameter"); the only label that has
    every word ("hole diameter" for "Hole1 Diameter", "hole" when Hole1 has a single input)."""
    k = key_of(words)
    if not k:
        return None
    for test in (lambda d: k in (key_of(d["id"]), key_of(d["label"])),
                 lambda d: d["input"] and k == key_of(d["input"])):
        hits = [d for d in dims if test(d)]
        if len(hits) == 1:
            return hits[0]
        if hits:
            return None
    toks = [t for t in re.sub(r"[^a-z0-9_ ]+", " ", words.lower()).split() if t not in FILLER]
    hits = [d for d in dims if toks and all(key_of(t) in key_of(d["label"]) for t in toks)]
    return hits[0] if len(hits) == 1 else None


NUM = r"(?P<val>-?\d+(?:\.\d+)?)\s*(?P<unit>mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|in|inch(?:es)?|deg|degrees?)?"
DIRECT_SET = re.compile(r"^(?:(?:set|change|make|put)\s+)?(?P<name>.+?)\s*(?:\bto\b|=|\bequals\b|\bequal to\b|\bat\b)\s*" + NUM + "$")
DIRECT_BARE = re.compile(r"^(?:(?:set|change|make|put)\s+)?(?P<name>.+?)\s+" + NUM + "$")
DIRECT_REL = re.compile(r"^(?P<verb>increase|raise|grow|decrease|lower|reduce|shrink)\s+(?P<name>.+?)\s+by\s+" + NUM + "$")
UNITS = {"mm": "", "millimeter": "", "millimetre": "", "cm": " cm", "centimeter": " cm", "centimetre": " cm", "in": " in",
         "inch": " in", "inche": " in", "deg": " deg", "degree": " deg"}


def set_dimension(dim, value, why=""):
    """Set one dimension (a number in its own unit, or an expression). (reply, changed). A value that breaks
    a feature is taken back."""
    before = undo_depth()
    text, err = session.call("set_parameter", {"name": dim["id"], "value": value})
    log(f"direct: {dim['label']} ({dim['id']}) = {value!r}{why} -> {'ERROR ' if err else ''}{text[:160]}")
    if err:
        return f"SolidCraft refused {dim['label']} = {value}: {text[:120]}", False
    r = json.loads(text)
    if r.get("errors"):
        session.call("undo", {})
        shown = f"{value:g}" if isinstance(value, float) else value
        return f"{dim['label']} = {shown} breaks {len(r['errors'])} feature(s), so I put it back.", False
    ai_redo.clear()
    if before is not None:
        ai_undo.append((before, f"{dim['label']} = {value:g}" if isinstance(value, float) else f"{dim['label']} = {value}"))
    unit = f" {dim['unit']}" if dim["unit"] else ""
    return f"{dim['label']} is now {r.get('display_value', value):g}{unit} (was {dim['value']:g}).", True


def direct_request(text):
    """A request that only sets one dimension ("set thickness to 12", "plate_w = 80", "hole diameter 8 mm",
    "increase thickness by 2") is done here, with no AI: (reply, changed). Anything else returns None."""
    t = re.sub(r"\s+", " ", text.lower().strip(" .!?,"))
    t = re.sub(r"^(?:please|can you|could you)\s+", "", t)
    m = DIRECT_REL.match(t) or DIRECT_SET.match(t) or DIRECT_BARE.match(t)
    if not m:
        return None
    try:
        dim = find_dimension(re.sub(r"^the\s+", "", m["name"]), dimensions())
    except (RuntimeError, ValueError):
        return None
    if dim is None:
        return None
    val, unit = float(m["val"]), UNITS.get((m["unit"] or "mm").rstrip("s"), "")
    if "verb" in m.groupdict() and m["verb"]:
        if unit or not isinstance(dim["value"], (int, float)):
            return None  # a change by so much in another unit: leave it to the AI
        val = dim["value"] + (val if m["verb"] in ("increase", "raise", "grow") else -val)
    value = f"{val:g}{unit}" if unit else val
    return set_dimension(dim, value, " (spoken)")


def dimensions_text():
    """For the headset, one dimension per line: id, label, value, unit, kind (user/feature/sketch), expression."""
    if session is None:
        return b""
    try:
        return "".join(f"{d['id']}\t{d['label']}\t{d['value']}\t{d['unit']}\t{d['kind']}\t{d['expression']}\n" for d in dimensions()).encode()
    except (RuntimeError, ValueError):
        return b""


def est_tokens(obj):
    """Rough size of a request in tokens (about 4 characters each)."""
    return len(json.dumps(obj)) // 4


def ask(request_text, selected=(), pointed=None, faces=(), allow_confirm=True, edges=()):
    """Run one request through Claude with the session's tools; returns (reply, changed). When the context grows
    past --context-confirm the request waits (pending["st"] is set) for /ask/confirm; past --context-max it stops."""
    key = api_key() if AI["backend"] == "anthropic" else "local"
    if not key:
        return ("No API key: run  security add-generic-password -a \"$USER\" -s scview-anthropic -w  on a Mac, or "
                "secret-tool store --label scview service scview-anthropic  on Linux."), False
    tools = [{"name": n, "description": session.tools[n].get("description", "")[:1500],
              "input_schema": session.tools[n].get("inputSchema", {"type": "object"})} for n in AI_TOOLS
             if n in session.tools]
    # A lookup the server answers itself: the SolidCraft tools cut long body lists short.
    tools.append({"name": "find_bodies", "description": "Exact names of the bodies whose names match a pattern (* matches any "
                  "characters, case does not matter), e.g. flat_sail*. Use it when you need the names, for example to treat each "
                  "matching body differently; the other tools truncate long body lists.",
                  "input_schema": {"type": "object", "properties": {
                      "pattern": {"type": "string", "description": "e.g. flat_sail* or *porthole*"},
                      "limit": {"type": "integer", "description": "most names to return (default 200)"}},
                      "required": ["pattern"]}})
    tools.append({"name": "colour_each", "description": "Give each of many bodies its own different colour in one call (use this "
                  "for 'each one a different colour', never one command per body). bodies: names or patterns such as flat_sail*; "
                  "colors: optional #rrggbb list to cycle through (default: a spread of distinct mid-tone colours).",
                  "input_schema": {"type": "object", "properties": {
                      "bodies": {"type": "array", "items": {"type": "string"}},
                      "colors": {"type": "array", "items": {"type": "string"}}},
                      "required": ["bodies"]}})
    # Prompt caching: the system prompt and tools are the same every time.
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    system = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
    sel = f"Selected bodies: {json.dumps(list(selected))}" if selected else "Nothing is selected."
    if faces:
        sel += "\nSelected faces (picked in the headset; \"this face\", \"these faces\", \"the selected faces\"):\n" + \
            "\n".join(f"- {d}" for d in faces)
    if edges:
        sel += ("\nSelected edges (picked in the headset; \"this edge\", \"these edges\", \"the selected edges\"). "
                "Use these indices in solid.fillet / solid.chamfer \"edges\" ([{\"body\": ..., \"index\": i}]):\n" +
                "\n".join(f"- {d}" for d in edges))
    if pointed:
        body_name, (x, y, z), pointed_face = pointed
        sel += (f"\nThe wearer is pointing at the body \"{body_name}\", at the point [{x:.2f}, {y:.2f}, {z:.2f}] mm on "
                f"its surface: \"here\", \"this point\" and \"there\" mean that point, and \"this\", \"this part\" "
                f"and \"it\" mean that body. Change only the bodies meant (pass them as \"targets\" where a command takes them).")
        if pointed_face:
            sel += f" The point is on {pointed_face}; \"this face\" means that face."
    focus = list(selected) + ([pointed[0]] if pointed else [])
    messages = [{"role": "user", "content": f"Request: {request_text}\n{sel}\n\n{history_text()}The design now:\n{design_summary(focus)}"}]
    st = {"key": key, "system": system, "tools": tools, "messages": messages, "step": 0, "changed": False,
          "confirmed": not allow_confirm, "allow_confirm": allow_confirm,
          "selected": list(selected), "pointed": bool(pointed), "faces": bool(faces) or bool(edges), "steps": [], "cost": 0.0,
          "heard": request_text}
    return run_ai(st)


def finish_ai(heard, before, changed):
    """After an AI run that changed the design: keep its undo step and publish the new model."""
    if changed:
        ai_redo.clear()
        if before is not None:
            ai_undo.append((before, heard))
        export_scm()


def confirm_k():
    """The size in thousands of tokens of the request waiting for an OK, else None."""
    st = pending["st"]
    return str(st["size"] // 1000) if st else None


# Spoken answers to a big request that is waiting (after the punctuation is dropped).
# A request that ends with one of these is dropped, not sent to the AI ("make it red, cancel").
CANCEL_END = re.compile(r"(^|\W)(cancel( that)?|scratch that|never ?mind|ignore that|forget (it|that)|disregard( that)?)\W*$", re.I)
GO_WORDS = {"go on", "go ahead", "continue", "carry on", "proceed", "yes", "yes please", "ok", "okay", "do it", "send it"}
# A request that is only "undo" or "redo" does what the panel's Undo and Redo buttons do (the whole last change),
# not whatever one step the AI would pick.
UNDO_WORDS = {"undo", "undo that", "undo it", "undo this", "undo the last", "undo last", "undo the last change",
              "undo the last request", "undo the last one", "undo last change", "undo last request", "undo the change",
              "undo my last request", "undo the last thing", "go back", "step back"}
REDO_WORDS = {"redo", "redo that", "redo it", "redo this", "redo the last", "redo last", "redo the last change",
              "redo the last request", "redo the last one", "redo last change", "redo last request", "redo the change"}
STOP_WORDS = {"stop", "cancel", "no", "no thanks", "never mind", "nevermind", "forget it", "abort"}


def resume_pending():
    """Carry on with the big request that was waiting for an OK: (reply, changed, ok)."""
    st, pending["st"] = pending["st"], None
    if st is None:
        return "Nothing is waiting to go on.", False, False
    st["confirmed"] = True
    before = undo_depth()
    reply, changed = run_ai(st)
    remember(st.get("heard", ""), reply, st, changed, st.get("selected", []))
    finish_ai("(continued) " + st["messages"][0]["content"].split("\n")[0][:80], before, changed)
    return reply, changed, True


def run_ai(st):
    """The AI loop of ask(); resumes from st["step"] after the wearer confirms a big request."""
    key, system, tools, messages = st["key"], st["system"], st["tools"], st["messages"]
    changed = st["changed"]
    last_ai["st"] = st
    for step in range(st["step"], 12):
        body = {"model": CLAUDE_MODEL, "max_tokens": 4000, "system": system, "tools": tools, "messages": messages}
        if step == 11:
            # Out of steps: no more tools, just say what was done and what was not.
            body["tool_choice"] = {"type": "none"}
            messages[-1]["content"].append({"type": "text", "text": "No more steps: in one or two sentences, say what "
                                            "you changed and what you could not do."})
        size = est_tokens(body)
        if size > args.context_max:
            log(f"  context ~{size} tokens is over the limit {args.context_max}: stopped")
            return (f"Stopped: this request grew to about {size // 1000}k tokens (limit {args.context_max // 1000}k). "
                    f"Select fewer parts, or ask for something narrower."), changed
        # Only ask before the request has changed anything: after a change, Stop could not take it back.
        if size > args.context_confirm and not st["confirmed"] and not changed:
            st.update(step=step, changed=changed, size=size)
            pending["st"] = st
            log(f"  context ~{size} tokens is over {args.context_confirm}: waiting for the wearer's OK")
            return f"Big request: about {size // 1000}k tokens.", changed
        reply = ai_step(key, body)
        u = reply.get("usage", {})
        st["cost"] = st.get("cost", 0.0) + add_spend(u)
        log(f"  step {step}: context ~{size // 1000}k tokens (API: input {u.get('input_tokens')}, cached read "
            f"{u.get('cache_read_input_tokens')}, written {u.get('cache_creation_input_tokens')}, output {u.get('output_tokens')})")
        if AI["backend"] != "anthropic" and AI["ctx"] and (u.get("input_tokens") or 0) > 0.9 * AI["ctx"]:
            log(f"  WARNING: the prompt ({u.get('input_tokens')} tokens) is close to the model's context window "
                f"({AI['ctx']}): the server may be cutting it. Raise the window (Ollama: OLLAMA_CONTEXT_LENGTH).")
        messages.append({"role": "assistant", "content": reply["content"]})
        uses = [c for c in reply["content"] if c.get("type") == "tool_use"]
        if reply.get("stop_reason") == "max_tokens":
            # Cut off while writing (typically hundreds of commands): the half-written calls must not run.
            log("  the AI's answer was cut off at max_tokens")
            if not uses:
                return "My answer was too long and got cut off, so nothing more was done. Try a smaller group.", changed
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"], "is_error": True,
                              "content": "Your reply was cut off before this call was complete, so it did not run. Use fewer "
                                         "commands per reply, or the colour_each tool for many bodies."} for u in uses]})
            continue
        if not uses:
            text = " ".join(c.get("text", "") for c in reply["content"] if c.get("type") == "text").strip()
            return text or ("Done." if changed else "I did not do anything."), changed
        results = []
        for u in uses:
            try:
                if u["name"] == "find_bodies":
                    text, err = find_bodies(u.get("input")), False
                elif u["name"] == "colour_each":
                    text, err = colour_each(u.get("input"))
                else:
                    text, err = session.call(u["name"], expand_all_bodies(u["name"], tidy_args(u["name"], u.get("input") or {})))
            except Exception as e:  # the session reports bad calls as errors; keep going
                text, err = f"error: {e}", True
            if not err and u["name"] in ("execute", "batch", "set_parameter", "undo", "colour_each"):
                changed = True
                if u["name"] == "undo":
                    st["no_learn"] = True
                else:
                    st["steps"].append({"name": u["name"], "input": json.loads(json.dumps(u.get("input") or {}))})
            log(f"  tool {u['name']} {json.dumps(u.get('input'))[:160]} -> {'ERROR ' if err else ''}{text[:120]}")
            cap = 12000 if u["name"] == "find_bodies" else 6000
            if len(text) > cap:
                text = text[:cap] + f"\n... (cut: {len(text) - cap} more characters; ask for less, e.g. one body)"
            results.append({"type": "tool_result", "tool_use_id": u["id"], "content": text, "is_error": err})
        messages.append({"role": "user", "content": results})
    return "Stopped after too many steps.", changed


# Words the wearer is likely to say: whisper hears them better with them as its starting text.
WHISPER_HINT = ("CAD requests. Make these red. Make this blue. Colours: red, green, blue, yellow, orange, purple, white, "
                "black. Hide, isolate, move the whole ship one millimetre, rotate, fillet, chamfer, hole, extrude, "
                "push, pull, body, face, edge, sail, hull. Go on. Stop. Cancel.")

transcribe_lock = threading.Lock()


def transcribe(wav, name="ask.wav"):
    path = os.path.join(work, name)
    with open(path, "wb") as f:
        f.write(wav)
    r = subprocess.run(["whisper-cli", "-m", args.whisper_model, "-f", path, "-nt", "-np", "-l", "en", "--prompt", WHISPER_HINT],
                       capture_output=True, text=True, timeout=120)
    return " ".join(r.stdout.split()).strip()


# --- HTTP and beacon ---

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def authorised(self):
        """True when the request carries the access code; else a 403 is sent (a JSON answer for the headset to show)."""
        if ACCESS_CODE is None or hmac.compare_digest(self.headers.get("X-Token", "").encode(), ACCESS_CODE.encode()):
            return True
        out = json.dumps({"reply": "Access code needed: type the code from the Mac and press Enter.", "ok": False}).encode()
        self.send_response(403)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)
        return False

    def do_GET(self):
        if not self.authorised():
            return
        if self.path in ("/comments.txt", "/models.txt", "/requests.txt", "/dimensions.txt"):
            out = {"/comments.txt": notes_text, "/models.txt": models_text, "/requests.txt": requests_text,
                   "/dimensions.txt": dimensions_text}[self.path]()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        with lock:
            data, name = state["data"], state["name"]
        if self.path != "/model.scm" or data is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        log(f"sent {name} ({len(data) / 1e6:.1f} MB) to {self.client_address[0]}")

    def reply_json(self, obj):
        out = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        if not self.authorised():
            return
        if self.path not in ("/ask", "/transcribe", "/undo", "/redo", "/pushpull", "/comment", "/comment/delete", "/comment/restore", "/comment/do",
                             "/comment/edit", "/model/open", "/ask/confirm", "/requests/save", "/requests/delete", "/save", "/capture", "/dimension"):
            self.send_error(404)
            return
        if self.path == "/model/open":
            # Only a file from the models list can be opened.
            path = urllib.parse.unquote(self.headers.get("X-Id", ""))
            if path not in model_files():
                self.reply_json({"reply": "That model is no longer in the list.", "ok": False})
                return
            with lock:
                state["wanted"] = path
                state["reload"] = True  # also when it is the model already open: start again from the file
            mb = os.path.getsize(path) / 1e6
            log(f"headset asked for {path}")
            self.reply_json({"reply": f"Opening {os.path.basename(path)} ({mb:.0f} MB file): "
                                      f"{'this can take a few minutes' if mb > 5 else 'a few seconds'}...", "ok": True})
            return
        if self.path == "/comment/edit":
            # Move a note (X-Pin) and/or set its text (a text/plain body); it then counts as not done.
            note_id = urllib.parse.unquote(self.headers.get("X-Id", ""))
            pin = urllib.parse.unquote(self.headers.get("X-Pin", ""))
            text = self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 100_000)).decode("utf-8", "replace").strip()
            reply, ok = "Note not found.", False
            try:
                with lock:
                    note = next((n for n in state["notes"] if n["id"] == note_id), None)
                    if note is not None:
                        if pin:
                            body_name, x, y, z = pin.split("\t")
                            note["body"], note["point"] = body_name, [float(x), float(y), float(z)]
                        if text:
                            note["text"] = text
                        note["done"] = False
                        save_notes()
                        reply, ok = (f"Note moved to {note['body']}." if pin else f"Note changed: {note['text'][:60]}"), True
            except ValueError as e:
                reply = f"Bad note edit: {e}"
            log(f"edit {note_id}: {reply}")
            self.reply_json({"heard": "", "reply": reply, "ok": ok, "changed": False})
            return
        if self.path == "/comment/do":
            try:
                if session is None:
                    raise RuntimeError("AI changes need a STEP or .sc source (this one is view only).")
                reply, changes = do_notes(urllib.parse.unquote(self.headers.get("X-Id", "")))
            except Exception as e:
                reply, changes = str(e), 0
            self.reply_json({"heard": "", "reply": reply, "ok": True, "changed": changes > 0, "changes": changes})
            return
        if self.path in ("/comment/delete", "/comment/restore"):
            note_id = urllib.parse.unquote(self.headers.get("X-Id", ""))
            deleted = self.path.endswith("delete")
            with lock:
                note = next((n for n in state["notes"] if n["id"] == note_id), None)
                if note is not None:
                    note["deleted"] = deleted
                    save_notes()
            reply = ("Note not found." if note is None else
                     f"Note {'deleted' if deleted else 'restored'}: {note['text'][:60]}")
            log(f"{self.path}: {reply}")
            self.reply_json({"heard": "", "reply": reply, "ok": note is not None, "changed": False})
            return
        if self.path == "/pushpull":
            # Push/pull from the headset: offset one face by a distance (mm, positive = outward), no AI.
            # X-Face: "body\tx\ty\tz" (a point on the face); X-Distance: mm.
            reply, changed = "", False
            try:
                if session is None:
                    raise RuntimeError("Push/pull needs a STEP or .sc source (this one is view only).")
                f = urllib.parse.unquote(self.headers.get("X-Face", "")).split("\t")
                point = [float(v) for v in f[1:4]]
                dist = float(self.headers.get("X-Distance", "0"))
                if not dist:
                    raise RuntimeError("No distance.")
                before = undo_depth()
                # Press Pull moves a flat face (adds material outward, cuts inward); Offset Face as a fallback.
                text, err = session.call("execute", {"command": "solid.press_pull",
                                                     "params": {"face": point, "distance": dist, "body": f[0]}})
                if err:
                    log(f"press_pull: {text[:200]}; trying offset_face")
                    text, err = session.call("execute", {"command": "solid.offset_face",
                                                         "params": {"faces": [point], "distance": dist, "body": f[0]}})
                log(f"push/pull {f[0]} {point} {dist:+g} mm -> {'ERROR ' if err else ''}{text[:200]}")
                if err:
                    # SolidCraft's reasons, shortened for the panel (curved faces are often refused).
                    why = text.split("refused: ")[-1].split(";")[0].strip() if "refused" in text else text[:120]
                    raise RuntimeError(f"SolidCraft can't move that face ({why}). Flat faces work best.")
                ai_redo.clear()
                if before is not None:
                    ai_undo.append((before, f"push/pull {dist:+g} mm"))
                export_scm()
                reply, changed = f"Moved the face {dist:+g} mm.", True
            except (ValueError, IndexError) as e:
                reply = f"Bad push/pull request: {e}"
            except RuntimeError as e:
                reply = str(e)
            self.reply_json({"heard": "", "reply": reply, "ok": changed, "changed": changed})
            return
        if self.path == "/dimension":
            # Set one parameter or feature dimension from the headset (a slider, a handle, a typed number), no AI.
            # X-Name: its id or label from /dimensions.txt (percent-encoded); X-Value: a number or an expression.
            reply, changed = "", False
            try:
                if session is None:
                    raise RuntimeError("Changing dimensions needs a STEP or .sc source (this one is view only).")
                name = urllib.parse.unquote(self.headers.get("X-Name", ""))
                raw = urllib.parse.unquote(self.headers.get("X-Value", "")).strip()
                dim = find_dimension(name, dimensions())
                if dim is None:
                    raise RuntimeError(f"No single dimension is called \"{name}\".")
                if not raw:
                    raise RuntimeError("No value.")
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
                reply, changed = set_dimension(dim, value, " (panel)")
                if changed:
                    export_scm()
            except (RuntimeError, ValueError) as e:
                reply = str(e)
            self.reply_json({"heard": "", "reply": reply, "ok": changed, "changed": changed})
            return
        if self.path in ("/undo", "/redo"):
            try:
                if session is None:
                    reply = "Nothing to undo."
                else:
                    reply = undo_last_ai() if self.path == "/undo" else redo_last_ai()
            except Exception as e:
                reply = str(e)
            log(f"{self.path[1:]}: {reply}")
            self.reply_json({"heard": "", "reply": reply, "ok": True, "changed": False})
            return
        body = self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 20_000_000))
        if self.path == "/ask/confirm":
            # The wearer said OK to a big request that was waiting: carry on from where it stopped.
            reply, changed, ok = "", False, False
            try:
                reply, changed, ok = resume_pending()
            except Exception as e:
                reply = str(e)
            log(f"reply after OK: {reply}")
            self.reply_json({"heard": "", "reply": reply, "ok": ok, "changed": changed})
            return
        if self.path == "/capture":
            try:
                reply = do_capture()
            except Exception as e:
                reply = str(e)
            log(f"capture: {reply}")
            self.reply_json({"heard": "", "reply": reply, "ok": True, "changed": False})
            return
        if self.path == "/save":
            try:
                reply = do_save()
            except Exception as e:
                reply = str(e)
            log(f"save: {reply}")
            self.reply_json({"heard": "", "reply": reply, "ok": True, "changed": False})
            return
        if self.path == "/requests/save":
            text = body.decode("utf-8", "replace").strip()
            if not text:
                reply = "Nothing to save yet: make a request first."
            else:
                entry = learned.get(norm(text))
                with saved_lock:
                    old = next((r for r in saved if norm(r["text"]) == norm(text)), None)
                    if old is None:
                        saved.append({"id": uuid.uuid4().hex[:8], "text": text, "steps": entry["steps"] if entry else None})
                    elif entry:
                        old["steps"] = entry["steps"]
                    save_saved()
                reply = f"Saved: {text}" + ("" if entry else " (words only: the AI does it each time)")
            log(f"requests/save: {reply}")
            self.reply_json({"heard": "", "reply": reply, "ok": True, "changed": False})
            return
        if self.path == "/requests/delete":
            rid = urllib.parse.unquote(self.headers.get("X-Id", ""))
            with saved_lock:
                saved[:] = [r for r in saved if r["id"] != rid]
                save_saved()
            self.reply_json({"heard": "", "reply": "Removed.", "ok": True, "changed": False})
            return
        if self.path == "/transcribe":
            # Live text while the wearer is still speaking: the audio so far. One at a time; a request
            # that arrives while one runs gets an empty answer and the headset sends again later.
            text = ""
            if transcribe_lock.acquire(blocking=False):
                try:
                    text = transcribe(body, "partial.wav")
                finally:
                    transcribe_lock.release()
            self.reply_json({"text": text})
            return
        if self.path == "/comment":
            heard, reply, ok, note_id = "", "", False, None
            try:
                if not state["source"]:
                    raise RuntimeError("No model file to keep notes with.")
                if self.headers.get("Content-Type", "").startswith("text/"):
                    heard = body.decode("utf-8", "replace").strip()  # typed on the headset
                else:
                    heard = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", transcribe(body, "note.wav")).strip()
                if heard.lower().strip(" .!?,") in ("", "you", "thank you", "thanks for watching", "bye"):
                    raise RuntimeError("I didn't catch the note: hold the left trigger while you speak.")
                redo_id = urllib.parse.unquote(self.headers.get("X-Id", ""))
                if redo_id:
                    # Re-recorded: new words for an existing note; the old ones go back to the headset for Undo.
                    with lock:
                        note = next((n for n in state["notes"] if n["id"] == redo_id), None)
                        if note is None:
                            raise RuntimeError("That note is gone.")
                        old_text = note["text"]
                        note["text"], note["done"] = heard, False
                        save_notes()
                    note_id = redo_id
                    reply = f"Note re-recorded on {note['body']}."
                    self.reply_json({"heard": heard, "reply": reply, "ok": True, "changed": False, "id": note_id,
                                     "old": old_text})
                    log(f"note re-recorded: {heard!r}")
                    return
                body_name, x, y, z = urllib.parse.unquote(self.headers.get("X-Pin", "")).split("\t")
                point = [float(x), float(y), float(z)]
                note_id = uuid.uuid4().hex[:8]
                with lock:
                    state["notes"].append({"id": note_id, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "body": body_name,
                                           "point": point, "text": heard, "author": "Quest"})
                    save_notes()
                reply = f"Note saved on {body_name}."
                ok = True
            except Exception as e:
                reply = str(e)
            log(f"note: {heard!r} -> {reply}")
            self.reply_json({"heard": heard, "reply": reply, "ok": ok, "changed": False, "id": note_id or ""})
            return
        t = time.time()
        heard, reply, ok, changed = "", "", False, False
        # The headset sends the selected bodies' names as percent-encoded JSON.
        try:
            selected = json.loads(urllib.parse.unquote(self.headers.get("X-Selected", "") or "[]"))
            selected = [str(s) for s in selected][:200] if isinstance(selected, list) else []
        except ValueError:
            selected = []
        # Where the right pointer was when the wearer started speaking: "body\tx\ty\tz" (optional).
        # Optionally followed by the surface normal there: "body\tx\ty\tz\tnx\tny\tnz".
        pointed = None
        try:
            if self.headers.get("X-Point"):
                f = urllib.parse.unquote(self.headers["X-Point"]).split("\t")
                pt = tuple(float(v) for v in f[1:4])
                face = resolve_face(f[0], pt, [float(v) for v in f[4:7]]) if len(f) >= 7 and session else None
                pointed = (f[0], pt, face)
        except (ValueError, RuntimeError):
            pointed = None
        # Selected faces: one per line, "body\tx\ty\tz\tnx\tny\tnz" (a point on the face and the normal there).
        faces = []
        try:
            for line in urllib.parse.unquote(self.headers.get("X-Faces", "")).split("\n")[:50]:
                f = line.split("\t")
                if len(f) == 7 and session:
                    d = resolve_face(f[0], [float(v) for v in f[1:4]], [float(v) for v in f[4:7]])
                    faces.append(d or f"a face of \"{f[0]}\" through the point {[round(float(v), 3) for v in f[1:4]]} (not matched exactly)")
        except (ValueError, RuntimeError):
            faces = []
        # Selected edges: one per line, "body\tstart xyz\tmiddle xyz\tend xyz\tlength".
        edges = []
        try:
            for line in urllib.parse.unquote(self.headers.get("X-Edges", "")).split("\n")[:50]:
                f = line.split("\t")
                if len(f) == 11 and session:
                    v = [float(x) for x in f[1:]]
                    d = resolve_edge(f[0], v[0:3], v[3:6], v[6:9], v[9])
                    edges.append(d or f"an edge of \"{f[0]}\" from {[round(x, 3) for x in v[0:3]]} to {[round(x, 3) for x in v[6:9]]} (not matched exactly)")
        except (ValueError, RuntimeError):
            edges = []
        try:
            if session is None:
                raise RuntimeError("Voice changes need a STEP or .sc source (this one is view only).")
            heard = body.decode("utf-8", "replace").strip() if self.headers.get("Content-Type", "").startswith(
                "text/") else transcribe(body)
            log(f"heard: {heard!r}, selected: {selected}" + (f", pointing at {pointed[0]} {pointed[1]} {pointed[2] or ''}" if pointed else "")
                + (f", faces: {faces}" if faces else "") + (f", edges: {edges}" if edges else ""))
            # whisper invents these for silence or a click, and marks non-speech as [BLANK_AUDIO], (music)...
            heard = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", heard).strip()
            if heard.startswith("CAD requests."):  # silence: whisper repeated its starting text
                heard = ""
            if heard.lower().strip(" .!?,") in ("", "you", "thank you", "thanks for watching", "bye"):
                raise RuntimeError("I didn't catch that: hold the left trigger while you speak.")
            spent_before = spend.get("spent", 0.0)
            word = norm(heard)
            if pending["st"] is not None and word in GO_WORDS:
                log("spoken OK to the waiting big request")
                reply, changed, ok = resume_pending()
            elif pending["st"] is not None and (word in STOP_WORDS or CANCEL_END.search(heard)):
                pending["st"] = None
                reply, ok = "Stopped: that request was not sent on.", True
                log("spoken stop to the waiting big request")
            elif word in UNDO_WORDS or word in REDO_WORDS:
                pending["st"] = None
                reply, ok = (undo_last_ai() if word in UNDO_WORDS else redo_last_ai()), True
                log(f"spoken {'undo' if word in UNDO_WORDS else 'redo'}: {reply}")
            elif CANCEL_END.search(heard):
                reply, ok = "Cancelled: nothing was sent to the AI.", True
                log(f"cancelled by voice: {heard!r}")
            else:
                before = undo_depth()
                pending["st"] = None  # a new request drops one still waiting for an OK
                replayed = None
                if self.headers.get("X-Replay") and not pointed and not faces and not edges:
                    entry = find_entry(heard)
                    replayed = replay(entry, selected) if entry else None
                # A request that only sets a dimension needs no AI. set_dimension() keeps its own undo step.
                done = None if (replayed or args.no_direct) else direct_request(heard)
                if done:
                    reply, changed = done
                    if changed:
                        export_scm()
                    remember(heard, reply, None, changed, selected)
                else:
                    if replayed:
                        reply, changed = replayed
                    else:
                        reply, changed = ask(heard, selected, pointed, faces, edges=edges)
                        learn(heard, last_ai["st"], changed)
                    if not pending["st"]:
                        remember(heard, reply, None if replayed else last_ai["st"], changed, selected)
                    finish_ai(heard, before, changed)
                ok = True
        except Exception as e:
            reply = str(e)
        log(f"reply ({time.time() - t:.1f} s): {reply}")
        out = {"heard": heard, "reply": reply, "ok": ok, "changed": changed}
        if ok:
            out["cost"] = f"{spend.get('spent', 0.0) - spent_before:.4f}"
        if confirm_k():
            out["confirm_k"] = confirm_k()
            out["confirm_cost"] = f"{pending['st']['size'] / 1e6 * PRICE_IN:.4f}"  # about, for one more step
        self.reply_json(out)

    def log_message(self, *a):
        pass


def notes_path():
    return state["source"] + ".comments.json" if state["source"] else None


def load_notes():
    """Read the model's notes file (missing or unreadable: no notes)."""
    path = notes_path()
    notes = []
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                notes = json.load(f).get("notes", [])
        except (OSError, ValueError) as e:
            log(f"notes: could not read {path}: {e}")
    with lock:
        state["notes"] = notes
        state["notes_version"] += 1
    log(f"notes: {len([n for n in notes if not n.get('deleted')])} in {path}")


def save_notes():
    """Write the notes file (atomically) and tell the headset there is a new version. Caller holds lock."""
    path = notes_path()
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump({"model": state["name"], "notes": state["notes"]}, f, indent=2)
    os.replace(tmp, path)
    state["notes_version"] += 1


def notes_text():
    """Live notes for the headset, one per line: id, body, x, y, z, done (0/1), text (tab-separated)."""
    clean = lambda v: str(v).replace("\t", " ").replace("\n", " ").replace("\r", " ")
    with lock:
        return "".join(f"{n['id']}\t{clean(n['body'])}\t{n['point'][0]}\t{n['point'][1]}\t{n['point'][2]}\t{1 if n.get('done') else 0}\t{clean(n['text'])}\n"
                       for n in state["notes"] if not n.get("deleted")).encode()


def do_notes(note_id):
    """Give one note (or "all" not yet done) to the AI. Returns (reply, number of changes)."""
    with lock:
        ids = set(note_id.split(","))
        todo = [n for n in state["notes"] if not n.get("deleted") and
                (n["id"] in ids if note_id != "all" else not n.get("done"))]
    if not todo:
        return ("No notes left for the AI to do." if note_id == "all" else "Note not found."), 0
    replies, changes = [], 0
    for n in todo:
        x, y, z = n["point"]
        request = (f"{n['text']}\n(This is a note pinned on the body \"{n['body']}\" at the point "
                   f"[{x:.2f}, {y:.2f}, {z:.2f}] mm, on its surface: \"here\", \"this\" and \"it\" mean that point "
                   f"and that body. Change only that body (pass it as \"targets\" where a command takes them). If the note is only "
                   f"a remark, not a change to make, change nothing and say so.)")
        log(f"note to AI: {n['text']!r} on {n['body']} at {n['point']}")
        before = undo_depth()
        reply, changed = ask(request, [n["body"]], allow_confirm=False)
        if changed:
            ai_redo.clear()
            if before is not None:
                ai_undo.append((before, n["text"]))
            export_scm()
            changes += 1
        log(f"note done ({'changed' if changed else 'no change'}): {reply}")
        with lock:
            n["done"] = True
            n["ai_reply"] = reply
            save_notes()
        replies.append(reply)
    return " ".join(replies) if len(replies) == 1 else f"{len(todo)} notes: " + " | ".join(replies), changes


def beacon(port, quest):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    while True:
        with lock:
            version, nv = state["version"], state["notes_version"]
        if version:
            msg = f"SCVIEW1 {port} {version} n{nv} b{balance_text()}".encode()
            for addr in ["255.255.255.255"] + ([quest] if quest else []):
                try:
                    s.sendto(msg, (addr, BEACON_PORT))
                except OSError as e:
                    log(f"beacon to {addr}: {e}")
        time.sleep(1)


def model_dirs():
    """Folders the headset's Models list shows: the source's folder and each --models folder."""
    src = args.source if os.path.isdir(args.source) else os.path.dirname(os.path.abspath(args.source))
    dirs = [src] + [os.path.expanduser(d) for d in args.models]
    return [d for i, d in enumerate(dirs) if os.path.isdir(d) and d not in dirs[:i]]


def model_files():
    """Model files in those folders (absolute paths), folder by folder, by name."""
    out = []
    for d in model_dirs():
        names = sorted(f for f in os.listdir(d) if f.lower().endswith(MODEL_EXTS) and not f.startswith("."))
        out += [os.path.join(d, f) for f in names if os.path.isfile(os.path.join(d, f))]
    return out


def do_capture():
    """Take a photo of what the headset shows with the Quest's own camera service (over adb), and copy it here
    to ~/Pictures/SC View. Needs adb and the headset reachable (USB, or adb over Wi-Fi)."""
    import shutil
    adb = shutil.which("adb") or "/opt/homebrew/bin/adb"
    folder = "/sdcard/Oculus/Screenshots"
    def run(*a, t=20):
        return subprocess.run([adb, *a], capture_output=True, text=True, timeout=t)
    try:
        before = run("shell", "ls", "-t", folder).stdout.split()
        r = run("shell", "am", "startservice", "-n", "com.oculus.metacam/.capture.CaptureService", "-a", "TAKE_SCREENSHOT",
                "-e", "exif_location", "false")
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"The Mac cannot reach the headset with adb ({e})."
    if r.returncode != 0 or "Error" in r.stdout:
        return "The Mac cannot reach the headset with adb: connect it by USB or adb over Wi-Fi."
    for _ in range(10):
        time.sleep(0.6)
        now = run("shell", "ls", "-t", folder).stdout.split()
        if now and (not before or now[0] != before[0]):
            break
    else:
        return "The photo did not appear on the headset."
    out = os.path.expanduser("~/Pictures/SC View")
    os.makedirs(out, exist_ok=True)
    r = run("pull", f"{folder}/{now[0]}", out, t=60)
    if r.returncode != 0:
        return f"Photo taken ({now[0]}) but could not be copied: {r.stderr.strip()[:100]}"
    return f"Photo saved on the Mac: Pictures/SC View/{now[0]}"


def do_save():
    """Save the open design as a new .sc file next to its source (the original is never overwritten)."""
    if session is None:
        raise RuntimeError("Saving needs a STEP or .sc source: this one is view only.")
    src = state["source"]
    stem = re.sub(r" \(Quest [^)]*\)$", "", os.path.splitext(os.path.basename(src))[0])
    name = f"{stem} (Quest {time.strftime('%Y-%m-%d %H.%M')}).sc"
    text, err = session.call("save", {"path": os.path.join(os.path.dirname(src), name)})
    if err:
        raise RuntimeError(f"Save failed: {text[:160]}")
    return f"Saved as {name}, next to the original."


def requests_text():
    """For the headset, one saved request per line: id, words, "r" if it can be repeated without the AI else "-"."""
    with saved_lock:
        return "".join(f"{r['id']}\t{r['text'].replace(chr(9), ' ').replace(chr(10), ' ')}\t{'r' if r.get('steps') else '-'}\n"
                       for r in saved).encode()


def models_text():
    """For the headset, one model per line: path (its id), name, size in MB, 1 if open (tab-separated)."""
    with lock:
        current = state["source"]
    return "".join(f"{p}\t{os.path.basename(p)}\t{os.path.getsize(p) / 1e6:.1f}\t{1 if p == current else 0}\n"
                   for p in model_files()).encode()


def pick_source(path):
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(MODEL_EXTS)]
        files = [f for f in files if os.path.isfile(f)]
        return max(files, key=os.path.getmtime) if files else None
    return path


def load(src):
    """Open or convert the source and publish it."""
    global session
    ext = os.path.splitext(src)[1].lower()
    # A different model: the AI's undo and redo steps belonged to the last one.
    if state["source"] != os.path.abspath(src):
        ai_undo.clear()
        ai_redo.clear()
        recent.clear()
    if ext in (".step", ".stp", ".sc") and not args.no_ai:
        if session is None:
            session = Mcp(args.cli)
        t = time.time()
        text, err = session.call("open", {"path": src})
        if err:
            raise RuntimeError(text[:300])
        log(f"opened in the SolidCraft session in {time.time() - t:.1f} s")
        data = export_scm()
    else:
        session = None
        if ext == ".scm":
            with open(src, "rb") as f:
                data = f.read()
        else:
            obj = src
            if ext in (".step", ".stp", ".sc"):
                obj = os.path.join(work, "model.obj")
                subprocess.run([args.cli, "run", "--in", src, "--out", obj], check=True, stdout=subprocess.DEVNULL)
            out = os.path.join(work, "model.scm")
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "obj_to_scm.py"), obj, out], check=True,
                           stdout=subprocess.DEVNULL)
            with open(out, "rb") as f:
                data = f.read()
        with lock:
            state["counter"] += 1
            state["data"] = data
            state["version"] = f"{int(time.time()):x}-{state['counter']}"
    with lock:
        state["name"] = os.path.basename(src)
        state["source"] = os.path.abspath(src)
    load_notes()
    log(f"ready: {len(data) / 1e6:.1f} MB; the headset will fetch it")


def main():
    global args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="model file (.step/.stp/.sc/.obj/.scm) or a folder (newest model in it)")
    ap.add_argument("--port", type=int, default=47801)
    ap.add_argument("--quest", help="headset IP address (optional; also sends the beacon straight to it)")
    ap.add_argument("--cli", default=DEFAULT_CLI, help="path to solidcraft-cli")
    ap.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    ap.add_argument("--no-auth", action="store_true", help="do not ask the headset for the access code (tests only)")
    ap.add_argument("--no-ai", action="store_true", help="view only: no SolidCraft session, no /ask")
    ap.add_argument("--context-confirm", type=int, default=30000,
                    help="an AI request bigger than this many tokens waits for the wearer's OK (default 30000)")
    ap.add_argument("--context-max", type=int, default=120000,
                    help="an AI request is stopped when it grows past this many tokens (default 120000)")
    ap.add_argument("--balance", type=float,
                    help="your API credit in dollars: the headset then shows what is left, estimated from what this server spends")
    ap.add_argument("--no-direct", action="store_true",
                    help="send every request to the AI, even one that only sets a parameter or dimension")
    ap.add_argument("--ai", choices=["anthropic", "ollama"], default=os.environ.get("SCVIEW_AI") or "anthropic",
                    help="who answers voice requests: the Claude API (default), or a local model behind an "
                         "OpenAI-compatible endpoint (Ollama, llama.cpp, LM Studio)")
    ap.add_argument("--ai-url", default=os.environ.get("SCVIEW_AI_URL") or AI["url"],
                    help="the local endpoint (default http://localhost:11434/v1)")
    ap.add_argument("--ai-model", default=os.environ.get("SCVIEW_AI_MODEL") or AI["model"],
                    help="the local model, which must support tool calling (default qwen3.8:27b)")
    ap.add_argument("--ai-think", default=os.environ.get("SCVIEW_AI_THINK") or AI["think"],
                    help="reasoning_effort sent to the local model: none (fastest, default), low, medium, high; "
                         "empty sends nothing")
    ap.add_argument("--ai-ctx", type=int, default=0,
                    help="the local model's context window in tokens, to be warned when a request nears it")
    ap.add_argument("--models", action="append",
                    default=[],
                    help="another folder for the headset's Models list (repeatable)")
    args = ap.parse_args()
    AI.update(backend=args.ai, url=args.ai_url, model=args.ai_model, think=args.ai_think, ctx=args.ai_ctx)
    global ACCESS_CODE
    if not args.no_auth:
        ACCESS_CODE = load_access_code()

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=beacon, args=(args.port, args.quest), daemon=True).start()
    try:  # keep the log file from growing for ever
        if os.path.getsize(os.path.join(SCV_DIR, "serve.log")) > 2_000_000:
            os.replace(os.path.join(SCV_DIR, "serve.log"), os.path.join(SCV_DIR, "serve.log.1"))
    except OSError:
        pass
    load_spend()
    load_saved()
    if args.balance is not None:
        spend.update(balance=args.balance, spent=0.0)
        save_spend()
    log(f"Serving on port {args.port}; watching {args.source}  (Ctrl-C to stop)")
    if ACCESS_CODE:
        log(f"access code for the headset: {ACCESS_CODE}  (kept in {os.path.join(SCV_DIR, 'token')})")
    else:
        log("WARNING: --no-auth: anyone on the network can use this server")
    if AI["backend"] != "anthropic":
        log(f"AI: local model {AI['model']} at {AI['url']} (no key, no cost)")
    elif not args.no_ai and not api_key():
        log("note: no Keychain item 'scview-anthropic' yet, so voice requests will be refused")

    seen = None
    while True:
        # A model picked on the headset replaces the one named on the command line.
        with lock:
            wanted = state.get("wanted")
        src = wanted or pick_source(args.source)
        if src and os.path.isfile(src):
            st = os.stat(src)
            key = (src, st.st_mtime_ns, st.st_size)
            with lock:
                forced = state.pop("reload", False)
            if forced:
                ai_undo.clear()
                ai_redo.clear()
                recent.clear()
            if key != seen or forced:
                seen = key
                time.sleep(0.5)  # let a save or export finish writing
                log(f"loading {os.path.basename(src)} ...")
                try:
                    load(src)
                except (subprocess.CalledProcessError, OSError, RuntimeError) as e:
                    log(f"  loading failed: {e}")
        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
