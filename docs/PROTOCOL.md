# SC View protocol, version 0 (draft)

How a viewer (the Quest app, the desktop build, or another client such as a tablet) talks to `tools/serve.py`.
Written from the server's code as a first draft for David to correct: where the code and this page disagree,
the code is right. Items marked **to confirm** are my reading, not a decision.

- Transport: UDP for discovery, plain HTTP/1.1 for everything else. Nothing is encrypted.
- Text is UTF-8. Header values that carry names or text are percent-encoded.
- Lengths and points are in the model's units, which for SolidCraft are millimetres.
- A field or line the client does not know should be ignored, so both sides can add to it.

## 1. Discovery: the beacon

Once a second, while a model is loaded, the server sends one UDP datagram to `255.255.255.255:47800`
(and to the address given with `--quest`, for networks that drop broadcasts):

```
SCVIEW1 <http port> <model version> n<notes version> b<balance>
```

| Field | Example | Meaning |
|---|---|---|
| `SCVIEW1` | | Protocol tag. A client ignores datagrams that start otherwise. |
| http port | `47801` | The server's HTTP port, at the datagram's source address. |
| model version | `6afe824a-1` | Opaque. When it changes, fetch `/model.scm` again. |
| `n` + number | `n3` | Notes version. When it changes, fetch `/comments.txt` again. |
| `b` + text | `b4.812`, `b-` | Estimated API balance in dollars, `-` when not given. |

Fields are separated by single spaces; later versions may append more. **To confirm:** a protocol version
number after `SCVIEW1` (the tag's `1` is the only version today).

## 2. The access code

Every HTTP request carries the header `X-Token: <code>`. The server makes the code on first run and keeps
it in `~/.scview/token`. A request without it gets `403` with a JSON body:

```json
{"reply": "Access code needed: type the code from the Mac and press Enter.", "ok": false}
```

`--no-auth` turns the check off (tests only).

## 3. Reading

| Request | Reply |
|---|---|
| `GET /model.scm` | The mesh, binary, format below. `404` while there is none. |
| `GET /comments.txt` | Notes, one per line: `id \t body \t x \t y \t z \t done (0/1) \t text` |
| `GET /models.txt` | Models the server can open: `path (its id) \t name \t size in MB \t open (0/1)` |
| `GET /requests.txt` | Saved requests: `id \t words \t r or -` (`r`: can be repeated without the AI) |
| `GET /dimensions.txt` | What can be set directly: `id \t label \t value \t unit \t kind \t expression` (kind: `user`, `feature`, `sketch`) |

Text replies are `text/plain; charset=utf-8`, tab-separated, one record per line, tabs and newlines in
values replaced by spaces.

### The mesh: `.scm` (little-endian)

```
"SCM1"  u32 vertex_count  u32 index_count
f32 min[3]  f32 max[3]                                   bounds
vertex_count x { f32 pos[3], f32 normal[3], u8 rgba[4] }  28 bytes each
index_count  x u32                                       triangles
optional:  "BODY"  u32 body_count
           body_count x { u32 first_index, u32 index_count, u16 name_len, name }
```

Each body's triangles are one contiguous index range. A vertex is unique per position, normal and colour.
The mesh carries no face or edge ids: a client groups triangles into faces itself and sends picks as points
(section 5), which the server matches to SolidCraft's own faces and edges.

## 4. Changing the design

All are `POST`. Every reply is JSON with at least:

```json
{"heard": "", "reply": "one sentence for the panel", "ok": true, "changed": false}
```

`changed: true` means the model changed; the new mesh is announced by the beacon's model version.

| Request | Headers and body | Notes |
|---|---|---|
| `/ask` | Body: a WAV recording (16 kHz mono), or text with `Content-Type: text/plain`. Optional `X-Selected`, `X-Point`, `X-Faces`, `X-Edges`, `X-Replay` (section 5). | The main request. Reply adds `cost` (dollars, as text) and, when a big request waits for an OK, `confirm_k` and `confirm_cost`. |
| `/ask/confirm` | | Carry on with the big request that was waiting. |
| `/transcribe` | Body: the recording so far. | Live text while speaking. Reply is `{"text": "..."}`; empty when the server is busy. |
| `/undo`, `/redo` | | The whole last change (all its steps), not one step. |
| `/pushpull` | `X-Face: body \t x \t y \t z` (a point on the face), `X-Distance: mm` (positive is outward) | No AI. |
| `/dimension` | `X-Name`: an id or label from `/dimensions.txt`; `X-Value`: a number in the dimension's unit, or an expression | No AI. A value that makes a feature fail is put back and `ok` is false. |
| `/save` | | Saves the design as a new `.sc` next to its source. |
| `/capture` | | Takes a headset photo over `adb` and copies it to the computer. |
| `/model/open` | `X-Id`: a path from `/models.txt` | Opens another model. |
| `/requests/save` | Body: the request's words, as text | Saves a request; the server adds what the AI did for it, if it still has that, so it can be repeated without the AI. |
| `/requests/delete` | `X-Id` | |

Spoken words the server handles itself, before any AI: "undo", "redo" and their variants; "go on" / "stop"
for a waiting big request; a request ending in "cancel" or "scratch that" is dropped; a request that only
sets one dimension ("set thickness to 12") is done directly unless the server runs with `--no-direct`.

## 5. What goes with a request

| Header | Value | Meaning |
|---|---|---|
| `X-Selected` | JSON array of body names | The selected bodies (at most 200). |
| `X-Point` | `body \t x \t y \t z` or `body \t x \t y \t z \t nx \t ny \t nz` | Where the pointer was when the wearer started speaking. With the normal, the server finds the exact face. |
| `X-Faces` | Lines of `body \t x \t y \t z \t nx \t ny \t nz` | Selected faces: a point on each and the normal there (at most 50). |
| `X-Edges` | Lines of `body \t start xyz \t middle xyz \t end xyz \t length` (11 fields) | Selected edges (at most 50). |
| `X-Replay` | any value | Allow a saved request to be repeated without the AI. |

Lines inside one header are separated by `\n` before percent-encoding.

## 6. Notes

| Request | Headers and body | Reply adds |
|---|---|---|
| `POST /comment` | `X-Pin: body \t x \t y \t z`; body is a recording, or text with `Content-Type: text/plain`. `X-Id` re-records an existing note. | `id` |
| `POST /comment/edit` | `X-Id`; optional `X-Pin` (move it) and a text body (new words) | |
| `POST /comment/delete`, `/comment/restore` | `X-Id` | |
| `POST /comment/do` | `X-Id`, or `all` for every note not yet done | `changes` |

Notes are kept in `<model file>.comments.json` next to the model.

## 7. Not in this version

- **Mac keys** (the computer's keyboard and mouse relayed to the headset over UDP) and the **sketch and
  dimension view** from Part 4 of the guide are not in the published server, so they are not described here.
- Encryption, more than one viewer at a time with their own selections, and a way for the client to learn
  the server's version or capabilities. **To confirm:** a `GET /version` or a field in the beacon would let
  two clients check they match before they send anything.
