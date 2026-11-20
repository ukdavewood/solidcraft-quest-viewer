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

Requests that touch the design are handled one at a time, in the order they arrive (except `/transcribe`).
One that waits more than 20 s behind another (an AI change can take a minute) is answered at once with
`"busy": true`, `ok: false` and a reply asking to try again. Numbers must be finite: `nan` or `inf` is
refused with `ok: false` and the design is not touched. Any request that fails still gets a JSON reply,
never a dropped connection. If SolidCraft itself stops (a crash), the server starts it again with the design
reopened from its file. Changes that were not saved are then lost, and the reply says so.

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

## 8. Added on the `xr-roadmap` branch

Not in David's published server; described here so another client can use them. All follow the rules above
(access code, tab-separated text, the JSON reply of section 4).

| Request | Reply or headers | Notes |
|---|---|---|
| `GET /history.txt` | `u` or `r` `\t` the change's words, newest first | What Undo and Redo would do next. |
| `GET /features.txt` | `id \t name \t type \t off (0/1) \t error` | The design's timeline, in order. |
| `GET /explodes.txt` | `id \t name \t on show (0/1)` | SolidCraft's own exploded views; empty when the design has none. |
| `GET /handles.txt` | `id \t label \t value \t unit \t x \t y \t z \t dx \t dy \t dz \t gain \t body \t min \t max` | A place on the model for each dimension that has one: a box's three sizes, a cylinder's height and radius, a hole's diameter, the distance of a sketch extruded into its own body, and each joint (a slider along its axis; a hinge at the turning part's far edge, `gain` in degrees per unit). `min` and `max` are the value's limits, `-` for none. A joint's line goes on with its type (`revolute`, `slider`), origin `x y z`, axis `x y z` and the bodies it carries (joined by `\x1f`), so a client can show them moving during a drag. The point and direction are in model units; a move of 1 along the direction adds `gain` to the value (2 for a diameter). `id` goes to `/dimension`. |
| `POST /feature` | `X-Id`: id or name; `X-Suppress`: `1` off, `0` on, absent to toggle | One undo step. |
| `POST /explode` | `X-View`: a view's name, `none` for assembled, absent for the next in turn | The mesh on show changes; the design does not. |
| `POST /visibility` | `X-Hidden`: JSON array of body names | What the wearer has hidden, for the AI's summary (`--sync-visibility` also sets SolidCraft's browser). |
| `POST /move` | `X-Body`: name; then `X-Translate: dx \t dy \t dz`, or `X-From`, `X-To` (`x \t y \t z` each) and optionally `X-Normal` | Moves one body by a distance, or sets it on another part at `X-To`. With `X-Normal` (the outward direction of the face there) the part's opposite side lands: it is moved, not turned, until the middle of that side touches the point. Without it: the point it was held by onto the point let go over. One undo step. |
| `POST /interference` | Reply adds `clashes` (a count) and, when there are any, `select` (the bodies involved, joined by `\x1f`) | SolidCraft's interference check on the design as it stands. Changes nothing in the design. When parts overlap, the mesh published next carries each overlap as a body of its own, named `Overlap 1`, `Overlap 2`...; `X-Clear: 1` publishes the plain mesh again. |
| `POST /sweep` | `X-Name`: a joint. Reply adds `hits`: `[part a, part b, from, to, most volume]` | The joint is taken through its travel (limit to limit) on a copy; where parts overlap on the way. Changes nothing. Also by words: "sweep hinge". |
| `GET /clashes.txt` | `body a \t body b \t volume mm3 \t x \t y \t z \t part a \t part b` | The last check's pairs, with the middle of each overlap. |
| `POST /preview/apply`, `/preview/cancel` | | With `--preview ai` (the default) an AI change is shown first; these keep it or drop it. The beacon's `p1` field says a preview is on show. |

The beacon has a seventh field, `m<id>`: a name for the model on show that stays the same through all its
edits and changes when another model is opened (`m-` while there is none). A viewer that fits its view to a
model's size should do so when this changes, and keep its scale and place otherwise: fitted afresh after
every edit, a plate made twice as long looks half as thick instead.

A reply to `/ask` may also carry `"command": "<id>"` (a viewer command from `tools/commands.tsv` that the
words named, done by the viewer itself) or `"select"` (body names joined by `\x1f`, to select).

Joints are in `/dimensions.txt` with kind `joint` and ids `joint:<id>` ("Hinge angle" in deg, "Slide distance"
in mm), ahead of the parameters, and are set through `/dimension` like any other: SolidCraft holds a value at
the joint's limits and the reply says so.
