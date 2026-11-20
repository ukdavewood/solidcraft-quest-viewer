# Direct Dimension Edits for the SolidCraft Quest Viewer (no AI)

"Set thickness to 12" does not need an AI. This guide adds a direct path to the viewer's server: a request
that only sets one parameter or one feature dimension is done by the server itself, in about 20 to 30
milliseconds, with no API call. The same path is open to the panel, so a slider or a grabbed dimension
handle can use it too.

It follows on from *Building a Quest 3 Viewer and AI Remote for SolidCraft with Claude Code* (Part 2 and
Part 4) and uses the sample code at <https://github.com/ukdavewood/solidcraft-quest-viewer>.

> **Scope and caveats.** Built and tested on 18 November 2026 on one Linux PC with SolidCraft v0.41.0,
> **without a headset**: requests were sent to the server as text and over HTTP. The viewer side (a slider,
> a handle) is not built. Unofficial: not affiliated with the SolidCraft project.

## The core idea

Every way of changing a dimension should end at one function on the server. Voice, a typed line, a slider
and a controller handle are then four ways in, not four implementations: they share one undo, one check and
one reply.

```
spoken or typed request ──> direct_request() ──┐
panel slider / handle  ──> POST /dimension  ──┼──> set_dimension() ──> SolidCraft set_parameter
(anything else)        ──> the AI, as before  ┘
```

SolidCraft already names everything that can be set. Its `parameters.list` command returns the user
parameters (`thickness`) and every feature input (`d4`, which is `Hole1`'s `Diameter`), and
`set_parameter` sets either kind by name.

## Step 1 — List what can be set

> Add a `dimensions()` function to `serve.py` that calls `parameters.list` (filter "all") and returns each
> user parameter and feature input with an id, a label people would say ("thickness", "Hole1 Diameter"),
> its value, unit and expression. Serve it to the headset as `/dimensions.txt`, tab-separated.

- **Verify:** `GET /dimensions.txt` lists `thickness`, `plate_w`, `Plate Length`, `Hole1 Diameter` and so on.

## Step 2 — One function that sets a dimension

> Add `set_dimension(dim, value)`: call `set_parameter`, and if SolidCraft reports features that now fail,
> undo and say so. On success, record an undo step the same way push/pull does, and reply in one sentence
> with the new and the old value.

- **What happened for us:** setting `thickness` to 0 made two features fail; the server put it back and
  replied "thickness = 0 breaks 2 feature(s), so I put it back."
- The value can be an expression: `plate_w = thickness * 5` works and stays parametric.

## Step 3 — Take simple requests before the AI

> Before a request goes to the AI, check whether it only sets one dimension: "set X to N", "X = N",
> "X N mm", "increase X by N". Find X among the dimensions, and only act when exactly one matches;
> otherwise return nothing and let the AI have the request. Add `--no-direct` to switch this off.

How a name is matched, in order: the exact name or label; the only input called that ("diameter"); the
only label containing every word ("hole diameter", or just "hole" when the hole has one input).

| Taken directly | Left for the AI |
|---|---|
| "set thickness to 12", "thickness = 12", "thickness 12" | "extrude 20mm", "make it red" |
| "change plate_w to 80", "make plate w 80" | "make the plate 80" (three inputs: which one?) |
| "set the hole diameter to 8", "make the hole 8mm" | "set the radius to 5" (nothing is called that) |
| "increase thickness by 2", "set thickness to 1.2 cm" | "make it 20 mm tall", "set thickness to twelve" |

- **Be strict.** A wrong direct edit is worse than a slow correct one, so anything ambiguous goes to the AI.
- **A surprise worth knowing:** "set plate width to 80" set the box's own `Width` input (40 → 80), not the
  user parameter `plate_w`, because "Plate Width" is that input's exact label. Name parameters so they
  don't collide with a feature's inputs, or say the parameter's name.

## Step 4 — The same path for the panel

> Add `POST /dimension` with the dimension's id or label in `X-Name` and a number or expression in
> `X-Value` (both percent-encoded), with the access code like every other request. It calls
> `set_dimension` and publishes the new model.

- **Verify:** the reply is the same JSON as `/pushpull` (`reply`, `ok`, `changed`), and Undo takes it back.
- The viewer side is still to build: a slider on the wrist panel for the selected dimension, or the
  dimension labels from Part 4 made draggable. Send the value when the trigger is released, not on every
  frame.

## What to expect

- **Speed:** 16 to 34 ms per edit in the server here, against 1 to 7 seconds for a local model and "a few
  seconds" for Haiku. In the headset the mesh export and download come on top.
- **Cost:** nothing.
- **Limits:** numbers must be digits ("12", not "twelve"; whisper usually writes digits). One dimension
  per request. Relative changes ("by 2") only in the dimension's own unit.

## Sample code

About 140 added lines in `tools/serve.py`: `dimensions`, `find_dimension`, `set_dimension`,
`direct_request`, `dimensions_text`, the `/dimension` handler and the hook in `/ask`. A test script
(`test_direct.py`) checks 15 requests that should be taken, 10 that should be left, and the put-back case,
by measuring the model.

**A tip for Claude Code:**

> Read `tools/serve.py` in <https://github.com/ukdavewood/solidcraft-quest-viewer> and this guide. Add the
> direct dimension path as described. Then write a test that builds a plate with a hole and two named
> parameters, and checks by volume which requests are taken, which are left for the AI, and that a
> breaking value is put back.
