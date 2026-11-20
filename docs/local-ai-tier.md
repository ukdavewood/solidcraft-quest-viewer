# Adding a Local AI Tier to the SolidCraft Quest Viewer with Claude Code

The Quest viewer's server sends each voice request to Claude Haiku. This guide adds a second tier: a model
running on your own computer through [Ollama](https://ollama.com), chosen with one option when you start the
server. Requests then cost nothing, need no API key, and nothing leaves your network.

It follows on from *Building a Quest 3 Viewer and AI Remote for SolidCraft with Claude Code* (Part 2,
Step C, "The Claude tier") and uses the sample code at <https://github.com/ukdavewood/solidcraft-quest-viewer>.

> **Scope and caveats.** Built and tested on 18 November 2026 on one Linux PC (Arch, RTX 4090 with 24 GB,
> Ollama 0.40.7, model `qwen3.8:27b`, SolidCraft v0.41.0) **without a headset**: requests were sent as text,
> straight to the server. Not tested on a Mac, with a real Quest, or with speech. Timings are from one
> machine and one small test part, so treat them as examples. Unofficial: not affiliated with the SolidCraft
> project, Ollama or Anthropic.

## The core idea

The server's AI loop already does everything in Anthropic's message format: the tools, the tool results,
the stop reason, the token counts. Don't rewrite the loop. Add a small **adapter** that turns one Anthropic
request into an OpenAI-style request, sends it to the local model, and turns the answer back. Every guard
from Step D (the cut-off check, the step limit, undo, saved requests) then works with either model.

```
serve.py  run_ai()  ──>  ai_step()  ──>  claude()        ──HTTPS──>  Claude Haiku API
                             │
                             └────────>  local_chat()    ──HTTP───>  Ollama on this computer
                                         to_openai() / from_openai()   (http://localhost:11434/v1)
```

Ollama, llama.cpp, LM Studio and vLLM all speak the same OpenAI chat format, so one adapter covers them.

## What you'll need

- The viewer's server working with Claude first (or with `--no-ai`), so you know SolidCraft and the model
  download are fine.
- **Ollama** running on the computer that runs the server.
- **A model that supports tool calling.** Check with `ollama show <model>`: "tools" must be listed under
  Capabilities. We used `qwen3.8:27b` (17 GB, fits a 24 GB graphics card with a 32K context window).
  Smaller cards need a smaller model; test it with the script below before trusting it.
- **A context window of at least 16K tokens.** One request is about 6K tokens before the model does
  anything (the system prompt and the tool descriptions). `ollama ps` shows the window in use; ours was
  32768. If yours is 4096, raise it (`OLLAMA_CONTEXT_LENGTH=32768` for the Ollama service), or the model
  will silently lose the start of the prompt and behave strangely.

## Step 1 — The adapter

> Read `tools/serve.py`, especially `claude()`, `ask()` and `run_ai()`. Add a second AI backend for a local
> model behind an OpenAI-compatible chat completions endpoint (Ollama at `http://localhost:11434/v1`). Keep
> `run_ai()` as it is: write `to_openai(body)` and `from_openai(reply)` that convert the Anthropic request
> and reply shapes, and an `ai_step(key, body)` that calls either `claude()` or the local model. Standard
> library only. Show me the plan first.

- **Claude will** map the system prompt to a `system` message, `tool_use` blocks to `tool_calls`,
  `tool_result` blocks to `tool` messages, `input_schema` to `parameters`, and `finish_reason: "length"` to
  the `max_tokens` stop reason the cut-off guard looks for.
- **Details that matter:** a tool call's arguments arrive as a JSON *string* and must be parsed; a failed
  tool result has no error flag in the OpenAI format, so prefix its text with `ERROR:`; drop the
  `cache_control` fields (prompt caching is an Anthropic feature; Ollama keeps the prompt prefix warm by
  itself); `tool_choice: {"type": "none"}` becomes `"none"`.
- **Verify:** `python3 -m py_compile tools/serve.py`, and the Claude tier still answers as before.

## Step 2 — Choosing the tier

> Add options `--ai anthropic|ollama` (default anthropic), `--ai-url`, `--ai-model` and `--ai-think`, each
> with an environment variable (`SCVIEW_AI`, `SCVIEW_AI_URL`, `SCVIEW_AI_MODEL`, `SCVIEW_AI_THINK`). With a
> local model, don't ask for an API key, count the cost as zero, and say which model is in use at start-up.

- **Verify:** start the server with `--ai ollama`. The log says
  `AI: local model qwen3.8:27b at http://localhost:11434/v1 (no key, no cost)`.

```sh
python3 -I tools/serve.py /path/to/model.step --quest <headset ip> --ai ollama
python3 -I tools/serve.py /path/to/model.step --quest <headset ip> --ai ollama --ai-model <other model>
```

## Step 3 — Turn thinking off

Many current local models "think" before they answer. For short CAD requests that made ours slower and
no more accurate, so the adapter sends `reasoning_effort: "none"` by default (`--ai-think low` to try it).

| `qwen3.8:27b`, 16 requests | Passed | Typical | Slowest |
|---|---|---|---|
| Thinking off (`none`) | 15 | 3 s | 7 s |
| Thinking `low` | 14 | 6 s | 46 s |

## Step 4 — Test it without the headset

> Write a test script that builds a small part in a headless SolidCraft session (a 60 x 40 x 10 plate with
> a 6 mm hole and a `thickness` parameter), sends requests through `ask()` and checks the result by
> measuring the model (volume and bounding box), not by trusting the AI's reply. Time each request.

What we got with thinking off, each request sent twice:

| Request | Passed | Seconds | What the model did |
|---|---|---|---|
| "make the hole 8mm" | 2 of 2 | 4 to 7 | Found the hole feature and edited its diameter in place |
| "set thickness to 12" | 2 of 2 | about 1 | `set_parameter` |
| "extrude 20mm" (pointing at the top face) | 1 of 2 | 3 to 4 | Once extruded the face 20 mm; once asked what to extrude |
| "put a 6 mm hole through here" (pointing) | 2 of 2 | about 1.5 | A hole at the pointed place |
| "make it red" (plate selected) | 2 of 2 | about 1 | |
| "fillet the top edges 1 mm" | 2 of 2 | 3 to 4 | |
| "make the plate twice as long" | 2 of 2 | about 4 | Read the length, then edited the box feature |
| "how thick is it?" | 2 of 2 | under 1 | Answered "10 mm", changed nothing |

- The times are the AI and SolidCraft only. In the headset, add whisper and the mesh download.
- The first request after Ollama has been idle also loads the model: about 30 seconds here. Ask Ollama to
  keep it loaded while you're in the headset (`OLLAMA_KEEP_ALIVE`), or send a warm-up request at start.
- "extrude 20mm" is a vague request. With thinking on, the model changed the `thickness` parameter to 20
  instead both times: a fair reading, but not what was meant. Say "extrude this face by 20 mm".
- **We did not run the same test against Claude Haiku** (no API key on the test machine), so this is not a
  side-by-side comparison. The guide's own figure for Haiku is "a few seconds" per request. Run the script
  with `--ai anthropic` to compare on your setup.

## What to expect

- **Speed:** on a 24 GB card, simple requests come back in 1 to 4 seconds; ones where the model has to
  look commands up first take 4 to 7.
- **Cost:** nothing per request. The headset's cost readout shows 0.
- **Quality:** good on single, clear edits. Not tested on big assemblies (the ship), long follow-up chains,
  or misheard speech, which is where Haiku earned its place. Keep Claude as the default and the local
  model as the cheap tier, and switch with the option.
- **Memory:** the model holds about 17 GB of graphics memory while loaded. On a Mac it would share memory
  with everything else.

## Troubleshooting

### "The local AI at ... does not answer"
Ollama isn't running, or the address is wrong. `curl http://localhost:11434/v1/models` should list your model.

### The model answers in words but never changes anything
It probably doesn't support tool calling, or its context window is too small and the tools were cut off.
Check `ollama show` (Capabilities: tools) and `ollama ps` (Context).

### It works, but every request takes 20 seconds or more
Thinking is on (`--ai-think none`), or the model doesn't fit in graphics memory and part of it runs on the
processor: `ollama ps` should say `100% GPU`.

### A tool call fails with odd arguments
Small models sometimes send `"false"` as text or invent parameter names. The server's `tidy_args` already
fixes the common cases; ask Claude to log the failing call and extend it.

## Sample code

The change is one file, about 110 added lines in `tools/serve.py`: `to_openai`, `from_openai`,
`local_chat`, `ai_step`, the options, and a zero cost for local requests. The test script is separate
(`run_tests.py`, about 90 lines).

**A tip for Claude Code:**

> Read `tools/serve.py` in <https://github.com/ukdavewood/solidcraft-quest-viewer>, then this guide. Add
> the local AI tier as described, keeping the Claude tier as the default and `run_ai()` unchanged. Then
> write the test script from Step 4 and run it against my local model. Tell me which requests fail and why.

## Resources

- Ollama: <https://ollama.com> (its OpenAI-compatible endpoint is `/v1` on port 11434)
- Sample code for the viewer and server: <https://github.com/ukdavewood/solidcraft-quest-viewer>
- The main guide: *Building a Quest 3 Viewer and AI Remote for SolidCraft with Claude Code*
