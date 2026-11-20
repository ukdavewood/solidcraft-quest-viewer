# Commands

Written from `src/commands.rs`; do not edit by hand (`SCVIEW_WRITE_COMMANDS=1 cargo test --lib commands`).
Every command can be reached from the panel, from the keyboard (a Bluetooth keyboard on the headset,
or the terminal of the desktop build as `^h`), and by saying or typing one of its phrases.

| Command | What it does | Keyboard | Say or type |
|---|---|---|---|
| `edit.undo` | Undo the last change | Ctrl+Z | undo, undo that, undo it, undo the last change, go back |
| `edit.redo` | Redo | Ctrl+Y, Ctrl+Shift+Z | redo, redo that, redo it |
| `view.hide` | Hide the selected parts | Ctrl+H, Del | hide, hide this, hide these, hide it, hide them, hide the selection |
| `view.isolate` | Show only the selected parts | Ctrl+I | isolate, isolate this, isolate these, isolate it, show only this, show only these |
| `view.show_all` | Show every part | Ctrl+A | show all, show everything, unhide all, unhide everything |
| `select.clear` | Deselect | Ctrl+D | deselect, deselect all, clear selection, clear the selection, select nothing |
| `view.recentre` | Bring the model back in front | Ctrl+R | recentre, recenter, bring it back, reset the view |
| `select.mode` | Pick parts, faces or edges | Ctrl+F | pick faces, pick edges, pick parts |
| `tool.push_pull` | Push or pull a face | Ctrl+P | push pull, pull |
| `tool.place` | Set the model down on a surface | Ctrl+L | set down, set it down |
| `view.size` | Step the model's size | Ctrl+J | size, next size |
| `view.snap` | Stand the model upright | Ctrl+U | upright, stand it up |
| `file.save` | Save as a new file | Ctrl+Shift+S | save the file, save the design |
| `tool.section` | Section cut | Ctrl+S | section, section cut, cut view |
| `tool.measure` | Measure | Ctrl+M | measure |
| `view.edges` | Edges: outline, all, none | Ctrl+E | edges, toggle edges |
| `view.explode` | Exploded view | Ctrl+X | explode, exploded view, explode it |
| `note.add` | Pin a voice note | Ctrl+N | note, add a note |
| `view.room` | Room on or off | Ctrl+B | room, toggle the room |
| `file.models` | Open another model | Ctrl+O | models, open a model, open another model |
| `view.guides` | Axes and grid | Ctrl+G | guides, axes, grid |
| `ui.keyboard` | On-screen keyboard | Ctrl+K | keyboard |
| `ai.repeat` | Repeat the last request | Ctrl+T | repeat, repeat that, again, do it again |
| `ai.keep` | Keep the last request | Ctrl+W | keep that, save that request |
| `ai.saved` | Saved requests | Ctrl+Q | saved requests, show saved requests |
| `ui.radial` | Radial menu at the pointing hand: move towards a choice, trigger to take it | Ctrl+Shift+R, Right thumbstick: press, push towards a choice, let go | menu, radial menu, open the menu |
| `tool.clash` | Check which parts overlap, and mark them red | Ctrl+Shift+C | check interference, interference, interference check, check for clashes, do any parts overlap, check overlaps |
| `tool.move` | Move a part: drag it, let go on another part's face to set it there | Ctrl+Shift+M | move, move parts, move a part, move this |
| `edit.handles` | Dimension handles on the model: drag one to change its size | Ctrl+Shift+H | handles, dimension handles, show the handles, hide the handles |
| `edit.features` | Feature tree: switch a feature off or on, or edit it | Ctrl+C | features, feature tree, show the features, timeline |
| `edit.dimensions` | Dimensions: change a size directly | Ctrl+V | dimensions, show dimensions, show the dimensions |
