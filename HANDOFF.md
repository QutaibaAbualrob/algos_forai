# Agent Handoff — AI Agents + Search Algorithms Lab

> **For the next agent:** Read this entire document before touching any file.
> All files live at `E:\visualcode\algos_forai\`. Standard: tkinter, Python 3.8+, zero pip deps.

---

## What we built

An educational GUI demonstrating 5 AI agent architectures and 7 search algorithms on grid worlds.

```
E:\visualcode\algos_forai\
├── README.md     ← Full spec (28 KB, verified)
├── plan.md       ← Build status + verify commands
├── world.py      ← GridWorld + Cell + 3 presets        [STABLE]
├── search.py     ← 7 search generators                  [STABLE]
├── agents.py     ← 5 agent classes                      [STABLE]
├── logger.py     ← Debug logger → logs/ folder           [STABLE]
├── main.py       ← Tkinter GUI (3 modes)                [STABLE]
└── logs/         ← Auto-created, one .log per session
```

---

## Current state — everything works

- All 3 modes functional: Agent / Search / Compare
- All 5 agent types operational
- All 7 search algorithms runnable
- Logger writes to `logs/###_HH-MM.log`
- Keyboard shortcuts: Space=Step, P=Play, R=Reset, 1/2/3=presets

---

## Last fix applied (most recent change)

**Problem:** Learning agent wasn't avoiding mud because mud gave same reward (−1) as normal cells, even though search algorithms (UCS/A*) treat mud as cost 5.

**Fix:** In `agents.py`, `LearningAgent.step_episode()`, the reward for moving to a cell now reads the actual cell cost:

```python
# OLD (line ~289):
reward = -1.0

# NEW:
cell_cost = self.world.get_cost(self.world.agent_pos)
reward = -float(cell_cost)
```

This means:
- Normal cell → −1 reward
- Mud cell → −5 reward  
- Wall hit → −10 reward (unchanged)
- Goal → +50 reward (unchanged)

**Result:** The learning agent now avoids mud — consistent with UCS/A* path optimization.

**README updated:** Line 48 changed from "−1 (same as normal...)" to "−5 (per cell cost)".

---

## Architecture rules (DO NOT BREAK)

1. **Neighbor order is URDL** (up, right, down, left) — canonical across ALL files. `GridWorld.get_neighbors()` returns in this order.
2. **Search generators yield SearchFrame** after every expansion. Step mode = `next(gen)`. Run mode = `root.after(delay, step_func)`.
3. **DFS/DLS use `in_stack` set** to prevent parent overwriting when node reached via multiple paths.
4. **Goal cost returns 0** in `GridWorld.get_cost()` — path cost = sum of intermediate cell costs only.
5. **All agent `step()` methods return `tuple[str, str]`** — (action, info_string). No single-string returns.
6. **Learning agent uses step-by-step API:** `start_episode()` → `step_episode(state)` → `end_episode()`. One Step click = one `step_episode()` call.
7. **`train_all()` runs all episodes in bulk** (no animation) — used by "Train All" button.

---

## Presets (verified by manual trace)

| Preset | Size | Key feature | Optimal search cost |
|---|---|---|---|
| A — Open Plain | 5×5 | All cost 1 | 8 (BFS/UCS/A*) |
| B — Short-Cut Trap | 5×6 | Short muddy path cost 16, long clean cost 8 | 8 (UCS/A*) |
| C — Mud Wall | 6×6 | Greedy cost 17, A* cost 9 | 9 (A*) |

---

## Known behaviors (not bugs)

- **Simple Reflex** oscillates up/down in corridors — correct, no memory.
- **Model Reflex** gets stuck at goal after exploring all neighbors — correct, everything visited.
- **Learning agent** reward = −5 on mud, so it learns to avoid mud (matches UCS/A*).
- **Compare mode** continues unfinished algorithm when one finishes early — by design.
- **Logger** uses `-` not `:` in filenames because Windows bans colons.
- **IDDFS** yields `DEPTH_LIMIT_CHANGE` frames between depth iterations.

---

## If something breaks

1. Run the verify command: `python -c "import world; import search; import agents; print('OK')"`
2. Check the latest log in `logs/` for errors
3. Re-read the agent class that's failing — all must return `tuple[str, str]` from `step()`
4. For learning agent issues: check `_learn_state` is reset in `_reset()`, and `episode_count` guard exists
