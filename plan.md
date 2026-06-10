# Implementation Plan — AI Agents + Search Algorithms Lab

> **Status:** All 4 files built. Ready for verification and GUI launch.
> **Project:** 5 agent architectures + 7 search algorithms on grid worlds.
> **Tech:** Python 3.8+, tkinter, zero dependencies.

---

## File Map

```
E:\visualcode\algos_forai\
├── README.md     ← Project spec (complete)
├── plan.md       ← This file — build status and verification
├── world.py      ← GridWorld + Cell + 3 presets         [DONE]
├── search.py     ← 7 search algorithm generators         [DONE]
├── agents.py     ← 5 agent classes                       [DONE]
└── main.py       ← Tkinter GUI (3 modes)                 [DONE]
```

---

## Task 1: world.py — GridWorld Environment

**Status:** ✅ Built (`world.py`, 175 lines)

### What it contains
- `Cell` dataclass — terrain (`normal|mud|wall`), cost, position, `is_wall` property
- `GridWorld` class with:
  - `get_neighbors(pos)` — URDL order (up, right, down, left), canonical
  - `get_cost(pos)` — returns cell cost; goal returns 0 (not counted in path cost)
  - `get_percept(pos)` — returns `{N:Cell|None, S:..., E:..., W:...}` for agents
  - `heuristic(pos)` — Manhattan distance to goal
  - `move_agent(action)` → `bool` — moves agent, returns success; stays if blocked
  - `reset()` — agent back to start
  - `set_cell(r,c,terrain,cost)` — builder helper
  - `get_state_index(pos)` — `row*cols+col` for Q-learning
- 3 static presets:
  - `preset_a()` — Open Plain 5×5, all cost 1
  - `preset_b()` — Short-Cut Trap 5×6, walls + mud (cost 5)
  - `preset_c()` — Mud Wall 6×6, mud zone + right-edge walls

### Verify

```bash
cd /e/visualcode/algos_forai
python -c "
from world import GridWorld
w = GridWorld.preset_a()
print('A:', w.rows, 'x', w.cols, 'h:', w.heuristic((0,0)))
w = GridWorld.preset_b()
print('B:', w.rows, 'x', w.cols, 'h:', w.heuristic((0,0)), 'mud:', w.get_cost((2,5)))
w = GridWorld.preset_c()
print('C:', w.rows, 'x', w.cols, 'mud:', w.get_cost((2,2)), 'clean:', w.get_cost((5,0)))
w = GridWorld.preset_a()
print('move right:', w.move_agent('right'), w.agent_pos)
w.reset(); print('reset:', w.agent_pos)
"
```

**Expected:** `A: 5 x 5 h: 8.0` / `B: 5 x 6 h: 9.0 mud: 5` / `C: 6 x 6 mud: 5 clean: 1` / `move right: True (0, 1)` / `reset: (0, 0)`

---

## Task 2: search.py — 7 Search Algorithm Generators

**Status:** ✅ Built (`search.py`, 310 lines)

### What it contains
- `SearchFrame` dataclass — frontier, visited, current, path, g, h, depth, depth_limit, special
- `SearchResult` dataclass — success, path, cost, nodes_expanded
- `reconstruct_path(parent, node)` — helper
- 7 generator functions, all `Generator[SearchFrame, None, SearchResult]`:

| Function | Data structure | Key pattern |
|---|---|---|
| `bfs_search` | `deque` (FIFO) | `visited` populated on expansion; `n not in frontier` guard |
| `dfs_search` | list (LIFO stack) | `in_stack` set prevents parent-overwrite + duplicate pushes |
| `dls_search(depth_limit)` | stack of `(node,depth)` | `in_stack` guard; stops expanding at depth limit |
| `iddfs_search` | repeated DLS | yields `DEPTH_LIMIT_CHANGE` frames; captures `yield from` return |
| `ucs_search` | `heapq` by `g(n)` | `cost_so_far` dict; stale-entry skipping |
| `greedy_search` | `heapq` by `h(n)` | tie-break: `(h, row, col, counter, node)` |
| `astar_search` | `heapq` by `f(n)=g+h` | `g_score` dict; stale-entry skipping |

- All use URDL neighbor order from `GridWorld.get_neighbors()`
- `world.get_cost(goal)` returns 0 → goal cost not counted
- All generators yield on every expansion (Step mode) and work with `root.after()` loop (Run mode)

### Verify

```bash
cd /e/visualcode/algos_forai
python -c "
from world import GridWorld
from search import bfs_search, ucs_search, astar_search, greedy_search

for name, fn, p in [('BFS',bfs_search,'a'),('UCS',ucs_search,'b'),
                     ('A*',astar_search,'c'),('Greedy',greedy_search,'c')]:
    w = getattr(GridWorld, f'preset_{p}')()
    g = fn(w)
    while True:
        try: next(g)
        except StopIteration as e:
            r = e.value
            print(f'{name:7s} preset {p.upper()}: success={r.success} cost={r.cost:3.0f} nodes={r.nodes_expanded}')
            break
"
```

**Expected:**
```
BFS     preset A: success=True cost=  8 nodes=25
UCS     preset B: success=True cost=  8 nodes=… 
A*      preset C: success=True cost=  9 nodes=…
Greedy  preset C: success=True cost= 17 nodes=…
```

---

## Task 3: agents.py — 5 Agent Classes

**Status:** ✅ Built (`agents.py`, 285 lines)

### What it contains

| Agent | Key method | Architecture |
|---|---|---|
| `SimpleReflexAgent` | `act(percept)` → picks first URDL non-wall; goal-if-visible | Percept → Action |
| `ModelReflexAgent` | `update_model()` + `act(percept)` → avoids visited, fallback: least-recently-visited | Percept → Update → Model+Percept → Action |
| `GoalBasedAgent` | `plan()` → runs search (BFS/DFS/UCS/A*); `step()` → follows plan | Search → Plan → Execute |
| `UtilityBasedAgent(GoalBasedAgent)` | inherits plan/step; adds `get_path_cost()` | Same, cost-optimizing |
| `LearningAgent` | `run_episode()` → Q-learning; `train_all()` → bulk; `reset_q()` / `reset_position()` | ε-greedy Q-learning |

- Learning agent: state = `row*cols+col`, 4 actions (URDL), linear ε decay, max `rows×cols×2` steps/episode, rewards: −1 move, −10 wall, +50 goal
- Goal/Utility agents import search functions from `search.py`

### Verify

```bash
cd /e/visualcode/algos_forai
python -c "
from world import GridWorld
from agents import GoalBasedAgent, UtilityBasedAgent, LearningAgent

# Goal-based
a = GoalBasedAgent(GridWorld.preset_b()); a.plan()
print('Goal path len:', len(a.get_planned_path()))

# Utility-based
a = UtilityBasedAgent(GridWorld.preset_b()); a.plan()
print('Utility cost:', a.get_path_cost())

# Learning (3 quick episodes)
a = LearningAgent(GridWorld.preset_a(), epsilon=1, episodes=3)
rewards = a.train_all()
print('Learning rewards:', [f'{r:.0f}' for r in rewards])
print('Epsilon:', a.get_epsilon())
"
```

**Expected:** Goal path len > 0, Utility cost = 8.0, Learning rewards show improvement, epsilon decays.

---

## Task 4: main.py — Tkinter GUI

**Status:** ✅ Built (`main.py`, 600 lines — larger than planned due to Compare mode + overlay system)

### Components

| Component | Class | Lines | Responsibility |
|---|---|---|---|
| Canvas | `GridCanvas(tk.Canvas)` | ~130 | Grid drawing + overlay system (visited, frontier, current, path, planned path, internal map, Q-heatmap, agent sprite) |
| Root | `App(tk.Tk)` | ~470 | Mode state, toolbar, info panel, log bar, step/run/reset logic, keyboard bindings |

### Modes

| Mode | Init | Step behavior |
|---|---|---|
| **Agent** | Instantiates agent class; Goal/Utility plan immediately | `agent.step()` — one action. Learning: `run_episode()` |
| **Search** | Creates generator from selected algorithm; DLS gets depth from slider | `next(generator)` — one expansion. Handles IDDFS special frames |
| **Compare** | Creates two generators; two canvases side-by-side | Advances both by one expansion each; if one finishes, the other continues |

### Controls

| Control | Widget | Default |
|---|---|---|
| Mode | Combobox | Agent |
| Sub-type | Combobox (agent type or algorithm) | Goal-based / BFS |
| Compare B | Combobox (shown in Compare mode) | Greedy |
| Preset | Combobox (A/B/C) | A |
| Run/Pause | Button | — |
| Step | Button | — |
| Reset | Button | — |
| Speed | Slider (50–1000ms) | 200ms |
| DLS depth | Slider (1–20, shown for DLS only) | 10 |
| Episodes | Spinbox (10–2000, shown for Learning only) | 100 |

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Step |
| `P` | Play / Pause |
| `R` | Reset |
| `1` / `2` / `3` | Switch preset A / B / C |

### Info panel (context-dependent)

**Agent mode**: Type, Preset, Position. Learning: episode count, ε, last reward.  
**Search mode**: Algorithm, Preset. Shows final result (cost, nodes) when done.  
**Compare mode**: Algorithm A/B, Preset. Shows per-algorithm results.

### Log bar

Last ~3 events displayed. Auto-scrolls. Agent: `→ action (rule)`. Search: `expanded (r,c) frontier=N`. IDDFS: `depth limit → N`. Goal: `✓ Goal! cost=X nodes=Y`.

### Verify

```bash
cd /e/visualcode/algos_forai
python main.py
```

**Manual checks:**
1. [ ] Window opens at 920×680
2. [ ] Agent mode → Goal-based → Preset B → Step: agent moves along planned path
3. [ ] Search mode → BFS → Preset A → Step: green wave expands
4. [ ] Preset C → A* → Step: finds path around left edge (cost 9)
5. [ ] Compare mode → A* vs Greedy → Preset C → Step: Greedy enters mud (cost 17)
6. [ ] Learning agent → Train All (button in info panel) → Q-table heatmap appears
7. [ ] DLS → depth slider appears → adjust → Step
8. [ ] Space = Step, P = Play/Pause, R = Reset, 1/2/3 = presets
9. [ ] Compare: one algorithm finishes → its canvas freezes, other continues
10. [ ] Reset clears everything for current mode

---

## File Sizes

| File | Lines | Size |
|---|---|---|
| `world.py` | 175 | 5.2 KB |
| `search.py` | 310 | 11.4 KB |
| `agents.py` | 285 | 11.3 KB |
| `main.py` | 600 | 30.8 KB |
| **Total** | **1,370** | **58.7 KB** |

---

## Quick Full Verify

```bash
cd /e/visualcode/algos_forai
python -c "import world; import search; import agents; import main; print('All imports OK')"
```

Then launch:

```bash
python main.py
```
