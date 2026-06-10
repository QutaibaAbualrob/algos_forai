# AI Agents + Search Algorithms — Visual Lab

> Academic demonstrator for Artificial Intelligence course.
> 5 agent architectures + 7 search algorithms, all running on grid-worlds.
> Zero dependencies. Ready to present to Dr. Ala Hamarsheh.

---

## What is this?

A single Python GUI (tkinter, stdlib only) that **shows AI concepts executing step by step** on grid worlds.

**Grid world**: N×M cells. Cell types = Normal (cost=1), Mud (cost=5), Wall (cost=∞). Start cell. Goal cell.

**Three modes**:

| Mode | What it does |
|---|---|
| **Agents** | Pick an agent type → it navigates Start→Goal. See its architecture at work. |
| **Search** | Pick an algorithm → watch it explore step-by-step. |
| **Compare** | Two algorithms side-by-side on the same grid. Synchronized step. |

**Keyboard shortcuts**: `Space` = Step, `P` = Play/Pause, `R` = Reset, `1-3` = switch presets.

---

## Section 1: Agents (5 types)

Each agent adds **one capability** to the decision pipeline:

| # | Agent | Percept | Internal State | Decision Logic | Behavior |
|---|---|---|---|---|---|
| 1 | **Simple Reflex** | 4 neighbor cells (N,S,E,W) | None | Pure neighbour-choice rules: checks N,S,E,W in URDL order (up,right,down,left). Picks the first non-wall neighbour. If goal among them → goes there. No memory — cannot avoid revisiting. | Reactive. Loops in dead-end corridors. |
| 2 | **Model-based Reflex** | Same 4 neighbors | Map of visited cells + known walls | Percept → update model (mark current, log walls) → model + percept → rules → action. Avoids visited + walls. **Fallback**: if all neighbors blocked/visited → pick least-recently-visited. | Systematic. Avoids revisiting. Map grows on screen. |
| 3 | **Goal-based** | Full grid (given) | Goal coords. Planned path. | Calls a search algorithm (BFS/DFS/UCS/A*). Plans full path, then executes. | One-time plan, smooth execution. Path shown on canvas. |
| 4 | **Utility-based** | Full grid | Cost function + path. | Same architecture but uses UCS or A*. Optimizes for cost. | Path arcs around mud. "Cost: 8 vs 16 if through mud." |
| 5 | **Learning** | 4 neighbors | Q-table (state × action) | Q-learning, ε-greedy. Episode: start→act→reward→update Q→repeat. Terminal: Q(goal,·)=0. | Train All (bulk) or Step Episode. Q-values shown as cell tint. ε decays. Max steps/episode = rows×cols×2. |

**Key insight:** Russell & Norvig hierarchy — each adds ONE component (memory → goal → utility → learning). Goal-based and Utility-based share architecture; only the metric differs. Model-based reflex does NOT detect dead ends (that requires search) — it just avoids revisiting and known walls.

### Learning Agent Rewards

| Event | Reward |
|---|---|
| Move to free cell | −1 |
| Hit wall (stay in place) | −10 |
| Reach goal | +50 |
| Step into mud | −1 (same as normal — agent learns to avoid mud because it takes more STEPS through it, not per-step cost) |

**Terminal state**: Q(goal_state, any_action) = 0. Final Q-update: `Q(s,a) ← Q(s,a) + α[R_goal + γ×0 − Q(s,a)]`.

**Episode cutoff**: max `rows × cols × 2` steps per episode (prevents infinite loops).

**Wall hits**: when the agent tries to move into a wall, `move_agent()` returns False, the agent stays in the same state, and receives −10 reward. This creates a self-loop transition: (s, a) → (s, r=−10). The Q-value for that action in that state will drop, discouraging the agent from repeating the failed action. This is correct Q-learning behavior — the agent learns to avoid walls by experiencing penalties.

**Episodes**: 100 (preset A/B) or 300 (preset C). ε decays linearly to 0.01.

### Agent Parameters (Agent mode only)

| Parameter | Default | Options |
|---|---|---|
| Goal-based: algorithm | A* | BFS / DFS / UCS / A* |
| Utility-based: algorithm | A* | UCS / A* |
| Learning: Episodes | 100 / 300 | 50–1000 |
| Learning: ε | 0.3 | decays to 0.01 |
| Learning: α | 0.1 | 0.01–1.0 |
| Learning: γ | 0.9 | 0.5–1.0 |

---

## Section 2: Search Algorithms (7 algorithms)

**Canonical neighbor order**: Up, Right, Down, Left (URDL) — used by ALL algorithms and `GridWorld.get_neighbors()`. This makes expansion order deterministic and reproducible.

| # | Algorithm | Data Structure | g(n) | h(n) | Behavior |
|---|---|---|---|---|---|
| 1 | **BFS** | Queue (FIFO) | — | — | Layer by layer. Optimal for uniform cost. |
| 2 | **DFS** | Stack (LIFO) | — | — | Deep first. Not optimal. |
| 3 | **DLS** | Stack + depth cutoff | — | — | DFS with depth limit (slider 1–20). |
| 4 | **IDDFS** | Repeated DLS | — | — | DLS at depth 1,2,3... |
| 5 | **UCS** | Priority queue by g(n) | ✓ | — | Cheapest-first. Optimal. |
| 6 | **Greedy** | Priority queue by h(n) | — | ✓ | Rushes toward goal. Cost-blind. Tie-break: lower row, then lower col. |
| 7 | **A\*** | Priority queue by g+h | ✓ | ✓ | Cost-aware + goal-directed. Optimal. |

**Critical**: BFS/UCS/A* identical on uniform grids — that's the teaching point. Presets A, B, C show divergence.

### 3 Preset Grids

Symbols: `S`=Start, `G`=Goal, `#`=Wall(∞), `~`=Mud(cost 5), `·`=Normal(cost 1).

#### Preset A: Open Plain (5×5)

```
S  ·  ·  ·  ·
·  ·  ·  ·  ·
·  ·  ·  ·  ·
·  ·  ·  ·  ·
·  ·  ·  ·  G
```
- S=(0,0), G=(4,4). All cost=1.
- BFS = UCS = A*: all find 8-step path, cost=8.
- DLS limit ≥8 finds goal; limit <8 fails.
- **Use for**: comparing uninformed algorithms without cost confusion.

#### Preset B: Short-Cut Trap (5×6)

```
S  ·  ·  ·  ·  ·
·  #  #  #  #  ·
·  ·  ·  ·  #  ~
·  ·  ·  ·  #  ~
·  ·  ·  ·  ·  G
```
- S=(0,0), G=(4,5). Walls at (1,1),(1,2),(1,3),(1,4),(2,4),(3,4). Mud at (2,5),(3,5).
- **Short muddy path** (8 steps): S→(0,5)→(1,5)→(2,5)=mud→(3,5)=mud→G. **Cost = 16.**
- **Long clean path** (9 steps): S→(1,0)→(2,0)→(3,0)→(4,0)→right→G. **Cost = 8.**
- **What happens**: BFS finds 8-step path (cost 16). UCS finds 9-step path (cost 8). Greedy also takes the 8-step muddy path (cost 16) — its heuristic ignores cost. A\* finds the clean path (cost 8).
- **Divergence**: 16 vs 8. Teaches why step-count ≠ cost, why UCS/A\* exist.

#### Preset C: Mud Wall (6×6)

```
S  ·  ·  ·  ·  ·
·  ·  ·  ·  ·  ·
·  ·  ~  ~  ~  #
·  ·  ~  ~  ~  #
·  ·  ·  ·  ·  #
·  ·  ·  ·  ·  G
```
- S=(0,0), G=(5,5). Mud at (2,2),(2,3),(2,4),(3,2),(3,3),(3,4). Walls block right edge: (2,5),(3,5),(4,5).
- **Greedy's path** (verified by manual trace): S→right along row 0 to (0,5)→(1,5)→(1,4)→enters mud at (2,4)→(3,4)→exits at (4,4)→(5,4)→G. **2 mud cells. Cost = 17.**
- **A\\*'s path**: goes around via left edge. S→down to (5,0)→right along row 5 to G. All cost 1. **Cost = 9.**
- **Divergence**: 17 vs 9 — nearly 2× worse. Greedy ignores cost, takes the shorter-heuristic muddy path. A* factors actual cost into f(n).
- **Teaches**: Why f(n)=g(n)+h(n) matters. Heuristic alone is blind to cost. Even on a small grid, the gap is real and visible.

---

## Section 3: Comparison Mode

**Two canvases, one grid, two algorithms, synchronized step.**

```
┌─────────────────────────────────────────────────────────────────┐
│  [Compare ▾]  A: [A* ▾]  B: [Greedy ▾]  Preset: [C ▾]  ▶ ⏭ ↺│
├──────────────────────┬──────────────────────────────────────────┤
│                      │                                          │
│    CANVAS A (A*)     │       CANVAS B (Greedy)                  │
│                      │                                          │
│  ·  ·  ·  ·  ·  ·   │  ·  ·  ·  ·  ·  ·                       │
│  ·  ·  ·  ·  ·  ·   │  ·  ·  ·  ·  ·  ·                       │
│  ·  ·  ~  ~  ~  #   │  ·  ·  ~  ~  🟩  #    ← mud + Greedy trail│
│  ·  ·  ~  ~  ~  #   │  ·  ·  ~  ~  🟩  #                       │
│  ·  ·  ·  ·  ·  #   │  ·  ·  ·  ·  🟩  #                       │
│  🟦  🟩  🟩  🟩  🟩  🟥  │  🟦  🟩  🟩  🟩  🟩  🟥                     │
│                      │                                          │
│  Nodes: 16  Cost: 9  │  Nodes: 27  Cost: 17                     │
├──────────────────────┴──────────────────────────────────────────┤
│  A*: 16 nodes, cost 9 (optimal, went around via left edge).     │
│  Greedy: 27 nodes, cost 17 (2 mud cells — nearly 2× worse).     │
└──────────────────────────────────────────────────────────────────┘
```

**Sync**: One Step advances both. If A finishes first → its canvas freezes, B continues. Step/Pause only affect the unfinished algorithm (finished one stays frozen).

**Play**: Both advance in lockstep. If one finishes early, the other continues at full speed.

**Preset C + Compare = strongest demo.** 16 nodes vs 27. 9 cost vs 17. One frame.

---

## How the files connect

```
  search.py          agents.py         world.py
 (7 generators)    (5 agent classes)   (GridWorld)
      │                  │                  │
      │                  ├─ goal-based ─────┤  (calls search algos)
      │                  ├─ utility-based ──┤  (calls ucs/astar)
      │                  ├─ learning ───────┤  (Q-learning on world)
      │                  ├─ model-reflex ───┤  (uses world percepts)
      │                  └─ simple-reflex ──┘  (uses world percepts)
      │                                     │
      └────────────── main.py ──────────────┘
```

**The connection**: `agents.py` imports from `search.py`. Goal-based agent calls `astar_search(world)`. Utility-based calls `ucs_search(world)`. Same algorithm code runs in both Search tab and Agent tab.

---

## Generator Trick + Algorithm Code

All search functions are generators yielding `SearchFrame` after each expansion.

### BFS

**Neighbor order**: URDL (up, right, down, left) — canonical across all algorithms.

```python
from collections import deque

def bfs_search(world):
    frontier = deque([world.start])
    visited = set()               # start NOT pre-marked
    parent = {world.start: None}
    expanded = 0

    while frontier:
        current = frontier.popleft()
        expanded += 1
        visited.add(current)      # late goal test

        yield SearchFrame(
            frontier=list(frontier),
            visited=visited.copy(),
            current=current,
            path=reconstruct_path(parent, current),
            g=float(len(reconstruct_path(parent, current))),
            h=world.heuristic(current),
        )

        if current == world.goal:
            return SearchResult(success=True,
                path=reconstruct_path(parent, current),
                cost=len(reconstruct_path(parent, current)),
                nodes_expanded=expanded)

        for n in world.get_neighbors(current):
            if n not in visited and n not in frontier:
                parent[n] = current
                frontier.append(n)

    return SearchResult(success=False, nodes_expanded=expanded)
```

### DFS

**Neighbor order**: up, right, down, left (URDL). This order is canonical across ALL algorithms for reproducibility.

```python
def dfs_search(world):
    stack = [world.start]
    in_stack = {world.start}   # prevents duplicate pushes + parent overwrite
    visited = set()
    parent = {world.start: None}
    expanded = 0

    while stack:
        current = stack.pop()
        in_stack.discard(current)
        expanded += 1
        visited.add(current)

        yield SearchFrame(
            frontier=list(stack), visited=visited.copy(),
            current=current, path=reconstruct_path(parent, current),
            g=float(len(reconstruct_path(parent, current))),
            h=world.heuristic(current))

        if current == world.goal:
            return SearchResult(success=True, ...)

        for n in world.get_neighbors(current):  # URDL order
            if n not in visited and n not in in_stack:
                parent[n] = current             # set ONCE, never overwritten
                in_stack.add(n)
                stack.append(n)

    return SearchResult(success=False, ...)
```

### UCS

```python
import heapq

def ucs_search(world):
    counter = 0
    frontier = [(0, counter, world.start)]; counter += 1
    visited = set()
    parent = {world.start: None}
    cost_so_far = {world.start: 0}
    expanded = 0

    while frontier:
        g, _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        expanded += 1
        visited.add(current)

        yield SearchFrame(
            frontier=[n for _, _, n in frontier],
            visited=visited.copy(), current=current,
            path=reconstruct_path(parent, current),
            g=float(g), h=world.heuristic(current))

        if current == world.goal:
            return SearchResult(success=True, ..., cost=g, nodes_expanded=expanded)

        for n in world.get_neighbors(current):
            new_cost = g + world.get_cost(n)
            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n] = new_cost
                parent[n] = current
                heapq.heappush(frontier, (new_cost, counter, n)); counter += 1

    return SearchResult(success=False, ...)
```

### Greedy Best-First

```python
def greedy_search(world):
    counter = 0
    h0 = world.heuristic(world.start)
    # tie-break: (h, row, col, counter, node)
    frontier = [(h0, world.start[0], world.start[1], counter, world.start)]
    counter += 1
    visited = set()
    parent = {world.start: None}
    expanded = 0

    while frontier:
        h, _, _, _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        expanded += 1
        visited.add(current)

        yield SearchFrame(
            frontier=[n for _, _, _, _, n in frontier],
            visited=visited.copy(), current=current,
            path=reconstruct_path(parent, current),
            g=float(len(reconstruct_path(parent, current))),
            h=world.heuristic(current))

        if current == world.goal:
            return SearchResult(success=True, ...)

        for n in world.get_neighbors(current):
            if n not in visited:
                parent[n] = current
                hn = world.heuristic(n)
                heapq.heappush(frontier, (hn, n[0], n[1], counter, n))
                counter += 1

    return SearchResult(success=False, ...)
```

### A\*

```python
def astar_search(world):
    counter = 0
    h0 = world.heuristic(world.start)
    frontier = [(h0, counter, world.start)]; counter += 1
    visited = set()
    parent = {world.start: None}
    g_score = {world.start: 0}
    expanded = 0

    while frontier:
        f, _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        expanded += 1
        visited.add(current)
        g = g_score[current]

        yield SearchFrame(
            frontier=[n for _, _, n in frontier],
            visited=visited.copy(), current=current,
            path=reconstruct_path(parent, current),
            g=float(g), h=world.heuristic(current))

        if current == world.goal:
            return SearchResult(success=True, ..., cost=g, nodes_expanded=expanded)

        for n in world.get_neighbors(current):
            tg = g + world.get_cost(n)
            if n not in g_score or tg < g_score[n]:
                g_score[n] = tg
                parent[n] = current
                heapq.heappush(frontier, (tg + world.heuristic(n), counter, n))
                counter += 1

    return SearchResult(success=False, ...)
```

### DLS (DFS with depth limit)

```python
def dls_search(world, depth_limit):
    stack = [(world.start, 0)]
    in_stack = {world.start}
    visited = set()
    parent = {world.start: None}
    expanded = 0

    while stack:
        current, depth = stack.pop()
        in_stack.discard(current)
        expanded += 1
        visited.add(current)

        yield SearchFrame(
            frontier=[n for n, _ in stack],
            visited=visited.copy(), current=current,
            path=reconstruct_path(parent, current),
            g=float(depth), h=world.heuristic(current),
            depth=depth, depth_limit=depth_limit)

        if current == world.goal:
            return SearchResult(success=True, ...)

        if depth < depth_limit:
            for n in world.get_neighbors(current):  # URDL order
                if n not in visited and n not in in_stack:
                    parent[n] = current
                    in_stack.add(n)
                    stack.append((n, depth + 1))

    return SearchResult(success=False, ...)
```

### IDDFS

```python
def iddfs_search(world):
    for depth_limit in range(1, 50):
        yield SearchFrame(special="DEPTH_LIMIT_CHANGE", depth=depth_limit)
        result = yield from dls_search(world, depth_limit)
        if result.success:
            return result
    return SearchResult(success=False, ...)
```

---

## GUI Layout

### Agent Mode

```
┌──────────────────────────────────────────────────────────────────┐
│  [🤖 Agent ▾: Goal-based] [Preset ▾]  ▶Run ⏭Step ↺Reset       │
│  Speed: ──●────────────  [Space]=Step [P]=Play [R]=Reset        │
├─────────────────┬────────────────────────────────────────────────┤
│   AGENT INFO    │            CANVAS                               │
│  Type: Goal     │  ·  ·  ·  ·  ·  ·                             │
│  Algo: A*       │  🟦 🟩 🟩 ·  ·  ·     🟦=Start 🟥=Goal        │
│  Step: 14       │  ·  · 🟩 🟩 #  ·     #=Wall 🟩=Agent/Visited │
│  Path cost: 8   │  ·  ·  · 🟧 🟥  ·     🟨=Frontier 🟧=Current   │
│  ───────────    │  ·  ·  · 🟨 🟨  ·     🟪=Planned path         │
│  LEARNING ONLY: │  ·  🟨 🟨 ·  ·  ·                             │
│  Episode: 47    │  ·  🟨  ·  ·  ·  ·                             │
│  ε: 0.12        │  ~ = Mud (cost 5)                             │
│  [Train All]    │  Q: brighter = higher value                   │
│  [Step Ep]      │                                                │
├─────────────────┴────────────────────────────────────────────────┤
│  Step 14 — Moved DOWN to (3,4), following planned path          │
└──────────────────────────────────────────────────────────────────┘
```

### Search Mode (info panel)

```
├─────────────────┬────────────────────────────────────────────────┤
│  SEARCH INFO    │            CANVAS                               │
│  Algo: BFS      │                                                │
│  Frontier: 8    │                                                │
│  Visited: 18    │                                                │
│  Expansions:18  │                                                │
│  ───────────    │                                                │
│  DLS/IDDFS:     │  ← DLS/IDDFS only                             │
│  Depth: 5/10    │                                                │
│  [Limit:1-20]   │                                                │
│  ───────────    │                                                │
│  UCS/A*/Greedy: │  ← cost-aware algos only                      │
│  g = 5  h = 2   │                                                │
│  f = 7          │                                                │
├─────────────────┴────────────────────────────────────────────────┤
│  Step 23 — Expanded (3,4), g=5 h=2 f=7, frontier: 8             │
└──────────────────────────────────────────────────────────────────┘
```

### States

| State | Behavior |
|---|---|
| **Empty** | Canvas: "Choose an agent or search algorithm from the toolbar above." |
| **Search failure** | Frontier exhausted. Canvas freezes with visited cells. Message: "No path found after N expansions." Run disables. |
| **Goal reached** | Path highlighted in purple. Stats displayed. Auto-stop. |

### Auto-Run Stop Conditions

`root.after()` loop stops on: (1) `StopIteration` caught, (2) user clicks Pause (`running=False` flag), (3) goal reached with `SearchResult(success=True)`.

### Reset

| Context | Action |
|---|---|
| Agent mode | New agent, same type + world + params |
| Search mode | Same algo + grid, fresh exploration |
| Learning | **Reset Q** (blank) or **Reset Pos** (keep Q, new episode) |
| Preset change | Auto-reset, fresh grid |
| Compare | Both canvases reset, both algorithms restart |

### Color Legend

| Color | Meaning |
|---|---|
| ⬜ White | Unexplored |
| ⬛ Black | Wall |
| 🟦 Blue | Start |
| 🟥 Red | Goal |
| 🟩 Green | Visited / Agent |
| 🟨 Yellow | Frontier |
| 🟧 Orange | Current node |
| 🟪 Purple | Final path |

Mud cells: darker background tint. Q-values: brighter green = higher max-Q *for that state* (best action value, not cell quality).

---

## Project Structure (4 files, ~1100 lines)

```
E:\visualcode\algos_forai\
│
├── README.md         ← This file
├── world.py          ← GridWorld + Cell (~130 lines)
├── agents.py         ← 5 agent classes (~220 lines)
├── search.py         ← 7 search generators (~320 lines)
└── main.py           ← Tkinter GUI (~480 lines)
```

### world.py — GridWorld

```python
from dataclasses import dataclass

@dataclass
class Cell:
    terrain: str       # "normal" | "mud" | "wall"
    cost: int          # 1 | 5 | float('inf')
    row: int
    col: int

class GridWorld:
    grid: list[list[Cell]]
    rows: int; cols: int
    start: tuple[int, int]; goal: tuple[int, int]
    agent_pos: tuple[int, int]   # mutable; agent classes only

    def get_neighbors(self, pos) -> list[tuple[int, int]]:
        """Walkable neighbors (excludes walls, in bounds).
        Returns in canonical URDL order: up, right, down, left."""

    def get_cost(self, pos) -> int:
        """Cost to step ON this cell. Walls→inf. Goal→0 (not counted)."""

    def get_percept(self, pos) -> dict:
        """{'N': Cell|None, 'S':..., 'E':..., 'W':...}.
        None = out of bounds or wall. Agents interpret None as obstacle."""

    def is_goal(self, pos) -> bool: ...
    def heuristic(self, pos) -> float:
        """Manhattan distance to self.goal."""

    def move_agent(self, action: str) -> bool:
        """Move in 'up'|'down'|'left'|'right'. Returns True if moved,
        False if blocked. Agent STAYS if blocked."""

    def reset(self): ...       # agent_pos → start
    def get_state_index(self, pos) -> int:
        """row * cols + col (for Q-learning)."""

    @staticmethod
    def preset_a() -> 'GridWorld': ...  # Open plain 5×5
    @staticmethod
    def preset_b() -> 'GridWorld': ...  # Short-cut trap 5×6
    @staticmethod
    def preset_c() -> 'GridWorld': ...  # Mud wall 6×6
```

### search.py

```python
from dataclasses import dataclass

@dataclass
class SearchFrame:
    frontier: list[tuple[int, int]]
    visited: set[tuple[int, int]]
    current: tuple[int, int]
    path: list[tuple[int, int]]
    g: float = 0.0           # cost so far
    h: float = 0.0           # heuristic to goal
    depth: int = 0
    depth_limit: int | None = None
    special: str | None = None   # "DEPTH_LIMIT_CHANGE"

@dataclass
class SearchResult:
    success: bool
    path: list[tuple[int, int]]
    cost: float
    nodes_expanded: int

def bfs_search(world): ...
def dfs_search(world): ...
def dls_search(world, depth_limit): ...
def iddfs_search(world): ...
def ucs_search(world): ...
def greedy_search(world): ...
def astar_search(world): ...
```

### agents.py

```python
class SimpleReflexAgent:
    """Percept → Action. No memory. None in percept = obstacle."""
    def perceive(self) -> dict: ...
    def act(self, percept: dict) -> str: ...   # "up"|"down"|"left"|"right"|"stop"
    def step(self) -> tuple[str, str]: ...     # (action, rule_name)

class ModelReflexAgent:
    """Percept → Update model → Model+percept → Action.
    Model: visited cells + known walls. No dead-end detection.
    Fallback: if all neighbors blocked/visited → least-recently-visited."""
    def perceive(self) -> dict: ...
    def update_model(self, percept: dict): ...
    def act(self, percept: dict) -> str: ...
    def step(self) -> tuple[str, str]: ...
    def get_internal_map(self) -> list[list[str]]: ...

class GoalBasedAgent:
    """Search → Plan → Execute."""
    def __init__(self, world, algorithm="astar"): ...
    def plan(self) -> None: ...         # runs search, stores path
    def step(self) -> str: ...          # next action from plan
    def has_path(self) -> bool: ...
    def get_planned_path(self) -> list: ...

class UtilityBasedAgent:
    """Same architecture, cost-optimizing (UCS/A*)."""
    def __init__(self, world, algorithm="astar"): ...
    def plan(self) -> None: ...
    def step(self) -> str: ...
    def get_path_cost(self) -> float: ...

class LearningAgent:
    """Tabular Q-learning, ε-greedy."""
    def __init__(self, world, alpha=0.1, gamma=0.9, epsilon=0.3, episodes=100): ...
    def get_state(self) -> int: ...
    def act(self, state: int) -> str: ...
    def learn(self, state, action, reward, next_state) -> None: ...
    def run_episode(self) -> float: ...     # one episode, max rows×cols×2 steps
    def train_all(self) -> list[float]: ... # all episodes, returns reward history
    def reset_q(self) -> None: ...
    def reset_position(self) -> None: ...
    def get_max_q_per_cell(self) -> list[list[float]]: ...
    def get_q_table(self) -> list[list[float]]: ...
    def get_epsilon(self) -> float: ...
```

### main.py (~480 lines)

| Component | Class | Purpose |
|---|---|---|
| Root window | `App(tk.Tk)` | Mode state, world creation, wiring |
| Toolbar | `Toolbar(tk.Frame)` | Mode/preset dropdowns, Run/Step/Reset, speed, DLS slider, episodes |
| Info panel | `InfoPanel(tk.Frame)` | Context-dependent stats |
| Canvas | `GridCanvas(tk.Canvas)` | Grid + overlay painting |
| Log bar | `LogBar(tk.Frame)` | Last 5 events |

**Speed**: 50–1000ms, default 200ms. **DLS slider**: 1–20, default 10. **Episodes**: 100/300, visible for Learning.

---

## Tech

| Item | Value |
|---|---|
| Python | 3.8+ |
| Deps | None (tkinter built-in) |
| Run | `python main.py` |

---

## Build Order

| Step | File | Verify |
|---|---|---|
| 1 | `world.py` | `python -c "from world import GridWorld; w=GridWorld.preset_b(); print(w.rows, w.cols, w.heuristic((0,0)))"` → `5 6 9` |
| 2 | `search.py` | `python -c "from world import GridWorld; from search import bfs_search; w=GridWorld.preset_b(); g=bfs_search(w); print(next(g).current)"` |
| 3 | `agents.py` | `python -c "from world import GridWorld; from agents import GoalBasedAgent; a=GoalBasedAgent(GridWorld.preset_b()); a.plan(); print(a.get_planned_path())"` |
| 4 | `main.py` | Window opens. All modes. All presets. Compare. |

---

## Presentation Script

**Opening**: *"Demonstrates every agent architecture and search algorithm from our AI course on grid worlds. Three modes: agents, search, side-by-side comparison."*

**Agents**: *"Five agents. Simple Reflex: no memory, loops. Model-based: avoids revisiting. Goal-based: plans full path. Utility-based: cheapest path. Learning: Q-table from scratch."*

**Search (Preset B)**: *"Short-cut trap. BFS finds 8-step path through mud, cost 16. UCS finds 9-step clean path, cost 8. Step-count ≠ cost."*

**Killer demo (Preset C + Compare)**: *"A* left, Greedy right. S→down left edge, cost 9, 16 nodes. Greedy enters mud at (2,4), costs 17, 27 nodes. Nearly 2× worse. Heuristic alone is blind to cost — that's f(n)=g(n)+h(n)."*

---

## Course Mapping

| File | Topic | Lab |
|---|---|---|
| `ch1.pdf` | Agent types, PEAS | 5 agents on same grid |
| `ch2.ppt` | Agent architectures | Info panel per agent |
| `m3-search.pdf` | BFS, DFS, DLS, IDDFS, UCS | Search + Compare |
| `m4-heuristics.pdf` | Greedy, A*, heuristics | Search + Compare (esp. Preset C) |
