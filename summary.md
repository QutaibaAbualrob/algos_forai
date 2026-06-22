# Summary — AI Agents + Search Algorithms Lab: Q&A Reference

> Answers to every question Amjad asked during the session, keyed to exact code locations.

---

## 1. How was the Utility Agent code written? How does it work step by step?

**Location:** `agents.py` lines 202–214 (class `UtilityBasedAgent`)

**It inherits everything from GoalBasedAgent — it's 12 lines of code:**

```python
class UtilityBasedAgent(GoalBasedAgent):
    def __init__(self, world, algorithm='astar'):
        if algorithm not in ('ucs', 'astar'):
            algorithm = 'astar'       # guard: BFS/DFS rejected
        super().__init__(world, algorithm)

    def get_path_cost(self) -> float:
        return sum(self.world.get_cost(p) for p in self.planned_path[1:])
```

**Step-by-step execution:**

| Step | Method | What happens |
|------|--------|-------------|
| 1 | `__init__()` | Validates algorithm (UCS/A* only). Calls parent `GoalBasedAgent.__init__` |
| 2 | `plan()` | Runs the search algorithm (UCS or A*) to completion, stores the cheapest path in `self.planned_path` |
| 3 | `step()` | Follows the planned path one cell at a time: compares current position to next target, maps delta to cardinal direction, calls `world.move_agent(action)` |
| 4 | `get_path_cost()` | Sums `world.get_cost(p)` for every cell in `planned_path[1:]` — this IS the utility function |

**Key difference from GoalBased:** GoalBased asks "does this path reach the goal?" (binary). UtilityBased asks "what's the CHEAPEST path?" (cost function). Same architecture, different metric.

**Proof on Preset B:** GoalBased + BFS → 8-step muddy path (cost 16). UtilityBased + A* → 9-step clean path (cost 8). Lower cost wins.

---

## 2. Where would you change the goal from "first decision" to "second decision" in the code?

**Location:** `world.py` lines 27–29

```python
self.start: tuple[int, int] = (0, 0)
self.goal: tuple[int, int] = (rows - 1, cols - 1)
```

The goal is set when the GridWorld is created. To change it to a "second decision" goal:

**Option A — temporary (for demo):** Before running, reassign:
```python
world.goal = (2, 3)   # some other cell
```

**Option B — permanent:** Add a new preset in `world.py`:
```python
@staticmethod
def preset_e():       # Two-goal variant
    w = GridWorld(5, 5)
    w.start = (0, 0)
    w.goal = (1, 4)   # first decision
    # then later: w.goal = (3, 0)  # second decision
    return w
```

The `heuristic()` function (line 78) automatically recomputes Manhattan distance to the new goal — no other code changes needed. Every search algorithm and agent reads `world.goal` dynamically on each expansion/decision.

---

## 3. How does depth-first search relate to growing nodes in the problem?

**Location:** `search.py` lines 89–125 (`dfs_search`)

DFS uses a **stack** (LIFO). It pushes ALL children at once, then pops the last-pushed child next.

**Node expansion order (URDL from start on Preset A):**
```
Push children of (0,0): up(out), right(0,1), down(1,0), left(out)
Stack: [(0,1), (1,0)]          ← right pushed first, down last
Pop → (1,0)  [down — last pushed, first popped]
Push children of (1,0): (0,0)visited, (1,1), (2,0), (0,0)out
Stack: [(0,1), (1,1), (2,0)]
Pop → (2,0)  [keeps going deeper]
```

DFS "grows deep" because it always chooses the most recently discovered node — the frontier is a stack of unexpanded nodes. This is why DFS can get lost in deep branches on Preset D while BFS expands layer by layer. The `in_stack` set (line 91) prevents duplicate pushes before a node is popped, which preserves correct parent pointers.

---

## 4. How did you implement the goal, and how does the system reach it?

**Goal definition:** `world.py` line 28 — `self.goal: tuple[int, int] = (rows - 1, cols - 1)`

**Two mechanisms to reach it:**

### A. Search Algorithms (Search/Compare mode)
Every search function checks `current == world.goal` after each expansion:
- **BFS** (`search.py` line 71): expands layer by layer until goal found
- **UCS** (`search.py` line 214): expands cheapest g-value first — guaranteed cost-optimal
- **A*** (`search.py` line 308): expands best f = g + h — optimal + efficient
- **Greedy** (`search.py` line 261): expands closest h to goal — fast but cost-blind

On goal found: returns `SearchResult(success=True, path=..., cost=..., nodes_expanded=...)`

### B. Agents (Agent mode)
| Agent | How it reaches goal |
|-------|-------------------|
| Simple Reflex | Rule 1: "if goal visible in any direction → go there" (`agents.py` line 36) |
| Model Reflex | Same rule 1 + avoids revisiting (`agents.py` line 90) |
| Goal-based | `plan()` runs search → `step()` follows the path (`agents.py` line 153, 169) |
| Utility-based | Same as Goal-based, but uses cost-optimal algorithm (`agents.py` line 202) |
| Learning | Q-learning: trial-and-error, receives +50 reward on goal → learns policy over episodes (`agents.py` line 283) |

---

## 5. Does the system take the smallest, closest, or biggest value to reach the goal?

**It depends on the algorithm:**

| Algorithm | What it optimizes | Value used |
|-----------|------------------|------------|
| **BFS** | Fewest steps (uniform cost) | Depth/layer count |
| **UCS** | Smallest **actual cost** g(n) | Sum of cell costs along path |
| **Greedy** | **Closest** heuristic distance h(n) | Manhattan distance to goal |
| **A*** | Combined: g(n) + h(n) | Actual cost + estimated remaining |
| **DFS** | Nothing (arbitrary path) | No optimization |

**Teaching example (Preset B):**
- BFS: 8 steps (fewest steps) but cost = 16 (went through mud)
- UCS: 9 steps (not fewest) but cost = 8 (smallest total cost)
- Greedy: 8 steps, cost = 16 (closest to goal — same as BFS here because h ignores cost)

The algorithm **does not take the biggest value** for anything — all optimizations are minimizations.

---

## 6. Where would you change the intention in the code?

**Intention = the committed plan the agent is currently executing.**

**Location:** `agents.py` line 150 — `self.planned_path: list[tuple[int, int]] = []`

To change the intention, modify or replace the planned path:

```python
# In GoalBasedAgent (agents.py line 153-167)
def plan(self):
    fn = self.ALGOS[self.algorithm]
    result = ...           # runs search to completion
    self.planned_path = result.path    # ← THIS is the intention
    self.path_index = 0
```

**Specific places to change intention:**
- **Change the algorithm** → different path: GUI dropdown or `self.algorithm = 'bfs'` then `plan()`
- **Replan mid-execution**: call `agent.plan()` again with a different algorithm
- **Manual intention override**: `agent.planned_path = [(0,0), (0,1), (0,2), ...]`
- **In `main.py`**: the algorithm dropdown (`self.algo_cb`) changes `self.agent_algo` which is passed to `_init_agent()` at line 518

---

## 7. BDI — How was BDI defined and applied in the project?

**Honest answer: The project uses the Russell & Norvig agent hierarchy, NOT BDI.**

BDI (Belief-Desire-Intention) is a different agent architecture model (Rao & Georgeff, 1995). Your project implements the progression from the AIMA textbook (Russell & Norvig):

| Russell & Norvig | BDI equivalent | Where in code |
|-----------------|----------------|---------------|
| **Model** (internal state) | ≈ **Beliefs** | `ModelReflexAgent.visited`, `.known_walls` → `agents.py` line 69-71 |
| **Goal** | ≈ **Desires** | `world.goal` + `GoalBasedAgent.plan()` → `world.py` line 28, `agents.py` line 153 |
| **Plan** | ≈ **Intentions** | `GoalBasedAgent.planned_path` + `.path_index` → `agents.py` line 150-151 |

**Where to find each in code:**

| Question | Answer |
|----------|--------|
| "Where would you change the belief?" | `agents.py` line 76 — `update_model()`: modifies `self.visited` and `self.known_walls` |
| "Where would you change the desire?" | `world.py` line 28 — `self.goal = (r, c)` |
| "Where would you change the intention?" | `agents.py` line 150 — `self.planned_path = new_path` |

**No single agent has all three** — they're split across the hierarchy intentionally to teach each component in isolation.

---

## 8. How was the Simple Reflex Agent built with BDI?

**It wasn't.** Simple Reflex has **no B, no D, no I.** That's the whole point.

```python
# agents.py line 15-51
class SimpleReflexAgent:
    def act(self, percept):
        # Rule 1: goal visible? → go there
        # Rule 2: first open neighbor (URDL) → go there
        # Rule 3: all blocked → stop
```

- **No Beliefs** — no memory, no internal state
- **No Desires** — doesn't know the goal's location (only sees it when adjacent)
- **No Intentions** — no plan, reacts purely to current percept

**Conditions and rules (agents.py lines 31–45):**

| Priority | Condition | Action | Rule name |
|----------|-----------|--------|-----------|
| 1 | Goal cell visible in any direction | Move toward goal | `goal-visible` |
| 2 | First non-wall neighbor in N,E,S,W order | Move there | `open:N` / `open:E` / etc. |
| 3 | All 4 directions blocked or walls | `stop` | `stuck` |

The URDL order is canonical: checks North first, then East, South, West. This makes behavior deterministic.

**"Where would you change the belief?"** — You can't. Simple Reflex HAS no belief. But you could add one by modifying the rules at `agents.py` line 41: change the neighbor selection priority, or add a memory of the last direction.

---

## 9. Learning Agent — show the code, explain how it works, how learning and feedback happen

### Core code location: `agents.py` lines 219–350

**The Q-learning update (lines 255–261):**
```python
def learn(self, state, action, reward, next_state):
    ai = self.ACTIONS.index(action)
    old = self.q_table[state][ai]
    nxt = max(self.q_table[next_state])
    self.q_table[state][ai] = old + self.alpha * (reward + self.gamma * nxt - old)
```

**How learning works:**

| Phase | What happens | Code |
|-------|-------------|------|
| **Explore** | With probability ε, pick random action | `agents.py` line 248 |
| **Exploit** | Otherwise, pick action with highest Q-value | `agents.py` line 251–253 |
| **Learn** | Update Q(s,a) using reward + max future Q | `agents.py` line 260–261 |
| **Feedback** | Reward signal: +50 (goal), −5 (mud), −1 (step), −10 (wall) | `agents.py` lines 283–291 |
| **Decay** | ε decreases linearly from 0.3 → 0.01 over all episodes | `agents.py` line 305–306 |

**Step-by-step episode flow (for visual animation):**
1. `start_episode()` → reset agent to start, return initial state (`agents.py` line 265)
2. `step_episode(state)` → one action + Q-update, returns (action, reward, next_state, done) (`agents.py` line 274)
3. `end_episode()` → decay epsilon, increment counter (`agents.py` line 302)

**"Where is the show learning function?"** — `get_max_q_per_cell()` at `agents.py` line 336. It collapses the Q-table (4 actions per state) into one max value per cell → painted as yellow heatmap in `main.py` at lines 99–111 (`update_overlay` Q-value heatmap section).

**Feedback visualization:** The heatmap starts invisible (all Q=0). After the agent first reaches the goal (+50 reward), positive Q-values propagate outward each episode — the yellow glow spreads. This IS the visual feedback.

---

## 10. Show the references and summarize the first one

**References cited in README.md:**

| Reference | Where used |
|-----------|-----------|
| Russell & Norvig, *Artificial Intelligence: A Modern Approach* (AIMA) | Agent hierarchy (Ch 2), search algorithms (Ch 3-4) |
| Rao & Georgeff, "BDI Agents: From Theory to Practice" (1995) | BDI architecture concepts |
| Sutton & Barto, *Reinforcement Learning: An Introduction* | Q-learning algorithm (tabular TD control) |

**First reference (Russell & Norvig) — summary:**
The AIMA textbook defines the standard agent taxonomy used in this project: Simple Reflex → Model-based Reflex → Goal-based → Utility-based → Learning. Each adds exactly one capability to the decision pipeline. The search algorithms (BFS, DFS, UCS, Greedy, A*) are covered in Chapters 3-4 as uninformed and informed search strategies. The Manhattan distance heuristic used in A* and Greedy is proven admissible and consistent for grid-world pathfinding (Section 3.6).

---

## 11. General: Show the code running

**Command:** `python main.py` (from `E:\visualcode\algos_forai`)

**Zero dependencies** — only Python 3.8+ and tkinter (built-in).

**Quick demo script (60 seconds):**
1. Launch → Agent mode, Goal-based + A*, Preset A
2. Click **▶ Run** → agent follows planned path (purple outline → green dot moves)
3. Switch mode to **Search** → BFS, Preset B
4. Click **▶ Run** → watch frontier expand, 8-step muddy path found (cost 16)
5. Switch algo to **A*** → **↺ Reset**, **▶ Run** → 9-step clean path (cost 8)
6. **Killer demo:** Mode → **Compare**, A: A*, B: Greedy, Preset C, **▶ Run**
   → Side-by-side: A* (left) goes around, Greedy (right) goes through mud

---

## 12. General: How did you apply BDI?

**The project maps to BDI as follows:**

| BDI | Code equivalent | File:Line |
|-----|----------------|-----------|
| **Belief** | Agent's internal model — `visited` set, `known_walls`, `q_table` | `agents.py:69` (ModelReflex), `agents.py:239` (Q-table) |
| **Desire** | Goal state — `world.goal` | `world.py:28` |
| **Intention** | Planned path — `planned_path` + `path_index` | `agents.py:150-151` |

But these are **split across different agents** — no single agent has all three simultaneously. This is by design: the Russell & Norvig hierarchy teaches each component in isolation before combining them conceptually.

---

## Quick Reference: Code Location Map

| Concept | File | Lines |
|---------|------|-------|
| Cell terrain + cost model | `world.py` | 10–19 |
| GridWorld class | `world.py` | 22–171 |
| Neighbor order (URDL) | `world.py` | 39–47 |
| Manhattan heuristic | `world.py` | 78–80 |
| Agent movement | `world.py` | 84–97 |
| 4 grid presets | `world.py` | 111–171 |
| SearchFrame dataclass | `search.py` | 15–26 |
| SearchResult dataclass | `search.py` | 29–34 |
| BFS algorithm | `search.py` | 51–84 |
| DFS algorithm | `search.py` | 89–125 |
| DLS algorithm | `search.py` | 130–169 |
| IDDFS algorithm | `search.py` | 174–184 |
| UCS algorithm | `search.py` | 189–230 |
| Greedy algorithm | `search.py` | 235–276 |
| A* algorithm | `search.py` | 281–324 |
| Simple Reflex agent | `agents.py` | 15–51 |
| Model Reflex agent | `agents.py` | 56–131 |
| GoalBased agent | `agents.py` | 136–197 |
| UtilityBased agent | `agents.py` | 202–214 |
| Learning agent (Q-learning) | `agents.py` | 219–350 |
| ε-greedy action selection | `agents.py` | 247–253 |
| Q-learning update rule | `agents.py` | 255–261 |
| Episode step-by-step API | `agents.py` | 265–308 |
| Reward model | `agents.py` | 283–291 |
| ε decay formula | `agents.py` | 305–306 |
| Q-value heatmap export | `agents.py` | 336–344 |
| GridCanvas rendering | `main.py` | 26–162 |
| Canvas overlay system | `main.py` | 84–162 |
| App state machine | `main.py` | 169–206 |
| Toolbar + dropdowns | `main.py` | 212–279 |
| UI refresh dispatcher | `main.py` | 348–428 |
| Agent initialization | `main.py` | 506–524 |
| Search initialization | `main.py` | 530–537 |
| Compare initialization | `main.py` | 543–549 |
| Step logic (all modes) | `main.py` | 555–670 |
| Agent step (learning + non-learning) | `main.py` | 564–594 |
| Search step | `main.py` | 596–641 |
| Compare step (sync) | `main.py` | 643–670 |
| Run loop (root.after) | `main.py` | 776–787 |
| Session logger | `logger.py` | 32–108 |

---

# Search Algorithms — Complete Reference

> All 7 algorithms, what they do, when to use them, and what they suck at.

---

## BFS — Breadth-First Search

**Data structure:** Queue (FIFO)  
**Code:** `search.py` lines 51–84  
**Preset demo:** Preset A (all uniform cost — BFS = UCS = A*)

| | |
|---|---|
| **How it works** | Expands nodes in order of discovery — layer by layer from start. Uses `deque.popleft()` to always expand the oldest node in frontier. |
| **g(n)** | — (doesn't use costs) |
| **h(n)** | — |
| **Pros** | ✅ **Complete** — always finds a path if one exists. ✅ **Optimal for uniform cost** — finds fewest steps. ✅ Simple to implement and explain. |
| **Cons** | ❌ Memory-hungry: stores entire frontier (O(b^d)). ❌ **Not cost-optimal** — ignores terrain costs, can take expensive shortcuts. ❌ Expands EVERY node at depth d before any at d+1. |
| **When to use** | Uniform-cost grids. Teaching uninformed search. Baseline to compare against. |
| **Preset B result** | 8 steps, cost = 16 (through mud — suboptimal) |

---

## DFS — Depth-First Search

**Data structure:** Stack (LIFO)  
**Code:** `search.py` lines 89–125  
**Preset demo:** Preset D (gets lost in deep corridors)

| | |
|---|---|
| **How it works** | Expands the most recently discovered node first — goes deep, backtracks when stuck. Uses list as stack: `pop()` from end. |
| **g(n)** | — |
| **h(n)** | — |
| **Pros** | ✅ **Low memory**: O(b·m) — only stores current path + siblings. ✅ Can find deep solutions faster than BFS. ✅ Good when solution is deep and branching factor is high. |
| **Cons** | ❌ **Not optimal** — finds SOME path, not best path. ❌ **Not complete on infinite graphs** (but fine on finite grids with visited set). ❌ Can wander down long dead-end corridors. ❌ `in_stack` set required to prevent parent-pointer corruption. |
| **When to use** | Memory-constrained environments. When any path is good enough. Contrasting against BFS. |
| **Preset D result** | Wanders deeply, may find a convoluted long path |

---

## DLS — Depth-Limited Search

**Data structure:** Stack + depth counter (DFS with cutoff)  
**Code:** `search.py` lines 130–169  
**Preset demo:** Preset A, limit=8 finds goal; limit=4 fails

| | |
|---|---|
| **How it works** | DFS that refuses to expand nodes beyond `depth_limit`. Stack stores `(node, depth)` tuples. Children only generated when `depth < depth_limit`. |
| **g(n)** | — (depth used as cost proxy) |
| **h(n)** | — |
| **Pros** | ✅ **Prevents infinite descent** — bounded DFS. ✅ Lower memory than BFS. ✅ Parameterized: slider 1–20 in GUI. |
| **Cons** | ❌ **Not complete** — can miss solutions deeper than the limit. ❌ **Not optimal**. ❌ User must know or guess the right depth. ❌ If limit is too shallow, fails even when solution exists. |
| **When to use** | When you know approximate solution depth. Teaching depth-bounding as a concept. Building block for IDDFS. |
| **Preset A result** | limit=8 → finds 8-step path (success). limit=4 → fails (frontier exhausted before reaching depth 8) |

---

## IDDFS — Iterative Deepening DFS

**Data structure:** Repeated DLS at depth 1, 2, 3, …  
**Code:** `search.py` lines 174–184  
**Preset demo:** Preset A (finds optimal path with BFS-like guarantees, DFS-like memory)

| | |
|---|---|
| **How it works** | Runs DLS at depth 1, then depth 2, then depth 3… until goal found. Each iteration rebuilds the tree from scratch. Uses `yield from dls_search(world, depth_limit)`. |
| **g(n)** | — |
| **h(n)** | — |
| **Pros** | ✅ **Complete + optimal for uniform cost** (like BFS). ✅ **Low memory** (like DFS): O(b·d). ✅ Combines best of both worlds. ✅ The repeated work is only a constant factor: total ≈ b/(b−1) × last iteration. |
| **Cons** | ❌ **Repeats work** — re-expands start node d times. ❌ Slower than BFS in practice for shallow solutions. ❌ Still not cost-optimal (ignores terrain costs). ❌ Max depth=50 hardcoded safety limit. |
| **When to use** | When memory is tight AND optimality matters AND costs are uniform. Teaching the memory-vs-time tradeoff. |
| **Preset A result** | Same 8-step path as BFS, but re-expands nodes each iteration |

---

## UCS — Uniform-Cost Search (Dijkstra)

**Data structure:** Priority queue by g(n)  
**Code:** `search.py` lines 189–230  
**Preset demo:** Preset B (9 steps, cost 8 — avoids mud, optimal)

| | |
|---|---|
| **How it works** | Always expands the node with the lowest PATH COST from start. Uses `heapq` ordered by `(g, counter, node)`. `cost_so_far` tracks best known g to each node; stale entries are skipped. |
| **g(n)** | ✅ Sum of actual cell costs from start |
| **h(n)** | — |
| **Pros** | ✅ **Always cost-optimal** — finds cheapest path regardless of terrain. ✅ **Complete**. ✅ Works on ANY cost model (normal=1, mud=5, walls=∞). ✅ Same result as BFS on uniform grids. |
| **Cons** | ❌ **Can expand many nodes** — explores in all directions equally. ❌ No goal-direction bias — might expand away from goal if costs are low. ❌ Higher memory than DFS (stores full frontier in heap). ❌ `counter` tiebreaker needed to prevent tuple comparison errors. |
| **When to use** | Non-uniform costs (mud, varied terrain). When path cost matters more than step count. Preset B/C/D. |
| **Preset B result** | 9 steps, cost = 8 (clean path around mud — optimal) |

---

## Greedy Best-First Search

**Data structure:** Priority queue by h(n) only  
**Code:** `search.py` lines 235–276  
**Preset demo:** Preset C (rushes into mud, cost 17 — nearly 2× worse)

| | |
|---|---|
| **How it works** | Always expands the node CLOSEST to goal by heuristic (Manhattan distance). Uses `heapq` ordered by `(h, row, col, counter, node)`. Tie-breaking by lower row, then lower col. |
| **g(n)** | — (ignored!) |
| **h(n)** | ✅ Manhattan distance to goal |
| **Pros** | ✅ **Fast** — often expands fewer nodes than A*. ✅ Goal-directed — "rushes" toward goal. ✅ Simple to understand (greedy = take what looks best now). |
| **Cons** | ❌ **Not optimal** — cost-blind. Can be led into expensive terrain. ❌ **Not complete on some graphs** (though fine on grids). ❌ The "greedy shortcut" is often the expensive shortcut. ❌ Deterministic tie-breaking required (row, col, counter) or behavior is unpredictable. |
| **When to use** | Teaching WHY cost matters. Demonstrating heuristic-only failure. Side-by-side vs A* (Preset C killer demo). |
| **Preset C result** | 27 nodes, cost = 17 (through 2 mud cells — suboptimal) |

---

## A* Search

**Data structure:** Priority queue by f(n) = g(n) + h(n)  
**Code:** `search.py` lines 281–324  
**Preset demo:** Preset C vs Greedy (16 nodes, cost 9 — optimal + efficient)

| | |
|---|---|
| **How it works** | Combines UCS and Greedy: expands by f = g + h (actual cost so far + estimated remaining). Uses `heapq` by `(f, counter, node)`. `g_score` tracks best known g; stale entries skipped. |
| **g(n)** | ✅ Actual cost from start |
| **h(n)** | ✅ Manhattan distance (admissible + consistent) |
| **Pros** | ✅ **Optimal** — guaranteed cheapest path with admissible heuristic. ✅ **Efficient** — expands fewer nodes than UCS (h biases toward goal). ✅ **Complete**. ✅ The gold standard for pathfinding. |
| **Cons** | ❌ **Needs a good heuristic** — bad heuristic = bad performance. ❌ Memory: stores frontier + g_score dict. ❌ Heuristic MUST be admissible (never overestimate) for optimality. ❌ More complex to implement than BFS/DFS. |
| **When to use** | ALWAYS for pathfinding with non-uniform costs. The go-to algorithm. Teaching f(n) = g(n) + h(n). Best demo: Preset C Compare mode. |
| **Preset C result** | 16 nodes, cost = 9 (avoids mud via left edge — optimal) |

---

## Algorithm Cheat Sheet

| Algorithm | Frontier | Optimal? | Complete? | Memory | Speed | Uses g? | Uses h? |
|-----------|----------|----------|-----------|--------|-------|---------|---------|
| **BFS** | Queue | Steps only | Yes | O(b^d) | Medium | — | — |
| **DFS** | Stack | No | On finite | O(b·m) | Varies | — | — |
| **DLS** | Stack+limit | No | No (if limit<d) | O(b·ℓ) | Varies | — | — |
| **IDDFS** | Repeated DLS | Steps only | Yes | O(b·d) | Slow (repeats) | — | — |
| **UCS** | Heap by g | **Cost** | Yes | O(b^(1+⌊C*/ε⌋)) | Slow (all dirs) | ✅ | — |
| **Greedy** | Heap by h | No | Yes (grids) | Depends | **Fastest** | — | ✅ |
| **A\*** | Heap by g+h | **Cost** | Yes | Depends | Fast (directed) | ✅ | ✅ |

---

## Best Comparison Pairs for Presentation

| Preset | Algorithm A | Algorithm B | What it shows |
|--------|------------|------------|---------------|
| **A** | BFS | DFS | Layer-by-layer vs deep-first divergence |
| **B** | BFS | UCS | Step-count (8) ≠ cost (16 vs 8) |
| **B** | BFS | A* | Same as UCS comparison, with heuristic |
| **C** | A* | Greedy | **Killer demo: 9 vs 17 cost, 16 vs 27 nodes** |
| **D** | BFS | DFS | DFS wanders deep, BFS expands methodically |
| **D** | A* | Greedy | Mud pocket divergence on larger maze |
