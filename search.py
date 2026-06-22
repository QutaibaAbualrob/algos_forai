"""Search algorithms as generators.
══════════════════════════════════════════════════════════════════════
ARCHITECTURE ROLE: This file is the ALGORITHM ENGINE of the project.
Every search function is a Python GENERATOR that yields a SearchFrame
after each node expansion, then returns a SearchResult on completion.

WHY GENERATORS: The GUI needs to PAINT between expansions for animation.
A regular function would compute everything at once → no animation.
A generator yields control back to the GUI after each step, which paints
the current state (visited, frontier, current node), then calls next()
to resume. The same code runs in Step mode (one next() per click) and
Run mode (next() in a root.after() loop). This is the "Generator Trick."

ALGORITHM LINEUP (7 total):
  1. BFS     — Queue (FIFO), layer-by-layer, optimal for uniform cost
  2. DFS     — Stack (LIFO), deep first, NOT optimal
  3. DLS  (Depth-Limited Search)   — DFS with depth cutoff, parameterized limit
  4. IDDFS (Iterative Deepening Depth-First Search)   — Repeated DLS at increasing depths (1,2,3...)
  5. UCS     — Priority queue by g(n), cost-optimal for any cost
  6. Greedy  — Priority queue by h(n) only, cost-blind, fast
  7. A*      — Priority queue by f(n)=g(n)+h(n), optimal + efficient

KEY PATTERNS ACROSS ALL ALGORITHMS:
  • Late goal test: check current == goal AFTER yielding the frame.
    This ensures the GUI paints the goal cell before stopping.
  • visited populated on EXPANSION (when popped), not on generation.
    Standard for graph-search (not tree-search).
  • parent dict: tracks best known parent for each node, enables
    path reconstruction via reconstruct_path().
  • Neighbor order: URDL (up,right,down,left) — CANONICAL, deterministic.

CONNECTIONS:
  • world.py — imports GridWorld, calls get_neighbors(), get_cost(), heuristic()
  • agents.py — GoalBasedAgent and UtilityBasedAgent call these functions
  • main.py — creates generators, calls next() in step loop, reads SearchFrame for canvas

PACKAGE: search module
Each yields SearchFrame after every node expansion and returns SearchResult.
Neighbor order: URDL (up, right, down, left) — canonical, deterministic.
"""

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Generator

from world import GridWorld


# ═══════════════════════════════════════════════════════════════════
#  DATA CLASSES — the communication protocol between algorithms and GUI
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SearchFrame:
    """
    SNAPSHOT of algorithm state after one node expansion.
    Yielded to the GUI, which reads these fields to paint the canvas.

    Every algorithm fills these fields — some are algorithm-specific:
      • frontier, visited, current, path — ALL algorithms
      • g, h — UCS, Greedy, A* (cost/heuristic-aware)
      • depth, depth_limit — DLS, IDDFS only
      • special — IDDFS only ("DEPTH_LIMIT_CHANGE" signal)

    The GUI uses isinstance() or field presence checks to decide what
    info panel fields to show (g/h only for cost-aware algorithms).
    """

    frontier: list[tuple[int, int]]       # nodes waiting to be expanded (queued)
    visited: set[tuple[int, int]]         # nodes already expanded (popped)
    current: tuple[int, int]               # the node being expanded RIGHT NOW
    path: list[tuple[int, int]]            # best path from start to current node
    g: float = 0.0                         # actual cost from start to current (UCS/A*/Greedy)
    h: float = 0.0                         # heuristic estimate from current to goal
    depth: int = 0                         # current depth in the search tree (DLS/IDDFS)
    depth_limit: int | None = None         # active depth cutoff (DLS/IDDFS only)
    special: str | None = None             # signal string: "DEPTH_LIMIT_CHANGE" for IDDFS


@dataclass
class SearchResult:
    """
    FINAL OUTCOME returned when the generator is exhausted.

    Captured by the GUI via `except StopIteration as e: result = e.value`.
    The generator uses `return SearchResult(...)` which sets StopIteration.value.

    Fields:
      • success — did we find the goal?
      • path — the solution path (empty if no path)
      • cost — total path cost (sum of intermediate cell costs)
      • nodes_expanded — how many nodes we popped from frontier (performance metric)
    """

    success: bool
    path: list[tuple[int, int]] = field(default_factory=list)
    cost: float = 0.0
    nodes_expanded: int = 0


# ── helper ───────────────────────────────────────────────────────

def reconstruct_path(parent: dict, node: tuple) -> list[tuple[int, int]]:
    """
    Walk backwards from node through parent pointers to start.

    The parent dict maps each node → node we came from (or None for start).
    This is standard graph-search path reconstruction — O(path_length).
    Returns [start, ..., node] in forward order.

    Used by ALL algorithms — path reconstruction is identical regardless
    of how nodes were expanded (queue, stack, heap).
    """
    path = []
    cur = node
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()                 # reverse from backward-walk to forward order
    return path


# ═══════════════════════════════════════════════════════════════════
#  BFS — Breadth-First Search
# ═══════════════════════════════════════════════════════════════════

def bfs_search(world: GridWorld) -> Generator[SearchFrame, None, SearchResult]:
    """
    BFS expands nodes in FIFO order — layer by layer.

    DATA STRUCTURE: collections.deque (O(1) popleft + append)
    COMPLETENESS: Yes (finite graph, finite branching factor)
    OPTIMALITY: Yes for UNIFORM cost. No for non-uniform cost (bfs ignores costs).
    TIME: O(b^d) where b = branching factor, d = solution depth
    SPACE: O(b^d) — keeps entire frontier in memory

    KEY IMPLEMENTATION DETAILS:
      • Visited is populated on EXPANSION (when popped), not on generation.
        This is the standard "late goal test" — we yield the frame first,
        then check goal. GUI paints the goal cell, then search stops.
      • `n not in visited and n not in frontier` — the double-check prevents
        duplicate frontier entries. Without this, BFS could enqueue the same
        node multiple times (slower, correct but wasteful).
      • `visited.copy()` in SearchFrame — we copy because the GUI might hold
        the reference and visited continues to grow.
    """

    frontier = deque([world.start])              # queue: start goes in first
    visited: set[tuple[int, int]] = set()        # empty — start NOT pre-marked visited
    parent: dict[tuple[int, int], tuple | None] = {world.start: None}
    expanded = 0

    while frontier:
        current = frontier.popleft()             # O(1) — the "FIFO" in BFS
        expanded += 1
        visited.add(current)                     # mark visited ON EXPANSION

        # Yield frame BEFORE goal check — GUI sees the expansion animation
        yield SearchFrame(
            frontier=list(frontier),
            visited=visited.copy(),
            current=current,
            path=reconstruct_path(parent, current),
            g=float(len(reconstruct_path(parent, current))),
            h=world.heuristic(current),
        )

        # Late goal test: check AFTER yielding (GUI paints goal first)
        if current == world.goal:
            return SearchResult(
                success=True,
                path=reconstruct_path(parent, current),
                cost=len(reconstruct_path(parent, current)),
                nodes_expanded=expanded,
            )

        # Expand: add unvisited, unqueued neighbors to frontier
        for n in world.get_neighbors(current):          # URDL order
            if n not in visited and n not in frontier:
                parent[n] = current                     # record path
                frontier.append(n)                      # push to back of queue

    # Frontier exhausted — no path exists
    return SearchResult(success=False, nodes_expanded=expanded)


# ═══════════════════════════════════════════════════════════════════
#  DFS — Depth-First Search
# ═══════════════════════════════════════════════════════════════════

def dfs_search(world: GridWorld) -> Generator[SearchFrame, None, SearchResult]:
    """
    DFS expands nodes in LIFO order — deep first, then backtrack.

    DATA STRUCTURE: Python list as stack (O(1) pop + append)
    COMPLETENESS: Yes for FINITE graphs (with visited set). No for infinite graphs.
    OPTIMALITY: NO — DFS finds SOME path, not the best one.
    TIME: O(b^m) where m = maximum depth
    SPACE: O(b·m) — only stores path from root + siblings (better than BFS for deep graphs)

    CRITICAL IMPLEMENTATION — the `in_stack` set:
      Without it, DFS can push the same node onto the stack multiple times
      through different paths before it's popped. This causes parent pointer
      overwrites (last push wins, wrong path) and duplicate expansions.
      `in_stack` = nodes currently IN the stack (generated but not expanded).
      `visited` = nodes already popped and expanded.
      Together they partition all seen nodes: in_stack ∪ visited = all seen.

    KEY DIFFERENCE FROM BFS: DFS uses a stack (pop from end), BFS uses a
    queue (popleft from front). Otherwise the logic is almost identical —
    demonstrating that the data structure IS the algorithm's behavior.
    """

    stack = [world.start]                            # LIFO stack
    in_stack = {world.start}                         # nodes still in stack (prevents duplicates)
    visited: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple | None] = {world.start: None}
    expanded = 0

    while stack:
        current = stack.pop()                        # O(1) — LIFO: last-pushed = first-popped
        in_stack.discard(current)                    # no longer in stack
        expanded += 1
        visited.add(current)                         # mark visited on expansion

        yield SearchFrame(
            frontier=list(stack),
            visited=visited.copy(),
            current=current,
            path=reconstruct_path(parent, current),
            g=float(len(reconstruct_path(parent, current))),
            h=world.heuristic(current),
        )

        if current == world.goal:
            return SearchResult(
                success=True,
                path=reconstruct_path(parent, current),
                cost=len(reconstruct_path(parent, current)),
                nodes_expanded=expanded,
            )

        # Expand neighbors: push to stack (URDL order inverted by LIFO —
        # last-pushed = first-popped, so left goes first, then down, right, up)
        for n in world.get_neighbors(current):          # URDL
            if n not in visited and n not in in_stack:
                parent[n] = current                     # set ONCE per node
                in_stack.add(n)                         # now in stack
                stack.append(n)

    return SearchResult(success=False, nodes_expanded=expanded)


# ═══════════════════════════════════════════════════════════════════
#  DLS — Depth-Limited Search (DFS with depth cutoff)
# ═══════════════════════════════════════════════════════════════════

def dls_search(world: GridWorld, depth_limit: int) -> Generator[SearchFrame, None, SearchResult]:
    """
    DFS that refuses to expand nodes beyond a given depth.

    PARAMETER: depth_limit — search stops expanding at this depth.
    The GUI exposes this via a slider (1–20, default 10).

    KEY DIFFERENCE FROM DFS:
      • Stack stores (node, depth) tuples instead of bare nodes.
      • Before expanding a node: if depth >= depth_limit → don't generate children.
        The current node is still expanded (visited, yielded), but its children
        are not added to the stack. This is "depth cutoff" — not "skip the node."

    USE CASE: On preset A with limit=8, finds goal. With limit=4, fails.
    Demonstrates the completeness-vs-efficiency tradeoff of depth bounding.
    """

    stack: list[tuple[tuple[int, int], int]] = [(world.start, 0)]
    in_stack = {world.start}
    visited: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple | None] = {world.start: None}
    expanded = 0

    while stack:
        current, depth = stack.pop()
        in_stack.discard(current)
        expanded += 1
        visited.add(current)

        yield SearchFrame(
            frontier=[n for n, _ in stack],
            visited=visited.copy(),
            current=current,
            path=reconstruct_path(parent, current),
            g=float(depth),                             # g = current depth (cost proxy)
            h=world.heuristic(current),
            depth=depth,
            depth_limit=depth_limit,                    # signal to GUI: show depth info
        )

        if current == world.goal:
            return SearchResult(
                success=True,
                path=reconstruct_path(parent, current),
                cost=depth,
                nodes_expanded=expanded,
            )

        # CUTOFF: only generate children if we haven't hit the depth limit
        if depth < depth_limit:
            for n in world.get_neighbors(current):      # URDL
                if n not in visited and n not in in_stack:
                    parent[n] = current
                    in_stack.add(n)
                    stack.append((n, depth + 1))        # child depth = parent + 1

    return SearchResult(success=False, nodes_expanded=expanded)


# ═══════════════════════════════════════════════════════════════════
#  IDDFS — Iterative Deepening DFS
# ═══════════════════════════════════════════════════════════════════

def iddfs_search(world: GridWorld) -> Generator[SearchFrame, None, SearchResult]:
    """
    Repeatedly runs DLS with increasing depth limits: 1, 2, 3, ...

    WHY IT EXISTS: Combines DFS's space efficiency (O(b·d)) with BFS's
    optimality for uniform costs. Each iteration discards the tree and
    rebuilds from scratch — seems wasteful but the last iteration dominates:
    the total work ≈ b/(b−1) × work of last iteration ≈ constant factor.

    GENERATOR TRICK: `yield from dls_search(world, depth_limit)` delegates
    all intermediate frames to the DLS generator. The GUI sees each frame
    as if it came directly from IDDFS.

    SPECIAL SIGNAL: Before each DLS iteration, yields a frame with
    special="DEPTH_LIMIT_CHANGE" so the GUI can log/display the new limit.
    This frame has empty frontier/visited — it's just a signal.

    MAX DEPTH: 50 is a safety limit. On an 8×8 grid with no obstacles,
    the longest possible path is 14 steps (Manhattan distance). 50 is
    generous — prevents infinite loops while handling worst-case mazes.
    """

    for depth_limit in range(1, 50):
        # Signal to GUI: starting a new depth iteration
        yield SearchFrame(
            frontier=[], visited=set(), current=world.start,
            path=[], g=0.0, h=0.0,
            special="DEPTH_LIMIT_CHANGE", depth=depth_limit,
        )
        # Delegate to DLS — `yield from` forwards all frames AND the return value
        result = yield from dls_search(world, depth_limit)
        if result.success:
            return result

    return SearchResult(success=False)


# ═══════════════════════════════════════════════════════════════════
#  UCS — Uniform-Cost Search (Dijkstra's algorithm for pathfinding)
# ═══════════════════════════════════════════════════════════════════

def ucs_search(world: GridWorld) -> Generator[SearchFrame, None, SearchResult]:
    """
    Expands the node with the lowest PATH COST from start (g(n)).

    DATA STRUCTURE: heapq (priority queue) ordered by (g, counter, node).
    COMPLETENESS: Yes (finite graph).
    OPTIMALITY: YES — always finds the cheapest path, regardless of costs.
    TIME: O(b^(1+⌊C*/ε⌋)) where C* = optimal cost, ε = min edge cost
    SPACE: O(b^(1+⌊C*/ε⌋))

    KEY IMPLEMENTATION DETAILS:
      • `counter` is a tiebreaker — prevents comparing tuples when g is equal.
        Without it, Python compares nodes (tuples) which raises TypeError
        if the heap contains (cost, tuple_a) and (cost, tuple_b).
      • Stale-entry skip: `if current in visited: continue`. Since we may push
        a node multiple times with different costs, we skip any entry for a
        node that's already been expanded (the first expansion was cheapest).
      • `cost_so_far` tracks the best known g-value TO each node. When we find
        a cheaper path, we update parent AND re-push (the old entry becomes stale).
      • parent can be overwritten when a cheaper path is found — correct behavior.

    THE TEACHING POINT: On uniform-cost grids, UCS = BFS (both expand in
    cost order = layer order). On non-uniform grids (Presets B/C/D), UCS
    diverges — it may take MORE steps but LOWER total cost than BFS.
    """

    counter = 0                                          # tiebreaker for heapq
    frontier = [(0.0, counter, world.start)]             # (g, counter, node)
    counter += 1
    visited: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple | None] = {world.start: None}
    cost_so_far: dict[tuple[int, int], float] = {world.start: 0.0}
    expanded = 0

    while frontier:
        g, _, current = heapq.heappop(frontier)          # pop cheapest g-value

        # Stale-entry skip: if this node was already expanded at a CHEAPER cost,
        # this entry is stale — ignore it
        if current in visited:
            continue

        expanded += 1
        visited.add(current)

        yield SearchFrame(
            frontier=[n for _, _, n in frontier],        # extract nodes from (g,counter,node)
            visited=visited.copy(),
            current=current,
            path=reconstruct_path(parent, current),
            g=float(g),
            h=world.heuristic(current),
        )

        if current == world.goal:
            return SearchResult(
                success=True,
                path=reconstruct_path(parent, current),
                cost=g,
                nodes_expanded=expanded,
            )

        # Expand: push neighbors with updated cost
        for n in world.get_neighbors(current):          # URDL
            new_cost = g + world.get_cost(n)             # cost to reach neighbor via current
            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n] = new_cost                # record cheaper cost
                parent[n] = current                      # update path to this cheaper route
                heapq.heappush(frontier, (new_cost, counter, n))
                counter += 1

    return SearchResult(success=False, nodes_expanded=expanded)


# ═══════════════════════════════════════════════════════════════════
#  Greedy Best-First Search
# ═══════════════════════════════════════════════════════════════════

def greedy_search(world: GridWorld) -> Generator[SearchFrame, None, SearchResult]:
    """
    Expands the node closest to the goal by HEURISTIC alone: h(n).

    DATA STRUCTURE: heapq ordered by (h, row, col, counter, node).
    COMPLETENESS: Yes for finite graphs with visited set.
    OPTIMALITY: NO — ignores actual path cost, can produce expensive paths.

    TIE-BREAKING: (h, row, col, counter, node).
      • h: primary key — Manhattan distance to goal
      • row, col: lower row first, then lower column → DETERMINISTIC behavior
        Without this, heap ordering is non-deterministic when h is equal
        (Python compares nodes as tuples which DOES work but gives arbitrary
        ordering based on coordinate values).
      • counter: final tiebreak to avoid comparing nodes directly

    TEACHING POINT: Greedy often finds a path FASTER than A* (fewer expansions)
    but the path can be MUCH more expensive. Preset C demonstrates this:
    Greedy takes a muddy shortcut costing 17 while A*'s clean path costs 9.
    Heuristic alone is blind to cost — that's why we need f(n)=g(n)+h(n).
    """

    counter = 0
    h0 = world.heuristic(world.start)
    # Tie-break tuple: (h, row, col, counter, node)
    # row and col ensure deterministic ordering when h is equal
    frontier = [(h0, world.start[0], world.start[1], counter, world.start)]
    counter += 1
    visited: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple | None] = {world.start: None}
    expanded = 0

    while frontier:
        h, _, _, _, current = heapq.heappop(frontier)

        # Stale-entry skip (same pattern as UCS/A*)
        if current in visited:
            continue

        expanded += 1
        visited.add(current)

        yield SearchFrame(
            frontier=[n for _, _, _, _, n in frontier],
            visited=visited.copy(),
            current=current,
            path=reconstruct_path(parent, current),
            g=float(len(reconstruct_path(parent, current))),
            h=world.heuristic(current),
        )

        if current == world.goal:
            return SearchResult(
                success=True,
                path=reconstruct_path(parent, current),
                cost=len(reconstruct_path(parent, current)),
                nodes_expanded=expanded,
            )

        # Expand: push neighbors ordered by heuristic value
        for n in world.get_neighbors(current):          # URDL
            if n not in visited:
                parent[n] = current
                hn = world.heuristic(n)
                heapq.heappush(frontier, (hn, n[0], n[1], counter, n))
                counter += 1

    return SearchResult(success=False, nodes_expanded=expanded)


# ═══════════════════════════════════════════════════════════════════
#  A* — the gold standard: f(n) = g(n) + h(n)
# ═══════════════════════════════════════════════════════════════════

def astar_search(world: GridWorld) -> Generator[SearchFrame, None, SearchResult]:
    """
    Expands nodes by f(n) = g(n) + h(n): combines actual cost + heuristic.

    DATA STRUCTURE: heapq ordered by (f, counter, node).
    COMPLETENESS: Yes.
    OPTIMALITY: YES — provided the heuristic is ADMISSIBLE (never overestimates)
               and CONSISTENT (h(n) ≤ cost(n,n') + h(n')). Manhattan distance
               on a grid-move model satisfies both.

    KEY DIFFERENCE FROM UCS: UCS uses g only; A* adds h to bias toward goal.
    This makes A* expand FEWER nodes than UCS in practice while remaining optimal.

    KEY DIFFERENCE FROM GREEDY: Greedy uses h only; A* adds g to stay optimal.
    Without g, Greedy can be led into expensive shortcuts (mud, Preset C).

    IMPLEMENTATION:
      • g_score tracks best known cost-from-start to each node (like cost_so_far in UCS).
      • f = g + h is used for heap ordering (NOT stored — computed on push).
      • When a cheaper path is found, parent is updated and node is re-pushed
        with the new f-value. Old entry becomes stale and is skipped.
      • `counter` is the tiebreaker (prevents tuple comparison).
    """

    counter = 0
    h0 = world.heuristic(world.start)
    frontier = [(h0, counter, world.start)]              # (f = g+h, counter, node)
    counter += 1
    visited: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple | None] = {world.start: None}
    g_score: dict[tuple[int, int], float] = {world.start: 0.0}
    expanded = 0

    while frontier:
        f, _, current = heapq.heappop(frontier)          # pop best f-value

        # Stale-entry skip
        if current in visited:
            continue

        expanded += 1
        visited.add(current)
        g = g_score[current]                             # retrieve actual g for this node

        yield SearchFrame(
            frontier=[n for _, _, n in frontier],
            visited=visited.copy(),
            current=current,
            path=reconstruct_path(parent, current),
            g=float(g),
            h=world.heuristic(current),
        )

        if current == world.goal:
            return SearchResult(
                success=True,
                path=reconstruct_path(parent, current),
                cost=g,
                nodes_expanded=expanded,
            )

        # Expand: compute f for each neighbor, push to heap
        for n in world.get_neighbors(current):          # URDL
            tg = g + world.get_cost(n)                   # tentative g = current g + step cost
            if n not in g_score or tg < g_score[n]:
                g_score[n] = tg                          # record better g
                parent[n] = current                      # update path
                # f = g + h — the A* formula
                heapq.heappush(frontier, (tg + world.heuristic(n), counter, n))
                counter += 1

    return SearchResult(success=False, nodes_expanded=expanded)
