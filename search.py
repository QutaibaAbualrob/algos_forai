"""Search algorithms as generators.

Each yields SearchFrame after every node expansion and returns SearchResult.
Neighbor order: URDL (up, right, down, left) — canonical, deterministic.
"""

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Generator

from world import GridWorld


@dataclass
class SearchFrame:
    """Yielded after each node expansion.  GUI reads this to paint canvas."""
    frontier: list[tuple[int, int]]
    visited: set[tuple[int, int]]
    current: tuple[int, int]
    path: list[tuple[int, int]]        # best path to current node
    g: float = 0.0                      # path cost so far
    h: float = 0.0                      # heuristic to goal
    depth: int = 0                      # current depth (DLS / IDDFS)
    depth_limit: int | None = None      # active limit (DLS / IDDFS only)
    special: str | None = None          # "DEPTH_LIMIT_CHANGE" for IDDFS


@dataclass
class SearchResult:
    success: bool
    path: list[tuple[int, int]] = field(default_factory=list)
    cost: float = 0.0
    nodes_expanded: int = 0


# ── helper ───────────────────────────────────────────────────────

def reconstruct_path(parent: dict, node: tuple) -> list[tuple[int, int]]:
    path = []
    cur = node
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


# ── BFS (queue) ──────────────────────────────────────────────────

def bfs_search(world: GridWorld) -> Generator[SearchFrame, None, SearchResult]:
    frontier = deque([world.start])
    visited: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple | None] = {world.start: None}
    expanded = 0

    while frontier:
        current = frontier.popleft()
        expanded += 1
        visited.add(current)

        yield SearchFrame(
            frontier=list(frontier),
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

        for n in world.get_neighbors(current):          # URDL
            if n not in visited and n not in frontier:
                parent[n] = current
                frontier.append(n)

    return SearchResult(success=False, nodes_expanded=expanded)


# ── DFS (stack) ──────────────────────────────────────────────────

def dfs_search(world: GridWorld) -> Generator[SearchFrame, None, SearchResult]:
    stack = [world.start]
    in_stack = {world.start}
    visited: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple | None] = {world.start: None}
    expanded = 0

    while stack:
        current = stack.pop()
        in_stack.discard(current)
        expanded += 1
        visited.add(current)

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

        for n in world.get_neighbors(current):          # URDL
            if n not in visited and n not in in_stack:
                parent[n] = current                     # set once
                in_stack.add(n)
                stack.append(n)

    return SearchResult(success=False, nodes_expanded=expanded)


# ── DLS (depth-limited DFS) ─────────────────────────────────────

def dls_search(world: GridWorld, depth_limit: int) -> Generator[SearchFrame, None, SearchResult]:
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
            g=float(depth),
            h=world.heuristic(current),
            depth=depth,
            depth_limit=depth_limit,
        )

        if current == world.goal:
            return SearchResult(
                success=True,
                path=reconstruct_path(parent, current),
                cost=depth,
                nodes_expanded=expanded,
            )

        if depth < depth_limit:
            for n in world.get_neighbors(current):      # URDL
                if n not in visited and n not in in_stack:
                    parent[n] = current
                    in_stack.add(n)
                    stack.append((n, depth + 1))

    return SearchResult(success=False, nodes_expanded=expanded)


# ── IDDFS (iterative deepening) ──────────────────────────────────

def iddfs_search(world: GridWorld) -> Generator[SearchFrame, None, SearchResult]:
    for depth_limit in range(1, 50):
        yield SearchFrame(
            frontier=[], visited=set(), current=world.start,
            path=[], g=0.0, h=0.0,
            special="DEPTH_LIMIT_CHANGE", depth=depth_limit,
        )
        result = yield from dls_search(world, depth_limit)
        if result.success:
            return result
    return SearchResult(success=False)


# ── UCS (uniform cost) ──────────────────────────────────────────

def ucs_search(world: GridWorld) -> Generator[SearchFrame, None, SearchResult]:
    counter = 0
    frontier = [(0.0, counter, world.start)]
    counter += 1
    visited: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple | None] = {world.start: None}
    cost_so_far: dict[tuple[int, int], float] = {world.start: 0.0}
    expanded = 0

    while frontier:
        g, _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        expanded += 1
        visited.add(current)

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

        for n in world.get_neighbors(current):          # URDL
            new_cost = g + world.get_cost(n)
            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n] = new_cost
                parent[n] = current
                heapq.heappush(frontier, (new_cost, counter, n))
                counter += 1

    return SearchResult(success=False, nodes_expanded=expanded)


# ── Greedy Best-First ───────────────────────────────────────────

def greedy_search(world: GridWorld) -> Generator[SearchFrame, None, SearchResult]:
    counter = 0
    h0 = world.heuristic(world.start)
    # tie-break: (h, row, col, counter, node)
    frontier = [(h0, world.start[0], world.start[1], counter, world.start)]
    counter += 1
    visited: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple | None] = {world.start: None}
    expanded = 0

    while frontier:
        h, _, _, _, current = heapq.heappop(frontier)
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

        for n in world.get_neighbors(current):          # URDL
            if n not in visited:
                parent[n] = current
                hn = world.heuristic(n)
                heapq.heappush(frontier, (hn, n[0], n[1], counter, n))
                counter += 1

    return SearchResult(success=False, nodes_expanded=expanded)


# ── A* ──────────────────────────────────────────────────────────

def astar_search(world: GridWorld) -> Generator[SearchFrame, None, SearchResult]:
    counter = 0
    h0 = world.heuristic(world.start)
    frontier = [(h0, counter, world.start)]
    counter += 1
    visited: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple | None] = {world.start: None}
    g_score: dict[tuple[int, int], float] = {world.start: 0.0}
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

        for n in world.get_neighbors(current):          # URDL
            tg = g + world.get_cost(n)
            if n not in g_score or tg < g_score[n]:
                g_score[n] = tg
                parent[n] = current
                heapq.heappush(frontier, (tg + world.heuristic(n), counter, n))
                counter += 1

    return SearchResult(success=False, nodes_expanded=expanded)
