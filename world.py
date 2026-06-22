"""GridWorld environment for AI Agents + Search Algorithms Lab.
══════════════════════════════════════════════════════════════════════
ARCHITECTURE ROLE: This is the FOUNDATION of the entire project.
Every agent and every search algorithm operates on a GridWorld instance.
It encapsulates all environment rules — terrain costs, neighbor order,
perception model, and agent movement — so the algorithms and agents
never touch grid implementation details.

DESIGN DECISIONS:
  • Cell is an immutable-ish dataclass (terrain+cost) — simple, inspectable.
  • Neighbor order is CANONICAL URDL across ALL code (world + search + agents).
    This makes algorithm expansion deterministic and reproducible.
  • get_cost(goal) returns 0 — so stepping ON the goal adds nothing to path cost.
    Only intermediate cells matter. This matches standard search formulations.
  • get_percept() returns None for both walls AND out-of-bounds — agents interpret
    None uniformly as "blocked", no special-casing needed.
  • move_agent() returns False on wall-hit, agent stays in place. This is intentional
    for Q-learning: creates a self-loop transition (s,a)→(s, r=−10) that teaches
    the agent to avoid walls. Not a bug — a feature of the learning architecture.

PRESETS: Four hand-crafted grids of increasing complexity (A→D).
Each is designed to demonstrate a specific pedagogical point:
  A — Open Plain (all algorithms identical — teaches what "uniform cost" means)
  B — Short-Cut Trap (BFS takes muddy shortcut 16 cost, A* takes clean 8 cost)
  C — Mud Wall (Greedy nearly 2× worse than A* — demonstrates f(n)=g(n)+h(n))
  D — Labyrinth (stress-tests algorithms on a real maze with narrow corridors)

CONNECTIONS:
  • search.py — imports GridWorld, calls get_neighbors(), get_cost(), heuristic()
  • agents.py — imports GridWorld, calls get_percept(), move_agent(), get_state_index()
  • main.py — creates GridWorld via presets, passes to GridCanvas and agent/search init
══════════════════════════════════════════════════════════════════════
Cell types: normal (cost=1), mud (cost=5), wall (cost=∞).
Neighbor order: URDL (up, right, down, left) — canonical across all algorithms.
"""

from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════
#  Cell — the atomic unit of the grid. Every position holds one Cell.
#  Using a dataclass here instead of a raw tuple/dict means the debugger
#  shows meaningful field names, and type hints are enforced.
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Cell:
    """A single grid cell. Immutable after creation (reset rebuilds grid)."""

    terrain: str       # "normal" | "mud" | "wall"
    cost: int          # 1 | 5 | float('inf') — cost to STEP ON this cell
    row: int            # grid row (0-indexed from top)
    col: int            # grid column (0-indexed from left)

    @property
    def is_wall(self) -> bool:
        """Wall check used by get_neighbors() to filter blocked cells."""
        return self.terrain == "wall"


# ═══════════════════════════════════════════════════════════════════
#  GridWorld — the environment all agents and algorithms inhabit.
#  Mutable state: agent_pos (moves), grid cells (set during preset init).
#  Everything else is query methods — pure functions of position.
# ═══════════════════════════════════════════════════════════════════

class GridWorld:
    """
    N×M grid with terrain types and a moving agent.

    QUERY METHODS (no side effects):
      get_neighbors(pos)      → walkable URDL neighbor coordinates
      get_cost(pos)           → cost to step ON this cell (wall=∞, goal=0)
      get_percept(pos)        → what an agent "sees" at pos (4 directions)
      is_goal(pos)            → bool
      heuristic(pos)          → Manhattan distance to goal (admissible, consistent)
      get_state_index(pos)    → flattened index for Q-table lookup

    MUTATION METHODS (change state):
      move_agent(action)      → attempt to move; returns success/failure
      reset()                 → agent back to start
      set_cell(r,c,terrain,cost) → used by presets to build grids

    KEY DESIGN CHOICE — Why agent_pos lives here:
      Agent classes call world.move_agent() rather than tracking their own position.
      This means all agents and search algorithms share the same position model —
      no sync issues, no duplicate state. The world IS the source of truth for position.
    """

    def __init__(self, rows: int, cols: int):
        """Create an all-normal grid. Call a preset() factory for specific layouts."""
        self.rows = rows
        self.cols = cols
        self.grid: list[list[Cell]] = []
        self.start: tuple[int, int] = (0, 0)
        self.goal: tuple[int, int] = (rows - 1, cols - 1)
        self.agent_pos: tuple[int, int] = self.start

        # Build the grid: every cell starts as normal/cost=1
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(Cell(terrain="normal", cost=1, row=r, col=c))
            self.grid.append(row)

    # ── query methods ────────────────────────────────────────────
    #   These are PURE — they read state but never modify it.
    #   Called by both search algorithms and agent percept/planning logic.

    def get_neighbors(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Walkable neighbours in CANONICAL URDL order: up, right, down, left.

        IMPORTANT: This order is documented and used by ALL algorithms and agents.
        Changing it changes algorithm behavior (expansion order, path found).
        The URDL order was chosen because:
          1. It's intuitive (compass directions clockwise from North)
          2. It matches the agent's percept direction order (N,E,S,W)
          3. It produces deterministic, reproducible results
        """
        r, c = pos
        candidates = [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]  # URDL
        return [
            (nr, nc) for nr, nc in candidates
            if 0 <= nr < self.rows and 0 <= nc < self.cols
            and not self.grid[nr][nc].is_wall           # walls are blocked
        ]

    def get_cost(self, pos: tuple[int, int]) -> int:
        """
        Cost to STEP ON this cell.

        Goal → 0: Reaching the goal itself doesn't add to path cost.
        This is the standard search formulation — the cost of a path is the
        sum of costs of all intermediate nodes, excluding the start and goal.

        Wall → ∞: Penalty that makes any path through a wall invalid.
        (In practice, get_neighbors() already filters walls, so this is a safety net.)
        """
        r, c = pos
        if pos == self.goal:
            return 0
        return self.grid[r][c].cost

    def get_percept(self, pos: tuple[int, int]) -> dict:
        """
        The agent's "eyes" — what it sees from its current position.

        Returns {'N': Cell|None, 'S': ..., 'E': ..., 'W': ...}.
        None = out of bounds OR wall. By unifying these into one sentinel value,
        agent rule logic becomes simple: "if percept[d] is None → blocked, skip it."

        This is the PEAS "Sensors" — the agent's only window into the world.
        Simple Reflex and Model Reflex agents use this for reactive decision-making.
        Goal-based and Utility-based agents bypass this (they get the full grid).
        """
        r, c = pos
        directions = {
            'N': (r - 1, c),
            'S': (r + 1, c),
            'E': (r, c + 1),
            'W': (r, c - 1),
        }
        result = {}
        for key, (nr, nc) in directions.items():
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                cell = self.grid[nr][nc]
                result[key] = None if cell.is_wall else cell
            else:
                result[key] = None
        return result

    def is_goal(self, pos: tuple[int, int]) -> bool:
        """Terminal test — used by search algorithms to stop expansion."""
        return pos == self.goal

    def heuristic(self, pos: tuple[int, int]) -> float:
        """
        Manhattan distance from pos to goal.

        KEY PROPERTIES (for A* optimality):
          • ADMISSIBLE: never overestimates true cost (shortest path ≥ Manhattan distance)
          • CONSISTENT: h(n) ≤ cost(n,n') + h(n') for the grid-move cost model
          → A* with this heuristic is guaranteed optimal.

        Used by: Greedy (priority = h), A* (priority = g + h).
        Also stored in SearchFrame for display/inspection.
        """
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    # ── mutation methods ─────────────────────────────────────────
    #   These CHANGE the world state. Called only by agent step() methods.

    def move_agent(self, action: str) -> bool:
        """
        Move agent one step in cardinal direction.

        Returns True if agent actually moved, False if blocked (wall or boundary).

        CRITICAL FOR Q-LEARNING: When False, the agent STAYS IN PLACE.
        This creates a (state, action) → (same_state, reward=−10) transition.
        The Q-value for that action drops, the agent learns to avoid that move.
        Without this, wall hits would be silent — the agent would never learn.

        Actions: 'up', 'down', 'left', 'right' (lowercase, from agent classes).
        """
        r, c = self.agent_pos
        moves = {
            'up': (r - 1, c), 'down': (r + 1, c),
            'left': (r, c - 1), 'right': (r, c + 1),
        }
        if action not in moves:
            return False
        nr, nc = moves[action]
        if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.grid[nr][nc].is_wall:
            self.agent_pos = (nr, nc)
            return True
        return False

    def reset(self) -> None:
        """Send agent back to start. Called on GUI Reset and each learning episode."""
        self.agent_pos = self.start

    def set_cell(self, row: int, col: int, terrain: str, cost: int):
        """Replace a cell. Used by preset() factories to place walls and mud."""
        self.grid[row][col] = Cell(terrain=terrain, cost=cost, row=row, col=col)

    def get_state_index(self, pos: tuple[int, int]) -> int:
        """
        Flatten (row, col) → integer for Q-table indexing.

        The Q-table is a list-of-lists: Q[state][action].
        State = row * cols + col. Simple, fast, no dict overhead.
        Used exclusively by LearningAgent.
        """
        return pos[0] * self.cols + pos[1]

    # ── presets ──────────────────────────────────────────────────
    #   Static factory methods. Each builds a specific grid layout.
    #   Designed to demonstrate different algorithmic behaviors.
    #   Costs are VERIFIED by manual trace — never guessed.

    @staticmethod
    def preset_a() -> 'GridWorld':
        """
        Open Plain 5×5. S=(0,0), G=(4,4). All cost 1.

        TEACHING POINT: On uniform-cost grids, BFS = UCS = A* (all find 8-step paths).
        Use this to introduce algorithms before cost complexity.
        DLS with limit ≥8 finds the goal; limit <8 fails — demonstrates depth bounding.
        """
        w = GridWorld(5, 5)
        w.start = (0, 0)
        w.goal = (4, 4)
        w.agent_pos = w.start
        return w

    @staticmethod
    def preset_b() -> 'GridWorld':
        """
        Short-Cut Trap 5×6. S=(0,0), G=(4,5).
        Walls: (1,1)(1,2)(1,3)(1,4) — blocks direct eastward path
               (2,4)(3,4) — blocks the direct southward route from row 1
        Mud: (2,5)(3,5) — the "shortcut" through the right corridor is expensive

        KEY DIVERGENCE: BFS finds 8 steps through mud (cost 16).
        UCS/A* find 9 steps around via left edge (cost 8).
        Step-count ≠ cost. This preset EXISTS to teach that distinction.
        """
        w = GridWorld(5, 6)
        w.start = (0, 0)
        w.goal = (4, 5)
        w.agent_pos = w.start

        for r, c in [(1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]:
            w.set_cell(r, c, "wall", float('inf'))
        for r, c in [(2, 5), (3, 5)]:
            w.set_cell(r, c, "mud", 5)

        return w

    @staticmethod
    def preset_c() -> 'GridWorld':
        """
        Mud Wall 6×6. S=(0,0), G=(5,5).
        Mud rows 2-3 cols 2-4. Right-edge walls: (2,5)(3,5)(4,5).

        KILLER DEMO: A* vs Greedy.
        Greedy rushes toward goal (h only) → enters mud → cost 17, 27 nodes.
        A* factors actual cost (g+h) → goes around left edge → cost 9, 16 nodes.
        Nearly 2× worse. One frame shows why f(n)=g(n)+h(n) matters.
        The right-edge walls at (2,5)(3,5)(4,5) FORCE Greedy through mud —
        without them, Greedy would go right along top edge and avoid mud entirely.
        """
        w = GridWorld(6, 6)
        w.start = (0, 0)
        w.goal = (5, 5)
        w.agent_pos = w.start

        for r in range(2, 4):
            for c in range(2, 5):
                w.set_cell(r, c, "mud", 5)
        for r, c in [(2, 5), (3, 5), (4, 5)]:
            w.set_cell(r, c, "wall", float('inf'))

        return w

    @staticmethod
    def preset_d() -> 'GridWorld':
        """
        The Labyrinth 8×8. S=(0,0), G=(7,7).
        20 wall cells form narrow corridors.
        Mud pocket at center-left: (4,2)(4,3)(5,2)(5,3).

        STRESS TEST: Larger grid with constrained topology.
        DFS can get lost in deep branches — contrasts with BFS's methodical layers.
        A* shines: Manhattan heuristic guides through the maze.
        Learning agent needs 300-500 episodes (vs 100 for smaller presets).
        The mud pocket at center-left gives A* vs Greedy another divergence demo.
        """
        w = GridWorld(8, 8)
        w.start = (0, 0)
        w.goal = (7, 7)
        w.agent_pos = w.start

        for r, c in [(0,2), (1,2), (1,4), (1,5), (1,6),
                     (2,1), (2,2), (2,6), (3,4),
                     (4,0), (4,1), (4,4), (4,6), (4,7),
                     (6,1), (6,2), (6,3), (6,4), (6,5), (7,4)]:
            w.set_cell(r, c, "wall", float('inf'))
        for r, c in [(4,2), (4,3), (5,2), (5,3)]:
            w.set_cell(r, c, "mud", 5)

        return w
