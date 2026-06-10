"""GridWorld environment for AI Agents + Search Algorithms Lab.

Cell types: normal (cost=1), mud (cost=5), wall (cost=inf).
Neighbor order: URDL (up, right, down, left) — canonical across all algorithms.
"""

from dataclasses import dataclass


@dataclass
class Cell:
    terrain: str       # "normal" | "mud" | "wall"
    cost: int          # 1 | 5 | float('inf')
    row: int
    col: int

    @property
    def is_wall(self) -> bool:
        return self.terrain == "wall"


class GridWorld:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.grid: list[list[Cell]] = []
        self.start: tuple[int, int] = (0, 0)
        self.goal: tuple[int, int] = (rows - 1, cols - 1)
        self.agent_pos: tuple[int, int] = self.start

        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(Cell(terrain="normal", cost=1, row=r, col=c))
            self.grid.append(row)

    # ── query methods ────────────────────────────────────────────

    def get_neighbors(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        """Walkable neighbours in URDL order: up, right, down, left."""
        r, c = pos
        candidates = [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]  # URDL
        return [
            (nr, nc) for nr, nc in candidates
            if 0 <= nr < self.rows and 0 <= nc < self.cols
            and not self.grid[nr][nc].is_wall
        ]

    def get_cost(self, pos: tuple[int, int]) -> int:
        """Cost to step ON this cell. Walls return inf. Goal returns 0."""
        r, c = pos
        if pos == self.goal:
            return 0
        return self.grid[r][c].cost

    def get_percept(self, pos: tuple[int, int]) -> dict:
        """Returns {'N': Cell|None, 'S': ..., 'E': ..., 'W': ...}.
        None = out of bounds or wall."""
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
        return pos == self.goal

    def heuristic(self, pos: tuple[int, int]) -> float:
        """Manhattan distance from pos to goal."""
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    # ── mutation methods ─────────────────────────────────────────

    def move_agent(self, action: str) -> bool:
        """Move agent one step. Returns True if moved, False if blocked."""
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
        self.agent_pos = self.start

    def set_cell(self, row: int, col: int, terrain: str, cost: int):
        self.grid[row][col] = Cell(terrain=terrain, cost=cost, row=row, col=col)

    def get_state_index(self, pos: tuple[int, int]) -> int:
        """Flattened index for Q-learning: row * cols + col."""
        return pos[0] * self.cols + pos[1]

    # ── presets ──────────────────────────────────────────────────

    @staticmethod
    def preset_a() -> 'GridWorld':
        """Open Plain 5x5.  S=(0,0), G=(4,4).  All cost 1."""
        w = GridWorld(5, 5)
        w.start = (0, 0)
        w.goal = (4, 4)
        w.agent_pos = w.start
        return w

    @staticmethod
    def preset_b() -> 'GridWorld':
        """Short-Cut Trap 5x6.  S=(0,0), G=(4,5).
        Walls: (1,1)(1,2)(1,3)(1,4)(2,4)(3,4).  Mud: (2,5)(3,5)."""
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
        """Mud Wall 6x6.  S=(0,0), G=(5,5).
        Mud rows 2-3 cols 2-4.  Right-edge walls: (2,5)(3,5)(4,5)."""
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
        """The Labyrinth 8x8.  S=(0,0), G=(7,7).
        Walls: 20 cells forming corridors.  Mud pocket: (4,2)(4,3)(5,2)(5,3)."""
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
