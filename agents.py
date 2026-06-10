"""AI Agent classes.

Five agent architectures demonstrating:
  simple reflex → model-based reflex → goal-based → utility-based → learning
"""

import random

from world import GridWorld
from search import bfs_search, dfs_search, ucs_search, astar_search


# ── Simple Reflex ────────────────────────────────────────────────

class SimpleReflexAgent:
    """Percept → Action.  No memory.  Pure reactive.

    Checks URDL neighbours; picks the first non-wall.  If goal is
    visible in any direction, goes there immediately.
    """

    URDL = ['N', 'E', 'S', 'W']                     # matches GridWorld URDL
    ACTION = {'N': 'up', 'E': 'right', 'S': 'down', 'W': 'left'}

    def __init__(self, world: GridWorld):
        self.world = world

    def perceive(self) -> dict:
        return self.world.get_percept(self.world.agent_pos)

    def act(self, percept: dict) -> tuple[str, str]:
        goal = self.world.goal

        # 1. goal visible?
        for d in self.URDL:
            cell = percept[d]
            if cell is not None and (cell.row, cell.col) == goal:
                return (self.ACTION[d], "goal-visible")

        # 2. first open neighbour (URDL)
        for d in self.URDL:
            if percept[d] is not None:
                return (self.ACTION[d], f"open:{d}")

        return ("stop", "stuck")

    def step(self) -> tuple[str, str]:
        action, rule = self.act(self.perceive())
        if action != "stop":
            self.world.move_agent(action)
        return (action, rule)


# ── Model-based Reflex ──────────────────────────────────────────

class ModelReflexAgent:
    """Percept → update model → model + percept → action.

    Maintains visited set and known-wall set, built incrementally.
    Fallback: least-recently-visited neighbour when all others blocked.
    """

    URDL = ['N', 'E', 'S', 'W']
    ACTION = {'N': 'up', 'E': 'right', 'S': 'down', 'W': 'left'}
    OFFSETS = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}

    def __init__(self, world: GridWorld):
        self.world = world
        self.visited: set[tuple[int, int]] = set()
        self.known_walls: set[tuple[int, int]] = set()
        self.visit_order: list[tuple[int, int]] = []

    def perceive(self) -> dict:
        return self.world.get_percept(self.world.agent_pos)

    def update_model(self, percept: dict):
        pos = self.world.agent_pos
        self.visited.add(pos)
        self.visit_order.append(pos)
        for d in self.URDL:
            if percept[d] is None:
                dr, dc = self.OFFSETS[d]
                self.known_walls.add((pos[0] + dr, pos[1] + dc))

    def act(self, percept: dict) -> tuple[str, str]:
        goal = self.world.goal
        pos = self.world.agent_pos

        # 1. goal visible?
        for d in self.URDL:
            cell = percept[d]
            if cell is not None and (cell.row, cell.col) == goal:
                return (self.ACTION[d], "goal-visible")

        # 2. unvisited non-wall neighbour
        for d in self.URDL:
            cell = percept[d]
            if cell is not None:
                npos = (cell.row, cell.col)
                if npos not in self.visited:
                    return (self.ACTION[d], f"unvisited:{d}")

        # 3. fallback — least-recently-visited reachable neighbour
        for past in reversed(self.visit_order):
            for d, (dr, dc) in self.OFFSETS.items():
                if (pos[0] + dr, pos[1] + dc) == past:
                    cell = percept.get(d)
                    if cell is not None:
                        return (self.ACTION[d], f"fallback:{d}")
            break   # only try the single most-recent (backtrack one step)

        return ("stop", "stuck")

    def step(self) -> tuple[str, str]:
        percept = self.perceive()
        self.update_model(percept)
        action, rule = self.act(percept)
        if action != "stop":
            self.world.move_agent(action)
        return (action, rule)

    def get_internal_map(self) -> list[list[str]]:
        """Returns grid of 'V' (visited), 'W' (known wall), '.' (unknown)."""
        rows, cols = self.world.rows, self.world.cols
        result = [['.' for _ in range(cols)] for _ in range(rows)]
        for (r, c) in self.visited:
            result[r][c] = 'V'
        for (r, c) in self.known_walls:
            if 0 <= r < rows and 0 <= c < cols:
                result[r][c] = 'W'
        return result


# ── Goal-based ───────────────────────────────────────────────────

class GoalBasedAgent:
    """Search → Plan → Execute.  Calls search.py functions.

    Plans the full path once, then executes it step-by-step.
    """

    ALGOS = {
        'bfs': bfs_search, 'dfs': dfs_search,
        'ucs': ucs_search, 'astar': astar_search,
    }

    def __init__(self, world: GridWorld, algorithm: str = 'astar'):
        self.world = world
        self.algorithm = algorithm
        self.planned_path: list[tuple[int, int]] = []
        self.path_index = 0

    def plan(self) -> None:
        fn = self.ALGOS[self.algorithm]
        gen = fn(self.world)
        result = None
        while True:
            try:
                next(gen)
            except StopIteration as e:
                result = e.value
                break
        if result and result.success:
            self.planned_path = result.path
            self.path_index = 0
        else:
            self.planned_path = []

    def step(self) -> tuple[str, str]:
        if self.path_index >= len(self.planned_path):
            return ("stop", "no path")

        # advance past cells we're already on
        while (self.path_index < len(self.planned_path) and
               self.world.agent_pos == self.planned_path[self.path_index]):
            self.path_index += 1

        if self.path_index >= len(self.planned_path):
            return ("stop", "path complete")

        target = self.planned_path[self.path_index]
        r, c = self.world.agent_pos
        tr, tc = target

        if tr < r:      action = 'up'
        elif tr > r:    action = 'down'
        elif tc > c:    action = 'right'
        else:           action = 'left'

        self.world.move_agent(action)
        return (action, f"→step {self.path_index}/{len(self.planned_path)}")

    def has_path(self) -> bool:
        return len(self.planned_path) > 0

    def get_planned_path(self) -> list[tuple[int, int]]:
        return self.planned_path


# ── Utility-based ────────────────────────────────────────────────

class UtilityBasedAgent(GoalBasedAgent):
    """Same architecture as GoalBased, but restricted to cost-optimising
    search (UCS / A*)."""

    def __init__(self, world: GridWorld, algorithm: str = 'astar'):
        if algorithm not in ('ucs', 'astar'):
            algorithm = 'astar'
        super().__init__(world, algorithm)

    def get_path_cost(self) -> float:
        if not self.planned_path:
            return 0.0
        return sum(self.world.get_cost(p) for p in self.planned_path[1:])


# ── Learning (Q-learning) ────────────────────────────────────────

class LearningAgent:
    """Tabular Q-learning with ε-greedy exploration.

    State  = flattened (row * cols + col).
    Action = up / right / down / left (URDL order).
    """

    ACTIONS = ['up', 'right', 'down', 'left']

    def __init__(self, world: GridWorld, alpha=0.1, gamma=0.9,
                 epsilon=0.3, episodes=100):
        self.world = world
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.episodes = episodes
        self.episode_count = 0

        n_states = world.rows * world.cols
        self.q_table = [[0.0] * 4 for _ in range(n_states)]
        self.reward_history: list[float] = []

    # ── core ─────────────────────────────────────────────────

    def get_state(self) -> int:
        return self.world.get_state_index(self.world.agent_pos)

    def act(self, state: int) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)
        qs = self.q_table[state]
        mx = max(qs)
        best = [a for a, q in zip(self.ACTIONS, qs) if q == mx]
        return random.choice(best)

    def learn(self, state: int, action: str,
              reward: float, next_state: int):
        ai = self.ACTIONS.index(action)
        old = self.q_table[state][ai]
        nxt = max(self.q_table[next_state])
        self.q_table[state][ai] = old + self.alpha * (
            reward + self.gamma * nxt - old)

    # ── episode (step-by-step) ──────────────────────────────

    def start_episode(self) -> int:
        """Begin a new episode. Returns initial state index."""
        self.world.reset()
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._episode_limit = self.world.rows * self.world.cols * 2
        self._episode_done = False
        return self.get_state()

    def step_episode(self, state: int) -> tuple[str, float, int, bool]:
        """Take one action. Returns (action, reward, next_state, done)."""
        if self._episode_done:
            return ("stop", 0.0, state, True)

        action = self.act(state)
        moved = self.world.move_agent(action)
        nxt = self.get_state()

        if self.world.is_goal(self.world.agent_pos):
            reward = 50.0
            self._episode_done = True
        elif not moved:
            reward = -10.0
        else:
            # mud cells cost more than normal — agent learns to avoid them
            cell_cost = self.world.get_cost(self.world.agent_pos)
            reward = -float(cell_cost)

        self.learn(state, action, reward, nxt)
        self._episode_reward += reward
        self._episode_steps += 1

        if self._episode_steps >= self._episode_limit:
            self._episode_done = True

        return (action, reward, nxt, self._episode_done)

    def end_episode(self):
        """Finish episode: decay epsilon, inc counter, reset position."""
        self.episode_count += 1
        progress = min(self.episode_count / self.episodes, 1.0)
        self.epsilon = max(0.01, self.initial_epsilon * (1.0 - progress))
        self.world.reset()
        return self._episode_reward

    def train_all(self) -> list[float]:
        self.reward_history = []
        for _ in range(self.episodes):
            if self.episode_count >= self.episodes:
                break
            state = self.start_episode()
            while True:
                _, _, state, done = self.step_episode(state)
                if done:
                    break
            reward = self.end_episode()
            self.reward_history.append(reward)
        return self.reward_history

    # ── GUI accessors ────────────────────────────────────────

    def reset_q(self) -> None:
        n = self.world.rows * self.world.cols
        self.q_table = [[0.0] * 4 for _ in range(n)]
        self.episode_count = 0
        self.epsilon = self.initial_epsilon
        self.reward_history = []

    def reset_position(self) -> None:
        self.world.reset()

    def get_max_q_per_cell(self) -> list[list[float]]:
        result = []
        for r in range(self.world.rows):
            row = []
            for c in range(self.world.cols):
                s = r * self.world.cols + c
                row.append(max(self.q_table[s]))
            result.append(row)
        return result

    def get_q_table(self) -> list[list[float]]:
        return self.q_table

    def get_epsilon(self) -> float:
        return self.epsilon
