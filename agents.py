"""AI Agent classes.
══════════════════════════════════════════════════════════════════════
ARCHITECTURE ROLE: This file implements the Russell & Norvig agent
hierarchy — five architectures, each adding ONE capability to the
decision pipeline. Presented as a progressive build:

  Simple Reflex   → Model Reflex   → Goal-based   → Utility-based   → Learning
  (react only)      (+ memory)       (+ planning)    (+ cost metric)   (+ experience)

This is the "PEAS" demonstration — each agent type shows a different
point on the reactivity ↔ deliberation spectrum.

UNIFIED INTERFACE: Every agent exposes:
  • step() → tuple[str, str]   — one decision cycle (action, rule_description)
    The return type MUST be tuple[str, str] across ALL agents.
    The GUI unpacks as `action, rule = agent.step()` — if any agent
    returns a bare string, the unpack crashes with ValueError.
  • Agent classes call world.move_agent() to execute actions.
    The world IS the source of truth for position — agents don't track
    their own position independently.

ARCHITECTURE PROGRESSION (teaching order):
  1. Simple Reflex  — percept → action. No memory. Loops in corridors.
  2. Model Reflex   — percept → update model → model+percept → action.
                       Avoids revisiting. Internal map grows visibly.
  3. Goal-based     — search → plan → execute. One-time full path planning.
                       Shows the power of deliberation.
  4. Utility-based  — same architecture, different metric (cost vs reachability).
                       "Is there a path?" vs "What's the BEST path?"
  5. Learning       — Q-table, ε-greedy. Learns from reward/penalty.
                       No prior knowledge — discovers the grid through experience.

CONNECTIONS:
  • world.py — imports GridWorld, calls get_percept(), move_agent(), get_state_index()
  • search.py — GoalBasedAgent and UtilityBasedAgent import search algorithms
  • main.py — creates agent instances, calls step() in the run loop

Five agent architectures demonstrating:
  simple reflex → model-based reflex → goal-based → utility-based → learning
"""

import random

from world import GridWorld
from search import bfs_search, dfs_search, ucs_search, astar_search


# ═══════════════════════════════════════════════════════════════════
#  1. SIMPLE REFLEX AGENT
#     Percept → Action. No internal state. Pure reactive.
#     The simplest possible intelligent agent.
# ═══════════════════════════════════════════════════════════════════

class SimpleReflexAgent:
    """
    REACTIVE agent: sees neighbors, picks the first open one.

    ARCHITECTURE: Percept → Rules → Action (no memory, no model).

    DECISION RULES (in priority order):
      1. Goal visible? → go there immediately
      2. First open neighbor in URDL order → go there
      3. All blocked → "stuck" (returns "stop")

    WEAKNESS: No memory — will revisit cells, loop in dead-end corridors,
    and oscillate between two positions. This is INTENTIONAL — it's the
    "before" picture that makes Model Reflex's memory the compelling "after."

    KEY DESIGN: URDL = ['N','E','S','W'] matches GridWorld's neighbor order.
    ACTION map translates compass directions to movement commands.
    """

    URDL = ['N', 'E', 'S', 'W']                     # matches GridWorld URDL neighbor order
    ACTION = {'N': 'up', 'E': 'right', 'S': 'down', 'W': 'left'}

    def __init__(self, world: GridWorld):
        self.world = world

    def perceive(self) -> dict:
        """Query the world for what's in each cardinal direction (PEAS: Sensors)."""
        return self.world.get_percept(self.world.agent_pos)

    def act(self, percept: dict) -> tuple[str, str]:
        """
        Condition-action rules (if-then).

        Returns (action, rule_name) — rule_name is for logging/debug display.
        The GUI shows "→ right (open:E)" so the presenter can explain
        which rule fired and why.
        """
        goal = self.world.goal

        # Rule 1: goal-visible — highest priority, win immediately
        for d in self.URDL:
            cell = percept[d]
            if cell is not None and (cell.row, cell.col) == goal:
                return (self.ACTION[d], "goal-visible")

        # Rule 2: open — first non-wall neighbor in URDL order
        for d in self.URDL:
            if percept[d] is not None:
                return (self.ACTION[d], f"open:{d}")

        # Rule 3: stuck — no open neighbors (surrounded by walls)
        return ("stop", "stuck")

    def step(self) -> tuple[str, str]:
        """
        One full decision cycle: perceive → decide → act.

        The act() method RETURNS the action; step() EXECUTES it via world.move_agent().
        This separation lets the GUI inspect the decision before execution if needed.

        Returns (action, rule) for GUI logging.
        """
        action, rule = self.act(self.perceive())
        if action != "stop":
            self.world.move_agent(action)
        return (action, rule)


# ═══════════════════════════════════════════════════════════════════
#  2. MODEL-BASED REFLEX AGENT
#     Percept → Update Model → Model + Percept → Action.
#     Adds MEMORY to Simple Reflex. Still reactive, but informed.
# ═══════════════════════════════════════════════════════════════════

class ModelReflexAgent:
    """
    REFLEX agent with INTERNAL STATE (memory).

    ADDS TO SIMPLE REFLEX:
      • visited set — cells the agent has been to
      • known_walls set — walls the agent has discovered
      • visit_order list — chronological record for backtracking

    ARCHITECTURE: Percept → update_model() → model + percept → act() → world.

    DECISION RULES (priority order):
      1. Goal visible? → go there
      2. Unvisited neighbor? → go there
      3. Fallback: backtrack to least-recently-visited reachable neighbor

    KEY DETAIL — Fallback logic:
      The agent scans visit_order from most-recent backwards (reversed).
      It tries the single most-recent position first (backtrack one step).
      If that's blocked, it stops ("stuck"). It does NOT scan the entire
      history — this is a deliberate simplification. A full backtrack would
      require search, which is what GoalBasedAgent is for.

    LIMITATION: Does NOT detect dead ends. If the agent enters a corridor
    that dead-ends, it will walk all the way in, then backtrack one step
    at a time (each step revisiting the dead end because it's "least recently
    visited"). This is visible behavior — a teaching point for why search
    matters.

    VISUAL: The internal map (visited + known walls) is painted on the canvas
    as green outlines, growing as the agent explores. This is the most visually
    compelling agent for "internal model" demonstrations.
    """

    URDL = ['N', 'E', 'S', 'W']
    ACTION = {'N': 'up', 'E': 'right', 'S': 'down', 'W': 'left'}
    OFFSETS = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}

    def __init__(self, world: GridWorld):
        self.world = world
        self.visited: set[tuple[int, int]] = set()        # cells the agent has occupied
        self.known_walls: set[tuple[int, int]] = set()    # walls discovered (via percept None)
        self.visit_order: list[tuple[int, int]] = []      # chronological visit history

    def perceive(self) -> dict:
        """Same sensor model as SimpleReflex — only sees immediate neighbors."""
        return self.world.get_percept(self.world.agent_pos)

    def update_model(self, percept: dict):
        """
        Update internal model from current percept.

        For each direction where percept is None, mark that position as a known wall.
        This is how the agent BUILDS its internal map incrementally — it starts
        knowing nothing and discovers walls as it encounters them (or their absence).
        """
        pos = self.world.agent_pos
        self.visited.add(pos)
        self.visit_order.append(pos)
        for d in self.URDL:
            if percept[d] is None:                      # wall or out-of-bounds
                dr, dc = self.OFFSETS[d]
                self.known_walls.add((pos[0] + dr, pos[1] + dc))

    def act(self, percept: dict) -> tuple[str, str]:
        """
        Decision logic with memory.

        Three-tier priority system: goal → unvisited → fallback.
        The fallback makes this agent "sticky" — it won't get permanently
        stuck in most configurations, unlike Simple Reflex which loops.
        """
        goal = self.world.goal
        pos = self.world.agent_pos

        # Rule 1: goal-visible (same as SimpleReflex)
        for d in self.URDL:
            cell = percept[d]
            if cell is not None and (cell.row, cell.col) == goal:
                return (self.ACTION[d], "goal-visible")

        # Rule 2: unvisited — prefer cells never seen before (exploration bias)
        for d in self.URDL:
            cell = percept[d]
            if cell is not None:
                npos = (cell.row, cell.col)
                if npos not in self.visited:
                    return (self.ACTION[d], f"unvisited:{d}")

        # Rule 3: fallback — backtrack to least-recently-visited neighbor
        # Scans visit_order from newest to oldest, tries the single most-recent
        # reachable neighbor. If that fails, agent is stuck.
        for past in reversed(self.visit_order):
            for d, (dr, dc) in self.OFFSETS.items():
                if (pos[0] + dr, pos[1] + dc) == past:
                    cell = percept.get(d)
                    if cell is not None:
                        return (self.ACTION[d], f"fallback:{d}")
            break   # only try the single most-recent (backtrack one step)

        return ("stop", "stuck")

    def step(self) -> tuple[str, str]:
        """
        Full cycle: perceive → update model → decide → act.

        Note the EXTRA step compared to SimpleReflex: update_model() runs
        between perceive() and act(). This is the architectural difference —
        the model is updated before the decision, so the decision uses the
        LATEST internal state.
        """
        percept = self.perceive()
        self.update_model(percept)
        action, rule = self.act(percept)
        if action != "stop":
            self.world.move_agent(action)
        return (action, rule)

    def get_internal_map(self) -> list[list[str]]:
        """
        Export the agent's internal model for visualization.

        Returns a grid of characters:
          'V' = visited (green outline on canvas)
          'W' = known wall
          '.' = unknown
        The GUI paints green borders on 'V' cells — showing the map
        growing as the agent explores. This is a compelling visual for
        "model-based" presentations.
        """
        rows, cols = self.world.rows, self.world.cols
        result = [['.' for _ in range(cols)] for _ in range(rows)]
        for (r, c) in self.visited:
            result[r][c] = 'V'
        for (r, c) in self.known_walls:
            if 0 <= r < rows and 0 <= c < cols:
                result[r][c] = 'W'
        return result


# ═══════════════════════════════════════════════════════════════════
#  3. GOAL-BASED AGENT
#     Search → Plan → Execute. Full deliberation before action.
#     Adds PLANNING to the agent architecture.
# ═══════════════════════════════════════════════════════════════════

class GoalBasedAgent:
    """
    DELIBERATIVE agent: runs a search algorithm to plan a full path,
    then executes it step-by-step.

    ARCHITECTURE: Search(goal) → Path → Execute(path).
    This is the first agent that "thinks before acting."

    KEY INSIGHT: A goal-based agent asks "does this reach the goal?" —
    a BINARY test. Any path to the goal satisfies it, regardless of cost.
    That's why BFS, DFS, UCS, and A* are all valid choices — they all
    produce SOME path. The choice affects which path, but the agent's
    goal is satisfied by any reachable path.

    ALGORITHM OPTIONS: bfs, dfs, ucs, astar (controlled by GUI dropdown).
    The agent calls the search generator, exhausts it (consuming all yields),
    and stores the final path from SearchResult.

    PRESENTER NOTE: Show Goal-based with BFS on Preset B (8-step muddy path,
    cost 16), then switch to A* (9-step clean path, cost 8). Both reach the
    goal — both satisfy the goal test. But the paths differ. This motivates
    UtilityBasedAgent — what if we want the BEST path, not just ANY path?
    """

    ALGOS = {
        'bfs': bfs_search,       # layer-by-layer, uniform-cost-optimal
        'dfs': dfs_search,       # deep-first, any path (not optimal)
        'ucs': ucs_search,       # cheapest-first, always cost-optimal
        'astar': astar_search,   # f=g+h, optimal with admissible heuristic
    }

    def __init__(self, world: GridWorld, algorithm: str = 'astar'):
        self.world = world
        self.algorithm = algorithm
        self.planned_path: list[tuple[int, int]] = []   # the computed path
        self.path_index = 0                              # current position in path

    def plan(self) -> None:
        """
        Run the search algorithm to completion, store the resulting path.

        Consumes the generator internally — yields intermediate frames
        for the algorithm to run but doesn't expose them (the agent doesn't
        need animation; the search algorithm's animation happens in Search mode).

        The path includes the start cell at index 0. The agent will skip
        past cells it's already on when executing.
        """
        fn = self.ALGOS[self.algorithm]
        gen = fn(self.world)
        result = None
        while True:
            try:
                next(gen)                            # consume all yields
            except StopIteration as e:
                result = e.value                     # capture SearchResult
                break
        if result and result.success:
            self.planned_path = result.path
            self.path_index = 0
        else:
            self.planned_path = []

    def step(self) -> tuple[str, str]:
        """
        Execute one step along the planned path.

        Skips past cells the agent is already on (important after reset
        or if the agent started at a position already in the path).

        Maps the difference between current position and next path target
        to a cardinal direction (up/down/left/right).
        """
        if self.path_index >= len(self.planned_path):
            return ("stop", "no path")

        # Advance past cells we're already on
        while (self.path_index < len(self.planned_path) and
               self.world.agent_pos == self.planned_path[self.path_index]):
            self.path_index += 1

        if self.path_index >= len(self.planned_path):
            return ("stop", "path complete")

        target = self.planned_path[self.path_index]
        r, c = self.world.agent_pos
        tr, tc = target

        # Determine cardinal direction from position delta
        if tr < r:      action = 'up'
        elif tr > r:    action = 'down'
        elif tc > c:    action = 'right'
        else:           action = 'left'

        self.world.move_agent(action)
        return (action, f"→step {self.path_index}/{len(self.planned_path)}")

    def has_path(self) -> bool:
        """Did planning succeed?"""
        return len(self.planned_path) > 0

    def get_planned_path(self) -> list[tuple[int, int]]:
        """Export path for GUI visualization (purple outline on canvas)."""
        return self.planned_path


# ═══════════════════════════════════════════════════════════════════
#  4. UTILITY-BASED AGENT
#     Same architecture as Goal-based, but optimizes for COST.
#     Adds a UTILITY FUNCTION to the decision pipeline.
# ═══════════════════════════════════════════════════════════════════

class UtilityBasedAgent(GoalBasedAgent):
    """
    COST-AWARE agent: same Search→Plan→Execute, but uses cost-optimal algorithms.

    INHERITS from GoalBasedAgent — the architecture is IDENTICAL.
    The only difference is the ALGORITHM CONSTRAINT (UCS/A* only) and
    the utility function (get_path_cost).

    KEY INSIGHT: A utility-based agent asks "what's the BEST path?" —
    it measures path quality via a cost function. Only cost-optimal
    algorithms (UCS, A*) make sense — BFS/DFS produce arbitrary paths
    and can't guarantee optimality.

    RESTRICTION: Constructor rejects non-cost-optimal algorithms.
    The GUI dropdown for Utility-based only shows UCS/A* (BFS/DFS blocked).

    get_path_cost() IS the utility function — it sums intermediate cell
    costs along the planned path. Lower = better. This is the numeric
    value the agent is optimizing.

    PRESENTER NOTE: On Preset B, compare GoalBased+BFS (cost 16) vs
    UtilityBased+A* (cost 8). Same architecture, same algorithm CAN work
    for both — the difference is in the METRIC: binary reachability vs
    continuous cost.
    """

    def __init__(self, world: GridWorld, algorithm: str = 'astar'):
        # Guard: only cost-optimal algorithms make sense for utility-based agents
        if algorithm not in ('ucs', 'astar'):
            algorithm = 'astar'
        super().__init__(world, algorithm)

    def get_path_cost(self) -> float:
        """
        THE UTILITY FUNCTION: sum of intermediate cell costs along the path.

        Skips index 0 (start cell — agent is already there).
        Sums world.get_cost() for each subsequent cell.
        On uniform grids: cost = path_length − 1.
        On mixed-cost grids: lower is better, mud avoidance shows in the sum.
        """
        if not self.planned_path:
            return 0.0
        return sum(self.world.get_cost(p) for p in self.planned_path[1:])


# ═══════════════════════════════════════════════════════════════════
#  5. LEARNING AGENT (Q-Learning)
#     Tabular Q-learning with ε-greedy exploration.
#     Learns from experience — no prior knowledge of the grid.
# ═══════════════════════════════════════════════════════════════════

class LearningAgent:
    """
    REINFORCEMENT LEARNING agent: discovers the optimal policy through trial and error.

    ALGORITHM: Tabular Q-learning (off-policy TD control).
      • State = flattened grid position (row * cols + col)
      • Action = one of 4 cardinal moves (URDL order)
      • Q-table = 2D list: Q[state][action]
      • Policy = ε-greedy (explore with probability ε, exploit best Q otherwise)

    UPDATE RULE: Q(s,a) ← Q(s,a) + α[R + γ·max_a' Q(s',a') − Q(s,a)]
      α (alpha) = learning rate (0.1) — how much new info overrides old
      γ (gamma) = discount factor (0.9) — how much future rewards matter
      ε (epsilon) = exploration rate (0.3 → 0.01, linear decay)

    REWARD MODEL:
      Goal reached       → +50  (sparse, terminal reward)
      Move to free cell  → −1   (step penalty — encourages short paths)
      Move to mud cell   → −5   (terrain penalty — learns to avoid mud)
      Hit wall / stuck   → −10  (heavy penalty — learns to avoid walls)

    EPISODE STRUCTURE:
      Each episode: start at 'S', act until goal or max_steps (rows×cols×2).
      The max_steps cutoff prevents infinite loops during exploration.

    STEP-BY-STEP API (for GUI animation — NOT a tight loop):
      start_episode() → sets up world, returns initial state
      step_episode(state) → one action + Q-update, returns (action, reward, next_state, done)
      end_episode() → decay epsilon, increment counter

      The GUI calls these one at a time so the canvas repaints between actions.
      train_all() is a convenience that runs all episodes in a tight loop
      (no animation, returns final reward history).

    KEY DESIGN DECISIONS:
      • Q-table initialized to ZEROS (optimistic in some formulations, neutral here).
      • ε decays linearly from initial_epsilon to 0.01 over all episodes.
      • Terminal state: Q(goal, any_action) = 0 (never updated after reaching goal —
        the agent doesn't take actions from goal state).
      • Wall hit = self-loop: (s,a)→(s,r=−10). The agent STAYS in the same state
        but receives the penalty. This is correct Q-learning — the Q-value for
        wall-directed actions drops, discouraging repeats.
      • Mud cost is the actual cell cost (5) — the agent learns terrain costs
        through experience, not from any prior knowledge.

    VISUAL: get_max_q_per_cell() collapses Q[state][4 actions] into max value
    per cell → painted as yellow heatmap on canvas. Intensity proportional to
    max Q value. Cells with max Q ≤ 0 are invisible — the heatmap only appears
    after the agent first reaches the goal and positive values propagate.
    """

    ACTIONS = ['up', 'right', 'down', 'left']           # matches URDL direction order

    def __init__(self, world: GridWorld, alpha=0.1, gamma=0.9,
                 epsilon=0.3, episodes=100):
        self.world = world
        self.alpha = alpha                                # learning rate
        self.gamma = gamma                                # discount factor
        self.initial_epsilon = epsilon                    # starting exploration rate
        self.epsilon = epsilon                            # current ε (decays over time)
        self.episodes = episodes                          # total training episodes
        self.episode_count = 0                            # episodes completed so far

        # Q-table: one row per state, one column per action
        n_states = world.rows * world.cols
        self.q_table = [[0.0] * 4 for _ in range(n_states)]
        self.reward_history: list[float] = []             # total reward per episode

    # ── core Q-learning ───────────────────────────────────────

    def get_state(self) -> int:
        """Current agent position as a flat state index."""
        return self.world.get_state_index(self.world.agent_pos)

    def act(self, state: int) -> str:
        """
        ε-greedy action selection.

        With probability ε: random action (explore).
        Otherwise: best action(s) by Q-value, random tie-break among equals.
        """
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)           # explore
        qs = self.q_table[state]
        mx = max(qs)
        best = [a for a, q in zip(self.ACTIONS, qs) if q == mx]
        return random.choice(best)                       # exploit (random tie-break)

    def learn(self, state: int, action: str,
              reward: float, next_state: int):
        """
        Q-learning update: Q(s,a) ← Q(s,a) + α[R + γ·max Q(s') − Q(s,a)].

        This is the TD(0) update — bootstraps from the max Q-value of the
        NEXT state (off-policy: uses max, not the action actually taken next).
        """
        ai = self.ACTIONS.index(action)
        old = self.q_table[state][ai]
        nxt = max(self.q_table[next_state])
        self.q_table[state][ai] = old + self.alpha * (
            reward + self.gamma * nxt - old)

    # ── episode (step-by-step, for GUI animation) ────────────

    def start_episode(self) -> int:
        """
        Begin a new episode. Resets agent to start, initializes counters.

        The episode limit (rows×cols×2) prevents infinite loops —
        on an 8×8 grid that's 128 steps max per episode, which is enough
        for exploration but cuts off truly stuck agents.
        """
        self.world.reset()
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._episode_limit = self.world.rows * self.world.cols * 2
        self._episode_done = False
        return self.get_state()

    def step_episode(self, state: int) -> tuple[str, float, int, bool]:
        """
        ONE action within an episode. Called by GUI on each Step click.

        Returns (action, reward, next_state, done).
        The GUI uses these to: paint the canvas, log the action, decide
        whether to call end_episode() or continue.

        Reward logic (order matters — goal check first):
          1. Goal reached → +50, mark episode done
          2. Wall hit (move returned False) → −10 (stay in same state)
          3. Otherwise: reward = −cell_cost (normal=−1, mud=−5)
        """
        if self._episode_done:
            return ("stop", 0.0, state, True)

        action = self.act(state)
        moved = self.world.move_agent(action)            # attempt move
        nxt = self.get_state()

        # Determine reward based on outcome
        if self.world.is_goal(self.world.agent_pos):
            reward = 50.0                                # WIN! large positive reward
            self._episode_done = True
        elif not moved:
            reward = -10.0                               # wall penalty (stay in place)
        else:
            # Terrain-based penalty: mud costs more than normal
            # Agent learns to prefer normal cells via Q-value differences
            cell_cost = self.world.get_cost(self.world.agent_pos)
            reward = -float(cell_cost)

        self.learn(state, action, reward, nxt)           # Q-learning update
        self._episode_reward += reward
        self._episode_steps += 1

        # Step cutoff — prevent infinite loops
        if self._episode_steps >= self._episode_limit:
            self._episode_done = True

        return (action, reward, nxt, self._episode_done)

    def end_episode(self):
        """
        Wrap up the episode: decay ε, increment counter, reset position.

        ε DECAY: linear from initial_epsilon to 0.01 over all episodes.
        progress = episodes_done / total_episodes (0.0 to 1.0).
        ε = max(0.01, initial_epsilon × (1.0 − progress)).
        At episode 1: ε ≈ 0.3 (mostly explore). At final episode: ε = 0.01 (mostly exploit).
        """
        self.episode_count += 1
        progress = min(self.episode_count / self.episodes, 1.0)
        self.epsilon = max(0.01, self.initial_epsilon * (1.0 - progress))
        self.world.reset()
        return self._episode_reward

    def train_all(self) -> list[float]:
        """
        Run ALL episodes in a tight loop (no animation, no GUI updates).

        This is the "bulk training" mode — click Train All, wait, see results.
        The GUI disables Step during this, shows the Q-heatmap after completion.
        Uses the same start/step/end API internally for consistency.
        """
        self.reward_history = []
        for _ in range(self.episodes):
            if self.episode_count >= self.episodes:      # guard: don't over-train
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
        """Wipe Q-table and counters. Fresh start for re-training."""
        n = self.world.rows * self.world.cols
        self.q_table = [[0.0] * 4 for _ in range(n)]
        self.episode_count = 0
        self.epsilon = self.initial_epsilon
        self.reward_history = []

    def reset_position(self) -> None:
        """Move agent to start, keep Q-table intact (for demo after training)."""
        self.world.reset()

    def get_max_q_per_cell(self) -> list[list[float]]:
        """
        Collapse Q-table to one value per cell: max(Q[state][all actions]).

        Used by the GUI to paint the Q-value heatmap.
        Cells with max Q ≤ 0 are invisible — the heatmap only appears
        after positive Q-values propagate from the goal.
        """
        result = []
        for r in range(self.world.rows):
            row = []
            for c in range(self.world.cols):
                s = r * self.world.cols + c
                row.append(max(self.q_table[s]))
            result.append(row)
        return result

    def get_q_table(self) -> list[list[float]]:
        """Raw Q-table for debugging/inspection."""
        return self.q_table

    def get_epsilon(self) -> float:
        """Current exploration rate (for GUI info panel)."""
        return self.epsilon
