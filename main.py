"""Tkinter GUI — AI Agents + Search Algorithms Lab.
══════════════════════════════════════════════════════════════════════
ARCHITECTURE ROLE: This is the FRONTEND — the visual layer that
consumes everything from world.py, search.py, and agents.py.
It's a single-file Tkinter application with zero external dependencies.

THREE MODES (selected via dropdown):
  AGENT   — Pick an agent type → watch it navigate the grid step-by-step.
            Shows agent architecture at work: percept → rules → action.
  SEARCH  — Pick an algorithm → watch it explore the grid node-by-node.
            Shows frontier expansion, visited set, and final path.
  COMPARE — Two algorithms side-by-side on the same grid, synchronized.
            Each canvas shows its own algorithm state; one Step advances both.

DESIGN PRINCIPLES:
  • Generator-driven animation: search algorithms yield SearchFrames;
    the GUI calls next(gen) on each Step, paints the canvas, then waits.
  • Run loop uses root.after(delay, callback) — non-blocking, keeps UI
    responsive. delay = speed slider value (50-1000ms).
  • Canvas uses LAYERED rendering: draw_grid() for terrain base,
    update_overlay() for dynamic state (visited, frontier, path, agent).
  • Keyboard shortcuts for fast demos: Space=Step, P=Play/Pause, R=Reset, 1-3=presets.

CLASS STRUCTURE:
  GridCanvas(tk.Canvas) — grid rendering + overlay painting
  App(tk.Tk)            — root window, toolbar, mode state machine, run loop

STATE MANAGEMENT (App instance variables):
  mode: 'agent' | 'search' | 'compare'
  agent_type / search_algo / cmp_a / cmp_b — selected algorithm/agent
  generator / gen_a / gen_b — active search generators (None when done)
  running — Play/Pause flag
  agent — active agent instance (or None)
  world — current GridWorld (mutated by agents, read by search algos)

Keyboard: Space=Step  P=Play/Pause  R=Reset  1/2/3=presets.
Logs: written to logs/ folder (one file per session).
"""

import tkinter as tk
from tkinter import ttk

from world import GridWorld
from search import (
    bfs_search, dfs_search, dls_search, iddfs_search,
    ucs_search, greedy_search, astar_search,
)
from agents import (
    SimpleReflexAgent, ModelReflexAgent, GoalBasedAgent,
    UtilityBasedAgent, LearningAgent,
)
from logger import SessionLogger


# ═══════════════════════════════════════════════════════════════════
#  GridCanvas — draws the grid + overlays
#
#  This is the VISUAL ENGINE. It handles all drawing: base terrain
#  (walls, mud, start, goal) and dynamic overlays (visited, frontier,
#  current node, path, agent position, Q-heatmap, internal map).
#
#  LAYERING: The canvas uses a "delete overlay, redraw" approach —
#  overlay items are tagged with "ol" so they can be efficiently
#  removed before each repaint. The base grid (terrain) is redrawn
#  only on preset change or reset.
# ═══════════════════════════════════════════════════════════════════

class GridCanvas(tk.Canvas):
    """
    Canvas widget that renders a GridWorld as colored rectangles.

    COLOR PHILOSOPHY: Each color means one thing, and the meaning is
    constant across all modes. Terrain colors are always the same.
    Overlay colors depend on MODE (search overlays vs agent overlays)
    but are never mixed — search colors don't appear in agent mode.

    TERRAIN COLORS (base layer, always visible):
      normal = white, mud = tan, wall = dark gray
      start = blue, goal = red

    SEARCH OVERLAYS (Search/Compare mode only):
      visited = light green, frontier = yellow, current = orange, path = purple

    AGENT OVERLAYS (Agent mode only):
      agent dot = green oval, planned path = purple outline,
      internal map = green border, Q-heatmap = yellow scale
    """

    COLORS = {
        "normal":  "#ffffff",        # white — standard traversable cell
        "mud":     "#c8b080",        # tan/brown — high-cost cell (cost=5)
        "wall":    "#333333",        # dark gray — impassable
        "start":   "#4488ff",        # blue — starting position
        "goal":    "#ff4444",        # red — goal position
        "visited": "#88cc88",        # light green — nodes expanded
        "frontier":"#ffff88",        # light yellow — nodes queued but not expanded
        "current": "#ff8844",        # orange — node being expanded right now
        "path":    "#cc66ff",        # purple — final solution path
        "agent":   "#00cc00",        # green — agent's current position (oval)
    }

    def __init__(self, parent, world: GridWorld, width=480, height=480, **kw):
        """
        Create canvas with dark background, compute cell size from grid dimensions.

        The canvas background (#1a1a1a) provides contrast for all overlay colors
        and gives the app a modern dark-theme look suitable for projection.
        """
        super().__init__(parent, width=width, height=height,
                         bg="#1a1a1a", highlightthickness=0, **kw)
        self.world = world
        self._cs = 0          # cell size in pixels (calculated per draw)
        self.draw_grid()

    # ── drawing ──────────────────────────────────────────────

    def _calc_cs(self):
        """
        Compute cell size to fill the canvas while keeping cells square.

        Uses min(canvas_w/cols, canvas_h/rows) so cells are square.
        Minimum cell size = 8px (prevents degenerate rendering on tiny canvases).
        In Compare mode, both canvases get the same width → same cell size.
        """
        w = int(self["width"])
        h = int(self["height"])
        self._cs = max(8, min(w // self.world.cols, h // self.world.rows))

    def draw_grid(self):
        """
        Redraw BASE grid: terrain colors + wall/mud labels + start/goal markers.

        Called on: initial creation, preset change, reset.
        NOT called every step — overlays are painted on TOP via update_overlay().
        This two-pass approach avoids flickering and is much faster.

        Each cell gets:
          1. A filled rectangle (terrain color, or start/goal override)
          2. A text label for walls ('#') and mud ('~')
        """
        self.delete("all")                     # clear entire canvas
        self._calc_cs()
        cs = self._cs
        for r in range(self.world.rows):
            for c in range(self.world.cols):
                x1, y1 = c * cs, r * cs
                x2, y2 = x1 + cs, y1 + cs
                cell = self.world.grid[r][c]

                # Base terrain color (overridden for start/goal)
                color = self.COLORS.get(cell.terrain, "#ffffff")
                if (r, c) == self.world.start:
                    color = self.COLORS["start"]
                elif (r, c) == self.world.goal:
                    color = self.COLORS["goal"]

                self.create_rectangle(x1, y1, x2, y2,
                                       fill=color, outline="#888", width=1)

                # Terrain labels — make walls and mud visually distinct
                if cell.terrain == "wall":
                    self.create_text(x1 + cs // 2, y1 + cs // 2,
                                      text="#", fill="white",
                                      font=("Consolas", cs // 3))
                elif cell.terrain == "mud":
                    self.create_text(x1 + cs // 2, y1 + cs // 2,
                                      text="~", fill="#664400",
                                      font=("Consolas", cs // 3))

    def update_overlay(self, *, visited=None, frontier=None, current=None,
                       path=None, planned_path=None,
                       internal_map=None, q_overlay=None, agent_pos=None):
        """
        Paint OVERLAYS on top of the base grid.

        ALWAYS called after draw_grid() in the same frame. Overlay items
        are tagged with "ol" — we delete "ol" items before each repaint
        so no stale overlay elements accumulate.

        Overlay types (mutually exclusive by mode):
          SEARCH MODE: visited, frontier, current, path
          AGENT MODE: internal_map, q_overlay, planned_path, agent_pos
          COMPARE MODE: same as search mode, on two canvases

        Parameters are keyword-only — the caller specifies exactly which
        overlays to paint. Unspecified overlays are not drawn.

        DRAWING ORDER (back to front):
          1. internal_map (green outlines) — bottom
          2. Q-heatmap (yellow rectangles) — above internal map
          3. visited cells (light green fill)
          4. frontier cells (yellow fill)
          5. current node (orange fill)
          6. final path (purple fill)
          7. planned path (purple outline) — separate from final path
          8. agent dot (green oval) — topmost
        """
        cs = self._cs

        # ── internal map (Model Reflex agent) ──
        # Green outlines on visited cells and known walls.
        # 'V' = visited (draw outline), 'W' = known wall (draw outline),
        # '.' = unknown (skip).
        if internal_map:
            for r in range(self.world.rows):
                for c in range(self.world.cols):
                    if internal_map[r][c] == 'V':
                        x1, y1 = c * cs + 2, r * cs + 2
                        self.create_rectangle(x1, y1, x1 + cs - 4, y1 + cs - 4,
                                               outline="#00cc00", width=2, tags="ol")

        # ── Q-value heatmap (Learning agent) ──
        # Yellow-scale rectangles: darker = lower Q, brighter = higher Q.
        # Only cells with max Q > 0 are painted (fresh Q-tables are all zeros).
        # Formula: intensity = 100 + 155*(v/mx)  → range [100, 255].
        # hex color: #{intensity:02x}{intensity:02x}00 → R=G=intensity, B=0.
        # This produces a yellow gradient: dark olive → bright yellow.
        # Cells with max Q ≤ 0 produce no rectangle (invisible).
        if q_overlay:
            mx = max((max(row) for row in q_overlay), default=1) or 1
            for r in range(self.world.rows):
                for c in range(self.world.cols):
                    v = q_overlay[r][c]
                    if v > 0:                      # only positive Q-values are visible
                        intensity = int(100 + 155 * (v / mx))
                        x1, y1 = c * cs + cs // 5, r * cs + cs // 5
                        self.create_rectangle(x1, y1, x1 + 3 * cs // 5,
                                               y1 + 3 * cs // 5,
                                               fill=f"#{intensity:02x}{intensity:02x}00",
                                               outline="", tags="ol")

        # ── visited cells (search mode) ──
        # Light green fill. Skip start and goal (they have their own colors).
        if visited:
            for (r, c) in visited:
                if (r, c) not in (self.world.start, self.world.goal):
                    x1, y1 = c * cs + 2, r * cs + 2
                    self.create_rectangle(x1, y1, x1 + cs - 4, y1 + cs - 4,
                                           fill=self.COLORS["visited"],
                                           outline="", tags="ol")

        # ── frontier (search mode) ──
        # Yellow fill. Skip goal (we don't color goal as frontier).
        if frontier:
            for (r, c) in frontier:
                if (r, c) != self.world.goal:
                    x1, y1 = c * cs + 3, r * cs + 3
                    self.create_rectangle(x1, y1, x1 + cs - 6, y1 + cs - 6,
                                           fill=self.COLORS["frontier"],
                                           outline="", tags="ol")

        # ── current node (search mode) ──
        # Orange fill — the node being expanded RIGHT NOW. Skip goal.
        if current and current != self.world.goal:
            r, c = current
            x1, y1 = c * cs + 1, r * cs + 1
            self.create_rectangle(x1, y1, x1 + cs - 2, y1 + cs - 2,
                                   fill=self.COLORS["current"],
                                   outline="", tags="ol")

        # ── final path (search mode, on completion) ──
        # Purple fill. Skip start and goal (index 0 and -1).
        # Only shown when path has 2+ cells (a real path, not just start).
        if path and len(path) > 1:
            for (r, c) in path[1:-1]:
                if (r, c) != current:              # don't overwrite current node
                    x1, y1 = c * cs + 5, r * cs + 5
                    self.create_rectangle(x1, y1, x1 + cs - 10, y1 + cs - 10,
                                           fill=self.COLORS["path"],
                                           outline="", tags="ol")

        # ── planned path (Goal-based / Utility-based agents) ──
        # Purple OUTLINE (hollow rectangle) — distinct from filled path.
        # Shows the agent's planned route BEFORE it executes.
        if planned_path:
            for (r, c) in planned_path[1:-1]:
                x1, y1 = c * cs + 8, r * cs + 8
                self.create_rectangle(x1, y1, x1 + cs - 16, y1 + cs - 16,
                                       fill="", outline=self.COLORS["path"],
                                       width=2, tags="ol")

        # ── agent dot (agent mode) ──
        # Green oval — the agent's current position. Drawn LAST (topmost).
        if agent_pos:
            r, c = agent_pos
            x1, y1 = c * cs + cs // 4, r * cs + cs // 4
            self.create_oval(x1, y1, x1 + cs // 2, y1 + cs // 2,
                              fill=self.COLORS["agent"],
                              outline="black", tags="ol")


# ═══════════════════════════════════════════════════════════════════
#  Main Application
#
#  App is a tk.Tk subclass — the root window. It owns:
#    • Toolbar (mode, sub-mode, preset, Run/Step/Reset, speed, DLS, episodes)
#    • Main area (info panel + canvas area with 1-2 GridCanvas instances)
#    • Log bar (bottom status line)
#
#  STATE MACHINE: The app has a single mode state (agent/search/compare)
#  and a sub-selection within that mode (agent type, search algorithm, etc.).
#  Changing mode triggers _refresh_ui() which shows/hides widgets and
#  creates/destroys the Compare panel.
#
#  RUN LOOP: root.after(delay, _run_loop) — schedules the next step.
#  delay comes from the Speed slider (50-1000ms). The loop checks
#  self.running flag and auto-stops when search/compare is finished.
# ═══════════════════════════════════════════════════════════════════

class App(tk.Tk):
    """
    Root Tkinter window — main application class.

    LIFE CYCLE:
      1. __init__() — build all widgets, set initial state, show window
      2. User interacts via toolbar dropdowns and buttons
      3. _refresh_ui() — called on every mode/agent/algo change
      4. _step() → _agent_step() / _search_step() / _compare_step() — one tick
      5. _run_loop() — auto-advance via root.after() when Play is active
      6. _stop() — pause the run loop (user clicked Pause or search finished)

    ALL STATE lives on self — no global variables except the COLORS dict
    and ALGO_MAP lookup. This makes the app testable and the state inspectable.
    """

    # ── Algorithm name → function mapping ──
    # Used by both Search mode and Compare mode to look up the generator function.
    # DLS is special-cased because it takes an extra depth_limit parameter.
    ALGO_MAP = {
        'BFS': bfs_search, 'DFS': dfs_search, 'DLS': dls_search,
        'IDDFS': iddfs_search, 'UCS': ucs_search,
        'Greedy': greedy_search, 'A*': astar_search,
    }

    def __init__(self):
        super().__init__()
        self.title("Qutaiba's AI Agents + Search Algorithms Lab")
        self.geometry("920x680")                       # comfortable size for projection
        self.configure(bg="#1e1e1e")                   # dark background — modern, low-glare

        # ── state variables ──────────────────────────────────
        # These are the "single source of truth" for the UI.
        # Dropdowns read from them; callbacks write to them.

        self.logger = SessionLogger()                  # creates a new log file
        self.logger.mode("agent", "Goal-based")

        self.world = GridWorld.preset_a()              # default grid
        self.mode = "agent"                            # current top-level mode
        self.agent_type = "Goal-based"                 # agent sub-type (in agent mode)
        self.search_algo = "BFS"                       # algorithm (in search mode)
        self.cmp_a = "A*"                              # algorithm A (in compare mode)
        self.cmp_b = "Greedy"                          # algorithm B (in compare mode)
        self.agent_algo = "A*"                         # algorithm for Goal/Utility agents

        # Run-loop control
        self.running = False                           # Play/Pause flag
        self.generator = None                          # active search generator (search mode)
        self.search_done = False                       # has search finished?
        self.search_result = None                      # SearchResult or None
        self.agent = None                              # active agent instance (agent mode)
        self._learn_state = None                       # current state during learning episode
        self.gen_a = self.gen_b = None                 # compare mode generators
        self.done_a = self.done_b = False              # has each compare algo finished?
        self.res_a = self.res_b = None                 # SearchResult for each

        # ── build UI ─────────────────────────────────────────
        self._toolbar()                                # top bar: dropdowns + buttons
        self._main_area()                              # middle: info panel + canvas
        self._logbar()                                 # bottom: status log
        self._bind_keys()                              # keyboard shortcuts
        self._refresh_ui()                             # initial widget visibility + reset

    # ══════════════════════════════════════════════════════════
    #  Toolbar
    #
    #  The toolbar packs widgets left-to-right:
    #    Mode: [Agent ▾]  [Sub-type ▾]  [Algo ▾]  [B: ▾]  Preset: [A ▾]  ▶ ⏭ ↺  Speed: [===]  DLS:[===]  Ep:[100]
    #
    #  Widget visibility is DYNAMIC — controlled by _refresh_ui():
    #    • Sub-type dropdown shows agents OR algorithms depending on mode
    #    • Algo dropdown (agent_algo) only shown for Goal-based / Utility-based
    #    • Compare B dropdown only shown in Compare mode
    #    • DLS slider only shown when DLS is active
    #    • Episodes spinbox only shown for Learning agent
    # ══════════════════════════════════════════════════════════

    def _toolbar(self):
        bar = tk.Frame(self, bg="#2d2d2d")
        bar.pack(fill="x", padx=4, pady=4)

        # ── Mode selector ──
        tk.Label(bar, text="Mode:", bg="#2d2d2d", fg="white").pack(side="left")
        self.mode_cb = ttk.Combobox(bar, values=["Agent", "Search", "Compare"],
                                     state="readonly", width=10)
        self.mode_cb.set("Agent")
        self.mode_cb.pack(side="left", padx=2)
        self.mode_cb.bind("<<ComboboxSelected>>", lambda e: self._set_mode())

        # ── Sub-selector (agent type OR search algorithm) ──
        # Values are set dynamically by _refresh_ui() based on mode.
        self.sub_cb = ttk.Combobox(bar, state="readonly", width=14)
        self.sub_cb.pack(side="left", padx=2)
        self.sub_cb.bind("<<ComboboxSelected>>", lambda e: self._set_sub())

        # ── Algorithm selector (agent mode: Goal/Utility only) ──
        # Packed/unpacked dynamically. Values: BFS/DFS/UCS/A* or UCS/A*.
        self.algo_cb = ttk.Combobox(bar, state="readonly", width=6)
        self.algo_cb.bind("<<ComboboxSelected>>", lambda e: self._set_agent_algo())

        # ── Compare algorithm B ──
        # Only visible in Compare mode. Packed/unpacked dynamically.
        self.cmp_b_cb = ttk.Combobox(bar,
            values=["BFS","DFS","DLS","IDDFS","UCS","Greedy","A*"],
            state="readonly", width=10)
        self.cmp_b_cb.set("Greedy")
        self.cmp_b_cb.bind("<<ComboboxSelected>>", lambda e: self._set_cmp_b())

        # ── Preset selector ──
        tk.Label(bar, text="  Preset:", bg="#2d2d2d", fg="white").pack(side="left")
        self.preset_cb = ttk.Combobox(bar,
            values=["A - Open Plain", "B - Short-Cut Trap", "C - Mud Wall",
                    "D - The Labyrinth"],
            state="readonly", width=18)
        self.preset_cb.set("A - Open Plain")
        self.preset_cb.pack(side="left", padx=2)
        self.preset_cb.bind("<<ComboboxSelected>>", lambda e: self._set_preset())

        # ── Action buttons ──
        # Run toggles Play/Pause (text changes dynamically).
        self.run_btn = tk.Button(bar, text="▶ Run", command=self._toggle_run,
                                  bg="#3a3a3a", fg="white", width=5)
        self.run_btn.pack(side="left", padx=(10, 2))
        tk.Button(bar, text="⏭ Step", command=self._step,
                  bg="#3a3a3a", fg="white", width=5).pack(side="left", padx=2)
        tk.Button(bar, text="↺ Reset", command=self._reset,
                  bg="#3a3a3a", fg="white", width=5).pack(side="left", padx=2)

        # ── Speed slider ──
        # Controls root.after() delay in ms (50-1000ms, default 200ms).
        tk.Label(bar, text="  Speed:", bg="#2d2d2d", fg="white").pack(side="left")
        self.speed_var = tk.IntVar(value=200)
        tk.Scale(bar, from_=50, to=1000, orient="horizontal",
                 variable=self.speed_var, length=100,
                 bg="#2d2d2d", fg="white", troughcolor="#555").pack(side="left")

        # ── DLS depth slider (hidden unless DLS is selected) ──
        self.dls_var = tk.IntVar(value=10)
        self.dls_slider = tk.Scale(bar, from_=1, to=20, orient="horizontal",
                                    variable=self.dls_var, length=60,
                                    bg="#2d2d2d", fg="white", troughcolor="#555",
                                    label="DLS")

        # ── Episodes spinbox (hidden unless Learning agent is selected) ──
        self.ep_var = tk.IntVar(value=100)
        self.ep_label = tk.Label(bar, text="Ep:", bg="#2d2d2d", fg="white")
        self.ep_spin = tk.Spinbox(bar, from_=10, to=2000,
                                   textvariable=self.ep_var, width=5,
                                   bg="#3a3a3a", fg="white")

    # ══════════════════════════════════════════════════════════
    #  Main area — info panel (left) + canvas area (right)
    #
    #  Layout:
    #  ┌──────────────┬────────────────────────────────────────┐
    #  │  Info Panel  │         Canvas Area                    │
    #  │  (22 chars)  │  ┌─────────────────┬─────────────────┐ │
    #  │              │  │   Panel A       │   Panel B       │ │
    #  │  Stats here  │  │  (always shown) │ (compare only)  │ │
    #  │              │  │                 │                 │ │
    #  └──────────────┴──┴─────────────────┴─────────────────┘ │
    #
    #  Panel B is created/destroyed dynamically in _refresh_ui().
    #  Both panels get equal width via fill=both, expand=True.
    # ══════════════════════════════════════════════════════════

    def _main_area(self):
        self.main_frame = tk.Frame(self, bg="#1e1e1e")
        self.main_frame.pack(fill="both", expand=True, padx=4)

        # ── Info panel (left sidebar) ──
        # Width=22 characters, monospace, dark background.
        # Shows mode-specific stats: algorithm, frontier size, costs, etc.
        self.info = tk.Text(self.main_frame, width=22, height=30,
                             bg="#252525", fg="#ddd", font=("Consolas", 10),
                             state="disabled", wrap="word")
        self.info.pack(side="left", fill="y", padx=(0, 4))

        # ── Canvas area (right) ──
        # Contains 1 or 2 panels, each with a label + GridCanvas.
        self.canvas_frame = tk.Frame(self.main_frame, bg="#1e1e1e")
        self.canvas_frame.pack(side="left", fill="both", expand=True)

        # Panel A — ALWAYS present, holds the primary canvas
        self.panel_a = tk.Frame(self.canvas_frame, bg="#1e1e1e")
        self.panel_a.pack(side="left", fill="both", expand=True)

        self.label_a = tk.Label(self.panel_a, text="", bg="#1e1e1e", fg="#aaa",
                                 font=("Consolas", 10, "bold"))

        self.canvas = GridCanvas(self.panel_a, self.world, width=480, height=480)
        self.canvas.pack(fill="both", expand=True)

        # Panel B — for Compare mode, created on demand
        # Destroyed when leaving Compare mode to free resources.
        self.panel_b = None
        self.label_b = None
        self.canvas_b = None

    # ══════════════════════════════════════════════════════════
    #  Log bar — bottom status line
    #
    #  3-line read-only text widget showing the most recent events.
    #  Auto-scrolls to bottom. Disabled for editing (state="disabled").
    # ══════════════════════════════════════════════════════════

    def _logbar(self):
        self.log_text = tk.Text(self, height=3, bg="#1a1a1a", fg="#aaa",
                                 font=("Consolas", 9), state="disabled")
        self.log_text.pack(fill="x", padx=4, pady=(0, 4))

    def log(self, msg: str):
        """
        Append a line to the on-screen log bar.

        Temporarily enables the widget, inserts text, scrolls to end,
        then disables again. This is the standard read-only Text pattern.

        The log bar shows the LAST 3 LINES (controlled by height=3).
        Older lines scroll off the top — it's a rolling window, not a
        full history (see logger.py for persistent logs).
        """
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ══════════════════════════════════════════════════════════
    #  Keyboard shortcuts
    #
    #  bind_all() catches keystrokes anywhere in the window.
    #  Space=Step  P=Play/Pause  R=Reset  1=Preset A  2=Preset B  3=Preset C
    #  These make live demos fast — no mouse needed.
    # ══════════════════════════════════════════════════════════

    def _bind_keys(self):
        self.bind_all("<space>", lambda e: self._step())
        self.bind_all("<p>", lambda e: self._toggle_run())
        self.bind_all("<r>", lambda e: self._reset())
        self.bind_all("<Key-1>", lambda e: self._quick_preset(0))
        self.bind_all("<Key-2>", lambda e: self._quick_preset(1))
        self.bind_all("<Key-3>", lambda e: self._quick_preset(2))

    def _quick_preset(self, idx):
        """Jump to preset by index (0=A, 1=B, 2=C). Mapped to keys 1-3."""
        self.preset_cb.current(idx)
        self._set_preset()

    # ══════════════════════════════════════════════════════════
    #  UI refresh helpers
    #
    #  _refresh_ui() is the CENTRAL DISPATCHER — called after every
    #  mode change, sub-type change, or preset change. It:
    #    1. Updates dropdown values and visibility
    #    2. Shows/hides conditional widgets (DLS slider, episodes, etc.)
    #    3. Creates/destroys Compare panel B
    #    4. Calls _reset() to reinitialize the current mode
    # ══════════════════════════════════════════════════════════

    def _refresh_ui(self):
        """
        Reconfigure toolbar widgets based on current mode.

        This is the "single source of truth" for widget visibility.
        All show/hide logic lives here — individual callbacks just set
        state variables and call _refresh_ui().

        CRITICAL: The order of pack() and pack_forget() matters.
        Widgets are packed after their preceding sibling using `after=`.
        """
        m = self.mode

        # ── Sub-menu values ──
        if m == "agent":
            self.sub_cb["values"] = ["Simple Reflex","Model Reflex",
                                      "Goal-based","Utility-based","Learning"]
            self.sub_cb.set(self.agent_type)
        elif m == "search":
            self.sub_cb["values"] = ["BFS","DFS","DLS","IDDFS","UCS","Greedy","A*"]
            self.sub_cb.set(self.search_algo)
        else:  # compare
            self.sub_cb["values"] = ["BFS","DFS","DLS","IDDFS","UCS","Greedy","A*"]
            self.sub_cb.set(self.cmp_a)

        # ── Compare B dropdown (visible only in Compare mode) ──
        if m == "compare":
            self.cmp_b_cb.pack(side="left", padx=2, after=self.sub_cb)
        else:
            self.cmp_b_cb.pack_forget()

        # ── DLS depth slider (visible when DLS is active) ──
        show_dls = ((m == "search" and self.search_algo == "DLS") or
                    (m == "compare" and "DLS" in (self.cmp_a, self.cmp_b)))
        if show_dls:
            self.dls_slider.pack(side="left", padx=(10, 2))
        else:
            self.dls_slider.pack_forget()

        # ── Episodes spinbox (visible only for Learning agent) ──
        show_ep = (m == "agent" and self.agent_type == "Learning")
        if show_ep:
            self.ep_label.pack(side="left", padx=(10, 2))
            self.ep_spin.pack(side="left")
        else:
            self.ep_label.pack_forget()
            self.ep_spin.pack_forget()

        # ── Algorithm dropdown for Goal-based / Utility-based agents ──
        # Goal-based: BFS, DFS, UCS, A* (any path-finding algorithm)
        # Utility-based: UCS, A* only (cost-optimal algorithms only)
        show_algo = (m == "agent" and self.agent_type in
                     ("Goal-based", "Utility-based"))
        if show_algo:
            if self.agent_type == "Goal-based":
                self.algo_cb["values"] = ["BFS", "DFS", "UCS", "A*"]
            else:
                self.algo_cb["values"] = ["UCS", "A*"]
                if self.agent_algo not in ("UCS", "A*"):
                    self.agent_algo = "A*"   # BFS/DFS invalid for Utility-based → reset
            self.algo_cb.set(self.agent_algo)
            self.algo_cb.pack(side="left", padx=(4, 2), after=self.sub_cb)
        else:
            self.algo_cb.pack_forget()

        # ── Compare panel B — created on demand ──
        # Both panels share canvas_frame space via fill=both, expand=True.
        # They get equal widths — same requested width (480) ensures
        # identical cell sizes on the same grid.
        if m == "compare":
            if self.panel_b is None:
                self.panel_b = tk.Frame(self.canvas_frame, bg="#1e1e1e")
                self.panel_b.pack(side="left", fill="both", expand=True,
                                  after=self.panel_a, padx=(4, 0))

                self.label_b = tk.Label(self.panel_b, text=self.cmp_b,
                                         bg="#1e1e1e", fg="#aaa",
                                         font=("Consolas", 10, "bold"))
                self.label_b.pack(pady=(2, 0))

                self.canvas_b = GridCanvas(self.panel_b, self.world,
                                            width=480, height=480)
                self.canvas_b.pack(fill="both", expand=True)

            # Show algorithm labels above each panel
            self.label_a.configure(text=self.cmp_a)
            self.label_a.pack(pady=(2, 0))
            self.label_b.configure(text=self.cmp_b)
        else:
            # Destroy panel B when leaving Compare mode (free resources)
            if self.panel_b is not None:
                self.panel_b.destroy()
                self.panel_b = None
                self.label_b = None
                self.canvas_b = None
            self.label_a.pack_forget()   # hide label in single-canvas mode

        # Reinitialize with current settings
        self._reset()

    # ── Dropdown callbacks ───────────────────────────────────
    # Each reads the dropdown value, updates the state variable,
    # then calls _refresh_ui() which resets and reconfigures.

    def _set_mode(self):
        """Mode dropdown changed: Agent / Search / Compare."""
        v = self.mode_cb.get()
        self.mode = {"Agent": "agent", "Search": "search",
                      "Compare": "compare"}.get(v, "agent")
        self.logger.mode(self.mode, v)
        self._refresh_ui()

    def _set_sub(self):
        """Sub-selector changed: agent type OR search/compare algorithm."""
        v = self.sub_cb.get()
        if self.mode == "agent":
            self.agent_type = v
        elif self.mode == "search":
            self.search_algo = v
        else:
            self.cmp_a = v
        self._refresh_ui()

    def _set_cmp_b(self):
        """Compare algorithm B dropdown changed."""
        self.cmp_b = self.cmp_b_cb.get()
        if self.label_b:
            self.label_b.configure(text=self.cmp_b)
        self._refresh_ui()

    def _set_agent_algo(self):
        """Algorithm dropdown for Goal-based/Utility-based agent changed."""
        self.agent_algo = self.algo_cb.get()
        self._reset()

    def _set_preset(self):
        """
        Preset dropdown changed. Loads a new GridWorld from the selected preset.

        Updates all canvases' world references, redraws base grids,
        then resets the active mode (agent/search/compare).
        """
        idx = self.preset_cb.current()
        presets = [GridWorld.preset_a, GridWorld.preset_b, GridWorld.preset_c,
                   GridWorld.preset_d]
        self.world = presets[idx]()
        self.logger.preset(chr(ord('A') + idx))
        self.canvas.world = self.world
        self.canvas.draw_grid()
        if self.canvas_b:
            self.canvas_b.world = self.world
            self.canvas_b.draw_grid()
        self._reset()

    # ══════════════════════════════════════════════════════════
    #  Reset — clear all state and reinitialize
    #
    #  Called on: mode change, sub-type change, preset change,
    #  user clicking Reset button or pressing R key.
    #
    #  What it does:
    #    1. Stops the run loop (if running)
    #    2. Resets world (agent back to start)
    #    3. Clears generators, search results
    #    4. Redraws canvas(es) with agent at start
    #    5. Initializes agent or search or compare (depending on mode)
    #    6. Updates info panel
    # ══════════════════════════════════════════════════════════

    def _reset(self):
        self._stop()                                   # stop run loop
        self.world.reset()                             # agent → start
        self.generator = None
        self.search_done = False; self.search_result = None
        self._learn_state = None                       # reset learning episode state
        self.gen_a = self.gen_b = None
        self.done_a = self.done_b = False
        self.res_a = self.res_b = None

        # Redraw base grid (no overlays yet)
        self.canvas.draw_grid()
        self.canvas.update_overlay(agent_pos=self.world.agent_pos)
        if self.canvas_b:
            self.canvas_b.draw_grid()
            self.canvas_b.update_overlay(agent_pos=self.world.agent_pos)

        # Re-initialize based on current mode
        if self.mode == "agent":
            self._init_agent()
        elif self.mode == "search":
            self._init_search()
        else:
            self._init_compare()

        self._update_info()
        self.logger.reset()
        self.log("— reset —")

    # ══════════════════════════════════════════════════════════
    #  Agent init
    #
    #  Creates an agent instance based on self.agent_type.
    #  Goal-based and Utility-based: also calls agent.plan() to
    #  compute the path immediately (so it's visible on canvas).
    #  Learning: passes episode count from the spinbox.
    #
    #  CRITICAL: The algorithm dropdown maps display names ("BFS")
    #  to internal keys ("bfs") — the agent constructor expects
    #  lowercase keys. A* is special: .lower() → "a*" (wrong!),
    #  so we use an explicit lookup dict.
    # ══════════════════════════════════════════════════════════

    def _init_agent(self):
        """Create and initialize an agent instance."""
        cls = {
            "Simple Reflex": SimpleReflexAgent,
            "Model Reflex":  ModelReflexAgent,
            "Goal-based":    GoalBasedAgent,
            "Utility-based": UtilityBasedAgent,
            "Learning":      LearningAgent,
        }.get(self.agent_type, GoalBasedAgent)

        if self.agent_type == "Learning":
            self.agent = cls(self.world, episodes=self.ep_var.get())
        elif self.agent_type in ("Goal-based", "Utility-based"):
            algo = {"BFS": "bfs", "DFS": "dfs", "UCS": "ucs", "A*": "astar"}[self.agent_algo]
            self.agent = cls(self.world, algorithm=algo)
            self.agent.plan()                          # compute path immediately
        else:
            self.agent = cls(self.world)

        self._display_agent()

    # ══════════════════════════════════════════════════════════
    #  Search init
    #
    #  Creates a generator for the selected search algorithm.
    #  DLS is special-cased because it takes an extra depth_limit parameter.
    #  Other algorithms take only the world.
    # ══════════════════════════════════════════════════════════

    def _init_search(self):
        """Create search generator for the selected algorithm."""
        fn = self.ALGO_MAP.get(self.search_algo)
        if self.search_algo == "DLS":
            self.generator = fn(self.world, self.dls_var.get())
        else:
            self.generator = fn(self.world)
        self.search_done = False
        self.search_result = None

    # ══════════════════════════════════════════════════════════
    #  Compare init
    #
    #  Creates TWO independent generators — one for each algorithm.
    #  Both use the SAME GridWorld instance (shared state is OK because
    #  search algorithms only READ the world, never mutate it).
    #  DLS is special-cased for either or both algorithms.
    # ══════════════════════════════════════════════════════════

    def _init_compare(self):
        """Create two independent search generators for side-by-side comparison."""
        fa = self.ALGO_MAP.get(self.cmp_a)
        fb = self.ALGO_MAP.get(self.cmp_b)
        self.gen_a = fa(self.world) if self.cmp_a != "DLS" else fa(self.world, self.dls_var.get())
        self.gen_b = fb(self.world) if self.cmp_b != "DLS" else fb(self.world, self.dls_var.get())
        self.done_a = self.done_b = False
        self.res_a = self.res_b = None

    # ══════════════════════════════════════════════════════════
    #  Step — one tick of the simulation
    #
    #  Dispatches to agent_step / search_step / compare_step based on mode.
    #  Called by: Step button, Space key, Run loop (via root.after).
    # ══════════════════════════════════════════════════════════

    def _step(self, *_):
        if self.mode == "agent":
            self._agent_step()
        elif self.mode == "search":
            self._search_step()
        else:
            self._compare_step()
        self._update_info()

    # ── Agent Step ──────────────────────────────────────────

    def _agent_step(self):
        """
        Advance the agent by one decision cycle.

        SPECIAL HANDLING for Learning agent:
          Uses the step-by-step episode API: start_episode() → step_episode() → end_episode().
          Each Step click runs ONE action within the current episode.
          When episode is done, end_episode() is called, and next Step starts a new episode.

        For non-learning agents:
          Simply calls agent.step() which returns (action, rule_string).
          The agent itself calls world.move_agent() — the GUI just logs and displays.
        """
        if self.agent is None:
            return

        if isinstance(self.agent, LearningAgent):
            # ── Learning agent: episode-based step-by-step ──
            if self.agent.episode_count >= self.agent.episodes:
                self.log("Training complete — all episodes done.")
                self._stop()
                return
            if self._learn_state is None:
                self._learn_state = self.agent.start_episode()
            state = self._learn_state
            action, reward, nxt, done = self.agent.step_episode(state)
            self._learn_state = nxt
            self._display_agent()
            self.log(f"→ {action}  reward={reward:+.0f}  ε={self.agent.get_epsilon():.3f}")
            if done:
                ep_reward = self.agent.end_episode()
                self._learn_state = None
                self.logger.learning_episode(self.agent.episode_count,
                    self.agent.episodes, ep_reward, self.agent.get_epsilon())
                self.log(f"Ep {self.agent.episode_count} done — total reward={ep_reward:.0f}")
        else:
            # ── Non-learning agent: simple step ──
            action, rule = self.agent.step()
            self.logger.step_agent(action, rule, self.agent_type)
            self.log(f"→ {action}  ({rule})")
            self._display_agent()
            if action == "stop":
                self.logger.agent_stuck(self.agent_type, self.world.agent_pos)
                self.log("Agent stopped.")
                self._stop()   # auto-stop the run loop

    # ── Search Step ─────────────────────────────────────────

    def _search_step(self):
        """
        Advance the search algorithm by one expansion.

        Calls next(gen) on the active generator. The generator yields a
        SearchFrame — we read it and paint the canvas. When the generator
        is exhausted (StopIteration), we capture the SearchResult and
        display the final path (or failure message).

        IDDFS special handling: when special="DEPTH_LIMIT_CHANGE", we log
        the new depth limit but don't paint (the frame has no data).
        """
        if self.generator is None or self.search_done:
            return
        try:
            frame = next(self.generator)
        except StopIteration as e:
            # Generator exhausted → capture SearchResult
            self.search_result = e.value
            self.search_done = True
            if self.search_result.success:
                self.logger.goal_found(self.search_algo,
                    self.search_result.cost, self.search_result.nodes_expanded,
                    self.search_result.path)
                self.log(f"✓ Goal!  cost={self.search_result.cost:.0f}  "
                         f"nodes={self.search_result.nodes_expanded}")
                self.canvas.draw_grid()
                self.canvas.update_overlay(
                    path=self.search_result.path,
                    agent_pos=self.world.agent_pos)
            else:
                self.logger.no_path(self.search_algo,
                    self.search_result.nodes_expanded)
                self.log(f"✗ No path.  expanded={self.search_result.nodes_expanded}")
            self._stop()
            return

        # IDDFS depth-limit-change signal — log it, don't paint
        if getattr(frame, "special", None) == "DEPTH_LIMIT_CHANGE":
            self.logger.iddfs_depth(frame.depth)
            self.log(f"IDDFS depth limit → {frame.depth}")
            return

        # Normal frame — paint the canvas with search state
        self.canvas.draw_grid()
        self.canvas.update_overlay(
            visited=frame.visited,
            frontier=frame.frontier,
            current=frame.current,
            path=frame.path,
        )

        # Log with g/h for cost-aware algorithms
        algo = self.search_algo
        if algo in ('UCS', 'Greedy', 'A*'):
            self.logger.step_search(algo, frame.current, len(frame.frontier),
                                     frame.g, frame.h)
            self.log(f"expanded {frame.current}  g={frame.g:.1f} h={frame.h:.1f}")
        else:
            self.logger.step_search(algo, frame.current, len(frame.frontier))
            self.log(f"expanded {frame.current}  frontier={len(frame.frontier)}")

    # ── Compare Step ────────────────────────────────────────

    def _compare_step(self):
        """
        Advance BOTH search generators by one expansion each.

        Iterates over (generator, done_flag, result_slot, canvas) tuples.
        For each algorithm that hasn't finished: calls next(gen), paints
        its canvas. Skip IDDFS depth-change signals (they carry no data).

        When BOTH are done, logs the comparison stats and auto-stops.
        If one finishes early, its canvas freezes and the other continues.

        This synchronized stepping is the key feature of Compare mode —
        you see both algorithms exploring SIDE BY SIDE at the same pace.
        """
        for gen, done_attr, res_attr, canvas in [
            (self.gen_a, "done_a", "res_a", self.canvas),
            (self.gen_b, "done_b", "res_b", self.canvas_b),
        ]:
            if getattr(self, done_attr) or gen is None:
                continue                                   # already finished or not initialized
            try:
                frame = next(gen)
            except StopIteration as e:
                setattr(self, res_attr, e.value)           # store SearchResult
                setattr(self, done_attr, True)
                continue
            if getattr(frame, "special", None) == "DEPTH_LIMIT_CHANGE":
                continue                                   # skip IDDFS signal frames
            canvas.draw_grid()
            canvas.update_overlay(
                visited=frame.visited,
                frontier=frame.frontier,
                current=frame.current,
                path=frame.path,
            )

        # Both done → log comparison, auto-stop
        if self.done_a and self.done_b:
            self.logger.compare_stats(self.cmp_a, self.res_a,
                                       self.cmp_b, self.res_b)
            self._stop()

    # ══════════════════════════════════════════════════════════
    #  Display helpers — update canvas overlays and info panel
    # ══════════════════════════════════════════════════════════

    def _display_agent(self):
        """
        Paint agent-specific overlays on the primary canvas.

        Checks the agent type and extracts the relevant visualization data:
          • ModelReflexAgent → internal_map (green outlines of visited/walls)
          • LearningAgent → q_overlay (yellow heatmap)
          • GoalBased/UtilityBased → planned_path (purple outline)
          • ALL agents → agent_pos (green oval)

        Calls update_overlay() which layers these on top of the base grid.
        """
        if self.agent is None:
            return
        self.canvas.draw_grid()

        kw = {}
        if isinstance(self.agent, ModelReflexAgent):
            kw["internal_map"] = self.agent.get_internal_map()
        if isinstance(self.agent, LearningAgent):
            kw["q_overlay"] = self.agent.get_max_q_per_cell()
        if hasattr(self.agent, "get_planned_path"):
            kw["planned_path"] = self.agent.get_planned_path()

        kw["agent_pos"] = self.world.agent_pos
        self.canvas.update_overlay(**kw)

    def _update_info(self):
        """
        Refresh the info panel (left sidebar) with mode-specific stats.

        Each mode has its own _info_* method that builds a list of lines.
        The info panel is a read-only Text widget — we temporarily enable
        it, clear, write, then disable again.
        """
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")

        if self.mode == "agent":
            self._info_agent()
        elif self.mode == "search":
            self._info_search()
        else:
            self._info_compare()

        self.info.configure(state="disabled")

    def _info_agent(self):
        """Info panel for Agent mode: type, algorithm, path cost, learning stats."""
        lines = [f"Mode: Agent",
                 f"Type: {self.agent_type}",
                 f"Preset: {self.preset_cb.get()[0]}",
                 f"Pos: {self.world.agent_pos}",
                 ""]
        if self.agent_type in ("Goal-based", "Utility-based") and self.agent:
            lines.append(f"Algorithm: {self.agent_algo}")
            lines.append(f"Path len: {len(getattr(self.agent, 'planned_path', []))}")
            if hasattr(self.agent, 'get_path_cost'):
                lines.append(f"Cost: {self.agent.get_path_cost():.0f}")
        if self.agent_type == "Learning" and self.agent:
            lines.append(f"Episode: {self.agent.episode_count}/{self.agent.episodes}")
            lines.append(f"ε: {self.agent.get_epsilon():.3f}")
            if self.agent.reward_history:
                lines.append(f"Last reward: {self.agent.reward_history[-1]:.0f}")
            lines.append("")
            lines.append("[Train All] runs all")
            lines.append("[Step Ep] one episode")
        self._write_info(lines)

    def _info_search(self):
        """Info panel for Search mode: algorithm, result (if done)."""
        lines = [f"Mode: Search",
                 f"Algo: {self.search_algo}",
                 f"Preset: {self.preset_cb.get()[0]}",
                 ""]
        if self.search_result:
            if self.search_result.success:
                lines.append(f"✓ Path found")
                lines.append(f"  cost: {self.search_result.cost:.0f}")
                lines.append(f"  nodes: {self.search_result.nodes_expanded}")
            else:
                lines.append(f"✗ No path")
                lines.append(f"  expanded: {self.search_result.nodes_expanded}")
        self._write_info(lines)

    def _info_compare(self):
        """Info panel for Compare mode: side-by-side stats for both algorithms."""
        lines = [f"Mode: Compare",
                 f"A: {self.cmp_a}",
                 f"B: {self.cmp_b}",
                 f"Preset: {self.preset_cb.get()[0]}",
                 ""]
        for label, res in [("A", self.res_a), ("B", self.res_b)]:
            if res and res.success:
                lines.append(f"{label}: ✓ cost={res.cost:.0f}  nodes={res.nodes_expanded}")
            elif res:
                lines.append(f"{label}: ✗ no path  exp={res.nodes_expanded}")
            else:
                lines.append(f"{label}: running…")
        self._write_info(lines)

    def _write_info(self, lines: list[str]):
        """Write a list of strings to the info panel Text widget."""
        for line in lines:
            self.info.insert("end", line + "\n")

    # ══════════════════════════════════════════════════════════
    #  Run / Stop — the animation loop
    #
    #  Play/Pause is a TOGGLE: clicking Run starts the loop,
    #  clicking Pause stops it. The button text changes accordingly.
    #
    #  The run loop uses root.after(delay, callback) — a non-blocking
    #  timer. Each tick calls _step() then schedules the next tick.
    #  The delay comes from the Speed slider (50-1000ms).
    #
    #  AUTO-STOP conditions:
    #    • Search mode: search_done (goal found or frontier exhausted)
    #    • Compare mode: both algorithms finished
    #    • Agent mode: agent returns "stop" action
    #    • User clicks Pause or presses P key
    # ══════════════════════════════════════════════════════════

    def _toggle_run(self, *_):
        """Run/Pause button clicked (or P key pressed)."""
        if self.running:
            self._stop()
            return
        self.running = True
        self.run_btn.configure(text="⏸ Pause")
        self._run_loop()

    def _stop(self):
        """Stop the animation loop."""
        self.running = False
        self.run_btn.configure(text="▶ Run")

    def _run_loop(self):
        """
        One iteration of the animation loop. Schedules the next iteration.

        Flow:
          1. Check running flag (stop if user paused)
          2. Call _step() — advance the simulation by one tick
          3. Check for completion (search done, compare done)
          4. If still running, schedule next tick via self.after(delay, self._run_loop)

        The `after` method is Tkinter's non-blocking timer — it schedules
        a callback after N milliseconds without freezing the GUI.
        """
        if not self.running:
            return
        self._step()
        if not self.running:
            return
        # Auto-stop when simulation is complete
        if self.mode == "search" and self.search_done:
            self._stop(); return
        if self.mode == "compare" and self.done_a and self.done_b:
            self._stop(); return
        self.after(self.speed_var.get(), self._run_loop)


# ═══════════════════════════════════════════════════════════════════
#  Entry point
#
#  Standard Python idiom: if this file is run directly (not imported),
#  create the App window and start the Tkinter event loop.
#  mainloop() blocks until the window is closed.
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
