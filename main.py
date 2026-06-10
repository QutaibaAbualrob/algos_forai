"""Tkinter GUI — AI Agents + Search Algorithms Lab.

Three modes: Agent | Search | Compare.
Keyboard: Space=Step  P=Play/Pause  R=Reset  1/2/3=presets.
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

# ══════════════════════════════════════════════════════════════════
#  GridCanvas — draws the grid + overlays
# ══════════════════════════════════════════════════════════════════

class GridCanvas(tk.Canvas):
    COLORS = {
        "normal":  "#ffffff",
        "mud":     "#c8b080",
        "wall":    "#333333",
        "start":   "#4488ff",
        "goal":    "#ff4444",
        "visited": "#88cc88",
        "frontier":"#ffff88",
        "current": "#ff8844",
        "path":    "#cc66ff",
        "agent":   "#00cc00",
    }

    def __init__(self, parent, world: GridWorld, width=480, height=480, **kw):
        super().__init__(parent, width=width, height=height,
                         bg="#1a1a1a", highlightthickness=0, **kw)
        self.world = world
        self._cs = 0          # cell size (calculated in draw)
        self.draw_grid()

    # ── drawing ──────────────────────────────────────────────

    def _calc_cs(self):
        w = int(self["width"])
        h = int(self["height"])
        self._cs = max(8, min(w // self.world.cols, h // self.world.rows))

    def draw_grid(self):
        """Redraw base grid (walls, mud, start, goal)."""
        self.delete("all")
        self._calc_cs()
        cs = self._cs
        for r in range(self.world.rows):
            for c in range(self.world.cols):
                x1, y1 = c * cs, r * cs
                x2, y2 = x1 + cs, y1 + cs
                cell = self.world.grid[r][c]

                color = self.COLORS.get(cell.terrain, "#ffffff")
                if (r, c) == self.world.start:
                    color = self.COLORS["start"]
                elif (r, c) == self.world.goal:
                    color = self.COLORS["goal"]

                self.create_rectangle(x1, y1, x2, y2,
                                       fill=color, outline="#888", width=1)

                # labels
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
        """Paint overlays on top of the base grid."""
        cs = self._cs

        # --- internal map (model-reflex) ---
        if internal_map:
            for r in range(self.world.rows):
                for c in range(self.world.cols):
                    if internal_map[r][c] == 'V':
                        x1, y1 = c * cs + 2, r * cs + 2
                        self.create_rectangle(x1, y1, x1 + cs - 4, y1 + cs - 4,
                                               outline="#00cc00", width=2, tags="ol")

        # --- Q-value heatmap (learning) ---
        if q_overlay:
            mx = max((max(row) for row in q_overlay), default=1) or 1
            for r in range(self.world.rows):
                for c in range(self.world.cols):
                    v = q_overlay[r][c]
                    if v > 0:
                        intensity = int(100 + 155 * (v / mx))
                        x1, y1 = c * cs + cs // 5, r * cs + cs // 5
                        self.create_rectangle(x1, y1, x1 + 3 * cs // 5,
                                               y1 + 3 * cs // 5,
                                               fill=f"#{intensity:02x}{intensity:02x}00",
                                               outline="", tags="ol")

        # --- visited cells ---
        if visited:
            for (r, c) in visited:
                if (r, c) not in (self.world.start, self.world.goal):
                    x1, y1 = c * cs + 2, r * cs + 2
                    self.create_rectangle(x1, y1, x1 + cs - 4, y1 + cs - 4,
                                           fill=self.COLORS["visited"],
                                           outline="", tags="ol")

        # --- frontier ---
        if frontier:
            for (r, c) in frontier:
                if (r, c) != self.world.goal:
                    x1, y1 = c * cs + 3, r * cs + 3
                    self.create_rectangle(x1, y1, x1 + cs - 6, y1 + cs - 6,
                                           fill=self.COLORS["frontier"],
                                           outline="", tags="ol")

        # --- current node ---
        if current and current != self.world.goal:
            r, c = current
            x1, y1 = c * cs + 1, r * cs + 1
            self.create_rectangle(x1, y1, x1 + cs - 2, y1 + cs - 2,
                                   fill=self.COLORS["current"],
                                   outline="", tags="ol")

        # --- final path ---
        if path and len(path) > 1:
            for (r, c) in path[1:-1]:
                if (r, c) != current:
                    x1, y1 = c * cs + 5, r * cs + 5
                    self.create_rectangle(x1, y1, x1 + cs - 10, y1 + cs - 10,
                                           fill=self.COLORS["path"],
                                           outline="", tags="ol")

        # --- planned path (goal/utility agents) ---
        if planned_path:
            for (r, c) in planned_path[1:-1]:
                x1, y1 = c * cs + 8, r * cs + 8
                self.create_rectangle(x1, y1, x1 + cs - 16, y1 + cs - 16,
                                       fill="", outline=self.COLORS["path"],
                                       width=2, tags="ol")

        # --- agent ---
        if agent_pos:
            r, c = agent_pos
            x1, y1 = c * cs + cs // 4, r * cs + cs // 4
            self.create_oval(x1, y1, x1 + cs // 2, y1 + cs // 2,
                              fill=self.COLORS["agent"],
                              outline="black", tags="ol")


# ══════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════

class App(tk.Tk):
    ALGO_MAP = {
        'BFS': bfs_search, 'DFS': dfs_search, 'DLS': dls_search,
        'IDDFS': iddfs_search, 'UCS': ucs_search,
        'Greedy': greedy_search, 'A*': astar_search,
    }

    def __init__(self):
        super().__init__()
        self.title("AI Agents + Search Algorithms Lab")
        self.geometry("920x680")
        self.configure(bg="#1e1e1e")

        # ── state ────────────────────────────────────────────
        self.world = GridWorld.preset_a()
        self.mode = "agent"                      # agent | search | compare
        self.agent_type = "Goal-based"
        self.search_algo = "BFS"
        self.cmp_a = "A*";  self.cmp_b = "Greedy"

        self.running = False
        self.generator = None
        self.search_done = False; self.search_result = None
        self.agent = None
        self.gen_a = self.gen_b = None
        self.done_a = self.done_b = False
        self.res_a = self.res_b = None

        # ── layout ───────────────────────────────────────────
        self._toolbar()
        self._main_area()
        self._logbar()
        self._bind_keys()
        self._refresh_ui()

    # ══════════════════════════════════════════════════════════
    #  Toolbar
    # ══════════════════════════════════════════════════════════

    def _toolbar(self):
        bar = tk.Frame(self, bg="#2d2d2d")
        bar.pack(fill="x", padx=4, pady=4)

        # mode
        tk.Label(bar, text="Mode:", bg="#2d2d2d", fg="white").pack(side="left")
        self.mode_cb = ttk.Combobox(bar, values=["Agent", "Search", "Compare"],
                                     state="readonly", width=10)
        self.mode_cb.set("Agent")
        self.mode_cb.pack(side="left", padx=2)
        self.mode_cb.bind("<<ComboboxSelected>>", lambda e: self._set_mode())

        # sub-selector
        self.sub_cb = ttk.Combobox(bar, state="readonly", width=14)
        self.sub_cb.pack(side="left", padx=2)
        self.sub_cb.bind("<<ComboboxSelected>>", lambda e: self._set_sub())

        # compare algo B
        self.cmp_b_cb = ttk.Combobox(bar,
            values=["BFS","DFS","DLS","IDDFS","UCS","Greedy","A*"],
            state="readonly", width=10)
        self.cmp_b_cb.set("Greedy")
        self.cmp_b_cb.bind("<<ComboboxSelected>>", lambda e: self._set_cmp_b())

        # preset
        tk.Label(bar, text="  Preset:", bg="#2d2d2d", fg="white").pack(side="left")
        self.preset_cb = ttk.Combobox(bar,
            values=["A - Open Plain", "B - Short-Cut Trap", "C - Mud Wall"],
            state="readonly", width=18)
        self.preset_cb.set("A - Open Plain")
        self.preset_cb.pack(side="left", padx=2)
        self.preset_cb.bind("<<ComboboxSelected>>", lambda e: self._set_preset())

        # buttons
        self.run_btn = tk.Button(bar, text="▶ Run", command=self._toggle_run,
                                  bg="#3a3a3a", fg="white", width=5)
        self.run_btn.pack(side="left", padx=(10, 2))
        tk.Button(bar, text="⏭ Step", command=self._step,
                  bg="#3a3a3a", fg="white", width=5).pack(side="left", padx=2)
        tk.Button(bar, text="↺ Reset", command=self._reset,
                  bg="#3a3a3a", fg="white", width=5).pack(side="left", padx=2)

        # speed
        tk.Label(bar, text="  Speed:", bg="#2d2d2d", fg="white").pack(side="left")
        self.speed_var = tk.IntVar(value=200)
        tk.Scale(bar, from_=50, to=1000, orient="horizontal",
                 variable=self.speed_var, length=100,
                 bg="#2d2d2d", fg="white", troughcolor="#555").pack(side="left")

        # DLS depth (hidden unless DLS)
        self.dls_var = tk.IntVar(value=10)
        self.dls_slider = tk.Scale(bar, from_=1, to=20, orient="horizontal",
                                    variable=self.dls_var, length=60,
                                    bg="#2d2d2d", fg="white", troughcolor="#555",
                                    label="DLS")

        # episodes spinbox (hidden unless Learning)
        self.ep_var = tk.IntVar(value=100)
        self.ep_label = tk.Label(bar, text="Ep:", bg="#2d2d2d", fg="white")
        self.ep_spin = tk.Spinbox(bar, from_=10, to=2000,
                                   textvariable=self.ep_var, width=5,
                                   bg="#3a3a3a", fg="white")

    # ══════════════════════════════════════════════════════════
    #  Main area
    # ══════════════════════════════════════════════════════════

    def _main_area(self):
        self.main_frame = tk.Frame(self, bg="#1e1e1e")
        self.main_frame.pack(fill="both", expand=True, padx=4)

        # info panel (left)
        self.info = tk.Text(self.main_frame, width=22, height=30,
                             bg="#252525", fg="#ddd", font=("Consolas", 10),
                             state="disabled", wrap="word")
        self.info.pack(side="left", fill="y", padx=(0, 4))

        # Canvas area (right) — uses a container frame for Compare mode
        self.canvas_frame = tk.Frame(self.main_frame, bg="#1e1e1e")
        self.canvas_frame.pack(side="left", fill="both", expand=True)

        # single canvas (agent / search mode)
        self.canvas = GridCanvas(self.canvas_frame, self.world, width=480, height=480)
        self.canvas.pack(side="left")

        # second canvas for Compare mode (hidden initially)
        self.canvas_b = None

    # ══════════════════════════════════════════════════════════
    #  Log bar
    # ══════════════════════════════════════════════════════════

    def _logbar(self):
        self.log_text = tk.Text(self, height=3, bg="#1a1a1a", fg="#aaa",
                                 font=("Consolas", 9), state="disabled")
        self.log_text.pack(fill="x", padx=4, pady=(0, 4))

    def log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ══════════════════════════════════════════════════════════
    #  Keyboard
    # ══════════════════════════════════════════════════════════

    def _bind_keys(self):
        self.bind_all("<space>", lambda e: self._step())
        self.bind_all("<p>", lambda e: self._toggle_run())
        self.bind_all("<r>", lambda e: self._reset())
        self.bind_all("<Key-1>", lambda e: self._quick_preset(0))
        self.bind_all("<Key-2>", lambda e: self._quick_preset(1))
        self.bind_all("<Key-3>", lambda e: self._quick_preset(2))

    def _quick_preset(self, idx):
        self.preset_cb.current(idx)
        self._set_preset()

    # ══════════════════════════════════════════════════════════
    #  UI refresh helpers
    # ══════════════════════════════════════════════════════════

    def _refresh_ui(self):
        """Reconfigure toolbar widgets based on current mode."""
        m = self.mode

        # sub-menu values
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

        # show/hide widgets
        if m == "compare":
            self.cmp_b_cb.pack(side="left", padx=2, after=self.sub_cb)
        else:
            self.cmp_b_cb.pack_forget()

        show_dls = ((m == "search" and self.search_algo == "DLS") or
                    (m == "compare" and "DLS" in (self.cmp_a, self.cmp_b)))
        if show_dls:
            self.dls_slider.pack(side="left", padx=(10, 2))
        else:
            self.dls_slider.pack_forget()

        show_ep = (m == "agent" and self.agent_type == "Learning")
        if show_ep:
            self.ep_label.pack(side="left", padx=(10, 2))
            self.ep_spin.pack(side="left")
        else:
            self.ep_label.pack_forget()
            self.ep_spin.pack_forget()

        # compare second canvas
        if m == "compare" and self.canvas_b is None:
            self.canvas_b = GridCanvas(self.canvas_frame, self.world,
                                        width=380, height=480)
            self.canvas_b.pack(side="left", padx=(4, 0))
        elif m != "compare" and self.canvas_b is not None:
            self.canvas_b.destroy()
            self.canvas_b = None

        self._reset()

    def _set_mode(self):
        v = self.mode_cb.get()
        self.mode = {"Agent": "agent", "Search": "search",
                      "Compare": "compare"}.get(v, "agent")
        self._refresh_ui()

    def _set_sub(self):
        v = self.sub_cb.get()
        if self.mode == "agent":
            self.agent_type = v
        elif self.mode == "search":
            self.search_algo = v
        else:
            self.cmp_a = v
        self._refresh_ui()

    def _set_cmp_b(self):
        self.cmp_b = self.cmp_b_cb.get()
        self._refresh_ui()

    def _set_preset(self):
        idx = self.preset_cb.current()
        presets = [GridWorld.preset_a, GridWorld.preset_b, GridWorld.preset_c]
        self.world = presets[idx]()
        self.canvas.world = self.world
        self.canvas.draw_grid()
        if self.canvas_b:
            self.canvas_b.world = self.world
            self.canvas_b.draw_grid()
        self._reset()

    # ══════════════════════════════════════════════════════════
    #  Reset
    # ══════════════════════════════════════════════════════════

    def _reset(self):
        self._stop()
        self.world.reset()
        self.generator = None
        self.search_done = False; self.search_result = None
        self.gen_a = self.gen_b = None
        self.done_a = self.done_b = False
        self.res_a = self.res_b = None

        self.canvas.draw_grid()
        self.canvas.update_overlay(agent_pos=self.world.agent_pos)
        if self.canvas_b:
            self.canvas_b.draw_grid()
            self.canvas_b.update_overlay(agent_pos=self.world.agent_pos)

        # init agent / search depending on mode
        if self.mode == "agent":
            self._init_agent()
        elif self.mode == "search":
            self._init_search()
        else:
            self._init_compare()

        self._update_info()
        self.log("— reset —")

    # ══════════════════════════════════════════════════════════
    #  Agent init
    # ══════════════════════════════════════════════════════════

    def _init_agent(self):
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
            algo = "astar" if self.agent_type == "Goal-based" else "astar"
            self.agent = cls(self.world, algorithm=algo)
            self.agent.plan()
        else:
            self.agent = cls(self.world)

        self._display_agent()

    # ══════════════════════════════════════════════════════════
    #  Search init
    # ══════════════════════════════════════════════════════════

    def _init_search(self):
        fn = self.ALGO_MAP.get(self.search_algo)
        if self.search_algo == "DLS":
            self.generator = fn(self.world, self.dls_var.get())
        else:
            self.generator = fn(self.world)
        self.search_done = False
        self.search_result = None

    # ══════════════════════════════════════════════════════════
    #  Compare init
    # ══════════════════════════════════════════════════════════

    def _init_compare(self):
        fa = self.ALGO_MAP.get(self.cmp_a)
        fb = self.ALGO_MAP.get(self.cmp_b)
        self.gen_a = fa(self.world) if self.cmp_a != "DLS" else fa(self.world, self.dls_var.get())
        self.gen_b = fb(self.world) if self.cmp_b != "DLS" else fb(self.world, self.dls_var.get())
        self.done_a = self.done_b = False
        self.res_a = self.res_b = None

    # ══════════════════════════════════════════════════════════
    #  Step
    # ══════════════════════════════════════════════════════════

    def _step(self, *_):
        if self.mode == "agent":
            self._agent_step()
        elif self.mode == "search":
            self._search_step()
        else:
            self._compare_step()
        self._update_info()

    def _agent_step(self):
        if self.agent is None:
            return

        if isinstance(self.agent, LearningAgent):
            # Step Episode — run one full episode
            reward = self.agent.run_episode()
            self.log(f"Ep {self.agent.episode_count}: reward={reward:.0f}  "
                     f"ε={self.agent.get_epsilon():.3f}")
            self._display_agent()
        else:
            action, rule = self.agent.step()
            self.log(f"→ {action}  ({rule})")
            self._display_agent()
            if action == "stop":
                self.log("Agent stopped.")

    def _search_step(self):
        if self.generator is None or self.search_done:
            return
        try:
            frame = next(self.generator)
        except StopIteration as e:
            self.search_result = e.value
            self.search_done = True
            if self.search_result.success:
                self.log(f"✓ Goal!  cost={self.search_result.cost:.0f}  "
                         f"nodes={self.search_result.nodes_expanded}")
                self.canvas.draw_grid()
                self.canvas.update_overlay(
                    path=self.search_result.path,
                    agent_pos=self.world.agent_pos)
            else:
                self.log(f"✗ No path.  expanded={self.search_result.nodes_expanded}")
            self._stop()
            return

        if getattr(frame, "special", None) == "DEPTH_LIMIT_CHANGE":
            self.log(f"IDDFS depth limit → {frame.depth}")
            return

        self.canvas.draw_grid()
        self.canvas.update_overlay(
            visited=frame.visited,
            frontier=frame.frontier,
            current=frame.current,
            path=frame.path,
        )

        algo = self.search_algo
        if algo in ('UCS', 'Greedy', 'A*'):
            self.log(f"expanded {frame.current}  g={frame.g:.1f} h={frame.h:.1f}")
        else:
            self.log(f"expanded {frame.current}  frontier={len(frame.frontier)}")

    def _compare_step(self):
        """Advance both generators by one expansion each."""
        for gen, done_attr, res_attr, canvas in [
            (self.gen_a, "done_a", "res_a", self.canvas),
            (self.gen_b, "done_b", "res_b", self.canvas_b),
        ]:
            if getattr(self, done_attr) or gen is None:
                continue
            try:
                frame = next(gen)
            except StopIteration as e:
                setattr(self, res_attr, e.value)
                setattr(self, done_attr, True)
                continue
            if getattr(frame, "special", None) == "DEPTH_LIMIT_CHANGE":
                continue
            canvas.draw_grid()
            canvas.update_overlay(
                visited=frame.visited,
                frontier=frame.frontier,
                current=frame.current,
                path=frame.path,
            )

        if self.done_a and self.done_b:
            self._stop()

    # ══════════════════════════════════════════════════════════
    #  Display helpers
    # ══════════════════════════════════════════════════════════

    def _display_agent(self):
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
        lines = [f"Mode: Agent",
                 f"Type: {self.agent_type}",
                 f"Preset: {self.preset_cb.get()[0]}",
                 f"Pos: {self.world.agent_pos}",
                 ""]
        if self.agent_type in ("Goal-based", "Utility-based") and self.agent:
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
        for line in lines:
            self.info.insert("end", line + "\n")

    # ══════════════════════════════════════════════════════════
    #  Run / Stop
    # ══════════════════════════════════════════════════════════

    def _toggle_run(self, *_):
        if self.running:
            self._stop()
            return
        self.running = True
        self.run_btn.configure(text="⏸ Pause")
        self._run_loop()

    def _stop(self):
        self.running = False
        self.run_btn.configure(text="▶ Run")

    def _run_loop(self):
        if not self.running:
            return
        self._step()
        if not self.running:
            return
        # stop if everything is done
        if self.mode == "search" and self.search_done:
            self._stop(); return
        if self.mode == "compare" and self.done_a and self.done_b:
            self._stop(); return
        self.after(self.speed_var.get(), self._run_loop)


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
