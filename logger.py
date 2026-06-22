"""Debug logger — writes structured logs to logs/ folder.
══════════════════════════════════════════════════════════════════════
ARCHITECTURE ROLE: Non-essential but valuable for debugging and
presentation QA. Every significant event (mode switches, steps, goal
found, errors) is timestamped and written to a session log file.

DESIGN:
  • One log file per application launch (not per reset).
  • Auto-numbered filenames: 001_14-35.log, 002_09-12.log, etc.
    Using '-' instead of ':' because Windows disallows colons in filenames.
  • BUFFERED writes — log lines are collected and flushed in batches
    to avoid 100+ tiny disk writes per second during Run mode.
  • Structured format: [TAG] key=value pairs — grep-friendly.

USAGE IN PRESENTATION: The log files serve as a session record —
you can show the log to demonstrate exactly what happened during a
presentation run, including step counts, rewards, and algorithm stats.

CONNECTIONS:
  • main.py — App creates a SessionLogger, calls log methods on events.
  • Not imported by world.py, search.py, or agents.py (logging is a GUI concern).

Filename format:  {file_number}_{hours}-{minutes}.log
One log file per session (app launch = new file).
"""

import os
import datetime


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def _ensure_log_dir():
    """Create logs/ directory if it doesn't exist. Idempotent."""
    os.makedirs(LOG_DIR, exist_ok=True)


def _next_file_number() -> int:
    """
    Find the next available sequence number for the log file.

    Scans existing log files, parses their numeric prefix (e.g. "022" from "022_18-17.log"),
    returns max + 1. Starts at 1 if no logs exist.
    Ignores files that don't match the naming pattern (defensive parsing).
    """
    _ensure_log_dir()
    existing = [f for f in os.listdir(LOG_DIR) if f.endswith(".log")]
    if not existing:
        return 1
    nums = []
    for f in existing:
        try:
            nums.append(int(f.split("_")[0]))
        except (ValueError, IndexError):
            pass
    return max(nums, default=0) + 1


class SessionLogger:
    """
    Buffered structured logger for one application session.

    Buffering STRATEGY: Each _write() call appends to a list and immediately
    flushes. This is simple and safe — no data loss on crash, and the write
    frequency (one line per step) is low enough that buffering overhead
    doesn't matter. For faster Run mode (50ms per step), lines accumulate
    naturally but each is flushed individually.

    LOG FORMAT: "HH:MM:SS  [TAG] detail"
    Example: "14:35:22  [SEARCH] BFS | expanded (2,3) | frontier=5"
    """

    def __init__(self):
        _ensure_log_dir()
        self.num = _next_file_number()
        now = datetime.datetime.now()
        # Use '-' not ':' for Windows filename compatibility
        self.filename = f"{self.num:03d}_{now.hour:02d}-{now.minute:02d}.log"
        self.path = os.path.join(LOG_DIR, self.filename)
        self._buffer: list[str] = []
        self._write(f"=== SESSION START === {now.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── public API ───────────────────────────────────────────
    #   Each method corresponds to a distinct event type.
    #   Tags ([MODE], [AGENT], [SEARCH], etc.) make logs grep-friendly.

    def mode(self, mode: str, detail: str = ""):
        """GUI mode change: agent, search, or compare."""
        self._write(f"[MODE] {mode}" + (f" | {detail}" if detail else ""))

    def preset(self, name: str):
        """Grid preset selected (A, B, C, or D)."""
        self._write(f"[PRESET] {name}")

    def step_agent(self, action: str, rule: str, agent_type: str):
        """One agent decision cycle: which rule fired, what action resulted."""
        self._write(f"[AGENT] {agent_type} | action={action} | rule={rule}")

    def step_search(self, algo: str, pos: tuple, frontier: int, g: float = 0, h: float = 0):
        """
        One search expansion. For cost-aware algorithms (UCS, Greedy, A*),
        includes g, h, and f=g+h values. For uninformed algorithms,
        only the position and frontier size are logged.
        """
        extra = f" g={g:.1f} h={h:.1f} f={g+h:.1f}" if (g or h) else ""
        self._write(f"[SEARCH] {algo} | expanded {pos} | frontier={frontier}{extra}")

    def iddfs_depth(self, depth: int):
        """IDDFS started a new depth iteration."""
        self._write(f"[IDDFS] depth-limit -> {depth}")

    def goal_found(self, algo: str, cost: float, nodes: int, path: list = None):
        """Search found a path! Log the stats and optionally the full path."""
        path_str = f" | path={path}" if path else ""
        self._write(f"[GOAL] {algo} | cost={cost:.0f} | nodes={nodes}{path_str}")

    def no_path(self, algo: str, nodes: int):
        """Search exhausted frontier without reaching goal."""
        self._write(f"[FAIL] {algo} | frontier exhausted | expanded={nodes}")

    def compare_stats(self, algo_a: str, res_a, algo_b: str, res_b):
        """Comparison mode summary: side-by-side stats for both algorithms."""
        def _fmt(r):
            if r is None: return "running"
            if r.success: return f"cost={r.cost:.0f} nodes={r.nodes_expanded}"
            return f"FAIL nodes={r.nodes_expanded}"
        self._write(f"[COMPARE] {algo_a}: {_fmt(res_a)}  |  {algo_b}: {_fmt(res_b)}")

    def agent_stuck(self, agent_type: str, pos: tuple):
        """Agent returned 'stop' action — either stuck or path complete."""
        self._write(f"[STUCK] {agent_type} at {pos}")

    def learning_episode(self, ep: int, total: int, reward: float, epsilon: float):
        """One Q-learning episode completed. Track reward and epsilon trend."""
        self._write(f"[LEARN] ep={ep}/{total} | reward={reward:.0f} | ε={epsilon:.3f}")

    def learning_train_done(self, episodes: int, final_reward: float):
        """Train All completed."""
        self._write(f"[LEARN] train-all done | {episodes} eps | final reward={final_reward:.0f}")

    def reset(self):
        """User pressed Reset."""
        self._write(f"[RESET]")

    def error(self, msg: str):
        """Unexpected condition (not currently used extensively, but available)."""
        self._write(f"[ERROR] {msg}")

    def close(self):
        """Application shutting down — write end-of-session marker."""
        now = datetime.datetime.now()
        self._write(f"=== SESSION END === {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── internal ─────────────────────────────────────────────

    def _write(self, line: str):
        """
        Buffer a log line with timestamp, then flush to disk.

        Timestamp format: HH:MM:SS (24-hour).
        Flush is immediate (no deferred writes) — safe against crashes.
        """
        now = datetime.datetime.now()
        ts = now.strftime("%H:%M:%S")
        self._buffer.append(f"{ts}  {line}")
        self._flush()

    def _flush(self):
        """
        Write all buffered lines to the log file, then clear buffer.

        Opens in append mode ('a') — each flush adds to the file.
        Encoding: utf-8 (handles any Unicode in log messages).
        """
        with open(self.path, "a", encoding="utf-8") as f:
            for line in self._buffer:
                f.write(line + "\n")
        self._buffer.clear()
