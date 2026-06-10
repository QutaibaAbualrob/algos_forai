"""Debug logger — writes structured logs to logs/ folder.

Filename format:  {file_number}_{hours}-{minutes}.log
One log file per session (app launch = new file).
"""

import os
import datetime


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def _ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _next_file_number() -> int:
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
    """Buffered logger for one application session."""

    def __init__(self):
        _ensure_log_dir()
        self.num = _next_file_number()
        now = datetime.datetime.now()
        self.filename = f"{self.num:03d}_{now.hour:02d}-{now.minute:02d}.log"
        self.path = os.path.join(LOG_DIR, self.filename)
        self._buffer: list[str] = []
        self._write(f"=== SESSION START === {now.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── public API ───────────────────────────────────────────

    def mode(self, mode: str, detail: str = ""):
        self._write(f"[MODE] {mode}" + (f" | {detail}" if detail else ""))

    def preset(self, name: str):
        self._write(f"[PRESET] {name}")

    def step_agent(self, action: str, rule: str, agent_type: str):
        self._write(f"[AGENT] {agent_type} | action={action} | rule={rule}")

    def step_search(self, algo: str, pos: tuple, frontier: int, g: float = 0, h: float = 0):
        extra = f" g={g:.1f} h={h:.1f} f={g+h:.1f}" if (g or h) else ""
        self._write(f"[SEARCH] {algo} | expanded {pos} | frontier={frontier}{extra}")

    def iddfs_depth(self, depth: int):
        self._write(f"[IDDFS] depth-limit -> {depth}")

    def goal_found(self, algo: str, cost: float, nodes: int, path: list = None):
        path_str = f" | path={path}" if path else ""
        self._write(f"[GOAL] {algo} | cost={cost:.0f} | nodes={nodes}{path_str}")

    def no_path(self, algo: str, nodes: int):
        self._write(f"[FAIL] {algo} | frontier exhausted | expanded={nodes}")

    def compare_stats(self, algo_a: str, res_a, algo_b: str, res_b):
        def _fmt(r):
            if r is None: return "running"
            if r.success: return f"cost={r.cost:.0f} nodes={r.nodes_expanded}"
            return f"FAIL nodes={r.nodes_expanded}"
        self._write(f"[COMPARE] {algo_a}: {_fmt(res_a)}  |  {algo_b}: {_fmt(res_b)}")

    def agent_stuck(self, agent_type: str, pos: tuple):
        self._write(f"[STUCK] {agent_type} at {pos}")

    def learning_episode(self, ep: int, total: int, reward: float, epsilon: float):
        self._write(f"[LEARN] ep={ep}/{total} | reward={reward:.0f} | ε={epsilon:.3f}")

    def learning_train_done(self, episodes: int, final_reward: float):
        self._write(f"[LEARN] train-all done | {episodes} eps | final reward={final_reward:.0f}")

    def reset(self):
        self._write(f"[RESET]")

    def error(self, msg: str):
        self._write(f"[ERROR] {msg}")

    def close(self):
        now = datetime.datetime.now()
        self._write(f"=== SESSION END === {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── internal ─────────────────────────────────────────────

    def _write(self, line: str):
        now = datetime.datetime.now()
        ts = now.strftime("%H:%M:%S")
        self._buffer.append(f"{ts}  {line}")
        self._flush()

    def _flush(self):
        with open(self.path, "a", encoding="utf-8") as f:
            for line in self._buffer:
                f.write(line + "\n")
        self._buffer.clear()
