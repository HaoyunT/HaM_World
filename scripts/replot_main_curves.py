from __future__ import annotations

import re
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import RESULTS_MAIN_FIG_ROOT, RUNS_MAIN_ROOT


LOG_DIR = RUNS_MAIN_ROOT / "model_based" / "batch" / "jobs"
LOW_BUDGET_LOG_DIR = RUNS_MAIN_ROOT / "model_free" / "batch" / "jobs"
OUT_DIR = RESULTS_MAIN_FIG_ROOT
OUT_DIR.mkdir(parents=True, exist_ok=True)
ASSEMBLED_STEM = RESULTS_MAIN_FIG_ROOT / "main_results_curves_assembled"

ALGOS = ["hamworld", "tdmpc2", "dreamerv3", "ppo_low", "sac_low"]
ALGO_LABELS = {
    "hamworld": "HaM-World",
    "tdmpc2": "TD-MPC2",
    "dreamerv3": "DreamerV3",
    "ppo_low": "PPO",
    "sac_low": "SAC",
}
ALGO_COLOR = {
    "hamworld": "#c63d2f",
    "tdmpc2": "#2563eb",
    "dreamerv3": "#0f766e",
    "ppo_low": "#94a3b8",
    "sac_low": "#b7791f",
}
LINEWIDTH = {algo: (2.4 if algo == "hamworld" else 1.7) for algo in ALGOS}
ZORDER = {"hamworld": 6, "tdmpc2": 5, "dreamerv3": 4, "sac_low": 3, "ppo_low": 2}
LINESTYLE = {
    "hamworld": "solid",
    "tdmpc2": "solid",
    "dreamerv3": "solid",
    "ppo_low": (0, (5.0, 2.2)),
    "sac_low": (0, (3.2, 1.6, 1.2, 1.6)),
}
BAND_ALPHA = {
    "hamworld": 0.16,
    "tdmpc2": 0.12,
    "dreamerv3": 0.12,
    "ppo_low": 0.08,
    "sac_low": 0.08,
}
TASKS = ["cartpole_swingup", "cheetah_run", "finger_spin", "reacher_easy"]
TASK_TITLE = {
    "cartpole_swingup": "Cartpole Swingup",
    "cheetah_run": "Cheetah Run",
    "finger_spin": "Finger Spin",
    "reacher_easy": "Reacher Easy",
}
SEEDS = [7, 8, 9]

STEP_RE = re.compile(r"^\[step\s+(\d+)\]\s+(.*)$")
EVAL_RE = re.compile(r"eval/return_mean=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10.5,
        "axes.labelcolor": "#334155",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 10.5,
        "legend.frameon": False,
        "figure.dpi": 220,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


def _parse_eval_curve(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        match = STEP_RE.match(line)
        if not match:
            continue
        eval_match = EVAL_RE.search(match.group(2))
        if eval_match:
            out[int(match.group(1))] = float(eval_match.group(1))
    return out


def _curve_for_algo_task(algo: str, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    per_seed = []
    for seed in SEEDS:
        if algo in {"ppo_low", "sac_low"}:
            path = LOW_BUDGET_LOG_DIR / f"{algo.replace('_low', '')}_{task}_seed{seed}.log"
        else:
            path = LOG_DIR / f"{algo}_{task}_seed{seed}.log"
        parsed = _parse_eval_curve(path)
        if parsed:
            per_seed.append(parsed)
    if not per_seed:
        return np.array([]), np.array([]), np.array([]), np.array([])
    steps = sorted({step for parsed in per_seed for step in parsed})
    xs, means, lows, highs = [], [], [], []
    for step in steps:
        values = np.array([parsed[step] for parsed in per_seed if step in parsed], dtype=float)
        if values.size == 0:
            continue
        xs.append(step)
        means.append(float(values.mean()))
        lows.append(float(values.min()))
        highs.append(float(values.max()))
    return np.array(xs), np.array(means), np.array(lows), np.array(highs)


def _smooth(y: np.ndarray, width: int = 3) -> np.ndarray:
    if y.size < width:
        return y
    kernel = np.ones(width) / width
    pad = width // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _fmt_steps(x, _pos) -> str:
    if x == 0:
        return "0"
    if x >= 1000:
        value = x / 1000
        return f"{value:.0f}k" if abs(value - round(value)) < 1e-6 else f"{value:.1f}k"
    return f"{x:.0f}"


def _xticks_for(x_max: float) -> list[int]:
    if x_max >= 99_000:
        return [0, 25_000, 50_000, 75_000, 100_000]
    return [0, int(round(x_max / 3)), int(round(2 * x_max / 3)), int(round(x_max))]


def _save_pair(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".pdf"))


def plot_task_curve(task: str) -> None:
    fig, ax = plt.subplots(figsize=(3.05, 2.35))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")
    has_any = False
    x_max = 0.0
    for algo in ALGOS:
        x, mean, low, high = _curve_for_algo_task(algo, task)
        if x.size == 0:
            continue
        color = ALGO_COLOR[algo]
        zorder = ZORDER[algo]
        mean_s = _smooth(mean)
        low_s = _smooth(low)
        high_s = _smooth(high)
        ax.fill_between(x, low_s, high_s, color=color, alpha=BAND_ALPHA[algo], linewidth=0, zorder=zorder - 0.5)
        ax.plot(
            x,
            mean_s,
            color=color,
            linewidth=LINEWIDTH[algo],
            linestyle=LINESTYLE[algo],
            solid_capstyle="round",
            solid_joinstyle="round",
            dash_capstyle="round",
            label=ALGO_LABELS[algo],
            zorder=zorder,
        )
        ax.scatter(
            x[-1],
            mean_s[-1],
            s=22 if algo == "hamworld" else 16,
            color=color,
            edgecolors="white",
            linewidths=0.9,
            zorder=zorder + 0.2,
        )
        x_max = max(x_max, float(x.max()))
        has_any = True
    if not has_any:
        plt.close(fig)
        return
    ax.set_title(TASK_TITLE[task], color="#0f172a", pad=6, loc="left")
    ax.set_xlabel("env steps")
    ax.set_ylabel("eval return")
    ax.set_xlim(0, x_max)
    ax.margins(y=0.08)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.set_xticks(_xticks_for(x_max))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_steps))
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, color="#e5e7eb", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="major", pad=2)
    ax.tick_params(axis="y", which="major", pad=2)
    fig.tight_layout(pad=0.3)
    _save_pair(fig, OUT_DIR / f"curve_{task}")
    plt.close(fig)


def plot_legend_strip() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 0.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=ALGO_COLOR[algo],
            linewidth=2.4 if algo == "hamworld" else 2.0,
            linestyle=LINESTYLE[algo],
            solid_capstyle="round",
            dash_capstyle="round",
            marker="o",
            markersize=4.2 if algo == "hamworld" else 3.6,
            markerfacecolor=ALGO_COLOR[algo],
            markeredgewidth=0.0,
            label=ALGO_LABELS[algo],
        )
        for algo in ALGOS
    ]
    ax.legend(
        handles=handles,
        loc="center",
        ncol=5,
        frameon=False,
        handlelength=2.0,
        handletextpad=0.55,
        columnspacing=1.45,
        borderaxespad=0.0,
    )
    _save_pair(fig, OUT_DIR / "legend")
    plt.close(fig)


def build_assembled_figure() -> None:
    curve_order = [
        "finger_spin",
        "reacher_easy",
        "cheetah_run",
        "cartpole_swingup",
    ]
    def _crop_curve_image(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        top = int(height * 0.01)
        bottom = int(height * 0.93)
        left = int(width * 0.015)
        right = int(width * 0.99)
        return image[top:bottom, left:right]

    curve_images = [_crop_curve_image(mpimg.imread(OUT_DIR / f"curve_{task}.png")) for task in curve_order]

    fig = plt.figure(figsize=(12.6, 3.52), facecolor="white")
    grid = fig.add_gridspec(1, 4, wspace=0.08)

    for idx, image in enumerate(curve_images):
        ax = fig.add_subplot(grid[0, idx])
        ax.imshow(image)
        ax.axis("off")

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=ALGO_COLOR[algo],
            linewidth=2.4 if algo == "hamworld" else 2.0,
            linestyle=LINESTYLE[algo],
            solid_capstyle="round",
            dash_capstyle="round",
            marker="o",
            markersize=4.4 if algo == "hamworld" else 3.8,
            markerfacecolor=ALGO_COLOR[algo],
            markeredgewidth=0.0,
            label=ALGO_LABELS[algo],
        )
        for algo in ALGOS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05),
        handlelength=1.9,
        handletextpad=0.45,
        columnspacing=1.1,
        fontsize=15.0,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(top=0.99, bottom=0.045, left=0.02, right=0.98)
    fig.savefig(ASSEMBLED_STEM.with_suffix(".png"))
    fig.savefig(ASSEMBLED_STEM.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    for task in TASKS:
        plot_task_curve(task)
        print(f"[curve] {task}")
    plot_legend_strip()
    build_assembled_figure()
    print(f"[assembled] {ASSEMBLED_STEM.with_suffix('.png')}")
    print(f"[done] {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
