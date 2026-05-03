#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
MPLCONFIGDIR = REPO_ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import ensure_repo_on_path

ensure_repo_on_path()

from hamworld.runtime import make_env
from hamworld.world_model import CanonicalDynamicsWorldModel, infer_checkpoint_step


@dataclass(frozen=True)
class PairScore:
    q_index: int
    p_index: int
    kind: str
    score: float
    coverage: float
    swirl: float
    speed: float
    occupancy: float
    q_span: float
    p_span: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 2D Hamiltonian slice flow maps from a HaM-World checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_*.pt.")
    parser.add_argument("--trace-path", required=True, help="Path to dynamics trace .npz file.")
    parser.add_argument("--output-dir", required=True, help="Directory for figures and pair summaries.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--grid-size", type=int, default=81, help="Grid size for slice evaluation.")
    parser.add_argument("--same-k", type=int, default=4, help="How many same-index (q_i, p_i) pairs to render.")
    parser.add_argument("--cross-k", type=int, default=2, help="How many cross-index (q_i, p_j) pairs to render.")
    parser.add_argument("--trace-episodes", type=int, default=3, help="How many trace episodes to overlay.")
    parser.add_argument("--trace-quantile-lo", type=float, default=2.0, help="Low percentile for slice bounds.")
    parser.add_argument("--trace-quantile-hi", type=float, default=98.0, help="High percentile for slice bounds.")
    return parser.parse_args()


def _load_checkpoint_world_model(checkpoint_path: Path, device: torch.device) -> tuple[CanonicalDynamicsWorldModel, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload["config"])

    env, spec = make_env(config["task"], int(config["experiment"]["seed"]) + 1000)
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()

    world_model = CanonicalDynamicsWorldModel(
        config=config,
        obs_dim=int(spec.observation_shape[0]),
        action_dim=int(spec.action_shape[0]),
    ).to(device)

    agent_state = payload.get("agent_state")
    if isinstance(agent_state, dict) and "model" in agent_state:
        state_dict = agent_state["model"]
    else:
        state_dict = payload.get("model")
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain a world model state: {checkpoint_path}")

    world_model.load_state_dict(state_dict)
    world_model.set_schedule_step(infer_checkpoint_step(payload, checkpoint_path))
    world_model.eval()
    return world_model, config


def _load_trace(trace_path: Path) -> dict[str, np.ndarray]:
    trace = np.load(trace_path, allow_pickle=True)
    return {key: trace[key] for key in trace.files}


def _flatten_valid_vectors(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(valid_mask, dtype=bool)
    return np.asarray(values[mask], dtype=np.float32)


def _robust_span(values: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> float:
    low, high = np.percentile(values, [lo, hi])
    return float(high - low)


def _histogram_entropy(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    probs = hist.ravel()
    total = probs.sum()
    if total <= 0.0:
        return 0.0
    probs = probs[probs > 0.0] / total
    entropy = -(probs * np.log(probs)).sum()
    max_entropy = np.log(float(bins * bins))
    return float(entropy / max(1e-6, max_entropy))


def _normalize(values: list[float]) -> list[float]:
    maximum = max(values) if values else 1.0
    if maximum <= 1e-6:
        return [0.0 for _ in values]
    return [float(value / maximum) for value in values]


def score_pairs(trace: dict[str, np.ndarray]) -> list[PairScore]:
    valid_mask = np.asarray(trace["valid_mask"], dtype=bool)
    q = _flatten_valid_vectors(trace["q"], valid_mask)
    p = _flatten_valid_vectors(trace["p"], valid_mask)
    dH_dq = _flatten_valid_vectors(trace["dH_dq"], valid_mask)
    dH_dp = _flatten_valid_vectors(trace["dH_dp"], valid_mask)

    q_dim = int(np.asarray(trace["q_dim"]).item())
    p_dim = int(np.asarray(trace["p_dim"]).item())

    raw_rows: list[dict[str, float | int | str]] = []
    for q_index in range(q_dim):
        for p_index in range(p_dim):
            qv = q[:, q_index]
            pv = p[:, p_index]
            dq = dH_dp[:, q_index]
            dp = -dH_dq[:, p_index]
            q_center = float(np.median(qv))
            p_center = float(np.median(pv))

            radial_norm = np.hypot(qv - q_center, pv - p_center)
            flow_norm = np.hypot(dq, dp)
            tangential = np.abs((qv - q_center) * dp - (pv - p_center) * dq)
            swirl = float(np.mean(tangential / np.maximum(1e-6, radial_norm * flow_norm)))

            q_span = _robust_span(qv)
            p_span = _robust_span(pv)
            coverage = float(q_span * p_span)
            speed = float(np.mean(flow_norm))
            occupancy = _histogram_entropy(qv, pv)
            raw_rows.append(
                {
                    "q_index": q_index,
                    "p_index": p_index,
                    "kind": "same" if q_index == p_index else "cross",
                    "coverage": coverage,
                    "swirl": swirl,
                    "speed": speed,
                    "occupancy": occupancy,
                    "q_span": q_span,
                    "p_span": p_span,
                }
            )

    coverage_norm = _normalize([float(row["coverage"]) for row in raw_rows])
    swirl_norm = _normalize([float(row["swirl"]) for row in raw_rows])
    speed_norm = _normalize([float(row["speed"]) for row in raw_rows])
    occupancy_norm = _normalize([float(row["occupancy"]) for row in raw_rows])

    scored: list[PairScore] = []
    for index, row in enumerate(raw_rows):
        score = (
            0.35 * coverage_norm[index]
            + 0.30 * swirl_norm[index]
            + 0.20 * speed_norm[index]
            + 0.15 * occupancy_norm[index]
        )
        scored.append(
            PairScore(
                q_index=int(row["q_index"]),
                p_index=int(row["p_index"]),
                kind=str(row["kind"]),
                score=float(score),
                coverage=float(row["coverage"]),
                swirl=float(row["swirl"]),
                speed=float(row["speed"]),
                occupancy=float(row["occupancy"]),
                q_span=float(row["q_span"]),
                p_span=float(row["p_span"]),
            )
        )

    return sorted(scored, key=lambda item: item.score, reverse=True)


def select_pairs(scores: list[PairScore], same_k: int, cross_k: int) -> list[PairScore]:
    same_pairs = [row for row in scores if row.kind == "same"][: max(0, int(same_k))]
    cross_pairs = [row for row in scores if row.kind == "cross"][: max(0, int(cross_k))]
    return same_pairs + cross_pairs


def _slice_bounds(values: np.ndarray, quantile_lo: float, quantile_hi: float) -> tuple[float, float]:
    low, high = np.percentile(values, [quantile_lo, quantile_hi])
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-5:
        center = float(np.median(values))
        return center - 1.0, center + 1.0
    margin = 0.18 * float(high - low)
    return float(low - margin), float(high + margin)


def evaluate_slice(
    world_model: CanonicalDynamicsWorldModel,
    q_reference: np.ndarray,
    p_reference: np.ndarray,
    pair: PairScore,
    q_values: np.ndarray,
    p_values: np.ndarray,
    grid_size: int,
    quantile_lo: float,
    quantile_hi: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    x_min, x_max = _slice_bounds(q_values, quantile_lo, quantile_hi)
    y_min, y_max = _slice_bounds(p_values, quantile_lo, quantile_hi)
    q_axis = np.linspace(x_min, x_max, grid_size, dtype=np.float64)
    p_axis = np.linspace(y_min, y_max, grid_size, dtype=np.float64)
    q_grid, p_grid = np.meshgrid(q_axis, p_axis, indexing="xy")

    q_eval = np.repeat(q_reference[None, :], grid_size * grid_size, axis=0)
    p_eval = np.repeat(p_reference[None, :], grid_size * grid_size, axis=0)
    q_eval[:, pair.q_index] = q_grid.reshape(-1)
    p_eval[:, pair.p_index] = p_grid.reshape(-1)

    q_tensor = torch.as_tensor(q_eval, dtype=torch.float32, device=device).requires_grad_(True)
    p_tensor = torch.as_tensor(p_eval, dtype=torch.float32, device=device).requires_grad_(True)
    energy = world_model.energy_head(q_tensor, p_tensor)
    dH_dq, dH_dp = torch.autograd.grad(energy.sum(), (q_tensor, p_tensor), create_graph=False, retain_graph=False)

    H = energy.detach().cpu().numpy().reshape(grid_size, grid_size)
    U = dH_dp.detach().cpu().numpy()[:, pair.q_index].reshape(grid_size, grid_size)
    V = (-dH_dq.detach().cpu().numpy()[:, pair.p_index]).reshape(grid_size, grid_size)
    speed = np.hypot(U, V)
    return {
        "q_axis": q_axis,
        "p_axis": p_axis,
        "H": H,
        "U": U,
        "V": V,
        "speed": speed,
        "q_grid": q_grid,
        "p_grid": p_grid,
    }


def _overlay_trace(axis: plt.Axes, trace: dict[str, np.ndarray], pair: PairScore, max_episodes: int) -> None:
    valid_mask = np.asarray(trace["valid_mask"], dtype=bool)
    q = np.asarray(trace["q"], dtype=np.float32)
    p = np.asarray(trace["p"], dtype=np.float32)
    episode_count = min(int(max_episodes), int(q.shape[0]))
    colors = ["#7dd3fc", "#38bdf8", "#0ea5e9"]
    for episode_index in range(episode_count):
        mask = valid_mask[episode_index]
        if not np.any(mask):
            continue
        x = q[episode_index, mask, pair.q_index]
        y = p[episode_index, mask, pair.p_index]
        axis.plot(x, y, color=colors[episode_index % len(colors)], linewidth=1.1, alpha=0.9)
        axis.scatter(x[0], y[0], color=colors[episode_index % len(colors)], s=14, marker="o", alpha=0.95)


def draw_pair_figure(
    output_path: Path,
    slice_eval: dict[str, np.ndarray],
    trace: dict[str, np.ndarray],
    pair: PairScore,
    q_reference: np.ndarray,
    p_reference: np.ndarray,
    checkpoint_name: str,
) -> None:
    q_axis = slice_eval["q_axis"]
    p_axis = slice_eval["p_axis"]
    H = slice_eval["H"]
    U = slice_eval["U"]
    V = slice_eval["V"]
    speed = slice_eval["speed"]

    fig, axis = plt.subplots(figsize=(6.4, 5.2), dpi=220)
    heat = axis.contourf(q_axis, p_axis, H, levels=24, cmap="magma")
    axis.contour(q_axis, p_axis, H, levels=10, colors="white", linewidths=0.45, alpha=0.55)
    axis.streamplot(
        q_axis,
        p_axis,
        U,
        V,
        density=1.1,
        color=speed,
        cmap="viridis",
        linewidth=0.9,
        arrowsize=0.9,
    )
    _overlay_trace(axis, trace, pair, max_episodes=3)
    axis.scatter(
        [q_reference[pair.q_index]],
        [p_reference[pair.p_index]],
        s=52,
        marker="x",
        linewidths=1.5,
        color="white",
        zorder=5,
    )
    axis.set_xlabel(f"q[{pair.q_index}]")
    axis.set_ylabel(f"p[{pair.p_index}]")
    axis.set_title(
        f"{pair.kind} pair q[{pair.q_index}] / p[{pair.p_index}]  score={pair.score:.3f}\n{checkpoint_name}",
        loc="left",
    )
    cbar = fig.colorbar(heat, ax=axis, fraction=0.046, pad=0.04)
    cbar.set_label("H(q, p)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_overview(
    output_path: Path,
    slices: list[tuple[PairScore, dict[str, np.ndarray]]],
    trace: dict[str, np.ndarray],
    q_reference: np.ndarray,
    p_reference: np.ndarray,
    checkpoint_name: str,
) -> None:
    columns = 3
    rows = max(1, int(np.ceil(len(slices) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(5.3 * columns, 4.5 * rows), dpi=220)
    axes_array = np.atleast_1d(axes).reshape(rows, columns)

    for axis, (pair, slice_eval) in zip(axes_array.flatten(), slices):
        heat = axis.contourf(slice_eval["q_axis"], slice_eval["p_axis"], slice_eval["H"], levels=20, cmap="magma")
        axis.contour(slice_eval["q_axis"], slice_eval["p_axis"], slice_eval["H"], levels=8, colors="white", linewidths=0.35, alpha=0.5)
        axis.streamplot(
            slice_eval["q_axis"],
            slice_eval["p_axis"],
            slice_eval["U"],
            slice_eval["V"],
            density=1.0,
            color=slice_eval["speed"],
            cmap="viridis",
            linewidth=0.8,
            arrowsize=0.8,
        )
        _overlay_trace(axis, trace, pair, max_episodes=2)
        axis.scatter(
            [q_reference[pair.q_index]],
            [p_reference[pair.p_index]],
            s=32,
            marker="x",
            linewidths=1.2,
            color="white",
            zorder=5,
        )
        axis.set_xlabel(f"q[{pair.q_index}]")
        axis.set_ylabel(f"p[{pair.p_index}]")
        axis.set_title(f"{pair.kind}  q[{pair.q_index}] / p[{pair.p_index}]  s={pair.score:.3f}", loc="left", fontsize=10)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for axis in axes_array.flatten()[len(slices) :]:
        axis.axis("off")

    fig.suptitle(f"Hamiltonian slice flow candidates\n{checkpoint_name}", x=0.055, y=0.995, ha="left", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_scores_csv(output_path: Path, scores: list[PairScore]) -> None:
    fieldnames = list(asdict(scores[0]).keys()) if scores else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in scores:
            writer.writerow(asdict(row))


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    trace_path = Path(args.trace_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    world_model, config = _load_checkpoint_world_model(checkpoint_path, device)
    trace = _load_trace(trace_path)

    valid_mask = np.asarray(trace["valid_mask"], dtype=bool)
    q_flat = _flatten_valid_vectors(trace["q"], valid_mask)
    p_flat = _flatten_valid_vectors(trace["p"], valid_mask)
    q_reference = np.median(q_flat, axis=0).astype(np.float32)
    p_reference = np.median(p_flat, axis=0).astype(np.float32)

    all_scores = score_pairs(trace)
    selected_pairs = select_pairs(all_scores, same_k=args.same_k, cross_k=args.cross_k)
    if not selected_pairs:
        raise RuntimeError("No q/p pairs were selected for rendering.")

    render_dir = output_dir / "pairs"
    render_dir.mkdir(parents=True, exist_ok=True)

    slice_bundle: list[tuple[PairScore, dict[str, np.ndarray]]] = []
    for pair in selected_pairs:
        q_values = q_flat[:, pair.q_index]
        p_values = p_flat[:, pair.p_index]
        slice_eval = evaluate_slice(
            world_model=world_model,
            q_reference=q_reference,
            p_reference=p_reference,
            pair=pair,
            q_values=q_values,
            p_values=p_values,
            grid_size=int(args.grid_size),
            quantile_lo=float(args.trace_quantile_lo),
            quantile_hi=float(args.trace_quantile_hi),
            device=device,
        )
        slice_bundle.append((pair, slice_eval))
        filename = f"hamiltonian_slice_{pair.kind}_q{pair.q_index}_p{pair.p_index}.png"
        draw_pair_figure(
            output_path=render_dir / filename,
            slice_eval=slice_eval,
            trace=trace,
            pair=pair,
            q_reference=q_reference,
            p_reference=p_reference,
            checkpoint_name=checkpoint_path.parent.parent.parent.name,
        )

    write_scores_csv(output_dir / "pair_scores.csv", all_scores)
    with (output_dir / "selected_pairs.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(pair) for pair in selected_pairs], handle, indent=2)

    draw_overview(
        output_path=output_dir / "hamiltonian_slice_overview.png",
        slices=slice_bundle,
        trace=trace,
        q_reference=q_reference,
        p_reference=p_reference,
        checkpoint_name=checkpoint_path.parent.parent.parent.name,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "trace_path": str(trace_path),
        "task": str(np.asarray(trace["task"]).item()),
        "seed": int(np.asarray(trace["seed"]).item()),
        "mode": str(np.asarray(trace["mode"]).item()),
        "q_reference": q_reference.tolist(),
        "p_reference": p_reference.tolist(),
        "selected_pairs": [asdict(pair) for pair in selected_pairs],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(output_dir / "hamiltonian_slice_overview.png")
    for pair in selected_pairs:
        print(render_dir / f"hamiltonian_slice_{pair.kind}_q{pair.q_index}_p{pair.p_index}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
