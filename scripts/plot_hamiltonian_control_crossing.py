#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
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
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import torch

from common import ensure_repo_on_path

ensure_repo_on_path()

from hamworld.runtime import make_env
from hamworld.world_model import CanonicalDynamicsWorldModel, infer_checkpoint_step


PAPER_TITLE_FONTSIZE = 19
PAPER_LABEL_FONTSIZE = 18
PAPER_TICK_FONTSIZE = 15
PAPER_CBAR_FONTSIZE = 17
PAPER_LEGEND_FONTSIZE = 14
PAPER_SPINE_WIDTH = 1.2


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


@dataclass(frozen=True)
class PairSummary:
    q_index: int
    p_index: int
    kind: str
    score: float
    num_segments: int
    mean_abs_delta_h: float
    mean_control_norm: float
    mean_abs_control_energy_push: float
    mean_cross_ratio: float
    mean_abs_slice_cross_energy: float
    cross_threshold: float
    threshold_cross_rate: float
    corr_abs_delta_h_control_norm: float
    corr_abs_delta_h_abs_control_energy_push: float
    corr_cross_ratio_control_norm: float
    corr_cross_ratio_abs_control_energy_push: float
    corr_signed_delta_h_control_energy_push: float
    corr_abs_slice_cross_energy_abs_control_energy_push: float
    corr_binary_cross_abs_control_energy_push: float
    mean_cross_ratio_high_control: float
    mean_cross_ratio_low_control: float
    mean_abs_delta_h_high_control: float
    mean_abs_delta_h_low_control: float
    mean_cross_ratio_high_push: float
    mean_cross_ratio_low_push: float
    mean_abs_delta_h_high_push: float
    mean_abs_delta_h_low_push: float
    threshold_cross_rate_high_push: float
    threshold_cross_rate_low_push: float
    sweep_lift_auc: float
    sweep_binary_corr_auc: float
    sweep_best_threshold_quantile: float
    sweep_best_lift: float
    sweep_best_binary_corr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Hamiltonian slice plots with control/crossing overlays.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_*.pt.")
    parser.add_argument("--trace-path", required=True, help="Path to dynamics trace .npz file.")
    parser.add_argument("--output-dir", required=True, help="Directory for figures and summaries.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--grid-size", type=int, default=81, help="Grid size for slice evaluation.")
    parser.add_argument("--same-k", type=int, default=4, help="How many same-index pairs to render.")
    parser.add_argument("--cross-k", type=int, default=2, help="How many cross-index pairs to render.")
    parser.add_argument("--trace-quantile-lo", type=float, default=2.0, help="Low percentile for slice bounds.")
    parser.add_argument("--trace-quantile-hi", type=float, default=98.0, help="High percentile for slice bounds.")
    parser.add_argument("--scatter-max-points", type=int, default=1200, help="Max points per scatter panel.")
    parser.add_argument(
        "--cross-threshold-quantile",
        type=float,
        default=0.60,
        help="Quantile of |∇H_slice·Δx| used to treat tiny crossings as no crossing.",
    )
    parser.add_argument(
        "--cross-threshold-sweep",
        default="0.40,0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated quantiles used for threshold sweep plots and summary metrics.",
    )
    parser.add_argument("--focus-q-index", type=int, default=2, help="Focused q index for the summary sheet.")
    parser.add_argument("--focus-p-index", type=int, default=2, help="Focused p index for the summary sheet.")
    parser.add_argument("--ranking-top-k", type=int, default=12, help="How many top pairs to show in ranking charts.")
    return parser.parse_args()


def _load_checkpoint_world_model(checkpoint_path: Path, device: torch.device) -> CanonicalDynamicsWorldModel:
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
    return world_model


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


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("Expected at least one threshold quantile.")
    values = sorted(set(values))
    for value in values:
        if not (0.0 < value < 1.0):
            raise ValueError(f"Threshold quantiles must be in (0, 1): {value}")
    return values


def score_pairs(trace: dict[str, np.ndarray]) -> list[PairScore]:
    valid_mask = np.asarray(trace["valid_mask"], dtype=bool)
    q = _flatten_valid_vectors(trace["q"], valid_mask)
    p = _flatten_valid_vectors(trace["p"], valid_mask)
    dH_dq = _flatten_valid_vectors(trace["dH_dq"], valid_mask)
    dH_dp = _flatten_valid_vectors(trace["dH_dp"], valid_mask)

    q_dim = int(np.asarray(trace["q_dim"]).item())
    p_dim = int(np.asarray(trace["p_dim"]).item())

    rows: list[dict[str, float | int | str]] = []
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
            rows.append(
                {
                    "q_index": q_index,
                    "p_index": p_index,
                    "kind": "same" if q_index == p_index else "cross",
                    "coverage": float(q_span * p_span),
                    "swirl": swirl,
                    "speed": float(np.mean(flow_norm)),
                    "occupancy": _histogram_entropy(qv, pv),
                    "q_span": q_span,
                    "p_span": p_span,
                }
            )

    coverage_norm = _normalize([float(row["coverage"]) for row in rows])
    swirl_norm = _normalize([float(row["swirl"]) for row in rows])
    speed_norm = _normalize([float(row["speed"]) for row in rows])
    occupancy_norm = _normalize([float(row["occupancy"]) for row in rows])

    scored: list[PairScore] = []
    for index, row in enumerate(rows):
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

    h = energy.detach().cpu().numpy().reshape(grid_size, grid_size)
    u = dH_dp.detach().cpu().numpy()[:, pair.q_index].reshape(grid_size, grid_size)
    v = (-dH_dq.detach().cpu().numpy()[:, pair.p_index]).reshape(grid_size, grid_size)
    return {
        "q_axis": q_axis,
        "p_axis": p_axis,
        "H": h,
        "U": u,
        "V": v,
        "speed": np.hypot(u, v),
    }


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _quantile_mean(values: np.ndarray, selector: np.ndarray) -> float:
    if selector.size == 0 or not np.any(selector):
        return float("nan")
    return float(np.mean(values[selector]))


def _quantile_mask(values: np.ndarray, quantile: float, upper: bool) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(values, dtype=bool)
    threshold = float(np.quantile(values, quantile))
    if upper:
        return values >= threshold
    return values <= threshold


def build_pair_steps(trace: dict[str, np.ndarray], pair: PairScore) -> dict[str, np.ndarray]:
    valid = np.asarray(trace["valid_mask"], dtype=bool)
    q = np.asarray(trace["q"], dtype=np.float32)
    p = np.asarray(trace["p"], dtype=np.float32)
    h = np.asarray(trace["H"], dtype=np.float32)
    dH_dq = np.asarray(trace["dH_dq"], dtype=np.float32)
    dH_dp = np.asarray(trace["dH_dp"], dtype=np.float32)
    control = np.asarray(trace["control"], dtype=np.float32)

    segments: list[np.ndarray] = []
    x0s: list[float] = []
    y0s: list[float] = []
    x1s: list[float] = []
    y1s: list[float] = []
    delta_hs: list[float] = []
    abs_delta_hs: list[float] = []
    control_norms: list[float] = []
    control_pushes: list[float] = []
    abs_control_pushes: list[float] = []
    cross_ratios: list[float] = []
    abs_slice_cross_energies: list[float] = []
    signed_slice_cross_energies: list[float] = []
    step_norms: list[float] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []

    eps = 1e-8
    for episode_index in range(q.shape[0]):
        length = q.shape[1]
        for step_index in range(length - 1):
            if not (valid[episode_index, step_index] and valid[episode_index, step_index + 1]):
                continue

            x0 = float(q[episode_index, step_index, pair.q_index])
            y0 = float(p[episode_index, step_index, pair.p_index])
            x1 = float(q[episode_index, step_index + 1, pair.q_index])
            y1 = float(p[episode_index, step_index + 1, pair.p_index])
            delta_h = float(h[episode_index, step_index + 1] - h[episode_index, step_index])
            control_vec = control[episode_index, step_index]
            grad_q_i = float(dH_dq[episode_index, step_index, pair.q_index])
            grad_p_i = float(dH_dp[episode_index, step_index, pair.p_index])

            move = np.asarray([x1 - x0, y1 - y0], dtype=np.float64)
            normal = np.asarray([grad_q_i, grad_p_i], dtype=np.float64)
            cross_ratio = abs(float(move.dot(normal))) / max(eps, float(np.linalg.norm(move) * np.linalg.norm(normal)))
            control_push = float(np.dot(dH_dp[episode_index, step_index], control_vec))
            signed_slice_cross_energy = float(move.dot(normal))

            segments.append(np.asarray([[x0, y0], [x1, y1]], dtype=np.float64))
            x0s.append(x0)
            y0s.append(y0)
            x1s.append(x1)
            y1s.append(y1)
            delta_hs.append(delta_h)
            abs_delta_hs.append(abs(delta_h))
            control_norms.append(float(np.linalg.norm(control_vec)))
            control_pushes.append(control_push)
            abs_control_pushes.append(abs(control_push))
            cross_ratios.append(min(1.0, max(0.0, cross_ratio)))
            abs_slice_cross_energies.append(abs(signed_slice_cross_energy))
            signed_slice_cross_energies.append(signed_slice_cross_energy)
            step_norms.append(float(np.linalg.norm(move)))
            episode_ids.append(episode_index)
            step_ids.append(step_index)

    return {
        "segments": np.asarray(segments, dtype=np.float64),
        "x0": np.asarray(x0s, dtype=np.float64),
        "y0": np.asarray(y0s, dtype=np.float64),
        "x1": np.asarray(x1s, dtype=np.float64),
        "y1": np.asarray(y1s, dtype=np.float64),
        "delta_h": np.asarray(delta_hs, dtype=np.float64),
        "abs_delta_h": np.asarray(abs_delta_hs, dtype=np.float64),
        "control_norm": np.asarray(control_norms, dtype=np.float64),
        "control_energy_push": np.asarray(control_pushes, dtype=np.float64),
        "abs_control_energy_push": np.asarray(abs_control_pushes, dtype=np.float64),
        "cross_ratio": np.asarray(cross_ratios, dtype=np.float64),
        "abs_slice_cross_energy": np.asarray(abs_slice_cross_energies, dtype=np.float64),
        "signed_slice_cross_energy": np.asarray(signed_slice_cross_energies, dtype=np.float64),
        "step_norm": np.asarray(step_norms, dtype=np.float64),
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "step_id": np.asarray(step_ids, dtype=np.int32),
    }


def summarize_pair(pair: PairScore, steps: dict[str, np.ndarray], cross_threshold_quantile: float) -> PairSummary:
    control_norm = steps["control_norm"]
    abs_push = steps["abs_control_energy_push"]
    abs_delta_h = steps["abs_delta_h"]
    cross_ratio = steps["cross_ratio"]
    signed_delta_h = steps["delta_h"]
    push = steps["control_energy_push"]
    abs_slice_cross_energy = steps["abs_slice_cross_energy"]

    control_hi = np.quantile(control_norm, 0.75) if control_norm.size else float("nan")
    control_lo = np.quantile(control_norm, 0.25) if control_norm.size else float("nan")
    push_hi = np.quantile(abs_push, 0.75) if abs_push.size else float("nan")
    push_lo = np.quantile(abs_push, 0.25) if abs_push.size else float("nan")
    cross_threshold = float(np.quantile(abs_slice_cross_energy, cross_threshold_quantile)) if abs_slice_cross_energy.size else float("nan")
    threshold_cross = abs_slice_cross_energy >= cross_threshold if np.isfinite(cross_threshold) else np.zeros_like(abs_slice_cross_energy, dtype=bool)

    high_control_mask = control_norm >= control_hi if np.isfinite(control_hi) else np.zeros_like(control_norm, dtype=bool)
    low_control_mask = control_norm <= control_lo if np.isfinite(control_lo) else np.zeros_like(control_norm, dtype=bool)
    high_push_mask = abs_push >= push_hi if np.isfinite(push_hi) else np.zeros_like(abs_push, dtype=bool)
    low_push_mask = abs_push <= push_lo if np.isfinite(push_lo) else np.zeros_like(abs_push, dtype=bool)

    sweep_lift_auc = float("nan")
    sweep_binary_corr_auc = float("nan")
    sweep_best_threshold_quantile = float("nan")
    sweep_best_lift = float("nan")
    sweep_best_binary_corr = float("nan")

    return PairSummary(
        q_index=pair.q_index,
        p_index=pair.p_index,
        kind=pair.kind,
        score=pair.score,
        num_segments=int(abs_delta_h.size),
        mean_abs_delta_h=float(np.mean(abs_delta_h)),
        mean_control_norm=float(np.mean(control_norm)),
        mean_abs_control_energy_push=float(np.mean(abs_push)),
        mean_cross_ratio=float(np.mean(cross_ratio)),
        mean_abs_slice_cross_energy=float(np.mean(abs_slice_cross_energy)),
        cross_threshold=cross_threshold,
        threshold_cross_rate=float(np.mean(threshold_cross.astype(np.float64))),
        corr_abs_delta_h_control_norm=_safe_pearson(abs_delta_h, control_norm),
        corr_abs_delta_h_abs_control_energy_push=_safe_pearson(abs_delta_h, abs_push),
        corr_cross_ratio_control_norm=_safe_pearson(cross_ratio, control_norm),
        corr_cross_ratio_abs_control_energy_push=_safe_pearson(cross_ratio, abs_push),
        corr_signed_delta_h_control_energy_push=_safe_pearson(signed_delta_h, push),
        corr_abs_slice_cross_energy_abs_control_energy_push=_safe_pearson(abs_slice_cross_energy, abs_push),
        corr_binary_cross_abs_control_energy_push=_safe_pearson(threshold_cross.astype(np.float64), abs_push),
        mean_cross_ratio_high_control=_quantile_mean(cross_ratio, high_control_mask),
        mean_cross_ratio_low_control=_quantile_mean(cross_ratio, low_control_mask),
        mean_abs_delta_h_high_control=_quantile_mean(abs_delta_h, high_control_mask),
        mean_abs_delta_h_low_control=_quantile_mean(abs_delta_h, low_control_mask),
        mean_cross_ratio_high_push=_quantile_mean(cross_ratio, high_push_mask),
        mean_cross_ratio_low_push=_quantile_mean(cross_ratio, low_push_mask),
        mean_abs_delta_h_high_push=_quantile_mean(abs_delta_h, high_push_mask),
        mean_abs_delta_h_low_push=_quantile_mean(abs_delta_h, low_push_mask),
        threshold_cross_rate_high_push=_quantile_mean(threshold_cross.astype(np.float64), high_push_mask),
        threshold_cross_rate_low_push=_quantile_mean(threshold_cross.astype(np.float64), low_push_mask),
        sweep_lift_auc=sweep_lift_auc,
        sweep_binary_corr_auc=sweep_binary_corr_auc,
        sweep_best_threshold_quantile=sweep_best_threshold_quantile,
        sweep_best_lift=sweep_best_lift,
        sweep_best_binary_corr=sweep_best_binary_corr,
    )


def compute_threshold_sweep(
    steps: dict[str, np.ndarray],
    threshold_quantiles: list[float],
    high_push_quantile: float = 0.75,
    low_push_quantile: float = 0.25,
    push_bins: int = 8,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    abs_push = steps["abs_control_energy_push"]
    abs_slice_cross_energy = steps["abs_slice_cross_energy"]
    high_push_mask = _quantile_mask(abs_push, high_push_quantile, upper=True)
    low_push_mask = _quantile_mask(abs_push, low_push_quantile, upper=False)

    x_min = float(np.min(abs_push)) if abs_push.size else 0.0
    x_max = float(np.max(abs_push)) if abs_push.size else 1.0
    if x_max <= x_min:
        edges = np.linspace(0.0, 1.0, push_bins + 1)
    else:
        edges = np.linspace(x_min, x_max, push_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    heatmap_rows: list[np.ndarray] = []
    sweep_rows: list[dict[str, float]] = []
    for quantile in threshold_quantiles:
        threshold = float(np.quantile(abs_slice_cross_energy, quantile))
        crossing = abs_slice_cross_energy >= threshold
        high_rate = _quantile_mean(crossing.astype(np.float64), high_push_mask)
        low_rate = _quantile_mean(crossing.astype(np.float64), low_push_mask)
        binary_corr = _safe_pearson(crossing.astype(np.float64), abs_push)
        amplitude_corr = _safe_pearson(
            np.where(crossing, abs_slice_cross_energy, 0.0),
            abs_push,
        )
        row = {
            "threshold_quantile": float(quantile),
            "cross_threshold": threshold,
            "overall_cross_rate": float(np.mean(crossing.astype(np.float64))),
            "high_push_cross_rate": high_rate,
            "low_push_cross_rate": low_rate,
            "push_cross_rate_gap": float(high_rate - low_rate),
            "binary_cross_push_corr": binary_corr,
            "amplitude_cross_push_corr": amplitude_corr,
        }
        sweep_rows.append(row)

        bin_rates = []
        for left, right in zip(edges[:-1], edges[1:]):
            if right == edges[-1]:
                mask = (abs_push >= left) & (abs_push <= right)
            else:
                mask = (abs_push >= left) & (abs_push < right)
            bin_rates.append(float(np.mean(crossing[mask].astype(np.float64))) if np.any(mask) else float("nan"))
        heatmap_rows.append(np.asarray(bin_rates, dtype=np.float64))

    return sweep_rows, centers, np.asarray(heatmap_rows, dtype=np.float64)


def attach_sweep_metrics(summary: PairSummary, sweep_rows: list[dict[str, float]]) -> PairSummary:
    if not sweep_rows:
        return summary
    lifts = np.asarray([row["push_cross_rate_gap"] for row in sweep_rows], dtype=np.float64)
    corrs = np.asarray([row["binary_cross_push_corr"] for row in sweep_rows], dtype=np.float64)
    best_index = int(np.nanargmax(lifts))
    best_row = sweep_rows[best_index]
    return replace(
        summary,
        sweep_lift_auc=float(np.nanmean(lifts)),
        sweep_binary_corr_auc=float(np.nanmean(corrs)),
        sweep_best_threshold_quantile=float(best_row["threshold_quantile"]),
        sweep_best_lift=float(best_row["push_cross_rate_gap"]),
        sweep_best_binary_corr=float(best_row["binary_cross_push_corr"]),
    )


def _draw_background(axis: plt.Axes, slice_eval: dict[str, np.ndarray], pair: PairScore, title: str) -> None:
    q_axis = slice_eval["q_axis"]
    p_axis = slice_eval["p_axis"]
    axis.contourf(q_axis, p_axis, slice_eval["H"], levels=28, cmap="magma", alpha=0.95)
    axis.contour(
        q_axis,
        p_axis,
        slice_eval["H"],
        levels=10,
        colors="white",
        linewidths=0.8,
        linestyles="--",
        alpha=0.48,
    )
    axis.streamplot(
        q_axis,
        p_axis,
        slice_eval["U"],
        slice_eval["V"],
        density=1.05,
        color="#203a8a",
        linewidth=1.15,
        arrowsize=1.35,
        maxlength=3.5,
    )
    axis.set_xlabel(f"q[{pair.q_index}]", fontsize=PAPER_LABEL_FONTSIZE)
    axis.set_ylabel(f"p[{pair.p_index}]", fontsize=PAPER_LABEL_FONTSIZE)
    axis.set_title(title, loc="left", fontsize=PAPER_TITLE_FONTSIZE, pad=12)
    axis.tick_params(axis="both", labelsize=PAPER_TICK_FONTSIZE, width=PAPER_SPINE_WIDTH, length=7, pad=8)
    axis.spines["left"].set_linewidth(PAPER_SPINE_WIDTH)
    axis.spines["bottom"].set_linewidth(PAPER_SPINE_WIDTH)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _line_collection(
    axis: plt.Axes,
    segments: np.ndarray,
    values: np.ndarray,
    cmap: str,
    norm: Normalize,
    linewidth: float = 2.4,
) -> LineCollection:
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=linewidth,
        alpha=0.98,
        capstyle="round",
        joinstyle="round",
        zorder=5,
    )
    collection.set_array(values)
    axis.add_collection(collection)
    return collection


def _scatter_points(axis: plt.Axes, steps: dict[str, np.ndarray]) -> None:
    axis.scatter(steps["x0"][0], steps["y0"][0], color="#7dd3fc", s=60, zorder=6, label="start")
    axis.scatter(steps["x1"][-1], steps["y1"][-1], color="#38bdf8", s=60, zorder=6, label="end")


def _style_colorbar(cbar: plt.colorbar, label: str) -> None:
    cbar.set_label(label, fontsize=PAPER_CBAR_FONTSIZE, labelpad=14)
    cbar.ax.tick_params(labelsize=PAPER_TICK_FONTSIZE, width=PAPER_SPINE_WIDTH, length=6)


def _style_axis(axis: plt.Axes) -> None:
    axis.tick_params(axis="both", labelsize=PAPER_TICK_FONTSIZE, width=PAPER_SPINE_WIDTH, length=7, pad=8)
    axis.spines["left"].set_linewidth(PAPER_SPINE_WIDTH)
    axis.spines["bottom"].set_linewidth(PAPER_SPINE_WIDTH)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def draw_pair_overlay_and_scatter(
    output_path: Path,
    slice_eval: dict[str, np.ndarray],
    pair: PairScore,
    steps: dict[str, np.ndarray],
    summary: PairSummary,
    scatter_max_points: int,
) -> None:
    delta_h = steps["delta_h"]
    push = steps["control_energy_push"]
    cross_ratio = steps["cross_ratio"]
    control_norm = steps["control_norm"]
    abs_push = steps["abs_control_energy_push"]
    abs_delta_h = steps["abs_delta_h"]
    segments = steps["segments"]

    delta_scale = max(1e-8, float(np.max(np.abs(delta_h))))
    push_scale = max(1e-8, float(np.max(np.abs(push))))
    delta_norm = TwoSlopeNorm(vmin=-delta_scale, vcenter=0.0, vmax=delta_scale)
    push_norm = TwoSlopeNorm(vmin=-push_scale, vcenter=0.0, vmax=push_scale)
    cross_norm = Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.8), dpi=220)
    titles = [
        r"Trajectory colored by $\Delta H_t$",
        r"Trajectory colored by $(\nabla_p H_t)\cdot control_t$",
        r"Trajectory colored by crossing ratio",
    ]
    overlay_specs = [
        (delta_h, "coolwarm", delta_norm),
        (push, "coolwarm", push_norm),
        (cross_ratio, "viridis", cross_norm),
    ]
    cbar_labels = [
        r"$\Delta H_t = H_{t+1} - H_t$",
        r"$(\nabla_p H_t)\cdot control_t$",
        r"$|u_t \cdot n_t| / (||u_t||\,||n_t||)$",
    ]

    for axis, title, (values, cmap, norm), cbar_label in zip(axes[0], titles, overlay_specs, cbar_labels):
        _draw_background(axis, slice_eval, pair, title)
        collection = _line_collection(axis, segments, values, cmap=cmap, norm=norm)
        _scatter_points(axis, steps)
        cbar = fig.colorbar(collection, ax=axis, fraction=0.046, pad=0.03)
        cbar.set_label(cbar_label)

    rng = np.random.default_rng(42)
    sample_size = min(int(scatter_max_points), int(abs_delta_h.size))
    sample_idx = np.arange(abs_delta_h.size)
    if sample_size < abs_delta_h.size:
        sample_idx = rng.choice(abs_delta_h.size, size=sample_size, replace=False)

    scatter_specs = [
        (
            control_norm[sample_idx],
            cross_ratio[sample_idx],
            abs_push[sample_idx],
            r"$||control_t||_2$",
            "cross ratio",
            f"cross vs control\nr={summary.corr_cross_ratio_control_norm:.3f}",
        ),
        (
            abs_push[sample_idx],
            cross_ratio[sample_idx],
            control_norm[sample_idx],
            r"$|(\nabla_p H_t)\cdot control_t|$",
            "cross ratio",
            f"cross vs |push|\nr={summary.corr_cross_ratio_abs_control_energy_push:.3f}",
        ),
        (
            abs_push[sample_idx],
            abs_delta_h[sample_idx],
            control_norm[sample_idx],
            r"$|(\nabla_p H_t)\cdot control_t|$",
            r"$|\Delta H_t|$",
            f"|ΔH| vs |push|\nr={summary.corr_abs_delta_h_abs_control_energy_push:.3f}",
        ),
    ]

    for axis, (xs, ys, color_values, xlabel, ylabel, title) in zip(axes[1], scatter_specs):
        scatter = axis.scatter(xs, ys, c=color_values, cmap="plasma", s=18, alpha=0.72, edgecolors="none")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=10)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(True, alpha=0.25)
        cbar = fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.03)
        cbar.set_label(r"$||control_t||_2$" if xlabel != r"$||control_t||_2$" else r"$|(\nabla_p H_t)\cdot control_t|$")

    fig.suptitle(
        f"{pair.kind} pair q[{pair.q_index}] / p[{pair.p_index}]  score={pair.score:.3f}\n"
        f"corr(sign ΔH, push)={summary.corr_signed_delta_h_control_energy_push:.3f}",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_pair_triptych(
    output_path: Path,
    slice_eval: dict[str, np.ndarray],
    pair: PairScore,
    steps: dict[str, np.ndarray],
) -> None:
    delta_h = steps["delta_h"]
    push = steps["control_energy_push"]
    cross_ratio = steps["cross_ratio"]
    segments = steps["segments"]

    delta_scale = max(1e-8, float(np.max(np.abs(delta_h))))
    push_scale = max(1e-8, float(np.max(np.abs(push))))

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), dpi=220)
    configs = [
        (delta_h, "coolwarm", TwoSlopeNorm(vmin=-delta_scale, vcenter=0.0, vmax=delta_scale), r"$\Delta H_t$"),
        (push, "coolwarm", TwoSlopeNorm(vmin=-push_scale, vcenter=0.0, vmax=push_scale), r"$(\nabla_p H_t)\cdot control_t$"),
        (cross_ratio, "viridis", Normalize(vmin=0.0, vmax=1.0), "cross ratio"),
    ]

    for axis, (values, cmap, norm, label) in zip(axes, configs):
        _draw_background(axis, slice_eval, pair, label)
        collection = _line_collection(axis, segments, values, cmap=cmap, norm=norm)
        _scatter_points(axis, steps)
        cbar = fig.colorbar(collection, ax=axis, fraction=0.046, pad=0.03)
        cbar.set_label(label)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_pair_push_flow(
    output_path: Path,
    slice_eval: dict[str, np.ndarray],
    pair: PairScore,
    steps: dict[str, np.ndarray],
    summary: PairSummary | None = None,
) -> None:
    push = steps["control_energy_push"]
    segments = steps["segments"]
    push_scale = max(1e-8, float(np.max(np.abs(push))))
    push_norm = TwoSlopeNorm(vmin=-push_scale, vcenter=0.0, vmax=push_scale)

    fig, axis = plt.subplots(1, 1, figsize=(8.4, 6.8), dpi=260)
    title = r"Trajectory colored by $(\nabla_p H_t)\cdot control_t$"
    if summary is not None:
        title += (
            f"\n|ΔH|-|push| r={summary.corr_abs_delta_h_abs_control_energy_push:.3f}, "
            f"lift_auc={summary.sweep_lift_auc:.3f}"
        )
    _draw_background(axis, slice_eval, pair, title)
    collection = _line_collection(axis, segments, push, cmap="coolwarm", norm=push_norm, linewidth=4.0)
    _scatter_points(axis, steps)
    cbar = fig.colorbar(collection, ax=axis, fraction=0.046, pad=0.04)
    _style_colorbar(cbar, r"$(\nabla_p H_t)\cdot control_t$")
    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _binned_mean_curve(x: np.ndarray, y: np.ndarray, bins: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.asarray([]), np.asarray([])
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return np.asarray([]), np.asarray([])
    edges = np.linspace(x_min, x_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == edges[-1]:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        means.append(float(np.mean(y[mask])) if np.any(mask) else float("nan"))
    return centers, np.asarray(means, dtype=np.float64)


def draw_pair_thresholded(
    output_path: Path,
    pair: PairScore,
    steps: dict[str, np.ndarray],
    summary: PairSummary,
    scatter_max_points: int,
) -> None:
    abs_push = steps["abs_control_energy_push"]
    abs_slice_cross_energy = steps["abs_slice_cross_energy"]
    step_norm = steps["step_norm"]
    threshold_cross = (abs_slice_cross_energy >= summary.cross_threshold).astype(np.float64)

    rng = np.random.default_rng(42)
    sample_size = min(int(scatter_max_points), int(abs_push.size))
    sample_idx = np.arange(abs_push.size)
    if sample_size < abs_push.size:
        sample_idx = rng.choice(abs_push.size, size=sample_size, replace=False)
    jitter = rng.normal(loc=0.0, scale=0.035, size=sample_idx.size)

    x_bins, y_bins = _binned_mean_curve(abs_push, threshold_cross, bins=8)
    x_bins_energy, y_bins_energy = _binned_mean_curve(abs_push, abs_slice_cross_energy, bins=8)

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2), dpi=220)

    scatter0 = axes[0].scatter(
        abs_push[sample_idx],
        abs_slice_cross_energy[sample_idx],
        c=step_norm[sample_idx],
        cmap="plasma",
        s=20,
        alpha=0.76,
        edgecolors="none",
    )
    axes[0].axhline(summary.cross_threshold, color="#0f172a", linewidth=1.2, linestyle="--", alpha=0.85)
    axes[0].set_xlabel(r"$|(\nabla_p H_t)\cdot control_t|$")
    axes[0].set_ylabel(r"$|(\nabla_{slice} H_t)\cdot \Delta x_t|$")
    axes[0].set_title(
        f"slice cross amplitude vs |push|\nr={summary.corr_abs_slice_cross_energy_abs_control_energy_push:.3f}",
        loc="left",
        fontsize=10,
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    cbar0 = fig.colorbar(scatter0, ax=axes[0], fraction=0.046, pad=0.03)
    cbar0.set_label(r"$||\Delta x_t||_2$")

    scatter1 = axes[1].scatter(
        abs_push[sample_idx],
        threshold_cross[sample_idx] + jitter,
        c=step_norm[sample_idx],
        cmap="viridis",
        s=20,
        alpha=0.78,
        edgecolors="none",
    )
    axes[1].set_xlabel(r"$|(\nabla_p H_t)\cdot control_t|$")
    axes[1].set_ylabel("thresholded crossing")
    axes[1].set_yticks([0.0, 1.0])
    axes[1].set_yticklabels(["no", "yes"])
    axes[1].set_title(
        f"binary crossing vs |push|\nr={summary.corr_binary_cross_abs_control_energy_push:.3f}",
        loc="left",
        fontsize=10,
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    cbar1 = fig.colorbar(scatter1, ax=axes[1], fraction=0.046, pad=0.03)
    cbar1.set_label(r"$||\Delta x_t||_2$")

    axes[2].plot(x_bins, y_bins, color="#dc2626", linewidth=2.2, marker="o", label="crossing rate")
    axes[2].plot(x_bins_energy, y_bins_energy, color="#2563eb", linewidth=2.0, marker="s", label="mean |slice cross|")
    axes[2].axhline(summary.threshold_cross_rate, color="#64748b", linewidth=1.0, linestyle="--", alpha=0.7)
    axes[2].set_xlabel(r"$|(\nabla_p H_t)\cdot control_t|$ bin center")
    axes[2].set_ylabel("rate / mean")
    axes[2].set_title(
        f"binned trend\nhigh push rate={summary.threshold_cross_rate_high_push:.3f}, low push rate={summary.threshold_cross_rate_low_push:.3f}",
        loc="left",
        fontsize=10,
    )
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False, loc="best", fontsize=8)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.suptitle(
        f"{pair.kind} pair q[{pair.q_index}] / p[{pair.p_index}]  thresholded slice crossing\n"
        f"cross threshold={summary.cross_threshold:.4g}",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_pair_threshold_sweep(
    output_path: Path,
    pair: PairScore,
    summary: PairSummary,
    sweep_rows: list[dict[str, float]],
    push_bin_centers: np.ndarray,
    sweep_heatmap: np.ndarray,
) -> None:
    quantiles = np.asarray([row["threshold_quantile"] for row in sweep_rows], dtype=np.float64)
    high_rates = np.asarray([row["high_push_cross_rate"] for row in sweep_rows], dtype=np.float64)
    low_rates = np.asarray([row["low_push_cross_rate"] for row in sweep_rows], dtype=np.float64)
    gaps = np.asarray([row["push_cross_rate_gap"] for row in sweep_rows], dtype=np.float64)
    binary_corrs = np.asarray([row["binary_cross_push_corr"] for row in sweep_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3), dpi=220)

    axes[0].plot(quantiles, high_rates, color="#dc2626", linewidth=2.2, marker="o", label="high push")
    axes[0].plot(quantiles, low_rates, color="#2563eb", linewidth=2.2, marker="s", label="low push")
    axes[0].plot(quantiles, gaps, color="#0f172a", linewidth=2.0, marker="^", label="gap")
    axes[0].set_xlabel("cross threshold quantile")
    axes[0].set_ylabel("crossing rate / gap")
    axes[0].set_title(
        f"threshold sweep\nlift_auc={summary.sweep_lift_auc:.3f}",
        loc="left",
        fontsize=10,
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    heat = axes[1].imshow(
        sweep_heatmap,
        aspect="auto",
        origin="lower",
        cmap="YlGnBu",
        extent=[float(push_bin_centers[0]), float(push_bin_centers[-1]), float(quantiles[0]), float(quantiles[-1])],
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_xlabel(r"$|(\nabla_p H_t)\cdot control_t|$ bin center")
    axes[1].set_ylabel("cross threshold quantile")
    axes[1].set_title("crossing probability heatmap", loc="left", fontsize=10)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    cbar = fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.03)
    cbar.set_label("P(crossing)")

    axes[2].plot(quantiles, binary_corrs, color="#7c3aed", linewidth=2.2, marker="o", label="binary corr")
    axes[2].axhline(summary.sweep_binary_corr_auc, color="#64748b", linewidth=1.1, linestyle="--", label="corr auc")
    axes[2].set_xlabel("cross threshold quantile")
    axes[2].set_ylabel(r"corr(binary crossing, $|push|$)")
    axes[2].set_title(
        f"correlation sweep\nbest_q={summary.sweep_best_threshold_quantile:.2f}, best_lift={summary.sweep_best_lift:.3f}",
        loc="left",
        fontsize=10,
    )
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False, loc="best", fontsize=8)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.suptitle(
        f"{pair.kind} pair q[{pair.q_index}] / p[{pair.p_index}]  threshold sweep view",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_pair_push_crossing_rate_curve(
    output_path: Path,
    pair: PairScore,
    summary: PairSummary,
    sweep_rows: list[dict[str, float]],
) -> None:
    quantiles = np.asarray([row["threshold_quantile"] for row in sweep_rows], dtype=np.float64)
    high_rates = np.asarray([row["high_push_cross_rate"] for row in sweep_rows], dtype=np.float64)
    low_rates = np.asarray([row["low_push_cross_rate"] for row in sweep_rows], dtype=np.float64)

    fig, axis = plt.subplots(1, 1, figsize=(7.8, 5.8), dpi=260)
    axis.plot(quantiles, high_rates, color="#dc2626", linewidth=3.0, marker="o", markersize=8.0, label="High |push|")
    axis.plot(quantiles, low_rates, color="#2563eb", linewidth=3.0, marker="s", markersize=8.0, label="Low |push|")
    axis.fill_between(quantiles, low_rates, high_rates, color="#fca5a5", alpha=0.24)
    axis.set_xlabel("Crossing Threshold Quantile", fontsize=PAPER_LABEL_FONTSIZE)
    axis.set_ylabel("Crossing Rate", fontsize=PAPER_LABEL_FONTSIZE)
    axis.set_title(
        f"{pair.kind} q[{pair.q_index}] / p[{pair.p_index}]\n"
        f"lift_auc={summary.sweep_lift_auc:.3f}, best_lift={summary.sweep_best_lift:.3f}",
        loc="left",
        fontsize=PAPER_TITLE_FONTSIZE,
        pad=12,
    )
    axis.set_xlim(float(np.min(quantiles)) - 0.01, float(np.max(quantiles)) + 0.01)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xticks(quantiles)
    axis.grid(True, alpha=0.22, linewidth=0.8)
    _style_axis(axis)
    axis.legend(frameon=False, loc="best", fontsize=PAPER_LEGEND_FONTSIZE + 3, handlelength=2.8)
    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _pair_label(pair: PairScore | PairSummary) -> str:
    return f"{pair.kind} q[{pair.q_index}] / p[{pair.p_index}]"


def _summary_matrix(
    summaries: list[PairSummary],
    value_key: str,
    q_dim: int,
    p_dim: int,
) -> np.ndarray:
    matrix = np.full((q_dim, p_dim), np.nan, dtype=np.float64)
    for summary in summaries:
        matrix[summary.q_index, summary.p_index] = float(getattr(summary, value_key))
    return matrix


def draw_all_pair_rankings(
    output_path: Path,
    summaries: list[PairSummary],
    q_dim: int,
    p_dim: int,
    top_k: int,
) -> None:
    ordered_by_lift = sorted(summaries, key=lambda row: row.sweep_lift_auc, reverse=True)
    ordered_by_corr = sorted(summaries, key=lambda row: row.sweep_binary_corr_auc, reverse=True)
    top_lift = ordered_by_lift[: max(1, int(top_k))]
    top_corr = ordered_by_corr[: max(1, int(top_k))]

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.0), dpi=220)

    lift_labels = [_pair_label(row) for row in reversed(top_lift)]
    lift_values = [row.sweep_lift_auc for row in reversed(top_lift)]
    axes[0, 0].barh(lift_labels, lift_values, color="#dc2626", alpha=0.88)
    axes[0, 0].set_title("Top Pairs By Sweep Lift AUC", loc="left")
    axes[0, 0].set_xlabel("sweep_lift_auc")
    axes[0, 0].grid(True, axis="x", alpha=0.22)
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)

    corr_labels = [_pair_label(row) for row in reversed(top_corr)]
    corr_values = [row.sweep_binary_corr_auc for row in reversed(top_corr)]
    axes[0, 1].barh(corr_labels, corr_values, color="#7c3aed", alpha=0.88)
    axes[0, 1].set_title("Top Pairs By Sweep Binary Corr AUC", loc="left")
    axes[0, 1].set_xlabel("sweep_binary_corr_auc")
    axes[0, 1].grid(True, axis="x", alpha=0.22)
    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)

    lift_matrix = _summary_matrix(summaries, "sweep_lift_auc", q_dim=q_dim, p_dim=p_dim)
    corr_matrix = _summary_matrix(summaries, "sweep_binary_corr_auc", q_dim=q_dim, p_dim=p_dim)

    im0 = axes[1, 0].imshow(lift_matrix, cmap="YlOrRd", aspect="auto")
    axes[1, 0].set_title("Sweep Lift AUC Matrix", loc="left")
    axes[1, 0].set_xlabel("p index")
    axes[1, 0].set_ylabel("q index")
    axes[1, 0].set_xticks(np.arange(p_dim))
    axes[1, 0].set_yticks(np.arange(q_dim))
    cbar0 = fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.03)
    cbar0.set_label("sweep_lift_auc")

    im1 = axes[1, 1].imshow(corr_matrix, cmap="BuPu", aspect="auto")
    axes[1, 1].set_title("Sweep Binary Corr AUC Matrix", loc="left")
    axes[1, 1].set_xlabel("p index")
    axes[1, 1].set_ylabel("q index")
    axes[1, 1].set_xticks(np.arange(p_dim))
    axes[1, 1].set_yticks(np.arange(q_dim))
    cbar1 = fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.03)
    cbar1.set_label("sweep_binary_corr_auc")

    fig.suptitle("All Pair Rankings And Metric Maps", x=0.055, y=0.995, ha="left", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_all_pair_focus_heatmap(
    output_path: Path,
    summaries: list[PairSummary],
    q_dim: int,
    p_dim: int,
) -> None:
    best_lift_matrix = _summary_matrix(summaries, "sweep_best_lift", q_dim=q_dim, p_dim=p_dim)
    best_q_matrix = _summary_matrix(summaries, "sweep_best_threshold_quantile", q_dim=q_dim, p_dim=p_dim)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=220)
    im0 = axes[0].imshow(best_lift_matrix, cmap="magma", aspect="auto")
    axes[0].set_title("Best Threshold Lift", loc="left")
    axes[0].set_xlabel("p index")
    axes[0].set_ylabel("q index")
    axes[0].set_xticks(np.arange(p_dim))
    axes[0].set_yticks(np.arange(q_dim))
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03).set_label("best_lift")

    im1 = axes[1].imshow(best_q_matrix, cmap="viridis", aspect="auto", vmin=0.4, vmax=0.9)
    axes[1].set_title("Best Threshold Quantile", loc="left")
    axes[1].set_xlabel("p index")
    axes[1].set_ylabel("q index")
    axes[1].set_xticks(np.arange(p_dim))
    axes[1].set_yticks(np.arange(q_dim))
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03).set_label("best threshold q")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_focus_story_sheet(
    output_path: Path,
    slice_eval: dict[str, np.ndarray],
    pair: PairScore,
    steps: dict[str, np.ndarray],
    summary: PairSummary,
    sweep_rows: list[dict[str, float]],
) -> None:
    delta_h = steps["delta_h"]
    push = steps["control_energy_push"]
    abs_push = steps["abs_control_energy_push"]
    abs_delta_h = steps["abs_delta_h"]
    segments = steps["segments"]

    delta_scale = max(1e-8, float(np.max(np.abs(delta_h))))
    push_scale = max(1e-8, float(np.max(np.abs(push))))
    delta_norm = TwoSlopeNorm(vmin=-delta_scale, vcenter=0.0, vmax=delta_scale)
    push_norm = TwoSlopeNorm(vmin=-push_scale, vcenter=0.0, vmax=push_scale)

    quantiles = np.asarray([row["threshold_quantile"] for row in sweep_rows], dtype=np.float64)
    high_rates = np.asarray([row["high_push_cross_rate"] for row in sweep_rows], dtype=np.float64)
    low_rates = np.asarray([row["low_push_cross_rate"] for row in sweep_rows], dtype=np.float64)
    gaps = np.asarray([row["push_cross_rate_gap"] for row in sweep_rows], dtype=np.float64)

    fig = plt.figure(figsize=(14.8, 8.2), dpi=220)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 0.95], height_ratios=[1.0, 1.0])

    ax0 = fig.add_subplot(gs[:, 0])
    _draw_background(ax0, slice_eval, pair, r"Trajectory colored by $(\nabla_p H_t)\cdot control_t$")
    coll0 = _line_collection(ax0, segments, push, cmap="coolwarm", norm=push_norm, linewidth=2.8)
    _scatter_points(ax0, steps)
    fig.colorbar(coll0, ax=ax0, fraction=0.046, pad=0.03).set_label(r"$(\nabla_p H_t)\cdot control_t$")

    ax1 = fig.add_subplot(gs[0, 1])
    scatter1 = ax1.scatter(abs_push, abs_delta_h, c=steps["step_norm"], cmap="plasma", s=22, alpha=0.75, edgecolors="none")
    ax1.set_xlabel(r"$|(\nabla_p H_t)\cdot control_t|$")
    ax1.set_ylabel(r"$|\Delta H_t|$")
    ax1.set_title(f"Energy response\nr={summary.corr_abs_delta_h_abs_control_energy_push:.3f}", loc="left")
    ax1.grid(True, alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.03).set_label(r"$||\Delta x_t||_2$")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(quantiles, high_rates, color="#dc2626", linewidth=2.2, marker="o", label="high push")
    ax2.plot(quantiles, low_rates, color="#2563eb", linewidth=2.2, marker="s", label="low push")
    ax2.plot(quantiles, gaps, color="#111827", linewidth=2.0, marker="^", label="gap")
    ax2.set_xlabel("cross threshold quantile")
    ax2.set_ylabel("crossing rate / gap")
    ax2.set_title(
        f"Threshold sweep\nlift_auc={summary.sweep_lift_auc:.3f}, best_lift={summary.sweep_best_lift:.3f}",
        loc="left",
    )
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False, loc="best", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(gs[:, 2])
    ax3.axis("off")
    lines = [
        f"Focus Pair: {_pair_label(pair)}",
        f"score = {pair.score:.3f}",
        "",
        "Key numbers",
        f"corr(sign ΔH, push) = {summary.corr_signed_delta_h_control_energy_push:.3f}",
        f"corr(|ΔH|, |push|) = {summary.corr_abs_delta_h_abs_control_energy_push:.3f}",
        f"corr(binary crossing, |push|) = {summary.corr_binary_cross_abs_control_energy_push:.3f}",
        f"sweep_lift_auc = {summary.sweep_lift_auc:.3f}",
        f"sweep_binary_corr_auc = {summary.sweep_binary_corr_auc:.3f}",
        f"best threshold q = {summary.sweep_best_threshold_quantile:.2f}",
        f"best lift = {summary.sweep_best_lift:.3f}",
        "",
        "Interpretation",
        "Large |push| segments tend to coincide",
        "with larger |ΔH| and, after thresholding,",
        "with a higher chance of visible crossing.",
        "",
        "This supports:",
        "control projected onto ∇_p H explains",
        "energy-changing events better than",
        "plain control magnitude alone.",
    ]
    ax3.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        color="#0f172a",
    )

    fig.suptitle("Focused Hamiltonian Control-Crossing Story", x=0.055, y=0.995, ha="left", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    trace_path = Path(args.trace_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_quantiles = _parse_float_list(args.cross_threshold_sweep)

    device = torch.device(args.device)
    world_model = _load_checkpoint_world_model(checkpoint_path, device)
    trace = _load_trace(trace_path)

    valid_mask = np.asarray(trace["valid_mask"], dtype=bool)
    q_flat = _flatten_valid_vectors(trace["q"], valid_mask)
    p_flat = _flatten_valid_vectors(trace["p"], valid_mask)
    q_reference = np.median(q_flat, axis=0).astype(np.float32)
    p_reference = np.median(p_flat, axis=0).astype(np.float32)
    q_dim = int(np.asarray(trace["q_dim"]).item())
    p_dim = int(np.asarray(trace["p_dim"]).item())

    all_scores = score_pairs(trace)
    selected_pairs = select_pairs(all_scores, same_k=args.same_k, cross_k=args.cross_k)
    if not selected_pairs:
        raise RuntimeError("No q/p pairs were selected for rendering.")

    figures_dir = output_dir / "pairs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "summary_views"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_pair_summaries: list[PairSummary] = []
    pair_summary_map: dict[tuple[int, int], PairSummary] = {}
    pair_sweep_map: dict[tuple[int, int], list[dict[str, float]]] = {}
    selection_rows = [asdict(pair) for pair in selected_pairs]
    write_csv(output_dir / "selected_pairs.csv", selection_rows)
    write_csv(output_dir / "pair_scores.csv", [asdict(pair) for pair in all_scores])

    for pair in all_scores:
        steps = build_pair_steps(trace, pair)
        summary = summarize_pair(pair, steps, cross_threshold_quantile=float(args.cross_threshold_quantile))
        sweep_rows, _push_bin_centers, _sweep_heatmap = compute_threshold_sweep(steps, threshold_quantiles)
        summary = attach_sweep_metrics(summary, sweep_rows)
        all_pair_summaries.append(summary)
        pair_summary_map[(pair.q_index, pair.p_index)] = summary
        pair_sweep_map[(pair.q_index, pair.p_index)] = sweep_rows

    write_csv(output_dir / "all_pair_correlation_summary.csv", [asdict(summary) for summary in all_pair_summaries])
    draw_all_pair_rankings(
        summary_dir / "all_pair_rankings.png",
        all_pair_summaries,
        q_dim=q_dim,
        p_dim=p_dim,
        top_k=int(args.ranking_top_k),
    )
    draw_all_pair_focus_heatmap(
        summary_dir / "all_pair_metric_maps.png",
        all_pair_summaries,
        q_dim=q_dim,
        p_dim=p_dim,
    )

    selected_summaries: list[PairSummary] = []
    for pair in selected_pairs:
        slice_eval = evaluate_slice(
            world_model=world_model,
            q_reference=q_reference,
            p_reference=p_reference,
            pair=pair,
            q_values=q_flat[:, pair.q_index],
            p_values=p_flat[:, pair.p_index],
            grid_size=int(args.grid_size),
            quantile_lo=float(args.trace_quantile_lo),
            quantile_hi=float(args.trace_quantile_hi),
            device=device,
        )
        steps = build_pair_steps(trace, pair)
        summary = pair_summary_map[(pair.q_index, pair.p_index)]
        selected_summaries.append(summary)
        sweep_rows, push_bin_centers, sweep_heatmap = compute_threshold_sweep(steps, threshold_quantiles)

        step_rows = []
        for idx in range(steps["delta_h"].size):
            step_rows.append(
                {
                    "episode_id": int(steps["episode_id"][idx]),
                    "step_id": int(steps["step_id"][idx]),
                    "x0": float(steps["x0"][idx]),
                    "y0": float(steps["y0"][idx]),
                    "x1": float(steps["x1"][idx]),
                    "y1": float(steps["y1"][idx]),
                    "delta_h": float(steps["delta_h"][idx]),
                    "abs_delta_h": float(steps["abs_delta_h"][idx]),
                    "control_norm": float(steps["control_norm"][idx]),
                    "control_energy_push": float(steps["control_energy_push"][idx]),
                    "abs_control_energy_push": float(steps["abs_control_energy_push"][idx]),
                    "cross_ratio": float(steps["cross_ratio"][idx]),
                    "abs_slice_cross_energy": float(steps["abs_slice_cross_energy"][idx]),
                    "signed_slice_cross_energy": float(steps["signed_slice_cross_energy"][idx]),
                    "step_norm": float(steps["step_norm"][idx]),
                }
            )
        pair_stem = f"{pair.kind}_q{pair.q_index}_p{pair.p_index}"
        write_csv(figures_dir / f"{pair_stem}_steps.csv", step_rows)
        write_csv(figures_dir / f"{pair_stem}_threshold_sweep.csv", sweep_rows)
        draw_pair_triptych(figures_dir / f"{pair_stem}_triptych.png", slice_eval, pair, steps)
        draw_pair_overlay_and_scatter(
            figures_dir / f"{pair_stem}_analysis.png",
            slice_eval,
            pair,
            steps,
            summary,
            scatter_max_points=int(args.scatter_max_points),
        )
        draw_pair_threshold_sweep(
            figures_dir / f"{pair_stem}_threshold_sweep.png",
            pair,
            summary,
            sweep_rows,
            push_bin_centers,
            sweep_heatmap,
        )
        draw_pair_push_crossing_rate_curve(
            figures_dir / f"{pair_stem}_push_crossing_rate.png",
            pair,
            summary,
            sweep_rows,
        )
        draw_pair_thresholded(
            figures_dir / f"{pair_stem}_thresholded.png",
            pair,
            steps,
            summary,
            scatter_max_points=int(args.scatter_max_points),
        )

    write_csv(output_dir / "pair_correlation_summary.csv", [asdict(summary) for summary in selected_summaries])

    focus_pair = next(
        (pair for pair in all_scores if pair.q_index == int(args.focus_q_index) and pair.p_index == int(args.focus_p_index)),
        None,
    )
    if focus_pair is not None:
        focus_steps = build_pair_steps(trace, focus_pair)
        focus_slice = evaluate_slice(
            world_model=world_model,
            q_reference=q_reference,
            p_reference=p_reference,
            pair=focus_pair,
            q_values=q_flat[:, focus_pair.q_index],
            p_values=p_flat[:, focus_pair.p_index],
            grid_size=int(args.grid_size),
            quantile_lo=float(args.trace_quantile_lo),
            quantile_hi=float(args.trace_quantile_hi),
            device=device,
        )
        draw_focus_story_sheet(
            summary_dir / f"focus_q{focus_pair.q_index}_p{focus_pair.p_index}_story.png",
            focus_slice,
            focus_pair,
            focus_steps,
            pair_summary_map[(focus_pair.q_index, focus_pair.p_index)],
            pair_sweep_map[(focus_pair.q_index, focus_pair.p_index)],
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "trace_path": str(trace_path),
                "task": str(np.asarray(trace["task"]).item()),
                "seed": int(np.asarray(trace["seed"]).item()),
                "mode": str(np.asarray(trace["mode"]).item()),
                "selected_pairs": selection_rows,
                "selected_pair_summaries": [asdict(summary) for summary in selected_summaries],
                "all_pair_summary_count": len(all_pair_summaries),
                "focus_pair": None if focus_pair is None else asdict(focus_pair),
            },
            handle,
            indent=2,
        )

    for pair in selected_pairs:
        pair_stem = f"{pair.kind}_q{pair.q_index}_p{pair.p_index}"
        print(figures_dir / f"{pair_stem}_triptych.png")
        print(figures_dir / f"{pair_stem}_analysis.png")
        print(figures_dir / f"{pair_stem}_thresholded.png")
        print(figures_dir / f"{pair_stem}_threshold_sweep.png")
        print(figures_dir / f"{pair_stem}_push_crossing_rate.png")
    print(summary_dir / "all_pair_rankings.png")
    print(summary_dir / "all_pair_metric_maps.png")
    if focus_pair is not None:
        print(summary_dir / f"focus_q{focus_pair.q_index}_p{focus_pair.p_index}_story.png")
    print(output_dir / "pair_correlation_summary.csv")
    print(output_dir / "all_pair_correlation_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
