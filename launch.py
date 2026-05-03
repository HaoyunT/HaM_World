from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

FINAL_ROOT = Path(__file__).resolve().parent
CODE_ROOTS = [
    FINAL_ROOT,
    FINAL_ROOT / "baselines",
]
for root in reversed(CODE_ROOTS):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

ALGORITHM_SPECS = {
    "hamworld": {
        "label": "HaM-World",
        "package": "hamworld",
        "default_config": FINAL_ROOT / "hamworld" / "configs" / "compare_dmcontrol.yaml",
        "preset_configs": {
            "compare_dmcontrol": FINAL_ROOT / "hamworld" / "configs" / "compare_dmcontrol.yaml",
            "finger_reacher": FINAL_ROOT / "hamworld" / "configs" / "finger_reacher.yaml",
        },
    },
    "dreamerv3": {
        "label": "DreamerV3",
        "package": "dreamerv3",
        "default_config": FINAL_ROOT / "baselines" / "dreamerv3" / "configs" / "compare_dmcontrol.yaml",
        "preset_configs": {
            "compare_dmcontrol": FINAL_ROOT / "baselines" / "dreamerv3" / "configs" / "compare_dmcontrol.yaml",
            "finger_reacher": FINAL_ROOT / "baselines" / "dreamerv3" / "configs" / "finger_reacher.yaml",
        },
    },
    "tdmpc2": {
        "label": "TD-MPC2",
        "package": "tdmpc2",
        "default_config": FINAL_ROOT / "baselines" / "tdmpc2" / "configs" / "compare_dmcontrol.yaml",
        "preset_configs": {
            "compare_dmcontrol": FINAL_ROOT / "baselines" / "tdmpc2" / "configs" / "compare_dmcontrol.yaml",
            "finger_reacher": FINAL_ROOT / "baselines" / "tdmpc2" / "configs" / "finger_reacher.yaml",
        },
    },
    "ppo": {
        "label": "PPO",
        "package": "ppo",
        "default_config": FINAL_ROOT / "baselines" / "ppo" / "configs" / "low_budget_compare_dmcontrol.yaml",
        "preset_configs": {
            "compare_dmcontrol": FINAL_ROOT / "baselines" / "ppo" / "configs" / "low_budget_compare_dmcontrol.yaml",
            "finger_reacher": FINAL_ROOT / "baselines" / "ppo" / "configs" / "low_budget_finger_reacher.yaml",
        },
    },
    "sac": {
        "label": "SAC",
        "package": "sac",
        "default_config": FINAL_ROOT / "baselines" / "sac" / "configs" / "low_budget_compare_dmcontrol.yaml",
        "preset_configs": {
            "compare_dmcontrol": FINAL_ROOT / "baselines" / "sac" / "configs" / "low_budget_compare_dmcontrol.yaml",
            "finger_reacher": FINAL_ROOT / "baselines" / "sac" / "configs" / "low_budget_finger_reacher.yaml",
        },
    },
}

DEFAULT_ALL_ALGORITHMS = ["hamworld", "dreamerv3", "tdmpc2", "ppo", "sac"]

PRESETS = {
    "compare_dmcontrol": ["cartpole_swingup", "cheetah_run"],
    "finger_reacher": ["finger_spin", "reacher_easy"],
}


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (FINAL_ROOT / path).resolve()
    return path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must decode to a dictionary.")
    data.setdefault("_config_path", str(path))
    return data


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _set_by_dotted_path(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        dotted_key, raw_value = override.split("=", 1)
        _set_by_dotted_path(updated, dotted_key, _coerce_value(raw_value))
    return updated


def expand_task_runs(config: dict[str, Any], tasks: list[str]) -> list[dict[str, Any]]:
    configured_tasks = config.get("benchmark", {}).get("tasks", [])
    task_map = {task.get("id"): task for task in configured_tasks}
    missing = [task_id for task_id in tasks if task_id not in task_map]
    if missing:
        available = ", ".join(sorted(task_map)) if task_map else "<none>"
        raise ValueError(f"Task(s) not found in {config['_config_path']}: {', '.join(missing)}. Available: {available}.")

    runs: list[dict[str, Any]] = []
    for task_id in tasks:
        run_config = copy.deepcopy(config)
        run_config["task"] = copy.deepcopy(task_map[task_id])
        run_config["experiment"] = copy.deepcopy(config.get("experiment", {}))
        base_name = run_config["experiment"].get("name", "experiment")
        run_config["experiment"]["resolved_name"] = f"{base_name}_{task_id}"
        runs.append(run_config)
    return runs


def normalize_tasks(tasks: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        if task not in seen:
            seen.add(task)
            deduped.append(task)
    return deduped


def select_tasks(args: argparse.Namespace) -> list[str]:
    if getattr(args, "tasks", None):
        return normalize_tasks(args.tasks)
    return PRESETS[args.preset]


def config_path_for_algorithm(algorithm: str, args: argparse.Namespace) -> Path:
    override_key = f"{algorithm}_config".replace("-", "_")
    override = getattr(args, override_key, None)
    if override:
        return resolve_repo_path(override)
    preset_name = getattr(args, "preset", None)
    if preset_name is not None:
        preset_configs = ALGORITHM_SPECS[algorithm].get("preset_configs", {})
        if preset_name in preset_configs:
            return preset_configs[preset_name]
    return ALGORITHM_SPECS[algorithm]["default_config"]


def build_overrides(args: argparse.Namespace) -> list[str]:
    overrides = list(getattr(args, "override", []) or [])
    output_root = resolve_repo_path(args.output_root)
    overrides.append(f"experiment.output_dir={output_root}")
    if getattr(args, "seed", None) is not None:
        overrides.append(f"experiment.seed={args.seed}")
    return overrides


def describe_task(task: dict[str, Any]) -> str:
    if task["suite"] == "dmcontrol":
        target = f"{task['domain']}/{task['task']}"
    else:
        target = task.get("env_name", "<unknown>")
    return (
        f"{task['id']}: suite={task['suite']}, target={target}, "
        f"action_repeat={task.get('action_repeat', 1)}, episode_length={task.get('episode_length', 'n/a')}"
    )


def list_command(args: argparse.Namespace) -> int:
    print("Available presets:")
    for preset_name, tasks in PRESETS.items():
        print(f"  {preset_name}: {', '.join(tasks)}")

    print("\nAlgorithms:")
    for algorithm in (args.algo if args.algo != "all" else DEFAULT_ALL_ALGORITHMS):
        spec = ALGORITHM_SPECS[algorithm]
        config_path = config_path_for_algorithm(algorithm, args)
        config = load_config(config_path)
        print(f"  {spec['label']} ({algorithm})")
        print(f"    config: {config_path}")
        for task in config.get("benchmark", {}).get("tasks", []):
            print(f"    - {describe_task(task)}")
    return 0


def build_run_configs(algorithm: str, args: argparse.Namespace, tasks: list[str]) -> tuple[Path, list[dict[str, Any]]]:
    if getattr(args, "resume", None):
        import torch

        checkpoint_path = resolve_repo_path(args.resume)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint["config"]
        checkpoint_algorithm = checkpoint_config["experiment"]["algorithm"]
        if algorithm != checkpoint_algorithm:
            raise ValueError(
                f"Resume checkpoint {checkpoint_path} is for algorithm '{checkpoint_algorithm}', not '{algorithm}'."
            )
        resolved = apply_overrides(checkpoint_config, build_overrides(args))
        resolved["_resume_checkpoint"] = str(checkpoint_path)
        return checkpoint_path, [resolved]

    config_path = config_path_for_algorithm(algorithm, args)
    config = load_config(config_path)
    resolved = apply_overrides(config, build_overrides(args))
    return config_path, expand_task_runs(resolved, tasks)


def print_run_plan(algorithm: str, config_path: Path, runs: list[dict[str, Any]], output_root: Path) -> None:
    for run_config in runs:
        task = run_config["task"]
        seed = run_config["experiment"]["seed"]
        print(
            f"[plan] algorithm={algorithm} task={task['id']} seed={seed} "
            f"config={config_path} outputs={output_root}"
        )


def train_runs(algorithm: str, runs: list[dict[str, Any]]) -> None:
    package_name = ALGORITHM_SPECS[algorithm]["package"]
    train_module = importlib.import_module(f"{package_name}.train")
    label = ALGORITHM_SPECS[algorithm]["label"]
    for run_config in runs:
        run_config_copy = copy.deepcopy(run_config)
        resume_checkpoint = run_config_copy.pop("_resume_checkpoint", None)
        task_id = run_config_copy["task"]["id"]
        print(f"[train] {label} -> {task_id}")
        train_module.main(config=run_config_copy, resume_checkpoint=resume_checkpoint)


def train_command(args: argparse.Namespace) -> int:
    if getattr(args, "resume", None):
        import torch

        checkpoint_path = resolve_repo_path(args.resume)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint["config"]
        tasks = [checkpoint_config["task"]["id"]]
        algorithms = [checkpoint_config["experiment"]["algorithm"]]
    else:
        tasks = select_tasks(args)
        algorithms = DEFAULT_ALL_ALGORITHMS if args.algo == "all" else [args.algo]
    output_root = resolve_repo_path(args.output_root)

    prepared: list[tuple[str, Path, list[dict[str, Any]]]] = []
    for algorithm in algorithms:
        config_path, runs = build_run_configs(algorithm, args, tasks)
        prepared.append((algorithm, config_path, runs))
        print_run_plan(algorithm, config_path, runs, output_root)

    if args.dry_run:
        return 0

    for algorithm, _, runs in prepared:
        train_runs(algorithm, runs)
    return 0


def add_task_selector_arguments(parser: argparse.ArgumentParser) -> None:
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task ids to run. Example: --tasks cartpole_swingup cheetah_run",
    )
    task_group.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="compare_dmcontrol",
        help="Named task preset. Defaults to compare_dmcontrol.",
    )


def add_shared_run_arguments(parser: argparse.ArgumentParser, include_algo: bool) -> None:
    if include_algo:
        parser.add_argument(
            "--algo",
            choices=["hamworld", "dreamerv3", "tdmpc2", "ppo", "sac", "all"],
            default="all",
            help="Algorithm selector.",
        )
    add_task_selector_arguments(parser)
    parser.add_argument("--output-root", default="outputs", help="Root directory used for all experiment outputs.")
    parser.add_argument("--seed", default=None, type=int, help="Optional seed override for all selected algorithms.")
    parser.add_argument("--override", action="append", default=[], help="Extra dotted config override, e.g. training.total_steps=10000")
    parser.add_argument("--hamworld-config", dest="hamworld_config", default=None, help="Optional config path override for HaM-World.")
    parser.add_argument("--dreamerv3-config", dest="dreamerv3_config", default=None, help="Optional config path override for DreamerV3.")
    parser.add_argument("--tdmpc2-config", dest="tdmpc2_config", default=None, help="Optional config path override for TD-MPC2.")
    parser.add_argument("--ppo-config", dest="ppo_config", default=None, help="Optional config path override for PPO.")
    parser.add_argument("--sac-config", dest="sac_config", default=None, help="Optional config path override for SAC.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume a single run from.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved run plan without starting training.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified launcher for the final 5-algorithm paper repository.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Show available presets, tasks, and default configs.")
    list_parser.add_argument(
        "--algo",
        nargs="+",
        choices=["hamworld", "dreamerv3", "tdmpc2", "ppo", "sac", "all"],
        default=["all"],
        help="Algorithm selector.",
    )
    list_parser.add_argument("--hamworld-config", dest="hamworld_config", default=None, help="Optional config path override for HaM-World.")
    list_parser.add_argument("--dreamerv3-config", dest="dreamerv3_config", default=None, help="Optional config path override for DreamerV3.")
    list_parser.add_argument("--tdmpc2-config", dest="tdmpc2_config", default=None, help="Optional config path override for TD-MPC2.")
    list_parser.add_argument("--ppo-config", dest="ppo_config", default=None, help="Optional config path override for PPO.")
    list_parser.add_argument("--sac-config", dest="sac_config", default=None, help="Optional config path override for SAC.")
    list_parser.set_defaults(handler=list_command)

    train_parser = subparsers.add_parser("train", help="Train one or more supported algorithms.")
    add_shared_run_arguments(train_parser, include_algo=True)
    train_parser.set_defaults(handler=train_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if getattr(args, "command", None) == "list" and args.algo == ["all"]:
            args.algo = DEFAULT_ALL_ALGORITHMS
        return args.handler(args)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
