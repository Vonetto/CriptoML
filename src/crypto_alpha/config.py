"""Utilities to load typed configurations from YAML files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class StrategyConfig:
    """Strongly-typed subset of the YAML strategy definition."""

    name: str
    description: str
    references: Dict[str, str]
    universe: Dict[str, Any]
    signal: Dict[str, Any]
    portfolio: Dict[str, Any]
    execution: Dict[str, Any]
    backtest: Dict[str, Any]
    risk_overlay: Dict[str, Any] | None = None


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_overrides(target: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        node = target
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value


def load_strategy_config(
    name: str,
    root: Path | None = None,
    overrides: Dict[str, Any] | None = None,
) -> StrategyConfig:
    """Load a strategy config from configs/strategy/<name>.yaml."""

    base = root or Path(__file__).resolve().parents[2]
    config_path = base / "configs" / "strategy" / f"{name}.yaml"
    data = load_yaml(config_path)
    if overrides:
        _apply_overrides(data, overrides)
    return StrategyConfig(**data)
