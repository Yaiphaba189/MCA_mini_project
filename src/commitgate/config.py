"""Dataclass config, YAML loading, `--set key=value` overrides, resolved-config dump."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    device: str = "auto"  # auto -> cuda, then mps, then cpu
    dtype: str = "float32"
    seed: int = 0
    seq_len: int = 256
    batch_size: int = 8
    lr: float = 2e-4
    steps_per_candidate: int = 50
    holdout_size: int = 128
    retention_probe_size: int = 64
    eval_every: int = 5
    lam: float = 1.0
    tau: float = 0.0
    eps: float = 0.02
    lora: LoRAConfig = field(default_factory=LoRAConfig)


def _build(cls, data: dict[str, Any]):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for k, v in data.items():
        sub = fields[k].default_factory() if fields[k].default_factory is not dataclasses.MISSING else None
        kwargs[k] = _build(type(sub), v) if dataclasses.is_dataclass(sub) and isinstance(v, dict) else v
    return cls(**kwargs)


def _set_path(d: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        _set_path(data, key.strip(), yaml.safe_load(raw))
    return _build(Config, data)


def dump_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
