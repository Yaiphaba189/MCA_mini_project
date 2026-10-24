import json

import pytest

from commitgate.config import dump_config, load_config
from commitgate.io_utils import RunLogger, make_run_dir


def test_base_config_loads_and_overrides():
    cfg = load_config("configs/base.yaml", ["lam=2.0", "lora.r=16"])
    assert cfg.lam == 2.0 and cfg.lora.r == 16 and cfg.lora.alpha == 16


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        load_config(None, ["nope=1"])


def test_roundtrip(tmp_path):
    cfg = load_config("configs/base.yaml", ["tau=0.01"])
    dump_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg


def test_run_dir_and_logger(tmp_path):
    cfg = load_config(None)
    run_dir = make_run_dir(tmp_path, "exp", "b0", "s1_abrupt", 3, cfg)
    assert run_dir.name == "seed3" and (run_dir / "config.yaml").exists()
    logger = RunLogger(run_dir)
    logger.log("chunk", t=0, nll=1.5)
    logger.write_summary({"ok": True})
    line = json.loads((run_dir / "log.jsonl").read_text().splitlines()[0])
    assert line["event"] == "chunk" and line["t"] == 0 and "ts" in line
    assert json.loads((run_dir / "summary.json").read_text())["ok"] is True
