from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from utils.env import EnvConfig
from utils.logger import get_logger
from utils.timing import trace

load_dotenv()


@dataclass
class ExpConfig:
    debug: bool = False
    seed: int = 42
    folds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    note: str = ""


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)


cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def init_output_dir(cfg: Config) -> Path:
    major = Path(__file__).resolve().parent.name  # exp000_sample
    minor = HydraConfig.get().runtime.choices.exp  # type: ignore
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = repo_root() / "outputs" / "runs" / major / minor / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg.env.output_dir = output_dir
    cfg.env.exp_name = f"{major}/{minor}"
    return output_dir


def dump_config(cfg: Config, output_dir: Path) -> None:
    (output_dir / "config_exp.yaml").write_text(
        OmegaConf.to_yaml(cfg.exp, resolve=True),
        encoding="utf-8",
    )
    (output_dir / "config_env.yaml").write_text(
        OmegaConf.to_yaml(cfg.env, resolve=True),
        encoding="utf-8",
    )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:
    output_dir = init_output_dir(cfg)
    logger = get_logger(__name__, output_dir)

    logger.info("exp_name=%s", cfg.env.exp_name)
    logger.info("output_dir=%s", str(output_dir))
    logger.info("overrides=%s", ", ".join(HydraConfig.get().overrides.task))
    logger.info("cfg.exp=%s", OmegaConf.to_container(cfg.exp, resolve=True))

    set_seed(cfg.exp.seed)
    dump_config(cfg, output_dir)

    with trace("dummy-step"):
        time.sleep(0.2 if cfg.exp.debug else 1.0)

    metrics: dict[str, Any] = {"dummy_metric": float(np.random.rand())}
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("metrics=%s", metrics)


if __name__ == "__main__":
    main()
