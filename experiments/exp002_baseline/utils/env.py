from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass
class EnvConfig:
    data_dir: Path = Path(os.getenv("DATA_DIR", str(_repo_root() / "data")))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", str(_repo_root() / "outputs")))
    exp_name: str = ""
