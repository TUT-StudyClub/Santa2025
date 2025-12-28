.PHONY: help venv sync fmt lint test clean notebook exp uv-exp submit log-exp docker-build docker-bash docker-jupyter docker-down

help:
	@echo "make venv      - create .venv and install deps"
	@echo "make sync      - install deps with uv (recommended)"
	@echo "make fmt       - format (ruff)"
	@echo "make lint      - lint (ruff)"
	@echo "make test      - run tests (pytest)"
	@echo "make notebook  - start jupyter lab"
	@echo "make exp EXP=000 CFG=000 - run experiment (requires env active)"
	@echo "make uv-exp EXP=000 CFG=000 - run experiment via uv"
	@echo "make submit EXP=000 CFG=000 [COMP=santa-2025] [MSG=...] [SUB_FILE=...] - submit latest"
	@echo "make log-exp EXP=000 CFG=000 [AUTHOR=...] [RUN_DIR=...] - append to docs/experiments.md"
	@echo "make docker-build [CPU=1]   - build Kaggle-like docker"
	@echo "make docker-bash [CPU=1]    - open bash in docker"
	@echo "make docker-jupyter [CPU=1] - run jupyter in docker"
	@echo "make docker-down [CPU=1]    - stop docker"
	@echo "make clean     - remove caches"

venv:
	python -m venv .venv
	. .venv/bin/activate && python -m pip install -U pip
	. .venv/bin/activate && python -m pip install -r requirements-dev.txt
	. .venv/bin/activate && python -m pip install -e .

sync:
	uv sync --dev

fmt:
	ruff format .

lint:
	ruff check .

test:
	pytest -q

notebook:
	jupyter lab

EXP ?= 000
CFG ?= 000
COMP ?= santa-2025
MSG ?= exp$(EXP)_$(CFG)
SUB_FILE ?=
AUTHOR ?= $(USER)
LOG_FILE ?= docs/experiments.md
EXP_DIR := $(shell ls -d ./experiments/exp$(EXP)_* 2>/dev/null | head -1)
EXP_NAME := $(notdir $(EXP_DIR))
SUB_DIR := $(shell ls -dt ./outputs/runs/$(EXP_NAME)/$(CFG)/* 2>/dev/null | head -1)
SUB_PATH := $(if $(SUB_FILE),$(SUB_FILE),$(SUB_DIR)/submission.csv)
RUN_DIR ?= $(SUB_DIR)
LOG_CMD ?= uv run python $(EXP_DIR)/run.py exp=$(CFG)

exp:
	@test -n "$(EXP_DIR)" || (echo "experiment not found: exp$(EXP)_*" && exit 1)
	python "$(EXP_DIR)/run.py" exp="$(CFG)"

uv-exp:
	@test -n "$(EXP_DIR)" || (echo "experiment not found: exp$(EXP)_*" && exit 1)
	uv run python "$(EXP_DIR)/run.py" exp="$(CFG)"

submit:
	@test -n "$(EXP_DIR)" || (echo "experiment not found: exp$(EXP)_*" && exit 1)
	@test -f "$(SUB_PATH)" || (echo "submission not found: $(SUB_PATH)" && exit 1)
	kaggle competitions submit -c "$(COMP)" -f "$(SUB_PATH)" -m "$(MSG)"

log-exp:
	@test -n "$(EXP_DIR)" || (echo "experiment not found: exp$(EXP)_*" && exit 1)
	@test -n "$(RUN_DIR)" || (echo "run dir not found for exp=$(EXP) cfg=$(CFG)" && exit 1)
	@RUN_DIR="$(RUN_DIR)" LOG_FILE="$(LOG_FILE)" AUTHOR="$(AUTHOR)" EXP_NAME="$(EXP_NAME)" LOG_CMD="$(LOG_CMD)" \
		python - <<'PY'
import datetime
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
log_file = Path(os.environ["LOG_FILE"])
author = os.environ.get("AUTHOR", "-")
exp_name = os.environ.get("EXP_NAME", "").strip()
log_cmd = os.environ.get("LOG_CMD", "-")

metrics_path = run_dir / "metrics.json"
if not metrics_path.exists():
    raise SystemExit(f"metrics not found: {metrics_path}")
metrics = json.loads(metrics_path.read_text())

score = metrics.get("score")
note = (metrics.get("note") or "").strip()
n_trees = metrics.get("n_trees")

seed = "-"
cfg_path = run_dir / "config_exp.yaml"
if cfg_path.exists():
    for line in cfg_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("seed:"):
            seed = line.split(":", 1)[1].strip() or "-"
            break

date = datetime.date.today().isoformat()
change = exp_name if exp_name else "-"
if note:
    change = f"{change} {note}".strip()

cv = "-"
if isinstance(score, (int, float)):
    cv = f"score={score:.4f}"

note_col = "metrics.json"
if n_trees is not None:
    note_col = f"n={n_trees}, metrics.json"

row = f"| {date} | {author} | - | {change} | {seed} | {cv} | - | `{log_cmd}` | {note_col} |"

lines = log_file.read_text().splitlines()
if row in lines:
    print("log entry already exists")
    raise SystemExit(0)

insert_at = None
for i, line in enumerate(lines):
    if line.startswith("| --- |"):
        insert_at = i + 1
        break
if insert_at is None:
    raise SystemExit("table header not found in experiments.md")

lines.insert(insert_at, row)
log_file.write_text("\n".join(lines) + "\n")
print(f"added log entry: {row}")
PY

COMPOSE_FILE := compose.yaml
ifneq ($(CPU),)
	COMPOSE_FILE := compose.cpu.yaml
endif

docker-build:
	docker compose -f "$(COMPOSE_FILE)" build

docker-bash:
	docker compose -f "$(COMPOSE_FILE)" run --rm kaggle bash

docker-jupyter:
	docker compose -f "$(COMPOSE_FILE)" up

docker-down:
	docker compose -f "$(COMPOSE_FILE)" down

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
