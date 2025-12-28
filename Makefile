.PHONY: help venv sync fmt lint test clean notebook exp uv-exp submit docker-build docker-bash docker-jupyter docker-down

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
EXP_DIR := $(shell ls -d ./experiments/exp$(EXP)_* 2>/dev/null | head -1)
EXP_NAME := $(notdir $(EXP_DIR))
SUB_DIR := $(shell ls -dt ./outputs/runs/$(EXP_NAME)/$(CFG)/* 2>/dev/null | head -1)
SUB_PATH := $(if $(SUB_FILE),$(SUB_FILE),$(SUB_DIR)/submission.csv)

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
