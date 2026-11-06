# Cross-platform: override with `make PY=python` if needed.
ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python.exe
else
VENV_PY := .venv/bin/python
endif
PY ?= $(VENV_PY)
CFG ?= configs/default.yaml

.PHONY: setup data features train backtest report test lint all
setup:
	python -m venv .venv
	$(VENV_PY) -m pip install -U pip
	$(VENV_PY) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"
data:
	$(PY) scripts/run_all.py data --config $(CFG)
features:
	$(PY) scripts/run_all.py features --config $(CFG)
train:
	$(PY) scripts/run_all.py train --config $(CFG)
backtest:
	$(PY) scripts/run_all.py backtest --config $(CFG)
report:
	$(PY) scripts/run_all.py report --config $(CFG)
all: data features train backtest report
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
