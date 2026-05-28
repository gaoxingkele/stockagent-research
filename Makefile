.PHONY: help setup data_d1 data_d2 data_d3 exp_e1_1 gate_1 gate_2 test qa clean report

help:
	@echo "Available targets:"
	@echo "  setup        Install dependencies"
	@echo "  data_d1      Build A-shares Tushare dataset"
	@echo "  data_d2      Build A-shares Qlib Alpha158 dataset"
	@echo "  data_d3      Build NASDAQ-100 dataset"
	@echo "  exp_e1_1     Run E1.1: LGBM + FH baseline on D1"
	@echo "  gate_1       Run Gate 1: TB + Fixed-PWC orthogonality"
	@echo "  gate_2       Run Gate 2: HIST + PWC model-agnostic"
	@echo "  test         Run pytest"
	@echo "  qa           Data quality checks"
	@echo "  report       Aggregate run results into reports"
	@echo "  clean        Remove cache + checkpoints"

setup:
	pip install -r requirements.txt
	pip install -e .

data_d1:
	python -m src.data.build_d1_ashares_tushare

data_d2:
	python -m src.data.build_d2_ashares_qlib

data_d3:
	python -m src.data.build_d3_nasdaq100

exp_e1_1:
	python -m src.train experiment=e1_1

gate_1:
	python -m src.experiments.gate_1

gate_2:
	python -m src.experiments.gate_2

test:
	pytest tests/ -v

qa:
	python -m tests.data_quality

report:
	python -m src.evaluation.aggregate

clean:
	rm -rf checkpoints/* results/* experiments/runs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
