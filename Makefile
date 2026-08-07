.PHONY: install test lint check pilot summarize paper clean-paper

install:
	python3 -m pip install -e '.[evaluation,dev]'

test:
	python3 -m pytest

lint:
	python3 -m ruff check scripts tests src/utils/reproducibility.py src/latent/vector_discovery.py

check: test lint

pilot:
	python3 run_race_vector_extraction.py --steps 25 --alphas -1.5 -0.75 0 0.75 1.5 --seed 999 --output experiments/runs/pilot_seed_999

summarize:
	python3 scripts/summarize_results.py experiments/runs --output experiments/summary

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean-paper:
	cd paper && latexmk -C
