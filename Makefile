.PHONY: install test lint check pilot summarize study-validate study-calibration \
	calibration-synthesize study-run study-analyze study-plot study-example \
	study-audit study-robustness direction-stability measurement-validate \
	replication-data replication-validate replication-freeze replication-run \
	replication-audit replication-analyze replication-assess replication-plot \
	paper paper-arxiv paper-audit clean-paper

PLANNED_CONFIG ?= configs/full_study.yaml
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
STUDY_CONFIG ?= configs/full_study_preregistered.yaml
RUN_DIR ?= experiments/runs/confirmatory_cuda
ANALYSIS_DIR ?= experiments/analysis/confirmatory_cuda
TECTONIC ?= tectonic
# Fix PDF metadata timestamps so unchanged source produces byte-identical PDFs.
SOURCE_DATE_EPOCH ?= 1787000000
REPLICATION_PLANNED_CONFIG ?= configs/replication_study.yaml
REPLICATION_CONFIG ?= configs/replication_study_preregistered.yaml
REPLICATION_RUN_DIR ?= experiments/runs/replication_cuda
REPLICATION_ANALYSIS_DIR ?= experiments/analysis/replication_cuda

install:
	$(PYTHON) -m pip install -e '.[evaluation,dev]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check generate_training_data.py run_race_vector_extraction.py scripts tests src

check: test lint

pilot:
	$(PYTHON) run_race_vector_extraction.py --steps 25 --alphas -1.5 -0.75 0 0.75 1.5 --seed 999 --output experiments/runs/pilot_seed_999

summarize:
	$(PYTHON) scripts/summarize_results.py experiments/runs --output experiments/summary

study-validate:
	$(PYTHON) scripts/run_study.py $(STUDY_CONFIG) --validate-only

measurement-validate:
	$(PYTHON) scripts/validate_skin_tone_metric.py $(PLANNED_CONFIG)

study-calibration:
	$(PYTHON) scripts/run_study.py configs/calibration_candidate_v2.yaml --allow-calibration --output experiments/runs/calibration_candidate_v2

calibration-synthesize:
	$(PYTHON) scripts/synthesize_calibration.py \
		experiments/runs/calibration_refined/results.jsonl \
		experiments/runs/calibration_refined_seeds_900003_900005/results.jsonl \
		experiments/runs/calibration_negative_extension/results.jsonl \
		experiments/runs/calibration_candidate_v2/results.jsonl \
		--output experiments/analysis/calibration_synthesis_five_seed \
		--expected-seeds 900002 900003 900004 900005 900006 \
		--target 5 --tolerance 3

study-run:
	$(PYTHON) scripts/run_study.py $(STUDY_CONFIG) --output $(RUN_DIR)

study-audit:
	$(PYTHON) scripts/audit_study_run.py $(STUDY_CONFIG) $(RUN_DIR) \
		--output $(ANALYSIS_DIR)/run_integrity_audit.json

study-analyze:
	$(PYTHON) scripts/analyze_study.py $(STUDY_CONFIG) $(RUN_DIR) --output $(ANALYSIS_DIR)

study-robustness:
	$(PYTHON) scripts/analyze_robustness.py $(STUDY_CONFIG) $(RUN_DIR) \
		--output experiments/analysis/robustness

direction-stability:
	$(PYTHON) scripts/analyze_direction_stability.py $(STUDY_CONFIG) \
		--output experiments/analysis/direction_stability \
		--reference-direction $(RUN_DIR)/direction/raw.pt

replication-data:
	$(PYTHON) generate_training_data.py --config $(REPLICATION_PLANNED_CONFIG) --n 96

replication-validate:
	$(PYTHON) scripts/validate_skin_tone_metric.py $(REPLICATION_PLANNED_CONFIG) \
		--output experiments/measurement_validation_replication

replication-freeze:
	$(PYTHON) scripts/freeze_confirmatory_config.py $(REPLICATION_PLANNED_CONFIG) \
		--manifest data/generated/training_manifest_replication_v1.json \
		--validation-report experiments/measurement_validation_replication/validation_report.json \
		--output $(REPLICATION_CONFIG)

replication-run:
	$(PYTHON) scripts/run_study.py $(REPLICATION_CONFIG) --output $(REPLICATION_RUN_DIR)

replication-audit:
	$(PYTHON) scripts/audit_study_run.py $(REPLICATION_CONFIG) $(REPLICATION_RUN_DIR) \
		--output $(REPLICATION_ANALYSIS_DIR)/run_integrity_audit.json

replication-analyze:
	$(PYTHON) scripts/analyze_study.py $(REPLICATION_CONFIG) $(REPLICATION_RUN_DIR) \
		--output $(REPLICATION_ANALYSIS_DIR)

replication-assess:
	$(PYTHON) scripts/analyze_replication.py $(REPLICATION_CONFIG) \
		$(REPLICATION_ANALYSIS_DIR) \
		--parent-analysis $(ANALYSIS_DIR) \
		--parent-direction $(RUN_DIR)/direction/raw.pt \
		--replication-direction $(REPLICATION_RUN_DIR)/direction/raw.pt \
		--output experiments/analysis/replication_assessment

replication-plot:
	$(PYTHON) scripts/plot_replication.py $(ANALYSIS_DIR) $(REPLICATION_ANALYSIS_DIR) \
		--output paper/figures

study-plot:
	$(PYTHON) scripts/plot_study.py $(ANALYSIS_DIR) --output paper/figures

study-example:
	$(PYTHON) scripts/make_qualitative_grid.py $(RUN_DIR) $(ANALYSIS_DIR) \
		--output paper/figures/confirmatory_qualitative.png

paper:
	cd paper && SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(TECTONIC) main.tex --keep-logs --keep-intermediates
	mkdir -p output/pdf
	cp paper/main.pdf output/pdf/denoising_time_skin_tone_steering_replication.pdf

paper-arxiv:
	cd paper && SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(TECTONIC) arxiv.tex --keep-logs --keep-intermediates
	mkdir -p output/pdf
	cp paper/arxiv.pdf output/pdf/arxiv_preprint.pdf

paper-audit:
	$(PYTHON) scripts/verify_manuscript_claims.py

clean-paper:
	cd paper && latexmk -C
