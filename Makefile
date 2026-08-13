.PHONY: install lock-evaluation evaluation-image metric-models test lint check pilot readiness readiness-linux confirmatory-plan evaluate summarize confirmatory-summarize paper clean-paper

install:
	python3 -m pip install -e '.[evaluation,dev]'

metric-models:
	python3 -m scripts.download_metric_models

lock-evaluation:
	uv pip compile requirements/evaluation-linux-py312.in --default-index https://pypi.org/simple --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --custom-compile-command 'make lock-evaluation' --output-file requirements/evaluation-linux-py312.lock

evaluation-image:
	docker build --platform linux/amd64 -f containers/evaluation/Dockerfile -t skin-tone-evaluation:step6 .

test:
	python3 -m pytest

lint:
	python3 -m ruff check scripts tests src/analysis src/metrics src/validation src/utils/reproducibility.py src/latent/vector_discovery.py

check: test lint

pilot:
	python3 run_race_vector_extraction.py --steps 25 --alphas -1.5 -0.75 0 0.75 1.5 --seed 999 --output experiments/runs/pilot_seed_999

readiness:
	python3 -m scripts.validate_readiness --output experiments/readiness/step6_readiness.json

readiness-linux: evaluation-image
	docker run --rm --platform linux/amd64 -v "$(CURDIR)/.artifacts/metrics:/artifacts/metrics:ro" skin-tone-evaluation:step6 python -m scripts.validate_readiness --artifact mediapipe_face_landmarker=/artifacts/metrics/mediapipe_face_landmarker/face_landmarker.task --artifact facenet_vggface2=/artifacts/metrics/facenet_vggface2/20180402-114759-vggface2.pt --artifact mtcnn_pnet=/artifacts/metrics/mtcnn_pnet/pnet.pt --artifact mtcnn_rnet=/artifacts/metrics/mtcnn_rnet/rnet.pt --artifact mtcnn_onet=/artifacts/metrics/mtcnn_onet/onet.pt --artifact alexnet_backbone=/artifacts/metrics/alexnet_backbone/alexnet-owt-7be5be79.pth --artifact lpips_alex_v0.1=/artifacts/metrics/lpips_alex_v0.1/lpips-alex-v0.1.pth --output experiments/readiness/step6_readiness_linux.json

confirmatory-plan:
	python3 scripts/run_confirmatory.py --config configs/full_study.yaml --output experiments/runs/confirmatory_v1

evaluate:
	python3 scripts/evaluate_sweep.py experiments/results --output experiments/evaluation --operating-max-alpha 0.75 --strict

summarize:
	python3 scripts/summarize_results.py experiments/runs/pilot_seed_999 --output experiments/summary/pilot --legacy-descriptive

confirmatory-summarize:
	python3 scripts/summarize_results.py experiments/runs/confirmatory_v1 --config configs/full_study.yaml --output experiments/summary --strict

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean-paper:
	cd paper && latexmk -C
