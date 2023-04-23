install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

install-dev: ## [Local development] Install test requirements
	python -m pip install -r requirements-test.txt

lint: ## [Local development] Run mypy, pylint and black
	python -m mypy img2dataset
	python -m pylint img2dataset
	python -m black --check -l 120 img2dataset

black: ## [Local development] Auto-format python code using black
	python -m black -l 120 .

build-pex:
	python3 -m venv .pexing
	. .pexing/bin/activate && python -m pip install -U pip && python -m pip install pex
	. .pexing/bin/activate && python -m pex setuptools scipy==1.9.0 gcsfs==2022.11.0 s3fs==2022.11.0 pyspark==3.2.0 requests==2.27.1 . -o img2dataset.pex -v
	rm -rf .pexing

test: ## [Local development] Run unit tests
	python -m pytest -x -s -v tests

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
