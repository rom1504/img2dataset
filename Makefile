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

venv-lint-test: ## [Continuous integration]
	python3 -m venv .env && . .env/bin/activate && make install install-dev lint test && rm -rf .env

test: ## [Local development] Run unit tests
	rm -rf tests/test_folder/
	python -m pytest -x -v --cov=img2dataset --cov-report term-missing --cov-fail-under .45 tests

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'