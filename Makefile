PYTHON := python3

install:
	python -m pip install -e .

test:
	py.test tests/test.py

.PHONY: install test
