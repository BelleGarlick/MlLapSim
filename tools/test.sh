#!/bin/sh

set -e

export PYTHONPATH=src:$PYTHONPATH
./venv/bin/python setup.py build_ext --inplace
./venv/bin/python -m coverage run -m pytest
./venv/bin/python -m coverage report -m
