#!/bin/sh

set -e

cd src
../venv/bin/python -m cli $@
