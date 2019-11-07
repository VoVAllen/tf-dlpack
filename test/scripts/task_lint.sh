#!/bin/bash
echo "Check codestyle of c++ code..."
python test/scripts/lint.py tfdlpack cpp include src
echo "Check codestyle of Python code..."
python3 -m pylint --reports=y -v --rcfile=test/scripts/pylintrc python/tfdlpack