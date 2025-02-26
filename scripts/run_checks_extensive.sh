#!/bin/zsh
python -m mypy library/ --namespace-packages --explicit-package-bases
pytest tests/unittests/*py -k 'critical'