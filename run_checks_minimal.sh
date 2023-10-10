#!/bin/zsh
python -m mypy library/mesh.py --namespace-packages --explicit-package-bases
python -m mypy library/initial_conditions.py --namespace-packages --explicit-package-bases
python -m mypy library/boundary_conditions.py --namespace-packages --explicit-package-bases
python -m mypy library/models/base.py --namespace-packages --explicit-package-bases
python -m mypy library/model.py --namespace-packages --explicit-package-bases
pytest tests/unittests/*py -k 'critical'