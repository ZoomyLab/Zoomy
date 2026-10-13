"""The 16 model goldens (M01-M16) — verbatim normalized SystemModel snapshots,
derived NO-CACHE through the full symbolic chain (approved spec §A).

SWE goldens are the DERIVED SME(level=0) composition, never the hand-built SWE
class (user mandate).  Golden bodies live in ``tests/goldens/<name>.txt``;
regenerate with ``python scripts/regen_goldens.py`` and review the git diff
(re-bless protocol: rederive tier green first when >1 family changes).

T1 gate members (v3 delta): m01/m02 (SWE 1D/2D), m03 (SME l1), m07
(SME, stress=Stress.full), m09 (ML-SWE 2-layer), m12 (VAM Escalante-bump).
Everything else is T2; m05/m13 (true 2-D derivations) are ``large``.
"""
import pytest

import goldenlib

pytestmark = pytest.mark.model


@pytest.mark.parametrize("name", goldenlib.golden_params("model"))
def test_model_golden(name):
    builder, _family, _tier = goldenlib.GOLDENS[name]
    goldenlib.assert_matches_golden(name, builder())
