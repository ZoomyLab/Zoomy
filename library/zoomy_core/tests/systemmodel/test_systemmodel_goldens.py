"""The 3 systemmodel goldens (approved spec §A):

* S01 ``from_model(SME(1)+BCs+IC)`` freeze — BC attach + kernels, single
  attach / no doubled aux, IC identity threading (REQ-87/103/154)   [gate]
* S02 ``chorin_split`` of the Escalante VAM bump, desingularize_hinv applied
  BEFORE the split — stage list (REQ-173), parent-aux prefix + hinv update
  (REQ-151), dt-in-corrector, elliptic J-width assert (REQ-169)     [gate]
* S03 ``chorin_split(MLVAM(2,1))`` — ML square DAE + singular-elliptic guard
  (downward pressure trace)                                          [T2]

Runtime identity asserts live in the builders; the snapshot is the pin.
"""
import pytest

import goldenlib

pytestmark = pytest.mark.systemmodel


@pytest.mark.parametrize("name", goldenlib.golden_params("systemmodel"))
def test_systemmodel_golden(name):
    builder, _family, _tier = goldenlib.GOLDENS[name]
    goldenlib.assert_matches_golden(name, builder())
