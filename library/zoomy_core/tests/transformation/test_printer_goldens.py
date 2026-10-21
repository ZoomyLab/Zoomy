"""The 2 printer goldens — FULL emitted source, default options (spec §A):

* P01 foam <- the VAM NSM (N02 base): SystemModel + Numerics printers —
  REQ-187 namespace-qualified opaque eigenvalue calls, REQ-183 p[]-lowered
  wet_dry_eps, REQ-185 time/dt/X signatures, REQ-190 dt_max, REQ-81
  projection x-h factor, REQ-91 no raw Integral.                     [gate]
* P02 generic-C / amrex path <- SME(1) via GenericCppModel: REQ-81 P3_h
  column factor through CSE, scalarization, registry-derived arg lists,
  REQ-91 no Integral.                                                [gate]

Emitted-source drift = golden diff; the absorbed signature/emit tests
(test_signature_registry, projection_no_integral, foam req183/req187 pins)
live in these two bodies now.
"""
import pytest

import goldenlib

pytestmark = pytest.mark.printer


@pytest.mark.parametrize("name", goldenlib.golden_params("printer"))
def test_printer_golden(name):
    builder, _family, _tier = goldenlib.GOLDENS[name]
    goldenlib.assert_matches_golden(name, builder())
