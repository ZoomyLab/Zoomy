"""The 3 NSM goldens + the N02 time canary (approved spec §A):

* N01 NSM(SWE-2D == SME(0,d3)), ReconstructionSpec(order=2) — defaults
  [normalize_face_normal, desingularize_hinv], KP hinv row, order-2 recon
  through hinv (REQ-156), no wet_dry_eps leak (REQ-194)             [gate]
* N02 NSM(VAM(1) Escalante) — lazy Max(|eigenvalues|) wavespeed (REQ-189
  structure, REQ-167 no-Gershgorin) incl. built face kernels        [gate]
* N03 SME(1) + guard_eigenvalue_powers + gate_eigenvalues_dry OPTED IN —
  exact conditional(h>eps,.,0) srepr, Max floor idempotent (REQ-181) [T2]

Time canary (v3 ruling): CPU-time, ONE run, threshold 1.5x the checked-in
baseline (tests/goldens/n02_time_baseline.json).  Catches the ~7x
factor_terms blowup class; median-of-3 is a MANUAL follow-up after a failure,
not part of the gate.
"""
import json
import platform

import pytest

import goldenlib

pytestmark = pytest.mark.nsm


@pytest.mark.parametrize("name", goldenlib.golden_params("nsm"))
def test_nsm_golden(name):
    builder, _family, _tier = goldenlib.GOLDENS[name]
    goldenlib.assert_matches_golden(name, builder())


@pytest.mark.gate
@pytest.mark.small
def test_n02_time_canary():
    """The N02 seam (NSM derive + numerics build + numpy lowering on a warm
    SM) must stay within 1.5x the recorded CPU-time baseline — ONE run."""
    if not goldenlib.TIME_BASELINE.exists():
        pytest.fail(
            "n02_time_baseline.json missing — run "
            "'python scripts/regen_goldens.py --baseline' and commit it")
    base = json.loads(goldenlib.TIME_BASELINE.read_text())
    if base.get("hostname") != platform.node():
        pytest.skip(
            f"canary baseline recorded on {base.get('hostname')!r}, running on "
            f"{platform.node()!r} — re-baseline on this host to arm the canary")
    cpu = goldenlib.measure_n02_cpu_seconds()
    limit = 1.5 * float(base["cpu_s"])
    assert cpu <= limit, (
        f"N02 derive+build CPU time {cpu:.2f}s exceeds 1.5x baseline "
        f"({base['cpu_s']}s -> limit {limit:.2f}s) — factor_terms-class "
        "blowup?  Investigate manually (median of 3) before re-baselining.")
