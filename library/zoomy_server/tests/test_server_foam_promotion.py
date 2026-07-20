"""#10 test_server_foam_promotion — foam-h5 promotion into the job dir
(absorbs test_foam_adapter): ``run/outputs/{swe,vof}_case.h5`` -> job
``simulation.h5`` with the SWE/SME side as the canonical participant.
Needs no OpenFOAM.
"""
from __future__ import annotations

import os

import pytest

from zoomy_server.adapters.foam import FoamAdapter, in_container

pytestmark = [pytest.mark.small, pytest.mark.postprocessing, pytest.mark.gui]


def test_in_container_predicate_returns_bool():
    assert isinstance(in_container(), bool)


def test_promote_swe_is_canonical_participant(tmp_path):
    case, out = tmp_path / "case", tmp_path / "job"
    outputs = case / "run" / "outputs"
    outputs.mkdir(parents=True)
    out.mkdir()
    (outputs / "swe_case.h5").write_bytes(b"SME-participant")
    (outputs / "vof_case.h5").write_bytes(b"VOF-participant-longer")

    FoamAdapter._promote_foam_h5(str(case), str(out))

    # both participants copied through; the SME side IS simulation.h5
    assert (out / "swe_case.h5").read_bytes() == b"SME-participant"
    assert (out / "vof_case.h5").read_bytes() == b"VOF-participant-longer"
    assert (out / "simulation.h5").read_bytes() == b"SME-participant"


def test_promote_falls_back_to_only_participant(tmp_path):
    case, out = tmp_path / "case", tmp_path / "job"
    outputs = case / "run" / "outputs"
    outputs.mkdir(parents=True)
    out.mkdir()
    (outputs / "vof_case.h5").write_bytes(b"only-vof")

    FoamAdapter._promote_foam_h5(str(case), str(out))
    assert (out / "simulation.h5").read_bytes() == b"only-vof"


def test_promote_without_outputs_is_quiet(tmp_path):
    case, out = tmp_path / "case", tmp_path / "job"
    case.mkdir()
    out.mkdir()
    FoamAdapter._promote_foam_h5(str(case), str(out))  # no run/outputs at all
    assert not os.path.exists(out / "simulation.h5")
