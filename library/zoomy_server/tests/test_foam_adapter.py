"""FoamAdapter unit tests that need no OpenFOAM: the in-container predicate and
the foam-h5 promotion (run/outputs/{swe,vof}_case.h5 -> job simulation.h5)."""
import os

from zoomy_server.adapters.foam import FoamAdapter, in_container


def test_in_container_returns_bool():
    assert isinstance(in_container(), bool)


def test_promote_foam_h5_swe_is_canonical(tmp_path):
    case = tmp_path / "case"
    out = tmp_path / "job"
    outputs = case / "run" / "outputs"
    outputs.mkdir(parents=True)
    out.mkdir()
    (outputs / "swe_case.h5").write_bytes(b"SME-participant")
    (outputs / "vof_case.h5").write_bytes(b"VOF-participant-longer")

    FoamAdapter._promote_foam_h5(str(case), str(out))

    # both participants copied through, SME side promoted to simulation.h5
    assert (out / "swe_case.h5").read_bytes() == b"SME-participant"
    assert (out / "vof_case.h5").read_bytes() == b"VOF-participant-longer"
    assert (out / "simulation.h5").read_bytes() == b"SME-participant"


def test_promote_foam_h5_falls_back_to_only_participant(tmp_path):
    case = tmp_path / "case"
    out = tmp_path / "job"
    outputs = case / "run" / "outputs"
    outputs.mkdir(parents=True)
    out.mkdir()
    (outputs / "vof_case.h5").write_bytes(b"only-vof")

    FoamAdapter._promote_foam_h5(str(case), str(out))
    assert (out / "simulation.h5").read_bytes() == b"only-vof"


def test_promote_foam_h5_no_outputs_is_quiet(tmp_path):
    case = tmp_path / "case"
    out = tmp_path / "job"
    (case).mkdir()
    out.mkdir()
    # no run/outputs at all -> no simulation.h5, no exception
    FoamAdapter._promote_foam_h5(str(case), str(out))
    assert not os.path.exists(out / "simulation.h5")
