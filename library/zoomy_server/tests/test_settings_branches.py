"""Two-level settings: the receiving adapter merges ITS branch over the
general keys and drops the other backends' branches."""
import json

from zoomy_server.adapter import SolverAdapter


class _A(SolverAdapter):
    tag = "jax"

    def solve(self, case_dir, output_dir, on_progress):  # pragma: no cover
        raise NotImplementedError


def test_branch_merge(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({
        "time_end": 0.6, "cfl": 0.45, "backend": "jax",
        "numpy": {"reconstruction_order": 1, "limiter": "minmod"},
        "jax": {"reconstruction_order": 2, "limiter": "venkatakrishnan"},
    }))
    s = _A().load_settings(str(tmp_path))
    assert s["time_end"] == 0.6                      # general survives
    assert s["reconstruction_order"] == 2            # own branch merged
    assert s["limiter"] == "venkatakrishnan"
    assert "numpy" not in s and "jax" not in s       # branches dropped
    assert s["backend"] == "jax"                     # informational, passes through


def test_no_branches_backward_compatible(tmp_path):
    raw = {"time_end": 0.1, "cfl": 0.45, "reconstruction_order": 1}
    (tmp_path / "settings.json").write_text(json.dumps(raw))
    assert _A().load_settings(str(tmp_path)) == raw
