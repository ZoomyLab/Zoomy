"""`Model.describe()` shows the equations of every kind of model, and a model written as
operators has the identity mass matrix."""
import sympy as sp

from zoomy_core.model.models import SWE, SME
from zoomy_core.systemmodel import SystemModel


def test_operator_model_describes_its_balance_law():
    text = SWE().describe()._repr_markdown_()
    assert "3 equations" in text
    assert r"**State $Q$:** $b$, $h$, $hu$" in text
    assert r"\frac{\partial}{\partial t} h + \frac{\partial}{\partial x} hu \;=\; 0" in text


def test_derived_model_describes_its_derivation_rows():
    text = SME(level=1).describe()._repr_markdown_()
    assert "4 equations" in text
    assert r"**Evolved fields:** $b$, $h$, $q_{0}$, $q_{1}$" in text
    assert r"\frac{\partial}{\partial t} h + \frac{\partial}{\partial x} q_{0} \;=\; 0" in text
    assert "0 = 0" not in text


def test_operator_model_has_identity_mass_matrix():
    sm = SystemModel.from_model(SWE())
    assert sp.Matrix(sm.mass_matrix.tolist()) == sp.eye(3)
