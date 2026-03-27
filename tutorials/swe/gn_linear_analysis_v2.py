import sympy as sp

from zoomy_core.model.analysis_linear import LinearWaveAnalyser
from tutorials.swe.simple_gn_v2 import make_model


def run():
    model = make_model()
    model.print_model_functions(function_names=["flux", "source"])


    # Base state and perturbation amplitudes
    h0, hu0 = sp.symbols("h0 hu0", real=True)
    h1, m1 = sp.symbols("h1 m1", real=True)

    analyser = LinearWaveAnalyser(model)
    result = analyser.analyse_phase_velocity(
        base_state={"h": h0, "hu": hu0},
        amplitude_symbols=[h1, m1],
        field_to_amplitude={"h": h1, "hu": m1},
        print_steps=True,
    )

    # Optional reduced output for the common rest-state case hu0 = 0.
    c_rest = [sp.simplify(c.subs({hu0: 0})) for c in result["phase_velocity_solutions"]]
    print("phase velocity for hu0=0:", c_rest)
    return result


if __name__ == "__main__":
    run()
