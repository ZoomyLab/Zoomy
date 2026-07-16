"""Template showcase — the ONE figure, three presets, plus the semantic palette.

Regenerates ``figures/template_showcase.png``: the identical multi-line overlay
rendered under ``screen`` / ``publication`` / ``thesis`` (fonts + figure size
come only from the preset), and a semantic-color panel where ``experiment`` /
``reference`` / ``analytic`` are fixed while model series ride the Okabe-Ito
cycle. Cases own ZERO styling — everything below is ``zp.subplots`` + a helper.

    python examples/template_showcase.py            # writes figures/ + prints checks
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import zoomy_plotting as zp

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS = ("screen", "publication", "thesis")

x = np.linspace(0.0, 2 * np.pi, 200)
SERIES = [
    {"x": x, "y": np.sin(x), "label": "SME(1)"},
    {"x": x, "y": np.sin(x) + 0.15 * np.sin(3 * x), "label": "SME(2)"},
    {"x": x, "y": np.sin(x) + 0.15 * np.sin(3 * x) + 0.05 * np.sin(5 * x),
     "label": "SME(3)"},
]


PANELS = [("phase", 1.0), ("decay", 0.5), ("mixed", 0.25)]


def preset_row(dst):
    """The same 1x3 overlay, once per preset — proof that format = preset."""
    outs = []
    for preset in PRESETS:
        fig, axes = zp.subplots(1, 3, preset=preset)     # size + fonts from here
        for ax, (title, damp) in zip(axes, PANELS):
            zp.line_plot(
                ax,
                [{"x": x, "y": s["y"] * np.exp(-damp * x), "label": s["label"]}
                 for s in SERIES],
                xlabel="x", ylabel="q", title=title)
        zp.figure_legend(fig)
        p = os.path.join(dst, f"template_showcase_{preset}.png")
        fig.savefig(p)                                    # dpi from the preset
        plt.close(fig)
        outs.append((preset, p, tuple(round(v, 2) for v in fig.get_size_inches())))
    return outs


def semantic_panel(dst):
    """experiment / reference / analytic fixed; model series on the cycle."""
    fig, ax = zp.subplots(preset="publication", width="2col")
    truth = np.sin(x)
    zp.line_plot(ax, [
        {"x": x[::12], "y": truth[::12] + 0.03 * np.random.default_rng(0).standard_normal(len(x[::12])),
         "label": "experiment", "color": zp.colors.experiment, "ls": "none",
         "marker": "o"},
        {"x": x, "y": truth, "label": "reference (DNS)", "color": zp.colors.reference},
        {"x": x, "y": np.sin(x) * 0.98, "label": "analytic", "color": zp.colors.analytic,
         "ls": "--"},
    ] + [dict(s) for s in SERIES], xlabel="x", ylabel="q",
        title="semantic roles fixed; model series cycled")
    zp.figure_legend(fig)
    p = os.path.join(dst, "template_showcase_semantic.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def checks():
    """Programmatic evidence: font + figure size actually change per preset."""
    rows = []
    for preset in PRESETS:
        zp.use(preset)
        rows.append((preset,
                     mpl.rcParams["font.size"],
                     mpl.rcParams["figure.titlesize"],
                     zp.figsize(preset, ncols=3)))
    zp.use("publication")   # leave a sane global default
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    dst = ap.parse_args().out
    os.makedirs(dst, exist_ok=True)

    for preset, p, size in preset_row(dst):
        print(f"[{preset:11s}] {size} -> {p}")
    print("[semantic   ]", semantic_panel(dst))

    print("\n=== per-preset font.size / figure.titlesize / 1x3 figsize ===")
    for preset, fs, ts, grid in checks():
        print(f"  {preset:11s} body={fs:>4}pt  title={ts:>4}pt  1x3={grid}")
    print("\nsemantic colors:",
          {"experiment": zp.colors.experiment, "reference": zp.colors.reference,
           "analytic": zp.colors.analytic})


if __name__ == "__main__":
    main()
