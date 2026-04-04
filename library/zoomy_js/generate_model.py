#!/usr/bin/env python3
"""Generate JavaScript Model2D code from a Zoomy symbolic model.

Usage:
    python generate_model.py                          # default: SWEWithTopo 2D
    python generate_model.py --model swe --dim 1      # SWE 1D
    python generate_model.py --output model_gen.js    # write to file
"""
import argparse
import sys

from zoomy_core.transformation.to_js import JsModel


def get_model(name, dimension):
    """Instantiate a Zoomy model by name."""
    if name in ("swe", "swe_topo"):
        from zoomy_core.model.models.shallow_water_topo import ShallowWaterEquationsWithTopo
        return ShallowWaterEquationsWithTopo(dimension=dimension)
    else:
        raise ValueError(f"Unknown model: {name}. Available: swe, swe_topo")


def main():
    parser = argparse.ArgumentParser(description="Generate JS Model2D from Zoomy symbolic model")
    parser.add_argument("--model", default="swe_topo", help="Model name (default: swe_topo)")
    parser.add_argument("--dim", type=int, default=2, help="Spatial dimension (default: 2)")
    parser.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")
    parser.add_argument("--object", action="store_true", help="Generate full Model2D object literal")
    args = parser.parse_args()

    model = get_model(args.model, args.dim)
    printer = JsModel(model)

    if args.object:
        code = printer.generate_model2d_object()
    else:
        code = printer.generate()

    if args.output:
        with open(args.output, "w") as f:
            f.write(code)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(code)


if __name__ == "__main__":
    main()
