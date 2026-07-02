"""zoomy-case — convert between the case representations.

    zoomy-case to-folder   case.py  case_dir/     # .py -> model.py/mesh.py/settings.json[...]
    zoomy-case from-folder case_dir/ case.py      # folder -> single .py
    zoomy-case to-ipynb    case.py  case.ipynb    # .py -> notebook (jupytext)
    zoomy-case from-ipynb  case.ipynb case.py     # notebook -> .py (jupytext)

The single-file form is a jupytext "percent" .py structured by markdown
headings (## Model, ## Mesh, ## Settings, ## Solver, optional
## Visualization) — the same file the GUI exports and the server ingests.
"""
import argparse

from . import case


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["to-folder", "from-folder", "to-ipynb", "from-ipynb"])
    ap.add_argument("source")
    ap.add_argument("dest")
    a = ap.parse_args()

    if a.command == "to-folder":
        case.to_folder(open(a.source).read(), a.dest)
    elif a.command == "from-folder":
        open(a.dest, "w").write(case.from_folder(a.source))
    elif a.command == "to-ipynb":
        open(a.dest, "w").write(case.to_notebook(open(a.source).read()))
    elif a.command == "from-ipynb":
        open(a.dest, "w").write(case.from_notebook(open(a.source).read()))
    print(f"{a.command}: {a.source} -> {a.dest}")


if __name__ == "__main__":
    main()
