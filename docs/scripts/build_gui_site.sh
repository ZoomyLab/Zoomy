#!/usr/bin/env bash
# Assemble the GUI the site serves under /Zoomy/gui/, from this checkout.
#
#   docs/scripts/build_gui_site.sh SITE      # -> SITE/gui/ (SITE: the site root, /Zoomy/)
#
# One recipe for the deploy and the tests: render-webpage.yml runs it into
# docs/book/_build/html, tests/qr_cases/test_qr_browser.py drives the result in a browser.
# Steps: the Theia extension and `theia build` (production), the GUI "brain" (card catalog,
# Pyodide worker, engine, projects) under gui/gui/, the zoomy_core / zoomy_plotting wheels
# both Pyodide workers install, the release data a browser session fetches, sw.js, the CLI
# package, version.json.
#
# Env: WHEEL_PYTHON (python that builds the wheels; python3), GUI_BUILD_VERSION (version.json;
# "local"), GUI_SKIP_THEIA=1 (reuse library/zoomy_gui/lib from an earlier build).
set -euo pipefail

SITE=${1:?usage: build_gui_site.sh SITE}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$SITE"
SITE=$(cd "$SITE" && pwd)
GUI="$SITE/gui"

cd "$ROOT/library/zoomy_gui"
if [ -z "${GUI_SKIP_THEIA:-}" ]; then
  npm ci --no-audit --no-fund || npm install --no-audit --no-fund
  # The Zoomy extension: Baukasten start page, code-editor route, and the native Theia
  # notebook driven by an in-browser Pyodide kernel. @theia/* resolve from the app's
  # node_modules (parent), so this only fetches typescript.
  ( cd zoomy-theia-ext && npm install --no-audit --no-fund && npm run build )
  npx theia build --mode production
fi
cd "$ROOT"
mkdir -p "$GUI"
cp -r library/zoomy_gui/lib/frontend/. "$GUI/"

# The GUI "brain" (card catalog + Pyodide worker + engine) is served as static assets under
# gui/ RELATIVE TO THE PAGE -- the model-config widget and the worker fetch it that way --
# hence /gui/gui/. Copied by name so the build tree (node_modules, lib, package.json) does
# not leak into the deploy.
mkdir -p "$GUI/gui"
for a in assets cards demo previews projects snippets tutorials \
         core.js engine.py param_extract.py pyodide-worker.js; do
  cp -r "library/zoomy_gui/$a" "$GUI/gui/"
done
# The zoomy_core / zoomy_plotting both Pyodide workers install: wheels built from this
# commit, named in gui/gui/wheels/index.json (PyPI only when that file is absent, i.e. local
# dev). The QR CI installs the same build.
"${WHEEL_PYTHON:-python3}" docs/scripts/build_gui_wheels.py "$GUI/gui/wheels"
# Release assets a browser session fetches, served same-origin: GitHub sends no CORS headers
# for release downloads (tests/qr_cases/test_qr_live.py MIRRORED).
mkdir -p "$GUI/gui/data"
for f in sme_vof_snap_gui_l1.tar.gz; do
  curl -fsSL --retry 3 -o "$GUI/gui/data/$f" \
    "https://github.com/ZoomyLab/Zoomy/releases/download/case-data-v1/$f"
done
# #10 offline + cross-origin isolation service worker (COOP/COEP)
cp library/zoomy_gui/sw.js "$GUI/sw.js"
# The isomorphic CLI package, imported as ./zoomy_cli/browser.mjs. Node-only and dev
# detritus stripped to keep the deploy small.
mkdir -p "$GUI/gui/zoomy_cli"
cp library/zoomy_cli/browser.mjs library/zoomy_cli/node.mjs library/zoomy_cli/package.json \
   "$GUI/gui/zoomy_cli/"
cp -r library/zoomy_cli/src "$GUI/gui/zoomy_cli/"

# The build version, shown in the footer
GUI_COMMIT=$(git -C library/zoomy_gui rev-parse HEAD)
BUILT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$GUI/version.json" <<EOF
{"version": "${GUI_BUILD_VERSION:-local}", "commit": "${GUI_COMMIT}", "built": "${BUILT}"}
EOF
echo "Built GUI version: ${GUI_COMMIT:0:7} at ${BUILT} -> $GUI"

# Cleanup: source maps are large and dev-only
rm -f "$GUI/.git" "$GUI/.gitignore"
rm -rf "$GUI/gui/__pycache__"
rm -f "$GUI"/*.map
