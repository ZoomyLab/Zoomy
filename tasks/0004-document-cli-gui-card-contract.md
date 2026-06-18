# 0004 — Document the CLI/GUI card contract in the docs (optional)

**What:** Capture the zoomy_cli / zoomy_gui **card contract** as an in-repo doc
page. It previously lived *only* in the deleted `~/.claude/agents/zoomy-cases.md`
and is not in the book; the slim `zoomy` agent now just points at the code +
tests. This task preserves the non-obvious parts so a future agent doesn't have
to reverse-engineer them.

**Where:** new `docs/book/authoring/cards.md` (or a section under an existing
authoring page), registered in `docs/book/_toc.yml`. Source of truth is the code:
`library/zoomy_cli/cli.js`, `library/zoomy_gui/{core.js,app.js,param_extract.py}`,
`library/zoomy_gui/cards/`, and their `tests/`.

**How (the non-obvious bits worth writing down):**
- Card folders merge in order **`default → generated → user`**; duplicate IDs
  keep the **first** occurrence (`cli.js:_loadCardsFolder`, mirrored in
  `core.js`).
- Card shape: `id`, `title`, `class` (model/solver) **or** a `template` string
  (mesh/viz) with `{placeholder}` substitution from `init`.
- Solver cards declare `requires_tag` (`numpy`/`jax`/`amrex`/`dmplex`); the
  Pyodide worker runs the numpy `template` verbatim with `model, mesh, Q, Qaux,
  store, open_hdf5, close_store` in scope.
- The GUI auto-builds widgets via
  `param_extract.extract_param_schema(class_path, init)` — **every tunable knob
  must be a `param.Number/Integer/Selector/Boolean/String`** or its widget
  disappears. (This `param.Parameterized` contract is also why solvers are
  param-classes; cross-link `docs/book/backends/numpy.md`.)
- `Project.buildCase()` (`core.js`) emits the `{version, model, mesh, solver}`
  JSON every adapter (`pyodide_adapter`, `http_adapter`) consumes.
- Tests: `library/zoomy_cli/tests/` (`test_card_commands.js`,
  `test_cli_contract.js`, `test_adapter_symmetry.js`) and
  `library/zoomy_gui/tests/` (`test_card_anatomy_v2.js`, `test_param_widgets.js`,
  …). Run `cd library/zoomy_cli && npm test`, `cd library/zoomy_gui && npm test`.

**Why:** the card contract is the only piece of the old `zoomy-cases` agent that
was neither in the docs nor in STEWARD. Keeping it discoverable in the book (not
in an out-of-git agent file) matches the project rule that architecture lives in
the docs, and lets the agents stay slim.

**Learned:**
- This is **optional** — the slim agent already points to the code + tests, so a
  motivated agent can rediscover it; do this only if the card surface is going to
  be touched often enough to deserve prose.
- Verify every detail against the JS before writing — the old agent's line
  numbers had drifted, which is exactly why it was retired.
