# 0003 — Align `zoomy` / `zoomy-cases` agent defs with steward + docs

**Decision:** keep both agents (do **not** remove them). They are *spawnable
subagent definitions* (own `tools:`/`model:` frontmatter) — a different
mechanism from the steward model, which governs main-session folder ownership.
Steward does not replace them. They are, however, drifted and partly obsolete.

**Where:**
- `~/.claude/agents/zoomy.md`
- `~/.claude/agents/zoomy-cases.md`
- Cross-refs: `STEWARD.md` §8 already points to `zoomy.md` for the
  root-cause checklist; the architecture lives in `docs/book/authoring/`.

**Nothing moves into STEWARD.** Process (ownership, git, where-work-goes) is
already complete in STEWARD.md. The agents' only process content is the
*obsolete* worktree + hub orchestration that steward replaced — delete it, do
not migrate it.

## How — three layers, three homes

**A. Must-fix (directly contradicts steward):**
- `zoomy.md:22` — remove the worktree permission ("Only use worktrees if
  needed… responsible that we do not have many open worktrees"). STEWARD §3 =
  never branch, never worktree.
- `zoomy.md:164–166` — delete the whole `## Hub membership` block. `hub/` dir
  is gone; `/hub model` + `~/git/Zoomy/hub/PROTOCOL.md` are dead. Replaced by
  `ORGANIZATION.md` + `STEWARD.md`.
- `zoomy.md:95`, `zoomy-cases.md:24` — `conda activate zoomy` →
  `micromamba run -n zoomy` (conda is **not** on PATH; micromamba is at
  `~/.local/bin/micromamba`).

**B. Should-fix (drifted from the rewritten docs — docs are authoritative):**
Replace the embedded architecture maps / recipes with a pointer to
`docs/book/authoring/{model,system-model,numerics}.md` + `backends/`. Stale
items found:
- solver names: `DaeSolver`→`DAESolver`, `ImexSolver`→`IMEXSolver`,
  `ChorinVamSolver`→`ChorinSplitVAMSolver`, `ColumnSolver`→
  `ColumnIntegratingSolver`; missing `RoeFreeSurfaceFlowSolver`, `FSFIMEXSolver`,
  `FSFSplittingSolver`.
- line numbers off: `HyperbolicSolver` :218 vs docs :274; `from_model`
  :550 vs docs :326; `FreeSurfaceFlowSolver` :762 vs :1072.
- stale import `from zoomy_core.model.models.system_model` (cases.md:303) —
  live path is `from zoomy_core.systemmodel import SystemModel`.
- legacy ops cited as current (`AffineProjection`, `EvaluateIntegrals`) that the
  doc pass removed as not-used-by-live-models.
Keep the **CLI/GUI card contract** (cases.md §"card contract") — it is
agent-unique and not in the docs.

**C. Recommended (the one thing that *should* move — to docs, not STEWARD):**
Promote the **working discipline** (reuse-before-new write-up;
root-cause-not-workaround + smell list + self-check; derivation-goal-first;
conventions: `M=I`, manual tagging, `Zstruct` dot-access, `param.Parameterized`
GUI contract, no `sys.path`; TDD loop; output style) into an in-repo
`docs/book/authoring/conventions.md`. Then have both STEWARD §8 and the agents
point there instead of duplicating. Rationale: an in-repo doc is reviewable and
CI-checkable; a `~/.claude/agents/*.md` file outside git silently drifts (this
task exists because it did).

**Why:** stale agent defs reintroduce abandoned workflows (hub, worktrees) and
wrong APIs (old solver names, old import path) that contradict both the steward
protocol and the just-corrected docs, so a freshly spawned `zoomy` agent would
act against current conventions.

**Learned:**
- The hub was removed in favour of `ORGANIZATION.md` + `STEWARD.md` (see the
  "Steward orchestration (replaces hub)" decision). `hub/` no longer exists.
- These files live under `~/.claude/agents/`, **outside** any git repo, so a
  steward cannot commit them — this is an edit-in-place task, flagged to the
  user. (That very fact is the argument for moving the durable content into
  `docs/` per layer C.)
