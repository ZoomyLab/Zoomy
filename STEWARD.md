# Steward protocol

A **steward** is one Claude session that owns **exactly one folder** of Zoomy
and nothing else. You start it with `/steward <path>`. This file is the
rulebook; the live coordination state lives in `ORGANIZATION.md` (same dir).

    ZOOMY = /Users/adam-obbpb5az1dhsjzf/git/Zoomy   (paths below are relative to it)

## 0. Layout — repos & where work goes

Zoomy is a **submodule superrepo** at `ZOOMY` (root branch `develop`):

- `library/zoomy_*` — the **solver core + backends** (`zoomy_core`, `zoomy_jax`,
  `zoomy_foam`, `zoomy_firedrake`, `zoomy_fenicsx`, `zoomy_amrex`,
  `zoomy_dmplex`, `zoomy_gui`). Each is its **own git repo** — a submodule on
  `main`.
- `thesis/` — an **independent git repo** (not a submodule), on `main`. This is
  expected and fine. `meshes/`, `data/` are likewise their own repos.

**Two repos to watch.** Because `thesis/` is a separate repo, `git status` /
`diff` / `commit` at the superrepo do **not** see thesis changes, and vice
versa. Always run git in the right place and know which repo each command hits:
`git -C $ZOOMY/thesis …` for thesis, `git -C $ZOOMY/library/<sub> …` for a
backend, the superrepo root for submodule pointers.

**What goes where:**

- In the **Zoomy library** we work on the **solver core only** — framework,
  models, numerics kernels. **No test cases live in the library.**
- **Every test case / experiment goes in `thesis/notebooks/<category>/`**, under
  a proper category, with a documenting notebook (jupytext, reproducible).
- **`thesis/cases/` is opt-in:** put a case there **only when the user
  explicitly says so**. Otherwise it stays in `thesis/notebooks/`.

## 1. Startup (the `/steward` skill runs this)

1. Read this file fully; adopt it.
2. Your folder = the `/steward` argument. Read `ORGANIZATION.md`.
3. **Ownership:** ensure exactly one row in the Ownership table is yours (add it
   if missing). If a *more-specific* sub-path of your folder is owned by someone
   else, that sub-path is theirs — never touch it. If your whole folder is
   already owned by a different agent, **STOP** and tell the user.
4. **Clean or stop:** `git -C <repo> status` must be clean and on the default
   branch (`main`; `develop` for the superrepo root). If dirty, on the wrong
   branch, or behind `origin` → report and stop; do not edit.
5. Set `status=active`, `last seen=now` in your row. Scan `## Requests` for items
   addressed to you. Report a slim status (owned path · branch ok · open
   requests), then idle-watch (§5).

## 2. Ownership — one folder, most-specific wins

- You may edit a path **iff** your row owns it **and** no more-specific row owns
  the sub-path. Reading anything is fine; writing only your path.
- Before any edit, confirm your row is registered and your tree is clean.
- When a task is done, **check whether you still need the path**. If not, delete
  your row (or the sub-path row). The table must always reflect reality — stale
  rows are removed, never left behind.

## 3. Git — never branch, never worktree

- Commit only on the repo's **default branch**: `main` for submodules / nested
  repos, `develop` for the superrepo root. **Never** create a branch or a
  worktree. (This guarantees we never diverge and never need a merge.)
- One commit per finished task; plain imperative message; **no `Co-Authored-By`**.
- Stage only the files you changed; never `git add -A` blindly.
- **Submodules:** edit a submodule's content on its own `main`. The **superrepo
  pointer bump** (recording the new submodule SHA) is the `root` steward's job —
  request it via `ORGANIZATION.md`.

## 4. Cross-agent work — only through `ORGANIZATION.md`

You never edit another steward's folder. To get a change there:

- **Default:** append a request to `## Requests` with three fields —
  **Problem · Desired end-state (UI/behaviour) · How to test**. The owner
  replies in-thread ("taking it"), commits with an agreed message, and posts
  `done — <repo>@<hash>`. You pull, run the test, then close it (`→ resolved`)
  or re-request.
- **A named commit is the done-signal** — no other handshake.
- **Fallback** (owner not running): end your turn with one copy-pasteable block
  for the user to hand over (same three fields).

## 5. Cadence — when to re-check `ORGANIZATION.md`

- **Executing a task** → do **not** poll; check only when it ends.
- After every task: commit → read `ORGANIZATION.md` → act on requests to you →
  prune your stale rows / resolved requests.
- **Just did a cross-agent job, or waiting on a reply** → check every ~5 min for
  ~2 cycles (catch the quick back-and-forth), then relax.
- **Idle, nothing pending** → check in ~every 30 min; stay silent if there's
  nothing to do.

## 6. Talking to the user — slim, abstract, bullets only

- While working: `- problem: …` and `- on it: …`. Nothing else.
- On completion: `✅ done: …` · `→ next: …` · `⚠ not achieved / blocked: …`.
- **Mandatory `⚠` flag:** whenever a result is hand-crafted, dimension-specific,
  single-test-case, or otherwise **not generalizable** — say so explicitly.
  Never omit it.
- No reasoning dumps. The user asks if they want more.

## 7. Examples

**Ownership row** (in `ORGANIZATION.md`):
```
| jax | library/zoomy_jax | main | active | 2026-06-13 19:40 |
```

**Request thread** (commit = done):
```
### REQ-07  jax → core  ·  open
- Problem: JAX FVM needs the flux Jacobian; `Model._dF` is private.
- End-state: public `model.flux_jacobian(q)`, importable from `zoomy_core`.
- Test: `pytest library/zoomy_core/tests/test_flux_jacobian.py` green.
- core: taking it. (19:20)
- core: done — zoomy_core@`a1b2c3d` "expose flux_jacobian"; test green. (19:40)
- jax: pulled, verified, integrated. → resolved (19:45)
```

**Copy-paste fallback** (owner offline):
```
@core: expose a public flux_jacobian(q) on Model (now _dF).
End-state: importable from zoomy_core. Test: pytest .../test_flux_jacobian.py.
```

**Slim reports to the user:**
```
- problem: JAX bench 3× slower than numpy on small grids
- on it: profiling the per-step jit recompile
```
```
✅ done: flux_jacobian wired into JAX FVM; bench now 1.1× numpy (zoomy_jax@e0f1a2)
→ next: extend to the well-balanced source term
⚠ HAND-CRAFTED: jacobian path covers dimension=1 only; 2-D is a stub
```

## 8. Working in Zoomy solver/library code

The framework architecture — the **Model → SystemModel → NumericalSystemModel →
Solver** pipeline, how models are authored, and the **reuse-don't-reinvent**
rules — lives in the docs, not here. Read before touching `library/`:
`docs/book/authoring/model.md` (then `system-model.md`, `numerics.md`). Build
only from the existing blocks; subclass + compose closures; **never** add flags /
`if`-branches / private attributes / hand-rolled solvers. The full
root-cause-not-workaround checklist is in `~/.claude/agents/zoomy.md`.
