# 0002 — Reconcile off-main / dirty submodules and untracked tutorials

**What:** Bring the working tree back to a clean state: submodules off `main`,
carrying unpushed/unmerged work, or dirty; plus untracked / modified tutorial
files. Each needs its owning steward to decide — **do not act on a guess; real
local-only work is at stake** (see the audit below).

**Status:** audited 2026-06-18 (read-only). Consolidation **paused** — the user
is checking in with the per-repo agents before any merge/push/delete.

## Per-repo audit (2026-06-18)

Counts are commits **ahead of `origin/main`**; "no remote" = the branch exists
ONLY locally (deleting it = permanent loss).

| repo | current branch | unique commits | notes |
|---|---|---|---|
| `library/zoomy_core` | `cstrong-opaque-derivative` | +32, **pushed** | also has `general/extract-nd` (0 ahead, pushed → safe delete) and local `main` (+5, FF-pushable). Stray worktree `/tmp/zoomy_core_main` (prunable). |
| `library/zoomy_jax` | `cstrong-opaque-derivative` | **+6, NO REMOTE** | ⚠ local-only. local `main` +4 (FF-pushable). |
| `library/zoomy_firedrake` | `restructure-adaptation` | **+15, NO REMOTE** | ⚠ local-only. local `main` **diverged: behind 37 / ahead 36** (not fast-forward). |
| `library/zoomy_foam` | `precice-coupling-interface-modes` | +15, **pushed** | local `main` clean at origin. |
| `library/zoomy_dmplex` | detached | — | dirty: `Model.H` rename `project_3d_to_2d`→`project_from_3d_to_2d` (1 line, generated header). |
| `library/zoomy_amrex` | detached, clean | — | only `main`; safe `checkout main`. |
| `library/zoomy_fenicsx` | detached, clean | — | only `main`; safe `checkout main`. |
| `library/zoomy_gui` | detached, clean | — | only `main`; safe `checkout main`. |
| `library/firedrake_animate` | detached, clean | — | only `main`; safe `checkout main`. |
| `meshes` | detached, clean | — | only `main`; safe `checkout main`. |
| `data` | `main`, clean | — | ✅ already good. |
| `tutorials/firedrake/malpasset_viscous_v2.py` | (superrepo) | — | modified, uncommitted; owner = root/firedrake. |

Superrepo `Zoomy`: on `develop`, clean, up to date with origin; one worktree
(good); a stale local `main` ref (behind/ahead of origin/main) — harmless.

## How (per STEWARD.md §0/§3) — once decisions are made

1. Per submodule, the **owning steward** decides merge-vs-keep-vs-discard for its
   feature branch, commits its content on `main`, pushes, removes extra branches.
2. **Unpushed-only branches** (`zoomy_jax/cstrong-opaque-derivative`,
   `zoomy_firedrake/restructure-adaptation`) MUST be merged or pushed **before**
   any delete — otherwise the commits are gone.
3. `zoomy_firedrake/main` is diverged — needs an explicit merge or force-push
   decision, independent of the feature branch.
4. The **`root`** steward then bumps the superrepo submodule pointers on
   `develop` and decides on the `tutorials/` file.

**Why:** a dirty/divergent submodule tree blocks a clean steward startup
(STEWARD §1.4 "clean or stop") and the superrepo pointer no longer matches the
checked-out submodule HEADs.

**Learned:**
- These are **not** the documentation thread's changes — never fold them into a
  docs/tasks commit.
- The blanket "everything to main, delete non-main branches" is **unsafe as
  stated**: it would lose 6 (jax) + 15 (firedrake) local-only commits and force
  a non-FF reconcile of `firedrake/main`. Consolidate (merge+push) first.
- Verify branch state with `git -C library/<sub> rev-list --left-right --count
  origin/main...<branch>` and `... origin/<branch>..<branch>` before deleting.
