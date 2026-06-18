# 0002 — Reconcile off-main / dirty submodules and untracked tutorials

**What:** Bring the working tree back to a clean state: submodules that carry
uncommitted content or sit off `main`, plus untracked / modified tutorial
files. None of these belong to the documentation thread, so each needs its
owning steward to decide.

**Where (superrepo `git status`, branch `develop`):**
- `library/zoomy_amrex` — modified content (uncommitted).
- `library/zoomy_core` — new commits **+ modified content + untracked content**.
- `library/zoomy_dmplex` — modified content (uncommitted).
- `library/zoomy_firedrake` — new commits (pointer not bumped in superrepo).
- `library/zoomy_jax` — new commits (pointer not bumped in superrepo).
- `tutorials/firedrake/malpasset_viscous_v2.py` — modified (untracked changes).
- `tutorials/shallow_moment/` — untracked (notebook + `.py` + `.md` +
  `_artifacts/`); already carries its own `sme_vs_mlsme_dambreak.md`.

**How (per STEWARD.md §0/§3):**
1. Run git **inside each repo** (`git -C library/<sub> status`) — the superrepo
   only sees pointers, not the submodule's own dirty tree.
2. The owning steward commits its submodule's content on that submodule's
   `main` (never branch/worktree), stages only files it changed.
3. The **`root`** steward then bumps the superrepo submodule pointers on
   `develop` (records the new SHAs) and decides on the `tutorials/` files.

**Why:** A dirty/divergent submodule tree blocks a clean steward startup
(STEWARD.md §1.4: "clean or stop") and means the superrepo pointer no longer
matches reality.

**Learned:**
- These were all present and flagged during the docs thread but are **not** its
  changes — do not fold them into a docs commit.
- `library/zoomy_core` is the messiest (new commits *and* uncommitted *and*
  untracked) — start there; it is the dependency root for the other backends.
- Earlier session notes recorded off-main branches in the wild:
  `zoomy_core` on `cstrong-opaque-derivative`, `zoomy_firedrake` on
  `restructure-adaptation`. Verify current branch with `git -C … branch
  --show-current` before committing.
