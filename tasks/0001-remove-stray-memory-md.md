# 0001 — Remove the stray `memory.md` at the Zoomy root

**What:** Delete the untracked file `memory.md` at the superrepo root.

**Where:** `/Users/adam-obbpb5az1dhsjzf/git/Zoomy/memory.md` (untracked,
~13,460 lines).

**How:** Confirm it is the stray agent transcript, then `rm memory.md`. It is
untracked, so no git history is touched. (It is *not* in `.gitignore`, so it
shows up as an untracked file in every `git status` and risks being swept into
an accidental `git add -A`.)

**Why:** It is a leaked agent/session transcript, not project content. It
pollutes `git status` for every steward and could be committed by mistake.

**Learned:**
- First lines are an agent transcript (`myst` / `Ran` / `Determine if MyST is
  mystmd-based…` / `Bash`), confirming it is session spillover, not a doc.
- It is **not** mine (the documentation thread never wrote it), which is why it
  was flagged rather than deleted during that thread.
- Consider adding `memory.md` (or `*.transcript.md`) to the root `.gitignore`
  if these keep appearing.
