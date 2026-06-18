# Tasks

The Zoomy backlog. **One markdown file per task** — this folder holds the full
detail; `ORGANIZATION.md` holds only the slim index (task · responsible
agent(s) · link). The two are kept in sync (see [STEWARD.md](../STEWARD.md) §4).

Each task file states **what · where · how (if clear) · why · learned**.

**Lifecycle:**
- *Add:* write `tasks/<nnnn>-<slug>.md` **and** add a row to the `ORGANIZATION.md`
  Tasks index.
- *Claim:* owning a folder makes you responsible for tasks in it. On taking new
  ownership, re-scan the index and add yourself (STEWARD §1.5).
- *Finish:* delete the `tasks/` file **and** its index row in the same commit
  that completes the work (a named commit is the done-signal).
