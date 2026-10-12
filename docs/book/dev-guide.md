# Developer Guide

```{warning}
Legacy: written for the previous solvers (`zoomy_core.fvm`, `zoomy_jax.fvm`), not the symbolic solvers; see the [user guide](guide/index.md).
```

```{warning}
**Work in progress.** These pages document Zoomy's internals for contributors.
They are being rewritten and parts are out of date — treat them as notes, not
as a specification. For anything user-facing, use the
[Authoring](authoring/model.md) and [Backends](backends/index.md) sections,
which are kept current.
```

This section covers the internal architecture: the component layout, how the
GUI and CLI share a card model, and how the pieces fit together.
