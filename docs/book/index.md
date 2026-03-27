# Zoomy Documentation

Zoomy is a flexible modeling and simulation framework for free-surface flows.

This documentation site is now based on **Jupyter Book + Sphinx**, with:

- Markdown-first authoring
- notebook support without forced re-execution in docs builds
- test report ingestion from CI artifacts
- a dedicated docs build image separated from solver toolchains

```{note}
Notebook execution for docs is intentionally disabled in this stage.
Only notebooks already executed in tests should be published.
```

## Explore Zoomy

::::{grid} 2
:gutter: 3

:::{grid-item-card} Software Overview
:link: software.html
:link-type: url
![Software](../../web/images/wip.png)
Explore software capabilities, backend structure, and runtime design.
:::

:::{grid-item-card} Tutorials
:link: tutorials.html
:link-type: url
![Tutorials](../../web/images/wip.png)
Browse tutorials by topic and backend folder grouping.
:::

:::{grid-item-card} API by Backend
:link: api/index.html
:link-type: url
![API](../../web/images/wip.png)
Find API references split into core, jax, amrex, petsc, and firedrake.
:::

:::{grid-item-card} Testing and CI Reports
:link: ci-reports.html
:link-type: url
![Testing](../../web/images/wip.png)
Review testing policy, marker model, and generated CI summaries.
:::

::::
