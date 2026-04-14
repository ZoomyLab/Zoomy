# Zoomy Documentation

Zoomy is a flexible modeling and simulation framework for free-surface flows.
Define a model once, run it across multiple backends (NumPy, JAX, AMReX, PETSc, Firedrake).

## Try It

::::{grid} 2
:gutter: 3

:::{grid-item-card} Open the GUI
:link: gui/index.html
:link-type: url
Launch the Zoomy GUI in your browser. Configure a simulation, select model + mesh + solver, and run -- no installation required.
:::

:::{grid-item-card} Load a Tutorial
:link: gui/index.html?project=tutorials/getting-started.zip
:link-type: url
Open a pre-configured project with three test cases (1D dam break, 2D dam break, scalar advection). Click a session and run.
:::

::::

## Learn

::::{grid} 3
:gutter: 3

:::{grid-item-card} User Guide
:link: user-guide.html
:link-type: url
How to use the GUI and CLI. Card-based workflow, sessions, `display()` output cells, URL-based project sharing.
:::

:::{grid-item-card} Tutorials
:link: tutorials/swe.html
:link-type: url
Step-by-step notebooks by topic: shallow water equations, moment equations, AMReX, OpenFOAM, DMPlex, and more.
:::

:::{grid-item-card} API Reference
:link: api/_apidoc_zoomy_core/zoomy_core.html
:link-type: url
Python reference for Zoomy Core and Zoomy JAX, plus backend overview pages.
:::

::::

## Develop

::::{grid} 3
:gutter: 3

:::{grid-item-card} Architecture
:link: dev-architecture.html
:link-type: url
System layers, data flow, how GUI and CLI stay in sync, deployment model.
:::

:::{grid-item-card} Component Reference
:link: dev-components.html
:link-type: url
Submodule map, card system, server adapter pattern, `display()` pipeline.
:::

:::{grid-item-card} Testing & CI
:link: testing.html
:link-type: url
Test markers, CI workflows, convergence baselines, and generated reports.
:::

::::
