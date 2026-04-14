# Using the CLI

The Zoomy CLI (`zoomy`) mirrors the GUI workflow from the terminal. It shares the same card system and produces identical results.

## Setup

The CLI requires Node.js (v18+). From the repository root:

```bash
# Verify it works
node library/zoomy_cli/cli.js --help
```

For convenience, create an alias:
```bash
alias zoomy="node $(pwd)/library/zoomy_cli/cli.js"
```

## Workflow

### 1. Initialize a project

```bash
mkdir my-simulation && cd my-simulation
zoomy start
```

This creates a `.zoomy/` directory with project state.

### 2. Browse available cards

```bash
zoomy overview          # all tabs at a glance
zoomy list model        # list model cards
zoomy list mesh         # list mesh cards
zoomy list solver       # list solver cards
zoomy show model sme-l0 # detailed card info
```

### 3. Select cards

```bash
zoomy select model sme-l0       # SWE (SME L0) model
zoomy select mesh create-1d     # 1D mesh
zoomy select solver numpy       # NumPy solver
zoomy status                    # verify selections
```

### 4. Inspect the case

```bash
zoomy case   # prints the simulation case as JSON
```

Output:
```json
{
  "version": "1.0",
  "model": {
    "class_path": "zoomy_core.model.models.sme_model.SMEInviscid",
    "init": {"level": 0},
    "parameters": {}
  },
  "mesh": {"type": "create_1d", "domain": [0, 1], "n_cells": 100},
  "solver": {"time_end": 0.1, "cfl": 0.45, "output_snapshots": 10}
}
```

### 5. Run the simulation

**Locally (no server needed):**
```bash
zoomy run --local
```

This generates a Python script from the selected cards and executes it via `python -c`. Output goes to stdout.

**With a backend server:**
```bash
zoomy connect http://localhost:8080
zoomy run --wait        # submit and block until complete
zoomy jobs              # list all jobs
zoomy watch <job_id>    # attach to a running job
```

### 6. Save the project

```bash
zoomy save my-project.zip    # export to ZIP
zoomy load my-project.zip    # restore later
```

## Session Management

```bash
zoomy session                    # show active session
zoomy session new "2D test"      # create and switch to new session
zoomy session switch "1D test"   # switch back
zoomy session list               # list all sessions
zoomy session rename "New name"  # rename active session
```

## Use Cases

### Batch simulation script

```bash
#!/usr/bin/env bash
set -euo pipefail

zoomy start
zoomy select model sme-l0
zoomy select mesh create-1d
zoomy select solver numpy
zoomy run --local
```

See `library/zoomy_cli/examples/tutorial-pipeline.sh` for a complete example.

### Parameter sweep (multiple sessions)

```bash
zoomy start
zoomy select model sme-l0
zoomy select solver numpy

# Coarse mesh
zoomy session new "Coarse (50 cells)"
zoomy select mesh create-1d
# (modify mesh params via code edit when available)
zoomy run --local

# Fine mesh
zoomy session new "Fine (200 cells)"
zoomy select mesh create-1d
zoomy run --local

zoomy save convergence-study.zip
```

### CI/CD integration

```bash
# In a GitHub Actions step:
- name: Run simulation
  run: |
    node library/zoomy_cli/cli.js start
    node library/zoomy_cli/cli.js select model sme-l0
    node library/zoomy_cli/cli.js select mesh create-1d
    node library/zoomy_cli/cli.js select solver numpy
    node library/zoomy_cli/cli.js run --local
```

## Command Reference

| Command | Description |
|---------|-------------|
| `zoomy start` | Initialize project in current directory |
| `zoomy overview` | Show all cards and current selections |
| `zoomy list <tab>` | List cards (tabs: model, mesh, solver, visu) |
| `zoomy show <tab> <name>` | Show card details |
| `zoomy select <tab> <name>` | Select a card |
| `zoomy status` | Show current selections and session |
| `zoomy case` | Print case JSON |
| `zoomy run --local` | Run with local Python |
| `zoomy run` | Submit to connected backend |
| `zoomy run --wait` | Submit and wait for completion |
| `zoomy connect <url>` | Connect to a backend |
| `zoomy disconnect <tag>` | Disconnect a backend |
| `zoomy backends` | List connected backends |
| `zoomy watch <job_id>` | Live progress for a job |
| `zoomy jobs` | List all jobs |
| `zoomy save <path.zip>` | Save project to ZIP |
| `zoomy load <path.zip>` | Load project from ZIP |
| `zoomy session new [name]` | Create new session |
| `zoomy session switch <name>` | Switch to session |
| `zoomy session list` | List sessions |
| `zoomy session rename <name>` | Rename active session |
