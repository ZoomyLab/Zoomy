# Installation

Use the installation workflows documented in the project root `README.md`:

- core (`zoomy_core`)
- optional backends (`zoomy_jax`, `zoomy_firedrake`, `zoomy_fenicsx`, `zoomy_amrex`, …)

For development environments, prefer containerized workflows where available.

## Git clone and submodules

Zoomy uses **git submodules** (`library/*`, `meshes`, `data`, …). `.gitmodules` sets `branch = main` for each so you can refresh to the latest upstream commit when you choose.

**Full clone with all submodules (pinned to the commits recorded on Zoomy’s branch):**

```bash
git clone --recurse-submodules https://github.com/ZoomyLab/Zoomy.git
cd Zoomy
```

**After clone, move every submodule to the latest `main` of its repository:**

```bash
cd Zoomy
git submodule sync --recursive
git submodule update --init --recursive
git submodule update --remote --merge --recursive
```

**Only one submodule — check it out and fast-forward to latest `main`:**

```bash
cd Zoomy
git submodule update --init library/zoomy_jax
git submodule update --remote --merge library/zoomy_jax
```

Replace `library/zoomy_jax` with any path from `.gitmodules`. To record the new submodule commit in your Zoomy branch: `git add <path>` and commit.

More detail and SSH URLs are in the root `README.md` under *Cloning the repository*.
