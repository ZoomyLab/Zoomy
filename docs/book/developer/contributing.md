# Contributing

## Branches

| branch | role | who changes it |
|---|---|---|
| `develop` | default branch, all work lands here | pull requests, reviewed |
| `main` | releases | the owner, by promoting `develop` |
| `feature/<topic>` | your work | you |

A push to `main` publishes the changed packages on PyPI and deploys the web page.

## A change in this repository

```bash
git switch develop && git pull
git switch -c feature/<topic>
# work, commit
git push -u origin feature/<topic>
```

1. Open a pull request into `develop`.
2. The tests of the changed components run on the pull request. They must pass.
3. A code owner reviews (`.github/CODEOWNERS`). Answer the comments with new commits.
4. After the approval, merge.

Without write access: fork the repository, push the branch to the fork and open the pull request from there.
The first run of the tests on a fork waits for an approval by a maintainer.

## A change in a backend repository

The backends with their own toolchain live in their own repositories, included here as submodules:
`library/zoomy_amrex`, `library/zoomy_foam`, `library/zoomy_gui`, `library/zoomy_dmplex`,
`library/zoomy_firedrake`, `library/zoomy_fenicsx`.
Each has one protected branch, `main`.

1. Pull request in the backend repository, into its `main`, reviewed as above.
2. After the merge, a pull request here into `develop` that moves the submodule pointer:

```bash
git -C library/zoomy_amrex switch main && git -C library/zoomy_amrex pull
git add library/zoomy_amrex
git commit -m "zoomy_amrex: <what changed>"
```

A pointer to a commit that exists only in a fork fails the check `Submodule pointers resolvable`.

## Tests

| tier | when | scope |
|---|---|---|
| on change | every pull request and push | the fast tests of the changed components, at most 5 minutes each |
| weekly | Sunday night | every test whose run takes at most 30 minutes |
| manual | before a release | the full cross-backend matrix and the timings |

How to run them locally: [Testing](testing.md). A new tutorial: [Adding a tutorial](tutorials.md).

## Conventions

- Docstrings and documentation in Markdown.
- Options are typed enums, not strings.
- Short commit messages that say what changed.
- New third-party components go into the acknowledgements (`docs/book/references/third_party.bib`).

## Releases

The version of a package is the `version` line of its `pyproject.toml`.
Raise it in the pull request that should be released. When `main` receives it, the package is published.
