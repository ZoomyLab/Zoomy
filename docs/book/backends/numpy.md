# NumPy backend

The NumPy backend is the reference implementation. It ships inside
`zoomy_core` and is what every other backend is verified against.

**Highlights**
- Pure Python / NumPy, no external solver dependency.
- Finite-volume HyperbolicSolver with NCP support.
- Same `Model` / `SystemModel` API as every other backend — no porting needed.

**Install**

```bash
pip install zoomy_core
```

**API reference**: see [zoomy_core](../api/_apidoc_zoomy_core/zoomy_core.rst).
