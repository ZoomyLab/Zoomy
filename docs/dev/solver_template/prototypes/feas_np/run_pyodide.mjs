// Run the jax-free 1-D single-source body under Pyodide (node).
// usage:  node run_pyodide.mjs <dir containing shared_body.py + pyodide_1d.py> [path/to/zoomy_core-*.whl]
import { loadPyodide } from "pyodide";
import fs from "node:fs";
import path from "node:path";
const dir = process.argv[2] ?? ".";
const wheel = process.argv[3];
const py = await loadPyodide();
await py.loadPackage(["numpy", "scipy", "sympy", "h5py", "micropip"]);
const mp = py.pyimport("micropip");
await mp.install(["param", "loguru"]);
if (wheel) {
  const dst = "/tmp/" + path.basename(wheel);
  py.FS.writeFile(dst, fs.readFileSync(wheel));
  await mp.install("emfs:" + dst);           // local tree (has the 2 unpushed commits)
} else {
  await mp.install("zoomy-core>=0.2.14");    // PyPI
}
py.FS.mkdirTree("/home/pyodide/feas");
for (const f of ["shared_body.py", "pyodide_1d.py"])
  py.FS.writeFile("/home/pyodide/feas/" + f, fs.readFileSync(path.join(dir, f)));
py.runPython(`
import os, sys, runpy
os.chdir('/home/pyodide/feas'); sys.path.insert(0, '/home/pyodide/feas')
sys.argv = ['pyodide_1d.py', '400']
runpy.run_path('pyodide_1d.py', run_name='__main__')
`);
