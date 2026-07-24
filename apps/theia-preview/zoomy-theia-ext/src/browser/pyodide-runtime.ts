/** Pyodide runtime for the native Theia notebook kernel.
 *  Loads Pyodide once, installs the SAME packages the GUI/JupyterLite page uses
 *  (zoomy-core + zoomy-plotting from PyPI), and runs a cell returning
 *  Jupyter-style outputs: stdout stream, the last-expression rich repr
 *  (html / markdown / latex / text) and any matplotlib figures as PNG.
 *  This mirrors the GUI's pyodide worker; jedi autocomplete / interrupts fold
 *  in here next, over the same kernel. */
export interface CellStreamOutput { type: 'stream'; text: string; }
export interface CellDataOutput { type: 'data'; mime: string; value: string; }
export interface CellErrorOutput { type: 'error'; ename: string; evalue: string; traceback: string; }
export type CellOut = CellStreamOutput | CellDataOutput | CellErrorOutput;

declare global { interface Window { loadPyodide: any; } }
let pyodidePromise: Promise<any> | undefined;

async function loadScript(src: string): Promise<void> {
    await new Promise<void>((res, rej) => {
        const s = document.createElement('script'); s.src = src;
        s.onload = () => res(); s.onerror = () => rej(new Error('load ' + src));
        document.head.appendChild(s);
    });
}

// Defines a persistent namespace + an exec helper that mimics a Jupyter cell:
// run the block, then display the value of a trailing expression (if any),
// plus flush matplotlib figures. Returns a JSON list of {mime,data}.
const INIT_PY = `
import sys, io, base64, json, ast
import matplotlib
matplotlib.use("AGG")
__zoomy_ns__ = {"__name__": "__main__"}

def __zoomy_exec__(src):
    outs = []
    tree = ast.parse(src, mode="exec")
    val = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        exec(compile(tree, "<cell>", "exec"), __zoomy_ns__)
        val = eval(compile(ast.Expression(last.value), "<cell>", "eval"), __zoomy_ns__)
    else:
        exec(compile(tree, "<cell>", "exec"), __zoomy_ns__)
    if val is not None:
        rich = None
        for meth, mime in (("_repr_html_", "text/html"),
                           ("_repr_markdown_", "text/markdown"),
                           ("_repr_latex_", "text/latex")):
            fn = getattr(val, meth, None)
            if fn:
                try:
                    r = fn()
                    if r:
                        rich = {"mime": mime, "data": r}
                        break
                except Exception:
                    pass
        outs.append(rich if rich else {"mime": "text/plain", "data": repr(val)})
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        for num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
            outs.append({"mime": "image/png",
                         "data": base64.b64encode(buf.getvalue()).decode()})
        plt.close("all")
    return json.dumps(outs)
`;

export function getPyodide(log: (m: string) => void): Promise<any> {
    if (!pyodidePromise) {
        pyodidePromise = (async () => {
            log('Loading Pyodide…');
            await loadScript('https://cdn.jsdelivr.net/pyodide/v0.28.0/full/pyodide.js');
            const py = await window.loadPyodide();
            log('Installing zoomy-core + zoomy-plotting…');
            // Let micropip resolve the whole tree (like JupyterLite's piplite):
            // it pulls the emscripten builds of numpy/scipy/matplotlib/h5py from
            // the Pyodide repo and sympy>=1.14 etc. from PyPI. Pre-loading the
            // stack here would pin sympy 1.13.3 and break zoomy-core's sympy>=1.14.
            await py.loadPackage('micropip');
            const micropip = py.pyimport('micropip');
            await micropip.install(['zoomy-core', 'zoomy-plotting']);
            log('Warming up kernel…');
            await py.runPythonAsync(INIT_PY);
            log('Pyodide ready.');
            return py;
        })();
    }
    return pyodidePromise;
}

export async function runCell(py: any, code: string): Promise<CellOut[]> {
    const outs: CellOut[] = [];
    py.setStdout({ batched: (s: string) => outs.push({ type: 'stream', text: s + '\n' }) });
    py.setStderr({ batched: (s: string) => outs.push({ type: 'stream', text: s + '\n' }) });
    try {
        py.globals.set('__zoomy_src__', code);
        const json = await py.runPythonAsync('__zoomy_exec__(__zoomy_src__)');
        const rich = JSON.parse(json || '[]') as Array<{ mime: string; data: string }>;
        for (const r of rich) { outs.push({ type: 'data', mime: r.mime, value: r.data }); }
    } catch (e: any) {
        const msg = (e && e.message) || String(e);
        outs.push({ type: 'error', ename: 'PythonError', evalue: msg, traceback: msg });
    }
    return outs;
}
