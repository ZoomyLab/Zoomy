import os, re, collections
root = os.path.expanduser("~/git/Zoomy")
pat = {
 "numpy_solver": r"from zoomy_core\.fvm\.solver_(numpy|imex_numpy|chorin_vam_numpy|dae_numpy|splitting_numpy) import",
 "ir_v6": r"zoomy_core\.solver(\.| import)|procedure_python|march_emitted|fvm\.solver2",
 "jax": r"(from|import) zoomy_jax",
}
cnt = collections.Counter(); where = collections.defaultdict(collections.Counter)
for dp, dn, fn in os.walk(root):
    if "/.git" in dp or "/output" in dp: continue
    for f in fn:
        if not f.endswith(".py"): continue
        p = os.path.join(dp, f)
        try: s = open(p, errors="ignore").read()
        except Exception: continue
        top = os.path.relpath(p, root).split(os.sep)
        key = "/".join(top[:2]) if top[0]=="library" else top[0]
        for k, r in pat.items():
            if re.search(r, s):
                cnt[k] += 1; where[k][key] += 1
for k in pat: print(k, cnt[k], dict(where[k].most_common(8)))
