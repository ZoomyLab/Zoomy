import os, re, collections
root = os.path.expanduser("~/git/Zoomy/thesis")
meth = collections.Counter(); priv = collections.Counter(); n=0
for dp, dn, fn in os.walk(root):
    if "/.git" in dp: continue
    for f in fn:
        if not f.endswith(".py"): continue
        p = os.path.join(dp, f); s = open(p, errors="ignore").read()
        if not re.search(r"from zoomy_core\.fvm\.solver_\w+ import", s) or re.search(r"zoomy_jax", s): continue
        n += 1
        for m in set(re.findall(r"\.(setup_simulation|run_simulation|solve|step|post_step)\(", s)): meth[m]+=1
        for m in set(re.findall(r"\._(sim_[a-z_]+)", s)): priv[m]+=1
print("numpy-only thesis scripts", n); print(meth.most_common()); print(priv.most_common(10))
