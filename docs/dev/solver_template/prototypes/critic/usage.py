import os, re, collections, sys
root = os.path.expanduser("~/git/Zoomy/thesis")
files = []
for dp, dn, fn in os.walk(root):
    if "/.git" in dp: continue
    for f in fn:
        if f.endswith(".py"):
            p = os.path.join(dp, f)
            try: s = open(p, errors="ignore").read()
            except Exception: continue
            if re.search(r"(from|import) zoomy_jax", s):
                files.append((p, s))
print("files", len(files))
priv = collections.Counter(); meth = collections.Counter(); sub = collections.Counter()
for p, s in files:
    for m in set(re.findall(r"\._(rt_[a-z_0-9]+|sim_[a-z_0-9]+|advance|couple_[a-z_]+|compile_time_loop|explicit_hyperbolic_step|time_loop_cache|run|halo_exchange|mood_active)\b", s)):
        priv[m] += 1
    for m in set(re.findall(r"\.(setup_simulation|run_simulation|run_observed|solve|step|post_step|compute_timestep|_advance)\(", s)):
        meth[m] += 1
    for m in set(re.findall(r"class \w+\((?:\w+\.)*(HyperbolicSolver|ChorinSplitVAMSolverJax|IMEXSourceSolverJax|JaxSolver|JH)\)", s)):
        sub[m] += 1
print("public methods:", meth.most_common())
print("private attrs (files):", priv.most_common(25))
print("subclassing:", sub.most_common())
n_priv = sum(1 for p,s in files if re.search(r"\._(rt_|sim_|advance|couple_|compile_time_loop|explicit_hyperbolic_step)", s))
print("files touching private internals:", n_priv)
