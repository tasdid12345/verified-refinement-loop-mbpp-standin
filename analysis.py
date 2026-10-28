"""Stand-in for generate-verify-refine: random-edit loop vs LightGBM-ranked edits + staged gate, on buggy MBPP programs."""
import io, json, os, sys, re, ast, copy, time, random, signal, tokenize, keyword, builtins, contextlib, warnings
from collections import Counter
import numpy as np, pandas as pd, requests, lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score

warnings.simplefilter("ignore")
URL = "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/full/test-00000-of-00001.parquet"
BUDGET, SEEDS = 30, (42, 43, 44)
OPS = {'+': ['-', '*'], '-': ['+'], '*': ['+', '/'], '/': ['*'], '//': ['%', '/'], '%': ['//']}
CMP = {'<': ['<=', '>'], '<=': ['<', '>='], '>': ['>=', '<'], '>=': ['>', '<='], '==': ['!='], '!=': ['==']}
SYM = sorted(set(OPS) | set(CMP) | {r for v in list(OPS.values()) + list(CMP.values()) for r in v})
BUILTINS = set(dir(builtins))

def download(url):  # retry with backoff on 429/5xx
    for k in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200: return r.content
            if r.status_code not in (429, 500, 502, 503, 504): r.raise_for_status()
        except requests.RequestException as e: print("retry:", e)
        time.sleep(2 ** k)
    raise RuntimeError("download failed")

# ---------- safe in-process execution with a timeout ----------
class TO(Exception): pass
def _alarm(*a): raise TO()
signal.signal(signal.SIGALRM, _alarm)
def run(fn, t=0.2):
    signal.setitimer(signal.ITIMER_REAL, t)
    try:
        with contextlib.redirect_stdout(io.StringIO()): return True, fn()
    except (Exception, SystemExit): return False, None
    finally: signal.setitimer(signal.ITIMER_REAL, 0)

def passes(code, asserts, setup=''):  # number of asserts that pass
    ns = {}
    if not run(lambda: exec(setup + '\n' + code, ns))[0]: return 0
    return sum(run(lambda a=a: exec(a, ns))[0] for a in asserts)

def syntax_ok(c):
    try: compile(c, '<p>', 'exec'); return True
    except Exception: return False

# ---------- token edits ----------
def toks(code):  # (type, string, start, end, line) for names/numbers/operators
    try:
        offs = [0]
        for l in io.StringIO(code).readlines(): offs.append(offs[-1] + len(l))
        return [(t.type, t.string, offs[t.start[0] - 1] + t.start[1], offs[t.end[0] - 1] + t.end[1], t.start[0])
                for t in tokenize.generate_tokens(io.StringIO(code).readline)
                if t.type in (tokenize.OP, tokenize.NUMBER, tokenize.NAME)]
    except (tokenize.TokenError, SyntaxError): return []

def isvar(T, i):
    ty, s = T[i][:2]
    return (ty == tokenize.NAME and not keyword.iskeyword(s) and s not in BUILTINS
            and not (i and T[i - 1][1] in ('.', 'def', 'import', 'from', 'class', 'as'))
            and not (i + 1 < len(T) and T[i + 1][1] == '('))

def candidates(code):  # all single-token edits: (start, end, replacement, features, token_idx)
    T = toks(code); lines = code.split('\n'); n = max(len(T), 1)
    cnt = Counter(t[1] for t in T if t[0] == tokenize.NAME)
    params = {t[1] for t in T if t[0] == tokenize.NAME and lines[t[4] - 1].lstrip().startswith('def ')}
    names = sorted({t[1] for i, t in enumerate(T) if isvar(T, i)})
    out = []
    for i, (ty, s, a, b, ln) in enumerate(T):
        L = lines[ln - 1]; st = L.lstrip()
        ctx = [i / n, ln / len(lines), st.startswith('return'), st.startswith(('for', 'while', 'if', 'elif')), len(L) - len(st), n]
        def add(kind, r, o=-1, rp=-1, c=0, d=0, oc=0, rc=0, op=0, rpm=0):
            out.append((a, b, r, [kind, o, rp, c, d, oc, rc, op, rpm] + ctx, i))
        if ty == tokenize.OP and s in OPS and not (s == '*' and i and T[i - 1][1] in ('(', ',')):
            for r in OPS[s]: add(0, r, SYM.index(s), SYM.index(r))
        elif ty == tokenize.OP and s in CMP:
            for r in CMP[s]: add(1, r, SYM.index(s), SYM.index(r))
        elif ty == tokenize.NUMBER and s.isdigit():
            for d in (1, -1):
                if int(s) + d >= 0: add(2, str(int(s) + d), c=int(s), d=d)
        elif isvar(T, i):
            for r in names:
                if r != s: add(3, r, oc=cnt[s], rc=cnt[r], op=s in params, rpm=r in params)
    return out

# ---------- tasks with injected bugs ----------
def perturb(v, rng):
    if isinstance(v, bool): return v
    if isinstance(v, int): return v + rng.choice([-2, -1, 1, 2, 3])
    if isinstance(v, (list, tuple)):
        l = list(v); k = rng.random()
        if l:
            if k < .33: rng.shuffle(l)
            elif k < .66 and len(l) > 1: l.pop(rng.randrange(len(l)))
            else: l.append(rng.choice(l))
        return type(v)(l)
    if isinstance(v, str) and len(v) > 1:
        i = rng.randrange(len(v)); return v[:i] + v[i + 1:]
    return v

def gen_inputs(ref, assert_str, rng):  # a few extra inputs derived from the visible test call
    try:
        names = re.findall(r'def\s+(\w+)', ref)
        call = next(n for n in ast.walk(ast.parse(assert_str)) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id in names)
        args = [ast.literal_eval(a) for a in call.args]
    except Exception: return None, []
    return call.func.id, [args] + [[perturb(v, rng) for v in args] for _ in range(4)]

def make_task(r, rng):
    tl = list(r.test_list); setup = r.test_setup_code or ''; ref = r.code.replace('\r\n', '\n')
    if len(tl) < 2 or passes(ref, tl, setup) < len(tl): return None
    for _ in range(30):
        code = ref
        for _ in range(rng.choice([1, 2])):  # 1-2 bugs, edit type chosen uniformly
            cs = candidates(code)
            if not cs: break
            k = rng.choice(sorted({c[3][0] for c in cs}))
            a, b, rep, _, _ = rng.choice([c for c in cs if c[3][0] == k]); code = code[:a] + rep + code[b:]
        if code != ref and syntax_ok(code) and passes(code, tl[:1], setup) == 0:  # draft must fail the visible test
            fname, inp = gen_inputs(ref, tl[0], rng)
            return dict(id=int(r.task_id), ref=ref, buggy=code, vis=tl[0], hidden=tl[1:], setup=setup, fname=fname, inputs=inp)
    return None

# ---------- ranker, gate and refinement loop ----------
def fit_ranker(X, y, g):  # small grid, group-CV AUC on train tasks only
    best = None
    for nl in (7, 15):
        for ne in (100, 250):
            m = lgb.LGBMClassifier(num_leaves=nl, n_estimators=ne, learning_rate=0.05, random_state=42, n_jobs=1, verbose=-1)
            auc = cross_val_score(m, X, y, cv=GroupKFold(3), groups=g, scoring='roc_auc').mean()
            if best is None or auc > best[0]: best = (auc, m)
    best[1].fit(X, y); return best

def rank(cands, model, rng):
    if not cands: return []
    if model is None: return rng.permutation(len(cands))  # baseline: random order
    p = np.clip(model.predict_proba(np.array([c[3] for c in cands], float))[:, 1], 1e-6, 1 - 1e-6)
    return np.argsort(-(np.log(p / (1 - p)) + rng.gumbel(size=len(p))))  # sample by model score

def probe(code, t):  # gate stage 3: number of generated inputs that crash
    ns = {}
    if not run(lambda: exec(t['setup'] + '\n' + code, ns))[0]: return len(t['inputs'])
    f = ns.get(t['fname'])
    return sum(not run(lambda a=a: f(*copy.deepcopy(a)))[0] for a in t['inputs'])

def refine(t, model, seed):  # model None = baseline; else ranked edits + staged gate. Same budget/pool/accept rule
    rng = np.random.default_rng([seed, t['id']]); code = t['buggy']
    cur = passes(code, [t['vis']], t['setup']); used = 1; cands = None
    while used < BUDGET and cur < 1:
        if cands is None: cands = candidates(code); order = rank(cands, model, rng); ptr = 0; prev_err = None
        if ptr >= len(order): break
        a, b, rep, _, _ = cands[order[ptr]]; ptr += 1; new = code[:a] + rep + code[b:]
        if model is not None and not syntax_ok(new): continue  # stage 1 (free)
        used += 1; s = passes(new, [t['vis']], t['setup'])  # stage 2
        if s < cur: continue
        if model is not None and s == 1:  # stage 3: not more crashes than previous version
            if used + 1 + (prev_err is None) > BUDGET: break
            if prev_err is None: prev_err = probe(code, t); used += 1
            used += 1
            if probe(new, t) > prev_err: continue
        code, cur, cands = new, s, None
    return code, used

def evaluate(tasks, model, seed):
    H = V = E = 0
    for t in tasks:
        code, used = refine(t, model, seed)
        H += passes(code, t['hidden'], t['setup']) == len(t['hidden'])
        V += passes(code, [t['vis']], t['setup']) == 1; E += used
    n = len(tasks); return H / n, V / n, E / n

def main():
    random.seed(42); np.random.seed(42)
    df = pd.read_parquet(io.BytesIO(download(URL)))
    rng = random.Random(42)
    tasks = [t for t in (make_task(r, rng) for r in df.itertuples()) if t]
    print(f"{len(tasks)} usable buggy tasks of {len(df)}")
    if len(tasks) < 150: sys.exit("too few tasks")
    train, test = train_test_split(tasks, test_size=0.3, random_state=42)
    X, y, g = [], [], []  # label: edit restores the original (reference) token
    for t in train:
        B = [x[1] for x in toks(t['buggy'])]; R = [x[1] for x in toks(t['ref'])]
        if len(B) != len(R): continue
        for a, b, r, f, i in candidates(t['buggy']):
            X.append(f); y.append(int(B[i] != R[i] and r == R[i])); g.append(t['id'])
    X, y = np.array(X, float), np.array(y)
    print(f"ranker data: {len(y)} edits, {y.sum()} positive")
    auc, model = fit_ranker(X, y, np.array(g))
    print(f"ranker CV AUC {auc:.3f}")
    res = {"baseline": [], "improved": []}
    for sd in SEEDS:  # test set used once, for both methods
        res["baseline"].append(evaluate(test, None, sd)); res["improved"].append(evaluate(test, model, sd))
        print(f"seed {sd}: baseline {res['baseline'][-1]} improved {res['improved'][-1]}")
    m = {}
    for k, name in enumerate(["score", "visible_pass", "mean_execs"]):
        for who in res:
            v = np.array([r[k] for r in res[who]]); m[f"{who}_{name}"] = float(v.mean()); m[f"{who}_{name}_std"] = float(v.std())
    m["improvement"] = m["improved_score"] - m["baseline_score"]; m["ranker_cv_auc"] = float(auc)
    if not all(np.isfinite(list(m.values()))): sys.exit("non-finite metrics")
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5)); xs = np.arange(2)
    for j, who in enumerate(res):
        ax.bar(xs + 0.35 * j, [m[f"{who}_score"], m[f"{who}_visible_pass"]], 0.35,
               yerr=[m[f"{who}_score_std"], m[f"{who}_visible_pass_std"]], label=who)
    ax.set_xticks(xs + 0.175); ax.set_xticklabels(["hidden pass", "visible pass"]); ax.legend(); ax.set_ylabel("rate (held-out tasks)")
    fig.tight_layout(); fig.savefig("figures/comparison.png", dpi=120)
    out = {
        "question": "In a simple non-LLM stand-in for a generate-verify-refine loop, can a LightGBM edit-ranker plus a staged verification gate repair more buggy programs on hidden tests than a standard random-edit refinement loop at the same evaluation budget?",
        "method": "MBPP reference solutions get 1-2 injected token bugs; first assert is visible, the rest hidden. Both loops use the same single-token edit pool, accept rule (visible pass count must not drop) and 30-execution budget. Improved ranks edits with LightGBM (trained on train-task edits, grid chosen by group-CV) and adds a syntax -> visible -> generated-input crash-comparison gate.",
        "dataset": {"name": "MBPP (full config, test split) with injected bugs", "source_url": URL, "n_rows": len(tasks),
                    "n_train": len(train), "n_test": len(test), "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "hidden-test pass rate (fraction of held-out tasks passing all hidden asserts)",
        "higher_is_better": True,
        "baseline": "Random single-token edit refinement loop, accepts an edit if visible-test pass count does not drop, 30 executions per task.",
        "improved": "Same loop, pool and budget, but edits sampled by LightGBM score and candidates pass a staged gate (syntax, visible test, crash comparison on generated inputs vs previous version).",
        "metrics": m,
        "notes": "Synthetic bugs on MBPP, not LiveCodeBench and no LLM; weak proxy for CodeGrad. Test split is 30% (not 50%) of usable tasks. Only drafts failing the visible test are kept. Gate executions count toward the 30-execution budget. Ranker label = edit restores the reference token (train tasks only). Loops stop when the visible test passes, so overfitting to the single visible test is possible. Std is over loop seeds 42-44 on one fixed task split."}
    json.dump(out, open("results.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()
