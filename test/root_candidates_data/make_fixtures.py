"""Deterministic simulated fixtures for root selection by sequence likelihood
under a strict clock. Run with the repo venv python from any directory."""
import io, json, os, random, shutil, subprocess, tempfile
from itertools import combinations
import numpy as np
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from treetime.gtr import GTR
from treetime.seqgen import SeqGen

OUT = os.path.dirname(os.path.abspath(__file__))
IQTREE = "/opt/conda/envs/bioinfo/bin/iqtree"
IQ_ARGS = ["--pathogen-force", "-m", "JC", "-redo", "-nt", "1"]


def coalescent(rng, n, ne, leaf_span):
    """Backward serial-sample coalescent. Returns (root Clade, node times)."""
    leaf_dates = sorted(rng.uniform(2010.0, 2010.0 + leaf_span) for _ in range(n - 1))
    leaf_dates = [2010.0] + leaf_dates
    names = [f"L{i+1:02d}" for i in range(n)]
    order = list(range(n)); rng.shuffle(order)
    leaves = [(names[order[i]], leaf_dates[i]) for i in range(n)]
    times = {}
    pending = sorted(leaves, key=lambda x: -x[1])  # latest first
    active, t, k = [], None, 0
    while pending or len(active) > 1:
        if not active:
            nm, d = pending.pop(0); c = Clade(name=nm); times[id(c)] = d
            active.append(c); t = d; continue
        m = len(active)
        wait = rng.expovariate(m * (m - 1) / 2 / ne) if m > 1 else float("inf")
        nxt = pending[0][1] if pending else -float("inf")
        if t - wait > nxt:
            t -= wait
            a, b = rng.sample(active, 2)
            active = [x for x in active if x is not a and x is not b]
            c = Clade(clades=[a, b]); times[id(c)] = t
            active.append(c)
        else:
            t = nxt; nm, d = pending.pop(0); c = Clade(name=nm); times[id(c)] = d
            active.append(c)
    return active[0], times


def set_lengths(root, times, rate):
    for cl in root.find_clades(order="level"):
        for ch in cl.clades:
            ch.branch_length = rate * (times[id(ch)] - times[id(cl)])


def leafset(c):
    return frozenset(t.name for t in c.get_terminals())


def splits(tree):
    allv = leafset(tree.root)
    out = set()
    for c in tree.find_clades(terminal=False):
        if c is tree.root:
            continue
        s = leafset(c)
        if 1 < len(s) < len(allv) - 1:
            out.add(min(s, allv - s, key=lambda x: (len(x), sorted(x))))
    return out


def fit_tree(aln_path, L, thr, tmp):
    prefix = os.path.join(tmp, "x")
    cmd = [IQTREE, "--pathogen-force", "-s", aln_path, "-m", "JC", "--prefix", prefix, "-redo", "-nt", "1"]
    subprocess.run(cmd, check=True, capture_output=True, cwd=tmp)
    tr = Phylo.read(prefix + ".treefile", "newick")
    ncol = 0
    for c in list(tr.find_clades(terminal=False, order="postorder")):
        if c is tr.root:
            continue
        if c.branch_length is None or c.branch_length < thr:
            tr.collapse(c); ncol += 1
    for c in tr.find_clades():
        if c.branch_length is None or c.branch_length < 0:
            c.branch_length = 0.0
    if tr.root.branch_length is not None:
        tr.root.branch_length = None
    return tr, ncol, cmd


def degrees(tr):
    r = len(tr.root.clades)
    ds = []
    for c in tr.find_clades(terminal=False):
        ds.append(len(c.clades) if c is tr.root else len(c.clades) + 1)
    if r == 2:  # bifurcating root is a degree-2 artefact of rooting
        ds.remove(2)
    return sorted(ds)


def build(name, n, L, ne, span, target_rtt, seed_base, accept):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    for attempt in range(500):
        seed = seed_base + attempt
        rng = random.Random(seed)
        root, times = coalescent(rng, n, ne, span)
        oldest = min(times[id(t)] for t in root.get_terminals())
        gap = oldest - times[id(root)]
        if not (10 <= gap <= 15) or len(root.clades) != 2:
            continue
        if min(len(leafset(c)) for c in root.clades) < 2:
            continue
        tips = list(root.get_terminals())
        mean_rtt = np.mean([times[id(t)] - times[id(root)] for t in tips])
        rate = round(target_rtt / mean_rtt, 5)
        set_lengths(root, times, rate)
        tree = Tree(root=root, rooted=True)
        sg = SeqGen(L, tree=tree, gtr=GTR.standard("JC69", alphabet="nuc_nogap"), rng_seed=seed)
        sg.evolve()
        aln = sg.get_aln()
        aln_path = os.path.join(d, "aln.fasta")
        from Bio import SeqIO
        SeqIO.write(sorted(aln, key=lambda r: r.id), aln_path, "fasta")
        thr = 0.5 / L
        with tempfile.TemporaryDirectory() as tmp:
            fit, ncol, cmd = fit_tree(aln_path, L, thr, tmp)
        info = dict(true=tree, fit=fit, times=times, aln=aln)
        if not accept(info):
            continue
        break
    else:
        raise RuntimeError("no acceptable seed for " + name)
    Phylo.write(tree, os.path.join(d, "true_tree.nwk"), "newick")
    Phylo.write(fit, os.path.join(d, "fitted_tree.nwk"), "newick")
    with open(os.path.join(d, "dates.tsv"), "w") as f:
        f.write("name\tdate\n")
        for t in sorted(tips, key=lambda t: t.name):
            f.write(f"{t.name}\t{times[id(t)]:.4f}\n")
    allv = leafset(root)
    small = min((leafset(c) for c in root.clades), key=lambda s: (len(s), sorted(s)))
    meta = dict(seed=seed, clock_rate=rate, L=L, gtr="JC69 (GTR.standard('JC69', alphabet='nuc_nogap'))",
                iqtree_cmd=" ".join(["iqtree"] + cmd[1:]).replace(aln_path, "aln.fasta").replace(
                    os.path.dirname(cmd[cmd.index("--prefix") + 1]), "<tmpdir>"),
                collapse_threshold=thr, n_internal_collapsed=ncol,
                true_root_split=sorted(small), true_root_date=round(times[id(root)], 4))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    ts, fs = splits(tree), splits(fit)
    print(f"== {name}: seed={seed} rate={rate} leaves={len(tips)}")
    print(" fitted internal-node degrees (unrooted):", degrees(fit), "collapsed:", ncol)
    print(" fitted splits subset of true:", fs <= ts, "| equal:", fs == ts,
          f"| true {len(ts)} fitted {len(fs)}")
    print(" true root split (smaller side):", sorted(small), "root date", meta["true_root_date"])


def acc_binary(i):
    return True


def acc_poly(i):
    if max(degrees(i["fit"])) < 4:
        return False
    seqs = {}
    for r in i["aln"]:
        seqs.setdefault(str(r.seq), []).append(r.id)
    dates = {t.name: i["times"][id(t)] for t in i["true"].get_terminals()}
    return any(len(v) >= 2 and len({round(dates[x], 3) for x in v}) > 1 for v in seqs.values())


def acc_mixed(i):
    ds = degrees(i["fit"])
    consts = sum(len({c for c in col}) == 1 for col in zip(*[str(r.seq) for r in i["aln"]]))
    return sum(d >= 4 for d in ds) >= 3 and consts >= 100


if __name__ == "__main__":
    build("binary20", 20, 600, 6.0, 10.0, 0.03, 1000, acc_binary)
    build("poly20", 20, 400, 6.0, 10.0, 0.008, 2000, acc_poly)
    build("mixed60", 60, 300, 6.0, 10.0, 0.01, 3000, acc_mixed)
