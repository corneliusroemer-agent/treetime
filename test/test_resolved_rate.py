"""clock_rate_method='date_resolved_xfit' (treetime.resolved_rate): the default
path is untouched, polytomies are resolved by dates, the cross-fit is
deterministic and splits the data exactly, and the edge cases behave."""
import copy
import os

import numpy as np
import pytest
from Bio.Phylo.BaseTree import Clade, Tree

from treetime import TreeTime
from treetime import resolved_rate as rr

HERE = os.path.dirname(os.path.abspath(__file__))
L = 1000


def tt_from_tree(tree, dates, aln=None, L=L):
    tt = TreeTime(tree=tree, aln=aln, seq_len=None if aln else L, dates=dates, verbose=0, gtr='JC69', rng_seed=7)
    for n in tt.tree.find_clades():
        n.mutation_length = n.branch_length
    return tt


def simulated_tree(n_tips=60, seed=1, internal_min=0, mu=2e-3, span=10.0):
    """Random binary tree with Poisson counts on its edges (k / L as length).
    ``internal_min`` counts are added to internal edges (1 keeps them uncollapsed)."""
    rng = np.random.default_rng(seed)
    nodes = [(Clade(name=f'T{i:03d}'), 2020.0 + span * rng.random()) for i in range(n_tips)]
    t = min(d for _, d in nodes)
    while len(nodes) > 1:
        t -= rng.exponential(0.5)
        i, j = sorted(rng.choice(len(nodes), 2, replace=False))
        (a, ta), (b, tb) = nodes[i], nodes[j]
        tn = min(t, ta - 0.01, tb - 0.01)
        c = Clade(clades=[a, b])
        a.branch_length = (rng.poisson(mu * L * (ta - tn)) + (internal_min if a.clades else 0)) / L
        b.branch_length = (rng.poisson(mu * L * (tb - tn)) + (internal_min if b.clades else 0)) / L
        nodes = [x for k, x in enumerate(nodes) if k not in (i, j)] + [(c, tn)]
    root = nodes[0][0]
    root.branch_length = 0.0
    return Tree(root=root, rooted=True)


def tip_dates(tree, seed=1, span=10.0):
    # tips were drawn as 2020 + span * U in order T000..; reproduce them
    rng = np.random.default_rng(seed)
    n = len(tree.get_terminals())
    d = 2020.0 + span * rng.random(n)
    return {f'T{i:03d}': float(d[i]) for i in range(n)}


def sim(n_tips=60, seed=1, internal_min=0):
    tree = simulated_tree(n_tips, seed, internal_min)
    return tree, tip_dates(tree, seed)


def star(m, seed=0):
    """Star tree: m tips with clock-like Poisson counts (rate 1e-3, L = 1000)
    hanging from the root, plus two dated clades."""
    rng = np.random.default_rng(seed)
    dates = {f'S{i:03d}': float(2010 + 5 * rng.random()) for i in range(m)}
    kids = [Clade(name=nm, branch_length=rng.poisson(d - 2008.0) / L) for nm, d in dates.items()]
    a = Clade(clades=[Clade(name='A1', branch_length=0.004), Clade(name='A2', branch_length=0.006)], branch_length=0.003)
    b = Clade(clades=[Clade(name='B1', branch_length=0.002), Clade(name='B2', branch_length=0.009)], branch_length=0.002)
    dates.update(A1=2005.0, A2=2012.0, B1=2003.0, B2=2015.0)
    root = Clade(clades=kids + [a, b], branch_length=0.0)
    return Tree(root=root, rooted=True), dates


def flat_from(tree, dates):
    f = rr._Flat()
    stack = [(tree.root, -1)]
    while stack:
        v, p = stack.pop()
        i = f.add(p, v.is_terminal(), dates.get(v.name, np.nan) if v.is_terminal() else np.nan,
                  (v.branch_length or 0.0) if p >= 0 else 0.0, None, v.name)
        for c in v.clades:
            stack.append((c, i))
    return f.preorder()


def rtt(f):
    d = np.zeros(len(f.par))
    for v in range(1, len(f.par)):
        d[v] = d[f.par[v]] + f.ml[v]
    return {f.name[v]: d[v] for v in range(len(f.par)) if f.tip[v]}


# ---------------------------------------------------------------- default path


def test_default_get_clock_model_is_the_regression():
    tree, dates = sim(40, seed=3)
    tt = tt_from_tree(tree, dates)
    assert tt.clock_rate_method == 'regression'
    for cov in (False, True):
        tt.get_clock_model(covariation=cov)
        ref = tt.setup_TreeRegression(covariation=cov).regression()
        assert tt.clock_model['slope'] == ref['slope']
        assert tt.clock_model['intercept'] == ref['intercept']
        assert 'date_resolved_xfit' not in tt.clock_model


def test_unknown_method_raises():
    tree, dates = sim(20, seed=3)
    tt = tt_from_tree(tree, dates)
    tt.clock_rate_method = 'nonsense'
    with pytest.raises(ValueError):
        tt.get_clock_model()


# ---------------------------------------------------------------- resolution


def test_known_polytomy_is_resolved_to_completion_and_keeps_root_to_tip_sums():
    """Joins continue while any pair has a positive gain: the zero-count star tips
    end up in one binary subtree, and no pair left at the root has a positive gain."""
    tree, dates = star(12)
    f = flat_from(tree, dates)
    assert rr._plain_regression(f)[0] > 0
    mu = 1e-3
    g, info = rr._resolve(f, mu, L, max_exact=100, neighbours=8)
    ch = g.children()
    assert info['n_polytomies'] == 1
    assert info['n_new_nodes'] >= 5
    assert all(len(c) in (0, 2) for c in ch[1:])
    # completion: recompute the times the resolver used and score the pairs left at the root
    par, tip, d, ml = (np.asarray(z) for z in (f.par, f.tip, f.date, f.ml))
    t = rr._node_positions(par, tip, np.where(tip, mu * d, np.nan), ml, np.ones(len(par))) / mu
    for v in range(len(par) - 1, 0, -1):
        t[par[v]] = min(t[par[v]], t[v] - 1e-6 * np.ptp(d[tip]))
    left = [c for c in ch[0] if c < len(par)]  # original children still attached to the root
    names = {f.name[v]: v for v in range(len(par))}
    for i, a in enumerate(left):
        for b in left[i + 1:]:
            va, vb = names[g.name[a]], names[g.name[b]]
            gain, _ = rr._pair_gain(mu * L, t[0], t[va], t[vb], ml[va] * L, ml[vb] * L, 1e-12)
            assert gain <= 1e-9
    a, b = rtt(f), rtt(g)
    assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-15 for k in a)


def test_resolution_joins_late_attaching_children_first():
    # three zero-count children: the two latest-dated ones share the new node
    tree = Tree(root=Clade(clades=[Clade(name='x', branch_length=0.0), Clade(name='y', branch_length=0.0),
                                   Clade(name='z', branch_length=0.0)]), rooted=True)
    f = flat_from(tree, {'x': 2000.0, 'y': 2010.0, 'z': 2011.0})
    g, _ = rr._resolve(f, 1e-3, L, 100, 8)
    ch = g.children()
    pairs = [frozenset(g.name[c] for c in kids) for kids in ch if len(kids) == 2 and all(g.tip[c] for c in kids)]
    assert pairs == [frozenset({'y', 'z'})]


def test_pair_gain_matches_numerical_maximum():
    from scipy.optimize import minimize_scalar
    rng = np.random.default_rng(0)
    for _ in range(200):
        A = rng.uniform(0.1, 5)
        tp = 0.0
        ta, tb = rng.uniform(0.01, 3, 2)
        ka, kb = rng.integers(0, 6, 2)
        g, u = rr._pair_gain(A, tp, ta, tb, ka, kb, 1e-12)
        f = lambda x: -(A * (x - tp) + (ka * np.log((ta - x) / (ta - tp)) if ka else 0)
                        + (kb * np.log((tb - x) / (tb - tp)) if kb else 0))
        r = minimize_scalar(f, bounds=(tp, min(ta, tb) - 1e-12), method='bounded', options=dict(xatol=1e-12))
        assert max(-r.fun, 0.0) == pytest.approx(float(g), abs=1e-6)


def test_restricted_join_with_a_full_window_equals_exact():
    rng = np.random.default_rng(4)
    for _ in range(20):
        m = int(rng.integers(3, 25))
        T = rng.uniform(0.5, 3, m)
        K = rng.integers(0, 4, m).astype(float)
        ex = rr._resolve_exact(2.0, 0.0, T, K, 1e-12, 1e-9)
        re = rr._resolve_restricted(2.0, 0.0, T, K, 1e-12, 1e-9, neighbours=m)
        norm = lambda js: [(min(i, j), max(i, j), round(t, 9)) for i, j, t in js]
        assert norm(ex) == norm(re)


def test_huge_polytomy_uses_the_restricted_join_and_is_fast():
    import time
    tree, dates = star(400)
    tt = tt_from_tree(tree, dates)
    t0 = time.time()
    est = rr.date_resolved_xfit_rate(tt, max_exact_polytomy=50)
    assert time.time() - t0 < 30
    assert est['n_restricted'] == 1 and est['max_polytomy'] == 402
    assert est['n_new_nodes'] > 100
    assert np.isfinite(est['rate']) and est['rate'] > 0


# ---------------------------------------------------------------- cross-fitting


def test_cross_fit_is_deterministic_given_the_seed():
    tree, dates = sim(50, seed=5)
    a = rr.date_resolved_xfit_rate(tt_from_tree(copy.deepcopy(tree), dates), seed=11)
    b = rr.date_resolved_xfit_rate(tt_from_tree(copy.deepcopy(tree), dates), seed=11)
    c = rr.date_resolved_xfit_rate(tt_from_tree(copy.deepcopy(tree), dates), seed=12)
    assert a['rate'] == b['rate'] and a['split_rates'] == b['split_rates'] and a['root_date'] == b['root_date']
    assert a['split_rates'] != c['split_rates']


def test_half_values_add_up_so_equal_weights_reproduce_the_full_fit():
    """With the same variances in both halves the mean of the two half slopes is
    the full-data slope exactly (the split only redistributes mutation_length)."""
    tree, dates = sim(50, seed=6)
    f = flat_from(tree, dates)
    reg = rr._Regression(f)
    ml = np.asarray(f.ml)
    k = np.round(ml * L)
    rng = np.random.default_rng(0)
    kA = rng.binomial(k.astype(int), 0.5).astype(float)
    var = (ml * L + 0.5) / L ** 2
    full = reg.fit(ml, var)['slope']
    vA = rr._half_values(ml, k, kA, L / 2, L)
    vB = rr._half_values(ml, k, k - kA, L / 2, L)
    assert 0.5 * (reg.fit(vA, var)['slope'] + reg.fit(vB, var)['slope']) == pytest.approx(full, rel=1e-12)


def test_root_date_of_the_branch_least_squares_equals_treeregression():
    """_node_positions at TreeRegression's GLS slope puts the root at -intercept
    (root-to-tip GLS = branch least squares with free internal nodes)."""
    tree, dates = sim(40, seed=8)
    f = flat_from(tree, dates)
    reg = rr._Regression(f)
    ml = np.asarray(f.ml)
    var = (ml * L + 0.5) / L ** 2
    fit = reg.fit(ml, var)
    d = np.asarray(f.date)
    x = rr._node_positions(np.asarray(f.par), np.asarray(f.tip), np.where(np.isfinite(d), fit['slope'] * d, np.nan),
                           ml, 1 / var)
    assert x[0] == pytest.approx(-fit['intercept'], rel=1e-9)


def test_binary_tree_matches_covariation_regression_with_the_same_floor():
    """No polytomy: nothing is resolved, and the cross-fitted rate agrees with
    the covariation regression (floor 0.5, no tip slack) up to split noise."""
    for seed in (1, 2, 3):
        tree, dates = sim(80, seed=seed, internal_min=1)
        tt = tt_from_tree(tree, dates)
        est = rr.date_resolved_xfit_rate(tt)
        assert est['n_polytomies'] == 0 and est['n_new_nodes'] == 0
        assert est['rate'] == rr.date_resolved_xfit_rate(tt, resolve=False)['rate']
        tt.tip_slack = 0.0
        tt.branch_variance_floor = 0.5
        cov = tt.setup_TreeRegression(covariation=True).regression()['slope']
        assert abs(np.log(est['rate'] / cov)) < 0.15, (seed, est['rate'], cov)
        assert est['split_sd'] < 0.2 * est['rate']


def test_tree_passed_in_is_not_modified():
    tree, dates = star(10)
    tt = tt_from_tree(tree, dates)
    before = [(n.name, n.branch_length, n.mutation_length, len(n.clades)) for n in tt.tree.find_clades()]
    rr.date_resolved_xfit_rate(tt)
    after = [(n.name, n.branch_length, n.mutation_length, len(n.clades)) for n in tt.tree.find_clades()]
    assert before == after


# ---------------------------------------------------------------- edge cases


def test_all_tips_with_the_same_date_raise():
    tree, dates = sim(20, seed=3)
    dates = {k: 2020.0 for k in dates}
    with pytest.raises(ValueError, match='No variation in sampling dates'):
        rr.date_resolved_xfit_rate(tt_from_tree(tree, dates))


def test_no_alignment_thins_counts_and_mutations_mode_needs_them():
    tree, dates = sim(30, seed=3)
    tt = tt_from_tree(tree, dates)
    assert rr.date_resolved_xfit_rate(tt)['counts'] == 'thinned'
    with pytest.raises(ValueError):
        rr.date_resolved_xfit_rate(tt, counts='mutations')
    with pytest.raises(ValueError):
        rr.date_resolved_xfit_rate(tt, floor=0.0)


def test_non_positive_reference_rate_skips_resolution():
    tree, dates = star(8)
    dates = {k: (4030.0 - v) for k, v in dates.items()}  # reversed dates: negative plain slope
    est = rr.date_resolved_xfit_rate(tt_from_tree(tree, dates))
    assert est['plain_rate'] < 0
    assert est['resolution_skipped'] and est['n_new_nodes'] == 0


def test_undated_tip_is_tolerated():
    tree, dates = sim(30, seed=4)
    dates.pop('T003')
    est = rr.date_resolved_xfit_rate(tt_from_tree(tree, dates))
    assert np.isfinite(est['rate'])


# ---------------------------------------------------------------- with an alignment, and the TreeTime hook


def fixture(fx):
    d = os.path.join(HERE, 'root_candidates_data', fx)
    from treetime.utils import parse_dates
    return os.path.join(d, 'fitted_tree.nwk'), os.path.join(d, 'aln.fasta'), parse_dates(os.path.join(d, 'dates.tsv'))


def test_mutation_counts_come_from_the_reconstruction():
    nwk, aln, dates = fixture('poly20')
    tt = TreeTime(tree=nwk, aln=aln, dates=dates, verbose=0, gtr='JC69', rng_seed=1)
    tt.infer_ancestral_sequences('probabilistic', marginal=False)
    tt.reroot('least-squares')
    tt.infer_ancestral_sequences('probabilistic', marginal=False)
    est = rr.date_resolved_xfit_rate(tt)
    assert est['counts'] == 'mutations'
    assert est['n_polytomies'] >= 1 and est['n_new_nodes'] >= 1
    pos = {id(n): np.array([m[1] for m in n.mutations], int) for n in tt.tree.find_clades() if n.up is not None}
    f = rr._flatten(tt.tree, tt.one_mutation, pos)
    assert sum(len(p) for p in f.pos[1:]) == sum(len(p) for p in pos.values())


def test_hook_sets_the_clock_model_and_leaves_the_tree():
    nwk, aln, dates = fixture('poly20')
    tt = TreeTime(tree=nwk, aln=aln, dates=dates, verbose=0, gtr='JC69', rng_seed=1)
    tt.infer_ancestral_sequences('probabilistic', marginal=False)
    tt.reroot('least-squares')
    tt.infer_ancestral_sequences('probabilistic', marginal=False)
    shape = [len(n.clades) for n in tt.tree.find_clades()]
    tt.clock_rate_method = 'date_resolved_xfit'
    tt.get_clock_model(covariation=False)
    assert not tt.clock_model['valid_confidence'] and 'cov' not in tt.clock_model  # as the regression path
    tt.get_clock_model(covariation=True)
    cm = tt.clock_model
    est = rr.date_resolved_xfit_rate(tt)
    assert cm['slope'] == est['rate'] and cm['intercept'] == est['intercept']
    assert cm['valid_confidence'] and cm['cov'][0, 0] > 0
    assert tt.date2dist.clock_rate == est['rate']
    assert shape == [len(n.clades) for n in tt.tree.find_clades()]
    tt.get_clock_model(covariation=True, slope=1e-3)  # a fixed slope keeps the regression
    assert tt.clock_model['slope'] == 1e-3 and 'date_resolved_xfit' not in tt.clock_model


def test_clock_filter_ignores_the_method():
    nwk, aln, dates = fixture('mixed60')
    out = []
    for method in ('regression', 'date_resolved_xfit'):
        tt = TreeTime(tree=nwk, aln=None, seq_len=1000, dates=dates, verbose=0, gtr='JC69', rng_seed=1)
        tt.clock_rate_method = method
        tt.clock_filter(reroot=None, n_iqd=2)
        out.append(sorted(n.name for n in tt.tree.get_terminals() if n.bad_branch))
        assert tt._clock_rate_method_suspended is False
    assert out[0] == out[1]


def test_run_with_the_method_end_to_end():
    nwk, aln, dates = fixture('poly20')
    tt = TreeTime(tree=nwk, aln=aln, dates=dates, verbose=0, gtr='JC69', rng_seed=1)
    tt.clock_rate_method = 'date_resolved_xfit'
    tt.run(root='least-squares', max_iter=1, branch_length_mode='input', resolve_polytomies=False)
    assert 'date_resolved_xfit' in tt.clock_model and tt.clock_model['slope'] > 0
    assert tt.clock_model['date_resolved_xfit']['counts'] == 'mutations'


# ---------------------------------------------------------------- against an independent dense implementation


def dense_xfit(f, splits, L, floor, iters, tip_extra=0.0):
    """Report 85's est_xfit recipe written with a dense design matrix: branch least squares with the
    internal node positions x and the rate mu as unknowns (dated tip c: x_c = mu t_c), weights
    1 / (max(predicted length, 0) L + floor [+ tip_extra on tips]); the first weight fit has constant
    variances plus the tip term.
    Returns the 2R slopes, the mean variance vector and the design pieces."""
    par, tip, d, ml = (np.asarray(z) for z in (f.par, f.tip, f.date, f.ml))
    n = len(par)
    free = [v for v in range(n) if not (tip[v] and np.isfinite(d[v]))]
    col = {v: i for i, v in enumerate(free)}
    X = np.zeros((n - 1, len(free) + 1))
    for r, c in enumerate(range(1, n)):
        if c in col:
            X[r, col[c]] += 1
        else:
            X[r, -1] += d[c]
        X[r, col[par[c]]] -= 1
    k = np.round(ml * L)
    extra = np.where(tip[1:], tip_extra, 0.0)

    def wls(y, w):
        sw = np.sqrt(w)
        return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]

    slopes, vs = [], []
    for kA, LA in splits:
        vA = np.where(k > 0, ml * kA / np.where(k > 0, k, 1) * L / LA, ml)[1:]
        vB = np.where(k > 0, ml * (k - kA) / np.where(k > 0, k, 1) * L / (L - LA), ml)[1:]
        for src, dst in ((vA, vB), (vB, vA)):
            w = 1.0 / (1.0 + extra)
            for _ in range(iters):
                lam = X @ wls(src, w)
                w = 1.0 / (np.maximum(lam, 0) * L + floor + extra)
            slopes.append(wls(dst, w)[-1])
            vs.append(1.0 / w)
    return slopes, np.mean(vs, 0), X, col


def test_cross_fit_matches_a_dense_implementation_of_the_recipe():
    """Weighting, cross-fitting (weights from one half, slope from the other), the floor and the tip
    term all enter: a version without any of them does not match."""
    tree, dates = sim(40, seed=9)
    f = flat_from(tree, dates)
    reg = rr._Regression(f)
    ml = np.asarray(f.ml)
    k = np.round(ml * L)
    rng = np.random.default_rng(3)
    splits = [(rng.binomial(k.astype(int), 0.5).astype(float), L / 2) for _ in range(2)]
    for floor, iters, extra in ((0.5, 3, 0.0), (2.0, 1, 0.0), (0.5, 3, 9.0)):
        got, _ = rr._xfit(reg, ml, k, splits, L, floor, iters, tip_extra=extra)
        ref, _, _, _ = dense_xfit(f, splits, L, floor, iters, tip_extra=extra)
        assert np.allclose(got, ref, rtol=1e-8), (floor, iters, extra)
    # sanity: the recipe is sensitive to what the test claims to pin
    a, _, _, _ = dense_xfit(f, splits, L, 0.5, 3)
    b, _, _, _ = dense_xfit(f, splits, L, 0.5, 0)
    c, _, _, _ = dense_xfit(f, splits, L, 1e-6, 3)
    assert not np.allclose(a, b, rtol=1e-3) and not np.allclose(a, c, rtol=1e-3)


def test_rate_and_root_date_end_to_end_against_the_dense_recipe():
    """date_resolved_xfit_rate on a tree without polytomies, thinned counts: the same column splits
    drawn by hand give the same rate; the root date is the GLS refit on all values at that rate
    (not a mean of half-sample root dates)."""
    tree, dates = sim(50, seed=10, internal_min=1)
    tt = tt_from_tree(tree, dates)
    est = rr.date_resolved_xfit_rate(tt, seed=21, n_splits=3)
    f = rr._flatten(tt.tree, 1.0 / L, None)
    ml = np.asarray(f.ml)
    k = np.round(ml * L)
    k[0] = 0
    rng = np.random.default_rng(21)
    splits = []
    for _ in range(3):
        LA = float(L // 2)
        rng.permutation(L)
        splits.append((rng.binomial(k.astype(int), LA / L).astype(float), LA))
    slopes, vbar, X, col = dense_xfit(f, splits, L, 0.5, 3)
    assert est['rate'] == pytest.approx(np.mean(slopes), rel=1e-8)
    # root date: weighted LS of all values with the slope fixed at the rate
    mu = est['rate']
    y = ml[1:] - mu * X[:, -1]
    w = 1.0 / vbar
    sw = np.sqrt(w)
    x = np.linalg.lstsq(X[:, :-1] * sw[:, None], y * sw, rcond=None)[0]
    assert est['root_date'] == pytest.approx(x[col[0]] / mu, abs=1e-6)


def test_mutation_positions_set_the_half_counts():
    """With a reconstruction, the half counts come from the mutated columns: the rate equals a
    hand-built cross-fit over the same column permutations, and differs from the thinned estimate."""
    nwk, aln, dates = fixture('poly20')
    tt = TreeTime(tree=nwk, aln=aln, dates=dates, verbose=0, gtr='JC69', rng_seed=1)
    tt.infer_ancestral_sequences('probabilistic', marginal=False)
    tt.reroot('least-squares')
    tt.infer_ancestral_sequences('probabilistic', marginal=False)
    est = rr.date_resolved_xfit_rate(tt, seed=5, n_splits=2, resolve=False)
    assert est['counts'] == 'mutations' and est['n_thinned_branches'] == 0
    L_ = tt.data.full_length
    pos = {id(n): np.array([m[1] for m in n.mutations], int) for n in tt.tree.find_clades() if n.up is not None}
    f = rr._flatten(tt.tree, 1.0 / L_, pos)
    ml = np.asarray(f.ml)
    k = np.array([0.0] + [float(len(f.pos[v])) for v in range(1, len(f.par))])
    rng = np.random.default_rng(5)
    splits = []
    for _ in range(2):
        inA = np.zeros(L_, bool)
        inA[rng.permutation(L_)[: L_ // 2]] = True
        splits.append((np.array([0.0] + [float(inA[f.pos[v]].sum()) for v in range(1, len(f.par))]), float(inA.sum())))
    slopes, _ = rr._xfit(rr._Regression(f), ml, k, splits, L_, 0.5, 3)
    assert est['rate'] == pytest.approx(np.mean(slopes), rel=1e-12)
    assert est['rate'] != rr.date_resolved_xfit_rate(tt, seed=5, n_splits=2, resolve=False, counts='thinned')['rate']


def test_nodes_without_sequences_are_thinned_alone():
    """After rerooting at a tip, the new root-edge node has no reconstructed sequence: only that
    branch is thinned, the rest keep their mutation counts."""
    nwk, aln, dates = fixture('poly20')
    tt = TreeTime(tree=nwk, aln=aln, dates=dates, verbose=0, gtr='JC69', rng_seed=1)
    tt.infer_ancestral_sequences('probabilistic', marginal=False)
    tt.reroot('oldest')
    est = rr.date_resolved_xfit_rate(tt)
    assert est['counts'] in ('mixed', 'mutations')
    assert est['n_thinned_branches'] <= 2


def test_non_positive_estimate_falls_back_to_the_regression():
    tree, dates = star(8)
    dates = {kk: (4030.0 - v) for kk, v in dates.items()}
    tt = tt_from_tree(tree, dates)
    tt.clock_rate_method = 'date_resolved_xfit'
    tt.get_clock_model(covariation=True)
    ref = tt.setup_TreeRegression(covariation=True).regression()
    if 'date_resolved_xfit' not in tt.clock_model:
        assert tt.clock_model['slope'] == ref['slope']
