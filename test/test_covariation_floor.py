"""Covariation regression with ``branch_variance_floor``: the O(n) tree
recursion must equal a dense GLS fit, at the current root and at every
internal node as root, and ``None`` must leave the old variance model intact."""
from io import StringIO

import numpy as np
import pytest
from Bio import Phylo

NWK = '((A:0.002,B:0.0)n1:0.0,((C:0.003,D:0.001)n3:0.002,(E:0.0,F:0.004)n4:0.001)n2:0.003,(G:0.005,H:0.002)n5:0.001)root;'
DATES = {'A': 2001.0, 'B': 2002.5, 'C': 2003.0, 'D': 2001.5, 'E': 2000.2, 'F': 2004.0, 'G': 2003.3, 'H': 2000.9}
L = 1000


def make_tt():
    from treetime import TreeTime
    tt = TreeTime(tree=Phylo.read(StringIO(NWK), 'newick'), aln=None, seq_len=L, dates=DATES, verbose=0,
                  gtr='JC69')
    for n in tt.tree.find_clades():
        n.mutation_length = n.branch_length
    return tt


def old_branch_variance(tt):
    """The variance model before ``branch_variance_floor`` existed."""
    om = tt.one_mutation
    return lambda x: (((max(0, x.clock_length) if hasattr(x, 'clock_length') else x.mutation_length)
                       + (tt.tip_slack ** 2 * om if x.is_terminal() else 0.0)) * om)


def dense_gls(tt, root, floor):
    """GLS of root-to-tip distance on date with the tree rooted at ``root``:
    Cov(d_i, d_j) = sum over edges shared by the root->i and root->j paths of
    that edge's variance; the variance of an edge is attached to its child in
    the original rooting."""
    om = tt.one_mutation
    adj = {}
    for n in tt.tree.find_clades():
        for c in n.clades:
            var = (c.mutation_length + (tt.tip_slack ** 2 * om if c.is_terminal() else 0.0) + (floor or 0.0) * om) * om
            e = (c.mutation_length, var, id(c))
            adj.setdefault(n, []).append((c, e))
            adj.setdefault(c, []).append((n, e))
    paths = {}
    stack = [(root, None, [])]
    while stack:
        v, parent, path = stack.pop()
        if v.is_terminal():
            paths[v.name] = path
        for w, e in adj.get(v, []):
            if w is not parent:
                stack.append((w, v, path + [e]))
    tips = sorted(paths)
    d = np.array([sum(e[0] for e in paths[t]) for t in tips])
    t = np.array([DATES[x] for x in tips])
    C = np.array([[sum(e[1] for e in paths[a] if e in paths[b]) for b in tips] for a in tips])
    Ci = np.linalg.inv(C)
    X = np.vstack([t, np.ones_like(t)]).T
    beta = np.linalg.solve(X.T @ Ci @ X, X.T @ Ci @ d)
    r = d - X @ beta
    return beta[0], beta[1], 0.5 * r @ Ci @ r


@pytest.mark.parametrize('floor', [None, 0.5, 3.0, 10.0])
def test_recursion_matches_dense_gls_at_every_node(floor):
    tt = make_tt()
    tt.tip_slack = 3.0
    tt.branch_variance_floor = floor
    if floor is None:
        # zero-length internal branches and tips get zero variance without a floor: skip the
        # singular dense inverse and just check the root fit is finite
        treg = tt.setup_TreeRegression(covariation=True)
        assert np.isfinite(treg.regression()['slope'])
        return
    treg = tt.setup_TreeRegression(covariation=True)
    treg._calculate_averages()
    from treetime.treeregression import base_regression
    for n in tt.tree.get_nonterminals():
        reg = base_regression(n.Qtot)
        slope, intercept, chisq = dense_gls(tt, n, floor)
        assert np.isclose(reg['slope'], slope, rtol=1e-9, atol=0)
        assert np.isclose(reg['intercept'], intercept, rtol=1e-9, atol=1e-12)
        assert np.isclose(reg['chisq'], chisq, rtol=1e-8, atol=1e-12)


def test_floor_changes_the_fit():
    tt = make_tt()
    tt.branch_variance_floor = 3.0
    a = tt.setup_TreeRegression(covariation=True).regression()['slope']
    tt.branch_variance_floor = None
    b = tt.setup_TreeRegression(covariation=True).regression()['slope']
    assert not np.isclose(a, b)


@pytest.mark.parametrize('floor', [None, 0, 0.0])
def test_no_floor_reproduces_old_model(floor):
    from treetime.treeregression import TreeRegression
    tt = make_tt()
    tt.tip_slack = 3.0
    tt.branch_variance_floor = floor
    new = tt.setup_TreeRegression(covariation=True)
    tt_old = make_tt()
    tt_old.tip_slack = 3.0
    old = TreeRegression(tt_old.tree, tip_value=new.tip_value, branch_value=new.branch_value,
                         branch_variance=old_branch_variance(tt_old))
    for n_new, n_old in zip(tt.tree.find_clades(), tt_old.tree.find_clades()):
        assert new.branch_variance(n_new) == old.branch_variance(n_old)
    rn = new.rank_roots(force_positive=False)
    ro = old.rank_roots(force_positive=False)
    assert [r['node'].name for r in rn] == [r['node'].name for r in ro]
    for a, b in zip(rn, ro):
        assert (a['slope'], a['intercept'], a['edge_chisq'], a['split']) == (b['slope'], b['intercept'],
                                                                               b['edge_chisq'], b['split'])


def test_root_search_along_edges_matches_dense_gls():
    """A root inside an edge splits the edge's length and variance; with the floor
    the split edge's two halves carry floor*x and floor*(1-x), so the dense fit on
    the tree with the root inserted there must agree with rank_roots."""
    tt = make_tt()
    tt.tip_slack = 3.0
    tt.branch_variance_floor = 3.0
    treg = tt.setup_TreeRegression(covariation=True)
    ranked = treg.rank_roots(force_positive=False)
    checked = 0
    for r in ranked:
        x = r['split']
        if not (0 < x < 1):
            continue
        tt2 = make_tt()
        tt2.tip_slack = 3.0
        node = next(n for n in tt2.tree.find_clades() if n.name == r['node'].name)
        parent = tt2.tree.get_path(node)[-2] if len(tt2.tree.get_path(node)) > 1 else tt2.tree.root
        # insert a degree-2 node at fraction x from the child; it carries floor*x on the lower half
        from Bio.Phylo.BaseTree import Clade
        mid = Clade(branch_length=node.branch_length * (1 - x), name='mid')
        mid.mutation_length = mid.branch_length
        parent.clades.remove(node)
        parent.clades.append(mid)
        node.branch_length *= x
        node.mutation_length = node.branch_length
        mid.clades.append(node)
        slope, intercept, chisq = dense_gls_split(tt2, mid, node, x, 3.0)
        assert np.isclose(r['slope'], slope, rtol=1e-7)
        assert np.isclose(r['edge_chisq'], chisq, rtol=1e-6)
        checked += 1
    assert checked > 0


def dense_gls_split(tt, mid, lower, x, floor):
    """dense_gls rooted at the inserted node ``mid``, with the floor of the split
    edge divided as floor*x (lower half) and floor*(1-x) (upper half), and the
    tip slack of a split terminal edge divided the same way."""
    om = tt.one_mutation
    adj = {}
    for n in tt.tree.find_clades():
        for c in n.clades:
            ts = tt.tip_slack ** 2 * om
            if c is lower:
                var = (c.mutation_length + x * ((ts if c.is_terminal() else 0.0) + floor * om)) * om
            elif c is mid:
                var = (c.mutation_length + (1 - x) * ((ts if lower.is_terminal() else 0.0) + floor * om)) * om
            else:
                var = (c.mutation_length + (ts if c.is_terminal() else 0.0) + floor * om) * om
            e = (c.mutation_length, var, id(c))
            adj.setdefault(n, []).append((c, e))
            adj.setdefault(c, []).append((n, e))
    paths = {}
    stack = [(mid, None, [])]
    while stack:
        v, parent, path = stack.pop()
        if v.is_terminal():
            paths[v.name] = path
        for w, e in adj.get(v, []):
            if w is not parent:
                stack.append((w, v, path + [e]))
    tips = sorted(paths)
    d = np.array([sum(e[0] for e in paths[t]) for t in tips])
    t = np.array([DATES[k] for k in tips])
    C = np.array([[sum(e[1] for e in paths[a] if e in paths[b]) for b in tips] for a in tips])
    Ci = np.linalg.inv(C)
    X = np.vstack([t, np.ones_like(t)]).T
    beta = np.linalg.solve(X.T @ Ci @ X, X.T @ Ci @ d)
    r = d - X @ beta
    return beta[0], beta[1], 0.5 * r @ Ci @ r


def test_negative_floor_rejected():
    tt = make_tt()
    tt.branch_variance_floor = -1.0
    with pytest.raises(ValueError):
        tt.setup_TreeRegression(covariation=True)


def test_regression_source_carries_floor():
    """End to end on a root_candidates fixture: floor None equals the default source; a floor reaches the
    regression through RegressionSource and regression_candidates and changes the objective."""
    from test_root_candidates import make_baseline
    from treetime.root_candidates import RegressionSource, regression_candidates
    b = make_baseline('binary20')
    kw = dict(covariation=True, tip_slack=3.0, slope=None)
    base = regression_candidates(b, k=5, **kw)
    assert regression_candidates(b, k=5, branch_variance_floor=None, **kw) == base
    fl = regression_candidates(b, k=5, branch_variance_floor=3.0, **kw)
    src = RegressionSource('cov', 5, branch_variance_floor=3.0, **kw).candidates(b)
    assert [c for c, _ in src] == [c for c, _ in fl]
    assert [i['regression_chisq'] for _, i in fl] != [i['regression_chisq'] for _, i in base]
    treg = b.regression(covariation=True, tip_slack=3.0, branch_variance_floor=3.0)
    assert fl[0][1]['regression_chisq'] == treg.rank_roots(force_positive=True, slope=None)[0]['edge_chisq']
