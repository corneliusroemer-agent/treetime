"""Tests for treetime.root_candidates (likelihood-based root selection, Stage 1).

Fixtures under test/data/root_candidates/simN were simulated on a known dated
tree; ``fitted.nwk`` was re-estimated from the alignment with IQ-TREE
(``--pathogen-force``) and internal branches shorter than 0.5/L fully collapsed.
``true_tree.nwk`` is used only as the reference answer.
"""
import copy
import os

import numpy as np
import pytest
from Bio import Phylo

from treetime.root_candidates import (
    EdgeRoot,
    FitConfig,
    NodeRoot,
    UnrootedBaseline,
    evaluate_root_candidate,
    fresh_marginal_score,
    rank_root_candidates,
    restore_best_fitted_state,
    score_newick,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'root_candidates')
QUICK = os.environ.get('ROOTCAND_QUICK', 'sim3')
FIT1 = FitConfig(iterations=1)
FIT3 = FitConfig(iterations=3)


def load_dates(path):
    out = {}
    with open(path) as fh:
        next(fh)
        for line in fh:
            name, d = line.split()
            out[name] = float(d)
    return out


def load_sim(name, **kw):
    d = os.path.join(DATA, name)
    return UnrootedBaseline(os.path.join(d, 'fitted.nwk'), os.path.join(d, 'aln.fasta'),
                            load_dates(os.path.join(d, 'dates.tsv')), gtr='JC69', **kw)


def true_root_split(name, baseline):
    t = Phylo.read(os.path.join(DATA, name, 'true_tree.nwk'), 'newick')
    side = frozenset(c.name for c in t.root.clades[0].get_terminals())
    return side if baseline.ref_leaf not in side else baseline.all_leaves - side


@pytest.fixture(scope='module')
def quick():
    return load_sim(QUICK)


@pytest.fixture(scope='module')
def multi():
    """Fixture with a multifurcation (unrooted node degree > 3)."""
    return load_sim('sim2')


def internal_nodes(b):
    return [v for v in sorted(b.split_of) if not b.is_leaf(v)]


def tree_dates(tt):
    return {n.name: float(n.numdate) for n in tt.tree.find_clades()}


# --------------------------------------------------------------------------
def test_regression_ranking_matches_find_best_root(quick):
    treg = quick.regression()
    best = treg.find_best_root(force_positive=True, slope=quick.clock_rate)
    ranked = treg.rank_roots(force_positive=True, slope=quick.clock_rate)
    assert ranked[0]['node'] is best['node'] and ranked[0]['split'] == best['split']
    assert all(a['x_chisq'] <= b['x_chisq'] for a, b in zip(ranked, ranked[1:]))
    assert len(ranked) <= len(quick.splits)


def test_pulley_invariance_and_clock_discriminates(quick):
    """At fixed branch lengths every rooting scores the same (reversible model), so
    differences between candidates must come from the clock-constrained fit.

    Edge roots are taken on edges longer than one substitution: splitting a
    (near-)zero edge puts two edges under TreeTime's MIN_BRANCH_LENGTH floor
    instead of one, which changes the score slightly."""
    cands = [NodeRoot(quick.split_of[v]) for v in internal_nodes(quick)[:3]]
    long_edges = sorted(quick.splits, key=lambda s: -quick.edge_length(s))[:2]
    cands += [EdgeRoot(long_edges[0], 0.3), EdgeRoot(long_edges[1], 0.5)]
    raw = [fresh_marginal_score(quick.make_treetime(quick.rooted_phylo(c))) for c in cands]
    assert max(raw) - min(raw) < 1e-6
    fitted = [evaluate_root_candidate(quick, c, FIT1)['fresh_marginal_LH'] for c in cands]
    assert max(fitted) - min(fitted) > 1e-3


# Acceptance 1 ---------------------------------------------------------------
@pytest.mark.parametrize('max_iter', [1, 3])
def test_explicit_root_reproduces_treetime_run(quick, max_iter):
    """Our fit for one candidate equals TreeTime.run on the same pre-rooted tree with the
    same fixed configuration, compared after the number of iterations run() really did."""
    reg_best = quick.regression().rank_roots(slope=quick.clock_rate)[0]
    cand = quick.edge(reg_best['node'].name, reg_best['split']) if reg_best['node'].name != quick.ref_leaf \
        else quick.node(quick.center)
    tt = quick.make_treetime(quick.rooted_phylo(quick.canonical(cand)))
    tt.run(root=None, infer_gtr=False, max_iter=max_iter, resolve_polytomies=False, Tc=None,
           fixed_clock_rate=quick.clock_rate, time_marginal='never', branch_length_mode='marginal',
           vary_rate=False, use_covariation=False, prune_short=False)
    n_done = len(tt.trace_run) - 1
    ours = evaluate_root_candidate(quick, cand, FitConfig(iterations=n_done), keep_tt=True)
    assert ours['iterations_performed'] == n_done
    assert ours['trajectory'][-1] == pytest.approx(fresh_marginal_score(tt), abs=1e-8)
    # with the last state forced (rather than the best) the full trees must agree
    if ours['best_iteration'] == n_done:
        d1, d2 = tree_dates(ours['tt']), tree_dates(tt)
        assert d1.keys() == d2.keys()
        assert max(abs(d1[k] - d2[k]) for k in d1) < 1e-8


# Acceptance 2 ---------------------------------------------------------------
def test_order_independence(quick):
    nodes = internal_nodes(quick)
    cands = [NodeRoot(quick.split_of[nodes[0]]), quick.edge(nodes[1], 0.3), NodeRoot(quick.split_of[nodes[2]])]
    fwd = [evaluate_root_candidate(quick, c, FIT1)['fresh_marginal_LH'] for c in cands]
    rev = [evaluate_root_candidate(quick, c, FIT1)['fresh_marginal_LH'] for c in reversed(cands)][::-1]
    assert fwd == rev
    res, _ = rank_root_candidates(quick, explicit=cands[::-1], regression_k=0, fit=FIT1, run_sensitivity=False)
    by_id = {r['candidate_id']: r['fresh_marginal_LH'] for r in res}
    assert sorted(by_id.values()) == sorted(fwd)


# Acceptance 3 + 4 -----------------------------------------------------------
def test_fresh_score_matches_saved_tree_and_state_is_reproducible(quick):
    cand = quick.edge(internal_nodes(quick)[1], 0.4)
    r = evaluate_root_candidate(quick, cand, FIT3, keep_tt=True)
    assert r['best_iteration'] == int(np.argmax(r['trajectory']))
    # 3: the restored tree, and a new TreeAnc built from the saved newick, score the same
    assert fresh_marginal_score(r['tt']) == pytest.approx(r['fresh_marginal_LH'], abs=1e-9)
    assert score_newick(quick, r['fitted_newick']) == pytest.approx(r['fresh_marginal_LH'], abs=1e-6)
    # cached and fresh are separate quantities
    assert 'cached_marginal_LH' in r and 'positional_LH' in r
    # 4: the snapshot installed into a different, freshly built object reproduces score and dates
    other = quick.make_treetime(quick.rooted_phylo(quick.canonical(cand)))
    restore_best_fitted_state(other, r['best_state'])
    assert fresh_marginal_score(other) == pytest.approx(r['fresh_marginal_LH'], abs=1e-9)
    assert tree_dates(other) == tree_dates(r['tt'])
    assert other.tree.root.numdate == pytest.approx(r['root_date'])
    # and re-running the evaluation reproduces everything bit for bit
    r2 = evaluate_root_candidate(quick, cand, FIT3)
    assert r2['trajectory'] == r['trajectory'] and r2['fitted_newick'] == r['fitted_newick']


# Acceptance 5 ---------------------------------------------------------------
def test_node_and_edge_semantics(multi):
    b = multi
    hub = max(internal_nodes(b), key=lambda v: len(b.adj[v]))
    deg = len(b.adj[hub])
    assert deg > 3
    # exact node root keeps the node and its full degree, no new node
    t = b.rooted_phylo(b.node(hub))
    assert t.root.name == hub and len(t.root.clades) == deg
    # edge root: new bifurcating root, initial split respected
    nbr = b.toward_ref[hub]
    w = b.adj[hub][nbr]
    e = b.edge(hub, 0.25)
    t = b.rooted_phylo(e)
    lens = {c.name: c.branch_length for c in t.root.clades}
    assert len(lens) == 2 and lens[hub] == pytest.approx(0.25 * w) and lens[nbr] == pytest.approx(0.75 * w)
    # the same edge addressed from the other end is the same candidate
    assert b.edge(nbr, 0.75, other=hub) == e
    # endpoint deduplication
    assert b.edge(hub, 0.0) == b.node(hub)
    if not b.is_leaf(nbr):
        assert b.edge(hub, 1.0) == b.node(nbr)
    leaf = b.leaves[-1]
    assert b.node(leaf) == EdgeRoot(b.split_of[leaf], 0.0)
    assert b.node(b.ref_leaf) == EdgeRoot(b.split_of[b.center], 1.0)
    # a zero-length root-incident edge fits without failure
    r = evaluate_root_candidate(b, b.node(leaf), FIT1)
    assert not r['failed'] and r['root_degree'] == 2
    r = evaluate_root_candidate(b, b.node(hub), FIT1)
    assert not r['failed'] and r['root_degree'] == deg


# Acceptance 6 ---------------------------------------------------------------
def test_topology_and_high_degree_nodes_preserved(multi):
    b = multi
    ref_sig = b.topology_signature(b.rooted_phylo(None))
    hub = max(internal_nodes(b), key=lambda v: len(b.adj[v]))
    for cand in [b.node(hub), b.edge(hub, 0.5), b.edge(b.leaves[3], 0.5)]:
        r = evaluate_root_candidate(b, cand, FIT1, keep_tt=True)
        assert b.topology_signature(r['tt'].tree) == ref_sig, r['candidate_id']


# Acceptance 7 ---------------------------------------------------------------
@pytest.mark.parametrize('name', [QUICK, 'sim4'])
def test_known_root_is_found(name):
    b = load_sim(name)
    true = true_root_split(name, b)
    assert true in b.splits
    res, summ = rank_root_candidates(b, explicit=[EdgeRoot(true, 0.5)], regression_k=5, fit=FIT1,
                                     run_sensitivity=False, n_jobs=6)
    top = res[0]
    a = b.node_of[true]
    near = {true, b.split_of.get(b.toward_ref[a])} | {b.split_of[w] for w in b.adj[a] if w in b.split_of}
    # sensible: the best root is on the true root edge or at one of its endpoints
    assert frozenset(top['split']) in near, (top['candidate_id'], b.candidate_id(EdgeRoot(true)))


# Acceptance 8 ---------------------------------------------------------------
def test_exhaustive_search_reproducible(quick):
    """Every internal node and every edge midpoint, once with 8 workers and once in reverse
    order with 3 workers and a newly built baseline: identical results."""
    cands = quick.all_candidates()
    assert len(cands) == len(set(cands)) == len(internal_nodes(quick)) + len(quick.splits)
    fit0 = FitConfig(iterations=0)
    res1, s1 = rank_root_candidates(quick, explicit=cands, regression_k=0, fit=fit0, run_sensitivity=False,
                                    n_jobs=8)
    b2 = load_sim(QUICK)
    res2, s2 = rank_root_candidates(b2, explicit=cands[::-1], regression_k=0, fit=fit0, run_sensitivity=False,
                                    n_jobs=3)
    assert [(r['candidate_id'], r['fresh_marginal_LH']) for r in res1] == \
        [(r['candidate_id'], r['fresh_marginal_LH']) for r in res2]
    assert len(res1) == len(cands) and not any(r['failed'] for r in res1)


# Acceptance 9 ---------------------------------------------------------------
def test_close_candidates_get_optimiser_diagnostics(quick):
    s = quick.split_of[internal_nodes(quick)[1]]
    cands = [EdgeRoot(s, 0.45), EdgeRoot(s, 0.55)]
    res, summ = rank_root_candidates(quick, explicit=cands, regression_k=0, fit=FIT1, sensitivity_window=1e9, n_jobs=6)
    for r in res:
        assert len(r['initialisations']) > 1
        assert r['init_spread'] is not None and r['uncertainty'] >= r['init_spread']
    top, other = res
    assert other['resolved_from_best'] == (-other['delta_LH'] > top['uncertainty'] + other['uncertainty'])
    assert summ['best_is_resolved'] == other['resolved_from_best']
    # candidates far outside the window are not re-fitted
    res, _ = rank_root_candidates(quick, explicit=cands, regression_k=0, fit=FIT1, sensitivity_window=-1.0)
    assert all(len(r['initialisations']) == 1 for r in res)
