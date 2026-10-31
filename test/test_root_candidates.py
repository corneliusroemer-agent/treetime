"""Tests for likelihood-based root selection (treetime.root_candidates) and the
regression ranking it draws candidates from (TreeRegression.rank_roots).

Fixtures in test/root_candidates_data/ were simulated on a known dated tree
(JC69, SeqGen). The tree TreeTime sees, ``fitted_tree.nwk``, was estimated from
the alignment with IQ-TREE 3.1.2 in CMAPLE mode (``--pathogen-force``) and every
internal edge shorter than 0.5/L (and every exact zero) was fully collapsed; see
make_fixtures.py and each meta.json. ``true_tree.nwk`` is only the reference
answer and is never fitted.
"""
import csv
import json
import os

import pytest
from Bio import Phylo

from treetime import TreeTime

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'root_candidates_data')


def load(fx):
    d = os.path.join(DATA, fx)
    with open(os.path.join(d, 'meta.json')) as fh:
        meta = json.load(fh)
    with open(os.path.join(d, 'dates.tsv')) as fh:
        dates = {r[0]: float(r[1]) for r in csv.reader(fh, delimiter='\t') if r[0] != 'name'}
    return os.path.join(d, 'fitted_tree.nwk'), os.path.join(d, 'aln.fasta'), dates, meta


# --- regression ranking --------------------------------------------------------


@pytest.mark.parametrize('covariation', [False, True])
@pytest.mark.parametrize('fixed_slope', [False, True])
def test_rank_roots_first_entry_is_find_best_root(covariation, fixed_slope):
    nwk, aln, dates, meta = load('binary20')
    tt = TreeTime(tree=nwk, aln=aln, dates=dates, gtr='JC69', verbose=0)
    treg = tt.setup_TreeRegression(covariation=covariation)
    slope = meta['clock_rate'] if fixed_slope else None
    best = treg.find_best_root(slope=slope)
    ranked = treg.rank_roots(slope=slope)
    assert ranked[0]['node'] is best['node'] and ranked[0]['split'] == best['split']
    assert ranked[0]['chisq'] == best['chisq']
    assert [r['edge_chisq'] for r in ranked] == sorted(r['edge_chisq'] for r in ranked)
    assert all(r['slope'] >= 0 for r in ranked)
    assert len({id(r['node']) for r in ranked}) == len(ranked)


# --- root candidates -------------------------------------------------------------

import copy  # noqa: E402

import numpy as np  # noqa: E402

from treetime.root_candidates import (  # noqa: E402
    EdgeRoot,
    FitConfig,
    NodeRoot,
    UnrootedBaseline,
    candidate_key,
    classify_trajectory,
    evaluate_root_candidate,
    fresh_marginal_score,
    rank_root_candidates,
    regression_candidates,
    restore_best_fitted_state,
    score_newick,
)

FIT1 = FitConfig(iterations=1)
FIT2 = FitConfig(iterations=2)


def make_baseline(fx, **kw):
    nwk, aln, dates, meta = load(fx)
    kw.setdefault('clock_rate', meta['clock_rate'])
    return UnrootedBaseline(nwk, aln, dates, gtr='JC69', **kw)


def true_split(fx, b):
    side = frozenset(load(fx)[3]['true_root_split'])
    return side if b.ref_leaf not in side else b.all_leaves - side


@pytest.fixture(scope='module')
def b20():
    return make_baseline('binary20')


@pytest.fixture(scope='module')
def top(b20):
    """Regression-best candidate on binary20 (an edge with non-zero length)."""
    c = regression_candidates(b20, k=1)[0][0]
    assert isinstance(c, EdgeRoot)
    return c


def hub(b):
    return max(b.internal_nodes(), key=lambda v: len(b.adj[v]))


def dates_by_name(tt):
    return {n.name: float(n.numdate) for n in tt.tree.find_clades()}


def test_top_regression_candidate_is_treetime_least_squares_root(b20, top):
    """The regression candidates are the edges TreeTime.reroot('least-squares') chooses from."""
    tt = b20.make_treetime(b20.rooted_phylo(None))
    tt.reroot('least-squares', clock_rate=b20.clock_rate)
    kids = tt.tree.root.clades
    side = frozenset(n.name for n in kids[0].get_terminals())
    assert len(kids) == 2 and top.split in (side, b20.all_leaves - side)


def test_pulley_invariance_and_clock_discriminates(b20):
    """At fixed branch lengths every rooting scores the same (reversible model), so candidate
    differences must come from the clock-constrained fit. Edge roots are on edges longer than one
    substitution: splitting a (near-)zero edge puts two edges under TreeTime's MIN_BRANCH_LENGTH
    floor instead of one, which shifts the score slightly."""
    cands = [NodeRoot(b20.split_of[v]) for v in b20.internal_nodes()[:3]]
    long_edges = sorted(b20.splits, key=lambda s: -b20.edge_length(s))[:2]
    cands += [EdgeRoot(long_edges[0], 0.3), EdgeRoot(long_edges[1], 0.5)]
    raw = [fresh_marginal_score(b20.make_treetime(b20.rooted_phylo(c))) for c in cands]
    assert max(raw) - min(raw) < 1e-6
    fitted = [evaluate_root_candidate(b20, c, FitConfig(0), keep_state=False).best_score for c in cands]
    assert max(fitted) - min(fitted) > 1e-3


# Acceptance 1 -----------------------------------------------------------------
@pytest.mark.parametrize('max_iter', [0, 2])
def test_explicit_root_reproduces_treetime_run(b20, top, max_iter):
    """One candidate's fit equals TreeTime.run on the same pre-rooted tree in the same fixed
    configuration, compared after the number of iterations run() actually performed (it stops
    early on ndiff == 0)."""
    tt = b20.make_treetime(b20.rooted_phylo(top))
    tt.run(root=None, infer_gtr=False, max_iter=max_iter, resolve_polytomies=False, Tc=None,
           fixed_clock_rate=b20.clock_rate, time_marginal='never', branch_length_mode='marginal',
           vary_rate=False, use_covariation=False, prune_short=False)
    n_done = len(tt.trace_run) - 1
    ours = evaluate_root_candidate(b20, top, FitConfig(iterations=n_done))
    assert ours.iterations_performed == n_done
    assert ours.scores[-1] == pytest.approx(fresh_marginal_score(tt), abs=1e-8)
    assert ours.positional_LH[-1] == pytest.approx(tt.tree.positional_LH, abs=1e-8)
    if ours.best_index == n_done:  # the retained state is the last one: whole trees agree
        d1, d2 = dates_by_name(restore_best_fitted_state(ours)), dates_by_name(tt)
        assert d1.keys() == d2.keys() and max(abs(d1[k] - d2[k]) for k in d1) < 1e-8


# Acceptance 2 + 8 ---------------------------------------------------------------
def test_exhaustive_search_reproducible_and_order_independent():
    """Every node and edge of poly20, forward; then all again in reverse order with a newly built
    baseline: bit-identical scores and the same ranking."""
    b = make_baseline('poly20')
    cands = b.all_candidates()
    assert len(cands) == len({candidate_key(c) for c in cands}) == len(b.internal_nodes()) + len(b.splits)
    kw = dict(regression_k=0, fit=FitConfig(0), sensitivity_window=None, keep_states=False)
    fwd = rank_root_candidates(b, explicit=cands, **kw)
    rev = rank_root_candidates(make_baseline('poly20'), explicit=cands[::-1], **kw)
    a = {r.candidate_id: r.score for r in fwd.results}
    assert len(a) == len(cands) and all(np.isfinite(s) for s in a.values())
    assert a == {r.candidate_id: r.score for r in rev.results}
    assert [r.candidate_id for r in fwd.results] == [r.candidate_id for r in rev.results]


def test_order_independence_with_iterations(b20, top):
    cands = [b20.node(hub(b20)), top, b20.edge(b20.internal_nodes()[1], 0.3)]
    fwd = [evaluate_root_candidate(b20, c, FIT1, keep_state=False).best_score for c in cands]
    rev = [evaluate_root_candidate(b20, c, FIT1, keep_state=False).best_score for c in reversed(cands)][::-1]
    assert fwd == rev


# Acceptance 3 + 4 ---------------------------------------------------------------
def test_fresh_score_matches_saved_tree_and_state_is_reproducible(b20, top):
    r = evaluate_root_candidate(b20, top, FitConfig(3))
    assert r.best_index == int(np.argmax(r.scores)) and r.best_score == max(r.scores)
    state = restore_best_fitted_state(r)
    # 3: the retained object, and a new TreeAnc built from the saved newick, score the same
    assert fresh_marginal_score(copy.deepcopy(state)) == pytest.approx(r.best_score, abs=1e-9)
    assert score_newick(b20, r.fitted_newick) == pytest.approx(r.best_score, abs=1e-6)
    assert state.tree.root.numdate == pytest.approx(r.root['root_date'])
    # cached and fresh are separate quantities; the cached value predates the time tree
    assert any(abs(c - f) > 1e-6 for c, f in zip(r.cached_LH, r.scores))
    # 4: re-running reproduces trajectory, best step, newick and every node date bit for bit
    r2 = evaluate_root_candidate(b20, top, FitConfig(3))
    assert r2.scores == r.scores and r2.best_index == r.best_index and r2.fitted_newick == r.fitted_newick
    assert dates_by_name(restore_best_fitted_state(r2)) == dates_by_name(state)


def test_iteration_count_is_not_cut_short_by_ndiff(b20, top):
    r = evaluate_root_candidate(b20, top, FitConfig(3), keep_state=False)
    assert r.iterations_performed == 3 and len(r.scores) == 4
    assert r.treetime_stop_iteration is not None and r.treetime_stop_iteration < 3


# Acceptance 5 ---------------------------------------------------------------------
def test_node_and_edge_semantics(b20):
    b = b20
    h = hub(b)
    deg = len(b.adj[h])
    assert deg > 3
    # exact node root keeps the node and its full degree, no new node
    t = b.rooted_phylo(b.node(h))
    assert t.root.name == h and len(t.root.clades) == deg
    # edge root: a new bifurcating root, initial split respected, edge length preserved
    nbr = b.toward_ref[h]
    w = b.adj[h][nbr]
    e = b.edge(h, 0.25)
    lens = {c.name: c.branch_length for c in b.rooted_phylo(e).root.clades}
    assert len(lens) == 2 and lens[h] == pytest.approx(0.25 * w) and lens[nbr] == pytest.approx(0.75 * w)
    # the same edge addressed from its other end is the same placement
    assert b.edge(nbr, 0.75, other=h) == e
    # endpoint deduplication
    assert b.edge(h, 0.0) == b.node(h)
    if not b.is_leaf(nbr):
        assert b.edge(h, 1.0) == b.node(nbr)
    leaf = b.leaves[-1]
    assert b.node(leaf) == EdgeRoot(b.split_of[leaf], 0.0)
    assert b.node(b.ref_leaf) == EdgeRoot(b.split_of[b.center], 1.0)
    # fractions on one edge are one candidate with several initialisations
    s = b.split_of[b.internal_nodes()[1]]
    res = rank_root_candidates(b, explicit=[EdgeRoot(s, 0.3), EdgeRoot(s, 0.7)], regression_k=0, fit=FitConfig(0),
                               sensitivity_window=None)
    assert len(res.results) == 1 and len(res.best.runs) == 2
    assert res.best.score == max(x.best_score for x in res.best.runs)


@pytest.mark.parametrize('optimize_baseline', [False, True])
def test_zero_length_root_edge_is_fitted(optimize_baseline):
    """binary20 has zero-length leaf edges (exact zeros in the input tree, ~1e-22 after the baseline
    branch-length optimisation). An edge root there gets a one-mutation initial root edge."""
    b = make_baseline('binary20', optimize_baseline=optimize_baseline)
    zero = [v for v in b.leaves if v != b.ref_leaf and b.edge_length(b.split_of[v]) < 1e-12]
    assert zero
    for cand in (b.edge(zero[0], 0.5), b.node(zero[0])):
        r = evaluate_root_candidate(b, cand, FIT1)
        assert r.error is None and np.isfinite(r.best_score) and r.root['root_degree'] == 2
        assert any('zero-length' in w for w in r.warnings)
        assert fresh_marginal_score(copy.deepcopy(restore_best_fitted_state(r))) == pytest.approx(r.best_score)


# Acceptance 6 ---------------------------------------------------------------------
@pytest.mark.parametrize('fx', ['binary20', 'poly20'])
def test_topology_and_high_degree_nodes_preserved(fx):
    b = make_baseline(fx)
    ref = b.topology_signature(b.rooted_phylo(None))
    h = hub(b)
    for cand in [b.node(h), b.edge(h, 0.5), b.edge(b.leaves[3], 0.5)]:
        r = evaluate_root_candidate(b, cand, FitConfig(0))
        assert b.topology_signature(restore_best_fitted_state(r).tree) == ref, r.candidate_id
    assert max(ref[1]) == len(b.adj[h]) > 3


# Acceptance 7 + 9 -------------------------------------------------------------------
def test_known_root_and_close_candidate_diagnostics(b20):
    true = true_split('binary20', b20)
    res = rank_root_candidates(b20, explicit=[EdgeRoot(true, 0.5)], regression_k=4, fit=FIT1,
                               sensitivity_window=3.0)
    assert res.best.candidate == EdgeRoot(true, res.best.candidate.fraction)
    close = [r for r in res.results if res.best.score - r.score <= 3.0]
    far = [r for r in res.results if res.best.score - r.score > 3.0]
    assert len(close) >= 2 and far
    for r in close:
        assert len(r.runs) > 1 and r.init_spread is not None and r.uncertainty >= r.init_spread
        assert any(x.initial_mode == 'joint' for x in r.runs)
    assert all(len(r.runs) == 1 for r in far)
    top_ = res.best
    for r in res.results[1:]:
        assert r.resolved_from_best == (-r.delta_logL > max(res.min_margin, top_.uncertainty + r.uncertainty))
    assert res.best_is_resolved == all(r.resolved_from_best for r in res.results[1:])
    # only the best run of each candidate keeps a TreeTime copy
    assert all((x.state is not None) == (x is r.best_run) for r in res.results for x in r.runs)
    rows = res.rows()
    assert rows[0]['delta_logL'] == 0 and all(x['delta_logL'] <= 0 for x in rows)
    assert res.to_tsv().count('\n') == len(rows) + 1
    assert json.loads(res.to_json())['summary']['best'] == res.best.candidate_id
    assert res.best.candidate_id in res.table()


def test_min_margin_flags_a_close_pair(b20, top):
    """Two placements whose fits differ by less than min_margin are never called resolved."""
    nbr = b20.node(b20.node_of[top.split]) if not b20.is_leaf(b20.node_of[top.split]) else b20.node(hub(b20))
    kw = dict(explicit=[top, nbr], regression_k=0, fit=FitConfig(0), sensitivity_window=None)
    res = rank_root_candidates(b20, min_margin=0.0, **kw)
    gap = res.margin
    assert gap > 0
    assert rank_root_candidates(b20, min_margin=gap + 1, **kw).best_is_resolved is False


def test_explicit_and_regression_candidates_are_merged(b20, top):
    res = rank_root_candidates(b20, explicit=[EdgeRoot(top.split, 0.2)], regression_k=1, fit=FitConfig(0),
                               sensitivity_window=None)
    assert len(res.results) == 1 and res.best.source == 'explicit+regression'
    assert res.best.regression_rank == 1 and len(res.best.runs) == 2
    assert res.best.runs[0].initial_fraction == pytest.approx(top.fraction)  # regression fraction first


def test_covariation_regression_candidates(b20):
    plain = regression_candidates(b20, k=3)
    cov = regression_candidates(b20, k=3, covariation=True, tip_slack=3)
    assert len(plain) == len(cov) == 3
    assert [i['regression_rank'] for _, i in cov] == [1, 2, 3]


def test_classify_trajectory():
    assert classify_trajectory([1.0, 2.0, 2.0]) == 'fixed_point'
    assert classify_trajectory([1.0, 2.0, 1.0, 2.0]) == 'cycling'
    assert classify_trajectory([-10.0, -18.0, -10.2, -17.0]) == 'oscillating'
    assert classify_trajectory([1.0, 2.0, 3.0]) == 'drifting'


def test_duplicate_internal_labels_are_renamed():
    """Tree-builder labels (e.g. support values) on internal nodes need not be unique."""
    nwk, aln, dates, meta = load('binary20')
    tree = Phylo.read(nwk, 'newick')
    for c in tree.get_nonterminals():
        c.name = '100'
    b = UnrootedBaseline(tree, aln, dates, optimize_baseline=False, clock_rate=1e-3)
    assert all(c.name == '100' for c in tree.get_nonterminals())  # input not mutated
    ref = make_baseline('binary20', optimize_baseline=False)
    assert b.splits == ref.splits and len(b.internal_nodes()) == len(ref.internal_nodes())


def test_example_data():
    """TreeTime's own example data (repository tree, not collapsed), if checked out."""
    d = os.path.join(HERE, 'treetime_examples', 'data', 'h3n2_na')
    if not os.path.isdir(d):
        pytest.skip('treetime_examples not checked out')
    from treetime.utils import parse_dates

    dates = parse_dates(os.path.join(d, 'h3n2_na_20.metadata.csv'))
    b = UnrootedBaseline(os.path.join(d, 'h3n2_na_20.nwk'), os.path.join(d, 'h3n2_na_20.fasta'), dates)
    assert b.clock_rate > 0
    res = rank_root_candidates(b, regression_k=3, fit=FIT1, sensitivity_window=None, keep_states=False)
    assert len(res.results) == 3 and all(np.isfinite(r.score) for r in res.results)
