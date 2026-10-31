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


def test_fixed_slope_chisq_is_the_intercept_only_residual():
    """With the slope fixed, chisq is the residual of the best intercept at that slope, not the
    free-slope residual (which base_regression returned whatever slope was passed)."""
    import numpy as np
    from treetime.treeregression import base_regression, dsqii, dtavgii, davgii, sii, tavgii, tsqii

    rng = np.random.default_rng(1)
    t = rng.uniform(2000, 2020, 30)
    d = 0.002 * (t - 1990) + rng.normal(0, 0.002, 30)
    Q = np.zeros(6)
    Q[tavgii], Q[davgii], Q[tsqii], Q[dtavgii], Q[dsqii], Q[sii] = t.sum(), d.sum(), (t * t).sum(), (d * t).sum(), (d * d).sum(), len(t)
    free = base_regression(Q)
    for slope in (1e-3, 2e-3, 2.0):
        r = base_regression(Q, slope=slope)
        res = lambda icpt: 0.5 * np.sum((d - slope * t - icpt) ** 2)
        assert r['chisq'] == pytest.approx(res(r['intercept']), rel=1e-6)
        assert res(r['intercept']) < min(res(r['intercept'] + 1e-4), res(r['intercept'] - 1e-4))
        assert r['chisq'] >= free['chisq'] - 1e-12 and r['chisq'] != free['chisq']


# --- root candidates -------------------------------------------------------------

import copy  # noqa: E402

import numpy as np  # noqa: E402

from treetime.root_candidates import _regression_edges  # noqa: E402
from treetime.root_candidates import (  # noqa: E402
    EdgeRoot,
    FitConfig,
    NodeRoot,
    UnrootedBaseline,
    RegressionSource,
    candidate_key,
    candidate_shortlist,
    classify_trajectory,
    default_candidate_sources,
    node_incident_edges,
    evaluate_root_candidate,
    fresh_marginal_score,
    plain_candidate_sources,
    rank_root_candidates,
    regression_candidates,
    resolve_candidate_sources,
    restore_best_fitted_state,
    score_newick,
    fit_rate_profile,
    profile_rate_point,
    rescale_times_for_rate,
    fitted_root_candidate,
    merge_duplicate_roots,
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
    kw = dict(regression_k=0, fit=FitConfig(0), sensitivity_window=None, keep_states=False, rate_rounds=0)
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
                               sensitivity_window=None, rate_rounds=0)
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
                               sensitivity_window=3.0, rate_rounds=3)
    assert res.best.candidate == EdgeRoot(true, res.best.candidate.fraction)
    close = [r for r in res.results if res.best.score - r.score <= 3.0]
    far = [r for r in res.results if res.best.score - r.score > 3.0]
    assert len(close) >= 2 and far
    for r in close:
        assert len(r.runs) > 1 and r.init_spread is not None and r.uncertainty >= r.init_spread
        assert any(x.initial_mode == 'joint' for x in r.runs)
    assert all(len(r.runs) == 1 for r in far if not r.rate_fitted)
    # rate-fit set: within rate_fit_window (5) of the best, at least 3, and the regression rank 1
    assert res.best.rate_fitted and sum(r.rate_fitted for r in res.results) >= 3
    assert all(r.rate_fitted for r in res.results if r.regression_rank == 1)
    top_ = res.best
    for r in res.distinct[1:]:
        assert r.resolved_from_best == (-r.delta_logL > max(res.min_margin, top_.uncertainty + r.uncertainty))
    assert all(r.resolved_from_best is None for r in res.results if r.duplicate_of)
    assert res.best_is_resolved == all(r.resolved_from_best for r in res.distinct[1:])
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
    kw = dict(explicit=[top, nbr], regression_k=0, fit=FitConfig(0), sensitivity_window=None, rate_rounds=0)
    res = rank_root_candidates(b20, min_margin=0.0, **kw)
    gap = res.margin
    assert gap > 0
    assert rank_root_candidates(b20, min_margin=gap + 1, **kw).best_is_resolved is False


def test_explicit_and_regression_candidates_are_merged(b20, top):
    res = rank_root_candidates(b20, explicit=[EdgeRoot(top.split, 0.2)], candidate_sources=plain_candidate_sources(1),
                               fit=FitConfig(0), sensitivity_window=None, rate_rounds=0)
    assert len(res.results) == 1 and res.best.source == 'explicit+plain'
    assert res.best.source_ranks == {'explicit': 1, 'plain': 1}
    assert res.best.regression_rank == 1 and len(res.best.runs) == 2
    assert res.best.runs[0].initial_fraction == pytest.approx(top.fraction)  # regression fraction first


def test_covariation_regression_candidates(b20):
    plain = regression_candidates(b20, k=3)
    cov = regression_candidates(b20, k=3, covariation=True, tip_slack=3)
    assert len(plain) == len(cov) == 3
    assert [i['regression_rank'] for _, i in cov] == [1, 2, 3]


def _keys(cands):
    return [candidate_key(c) for c, _ in cands]


@pytest.mark.parametrize('fx', ['binary20', 'poly20', 'mixed60'])
def test_default_shortlist_is_plain_union_covariation(fx):
    """Default candidates: plain fixed-slope top 5 and covariation (tip_slack 3, free slope) top 5,
    deduplicated by bipartition, each labelled with the ranks of every source that proposed it."""
    b = make_baseline(fx)
    plain = regression_candidates(b, k=5)
    cov = regression_candidates(b, k=5, covariation=True, tip_slack=3.0, slope=None)
    merged = candidate_shortlist(b, sources=default_candidate_sources(5))
    union = list(dict.fromkeys(_keys(plain) + _keys(cov)))
    assert list(merged) == union
    for name, lst in (('plain', plain), ('cov', cov)):
        for key, (c, info) in zip(_keys(lst), lst):
            assert merged[key].source_ranks[name] == info['regression_rank']
    for key, e in merged.items():
        assert set(e.source_ranks) == {n for n, lst in (('plain', plain), ('cov', cov)) if key in _keys(lst)}
        assert len(e.inits) == 1  # fitted once, from the first source's placement
        first = plain if 'plain' in e.source_ranks else cov
        assert e.inits[0] == dict(zip(_keys(first), [c for c, _ in first]))[key]


def test_default_shortlist_sources_differ_and_reach_the_ranking():
    """On mixed60 the two lists differ (so the union is larger than either), and
    rank_root_candidates' default evaluates exactly the union."""
    b = make_baseline('mixed60')
    plain = set(_keys(regression_candidates(b, k=5)))
    cov = set(_keys(regression_candidates(b, k=5, covariation=True, tip_slack=3.0, slope=None)))
    fixed_cov = set(_keys(regression_candidates(b, k=5, covariation=True, tip_slack=3.0)))
    assert plain & cov and cov - plain and plain - cov
    assert cov != fixed_cov  # the free slope matters: the legacy covariation=True path ranks differently
    res = rank_root_candidates(b, fit=FitConfig(0), keep_states=False)
    assert {candidate_key(r.candidate) for r in res.results} == plain | cov
    assert len(res.results) == len(plain | cov)
    for r in res.results:
        assert set(r.source_ranks) <= {'plain', 'cov'} and r.source == '+'.join(r.source_ranks)
        assert (r.regression_rank is None) == ('plain' not in r.source_ranks)
    assert {s['name'] for s in res.config['candidate_sources']} == {'plain', 'cov'}
    row = next(r for r in res.rows() if '+' in r['source'])
    assert row['source_ranks'].startswith('plain:') and ',cov:' in row['source_ranks']
    assert 'source_ranks' in res.to_tsv().splitlines()[0].split('\t')


def test_plain_only_shortlist_is_reachable():
    """The previous default (plain fixed-slope top k alone) via candidate_sources, and the
    legacy covariation/tip_slack arguments still give one fixed-slope source."""
    b = make_baseline('mixed60')
    plain = _keys(regression_candidates(b, k=5))
    for kw in (dict(candidate_sources=plain_candidate_sources(5)), dict(covariation=False)):
        res = rank_root_candidates(b, fit=FitConfig(0), keep_states=False, **kw)
        assert {candidate_key(r.candidate) for r in res.results} == set(plain)
        assert all(r.source == 'plain' and r.source_ranks['plain'] == r.regression_rank for r in res.results)
    m = candidate_shortlist(b, sources=(RegressionSource('cov_fixed_slope', 5, covariation=True, tip_slack=3.0),))
    assert resolve_candidate_sources(5, covariation=True, tip_slack=3.0) == (
        RegressionSource('cov_fixed_slope', 5, covariation=True, tip_slack=3.0, slope='fixed'),)
    assert list(m) == _keys(regression_candidates(b, k=5, covariation=True, tip_slack=3.0))
    assert resolve_candidate_sources(0) == () and resolve_candidate_sources(0, covariation=True) == ()
    with pytest.raises(ValueError):
        resolve_candidate_sources(5, candidate_sources=plain_candidate_sources(), covariation=True)


def test_shortlist_ranked_sources_and_explicit():
    b = make_baseline('binary20')
    plain = regression_candidates(b, k=2)
    edge = plain[0][0]
    assert isinstance(edge, EdgeRoot)
    other = EdgeRoot(edge.split, 0.3 if abs(edge.fraction - 0.3) > 0.05 else 0.7)
    extra = [b.node(hub(b)), plain[1][0]]
    m = candidate_shortlist(b, explicit=[other], ranked_sources={'attach': extra}, sources=plain_candidate_sources(2))
    e0, e1 = m[candidate_key(edge)], m[candidate_key(plain[1][0])]
    assert e0.source_ranks == {'explicit': 1, 'plain': 1} and e0.inits == [edge, other]  # regression first
    assert e1.source_ranks == {'attach': 2, 'plain': 2} and len(e1.inits) == 1  # same placement, one init
    assert m[candidate_key(b.canonical(extra[0]))].source_ranks['attach'] == 1
    with pytest.raises(ValueError):
        candidate_shortlist(b, ranked_sources={'plain': extra}, sources=plain_candidate_sources(2))


def test_node_candidate_fit_stays_on_the_node(b20):
    """A NodeRoot fit keeps the node as root with its full degree, so it can never reach the
    inside of an incident edge (why node_edges exists)."""
    h = hub(b20)
    run = evaluate_root_candidate(b20, b20.node(h), FIT2)
    assert run.root['root_degree'] == len(b20.adj[h])
    assert sorted(run.root['root_children']) == sorted(b20.adj[h])


@pytest.mark.parametrize('fx', ['poly20', 'mixed60'])
def test_node_edges_expands_node_candidates(fx):
    b = make_baseline(fx)
    for ne in (1, 2, None):
        src = RegressionSource('cov', 5, covariation=True, tip_slack=3.0, slope=None, node_edges=ne)
        cands, exp = src.candidates_and_expansions(b)
        nodes = [(c, i['regression_rank']) for c, i in cands if isinstance(c, NodeRoot)]
        assert nodes, 'fixture should have a node among the top 5'
        for c, rank in nodes:
            v = b.node_of[c.split]
            mine = [e for e, r in exp if r == rank]
            nz = [w for w, length in b.adj[v].items() if length > 0]
            assert len(mine) == (len(nz) if ne is None else min(ne, len(nz)))
            for e in mine:  # an incident, non-zero-length edge, started at its midpoint
                a = b.node_of[e.split]
                assert isinstance(e, EdgeRoot) and e.fraction == 0.5
                assert v in (a, b.toward_ref[a]) and b.adj[a][b.toward_ref[a]] > 0
        m = candidate_shortlist(b, sources=(src,))
        listed = {candidate_key(c) for c, _ in cands}
        assert len(m) == len(listed | {candidate_key(e) for e, _ in exp})
        for key, e in m.items():
            assert len(e.inits) == 1
            if key in listed:
                assert e.source_ranks['cov'] and e.inits[0] == dict((candidate_key(c), c) for c, _ in cands)[key]
            else:
                assert set(e.source_ranks) == {'cov_adj'}
    # the best-ranked incident edge comes first: n=1 is a prefix of n=2
    src = lambda ne: RegressionSource('plain', 5, node_edges=ne)  # noqa: E731
    one, two = src(1).candidates_and_expansions(b)[1], src(2).candidates_and_expansions(b)[1]
    for c, i in src(0).candidates(b):
        if isinstance(c, NodeRoot):
            assert node_incident_edges(b, c, _regression_edges(b), 1) == node_incident_edges(b, c, _regression_edges(b), 2)[:1]
    assert {candidate_key(e) for e, _ in one} <= {candidate_key(e) for e, _ in two}
    capped = RegressionSource('plain', 5, node_edges=None, node_edges_max_rank=2).candidates_and_expansions(b)[1]
    full = src(None).candidates_and_expansions(b)[1]
    assert capped == [(e, r) for e, r in full if r <= 2]


# Per-candidate clock rate ------------------------------------------------------------
def test_rate_fit_bounds_and_retained_state(b20, top):
    r = evaluate_root_candidate(b20, top, FitConfig(1, rate_rounds=4))
    mu0 = b20.clock_rate
    assert all(mu0 / 10 <= m <= mu0 * 10 and m > 0 for m in r.clock_rates)
    assert r.clock_rate == r.clock_rates[r.best_index] and r.rate_rounds_performed >= 1
    assert r.best_score >= max(r.scores[:2])  # never worse than the shared-rate part of the trajectory
    state = restore_best_fitted_state(r)
    assert fresh_marginal_score(copy.deepcopy(state)) == pytest.approx(r.best_score, abs=1e-9)
    assert score_newick(b20, r.fitted_newick) == pytest.approx(r.best_score, abs=1e-6)
    # the retained time tree is the clock fit at the recorded rate: branch length = rate x time span
    for n in state.tree.find_clades():
        if n.up is not None and n.branch_length > 1e-12:
            assert n.branch_length == pytest.approx(r.clock_rate * (n.numdate - n.up.numdate), rel=1e-6)
    # re-running reproduces it exactly, and evaluation order does not matter
    other = evaluate_root_candidate(b20, b20.node(hub(b20)), FitConfig(1, rate_rounds=4), keep_state=False)
    r2 = evaluate_root_candidate(b20, top, FitConfig(1, rate_rounds=4), keep_state=False)
    other2 = evaluate_root_candidate(b20, b20.node(hub(b20)), FitConfig(1, rate_rounds=4), keep_state=False)
    assert r2.scores == r.scores and r2.clock_rates == r.clock_rates and other.scores == other2.scores


def test_rate_fit_moves_towards_true_rate(b20, top):
    """binary20 was simulated at rate 0.0018: started from twice that, the fitted rate moves back."""
    true_rate = load('binary20')[3]['clock_rate']
    r = evaluate_root_candidate(b20, top, FitConfig(1, rate_rounds=6), clock_rate=2 * true_rate, keep_state=False)
    assert abs(np.log(r.clock_rate / true_rate)) < 0.5 * np.log(2)
    assert r.best_score > r.scores[0]


def test_classify_trajectory():
    assert classify_trajectory([1.0, 2.0, 2.0]) == 'fixed_point'
    assert classify_trajectory([1.0, 2.0, 1.0, 2.0]) == 'cycling'
    assert classify_trajectory([-10.0, -18.0, -10.2, -17.0]) == 'oscillating'
    assert classify_trajectory([1.0, 2.0, 3.0]) == 'drifting'


def test_stable_id_does_not_depend_on_input_rooting(b20):
    from treetime.root_candidates import stable_id
    nwk, aln, dates, meta = load('binary20')
    tree = Phylo.read(nwk, 'newick')
    tree.root_with_outgroup(tree.get_terminals()[5])
    other = UnrootedBaseline(tree, aln, dates, clock_rate=meta['clock_rate'], optimize_baseline=False)
    ref = make_baseline('binary20', optimize_baseline=False)
    assert {stable_id(c) for c in other.all_candidates()} == {stable_id(c) for c in ref.all_candidates()}


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
    res = rank_root_candidates(b, candidate_sources=plain_candidate_sources(3), fit=FIT1, sensitivity_window=None,
                               keep_states=False, rate_rounds=0)
    assert len(res.results) == 3 and all(np.isfinite(r.score) for r in res.results)


def test_plots_render(b20, top, tmp_path):
    mpl = pytest.importorskip('matplotlib')
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from treetime.root_candidates import plot_root_candidates, plot_root_to_tip

    res = rank_root_candidates(b20, explicit=[b20.node(hub(b20))], regression_k=3, fit=FitConfig(0),
                               sensitivity_window=None, keep_states=False, rate_rounds=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_root_candidates(res, b20, true_candidate=EdgeRoot(true_split('binary20', b20)), ax=a1)
    plot_root_to_tip(b20, top, ax=a2)
    fig.savefig(tmp_path / 'p.png')
    plt.close(fig)
    assert (tmp_path / 'p.png').stat().st_size > 0


def test_candidate_labels_do_not_depend_on_hash_seed():
    """Node names (INT_xxxx), candidate ids and the default shortlist's order and
    source labels are deterministic across processes with different
    PYTHONHASHSEED values (set/frozenset iteration order)."""
    import subprocess
    import sys
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from test_root_candidates import load\n"
            "from treetime.root_candidates import UnrootedBaseline, regression_candidates\n"
            "nwk, aln, dates, _ = load('poly20')\n"
            "b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')\n"
            "print([(b.candidate_id(c), sorted(c.split)) for c in b.all_candidates()])\n"
            "print([b.candidate_id(c) for c, _ in regression_candidates(b, 3)])\n"
            "from treetime.root_candidates import candidate_shortlist, default_candidate_sources, stable_id\n"
            "for fx in ('poly20', 'mixed60'):\n"
            "    nwk, aln, dates, meta = load(fx)\n"
            "    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69', clock_rate=meta['clock_rate'])\n"
            "    m = candidate_shortlist(b, sources=default_candidate_sources(5))\n"
            "    print([(b.candidate_id(e.candidate), stable_id(e.candidate), e.source, sorted(e.source_ranks.items()))\n"
            "           for e in m.values()])\n") % os.path.dirname(os.path.abspath(__file__))
    outs = []
    for seed in ('0', '7', '12345'):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.append(subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True,
                                   check=True).stdout)
    assert outs[0] == outs[1] == outs[2] and outs[0].strip()


# --------------------------------------------------------------------------
# final rate refit of the winner
# --------------------------------------------------------------------------
def _rows_without_runtime(res):
    return [{k: v for k, v in r.items() if k != 'runtime'} for r in res.rows()]


@pytest.fixture(scope='module')
def refit_pair():
    """binary20 ranked at a shared rate 2x the simulated one, with and without the final rate refit."""
    b = make_baseline('binary20', clock_rate=2 * load('binary20')[3]['clock_rate'])
    kw = dict(regression_k=2, fit=FIT1, candidate_sources=plain_candidate_sources(2))
    return b, rank_root_candidates(b, **kw), rank_root_candidates(b, final_rate_refit=True, **kw)


def test_final_rate_refit_keeps_the_winner_and_improves_logL(refit_pair):
    b, off, on = refit_pair
    ref = on.final_rate_refit
    # selection is untouched: same winner, same rows (scores, ranks, verdicts, root dates)
    assert on.best.candidate_id == off.best.candidate_id == ref['candidate_id']
    assert _rows_without_runtime(on) == _rows_without_runtime(off)
    assert ref['headline'] == 'shared' and ref['n_evals'] <= 15
    assert ref['ranking_logL'] == ref['shared_logL'] and not ref['ranking_rate_fitted']
    assert ref['converged'] is True and ref['at_bound'] is False and ref['warnings'] == []
    # the ranking state counts as a point: refit logL >= shared-rate logL
    assert ref['refit_logL'] >= ref['shared_logL'] == on.best.score
    assert ref['shared_clock_rate'] == b.clock_rate and ref['shared_root_date'] == on.best.best_run.root['root_date']
    # started at 2x the simulated rate, the refit moves the rate (and so the date) and improves the fit
    assert ref['refit_logL'] > ref['shared_logL'] and ref['refit_clock_rate'] < b.clock_rate
    # diagnostic only: the headline stays on the shared-rate ranking
    assert on.headline_root_date == ref['shared_root_date'] != ref['refit_root_date']
    assert on.headline_clock_rate == ref['shared_clock_rate']
    assert on.headline_treetime() is on.best_treetime()
    assert on.best_treetime() is on.best.best_run.state  # ranking state still reachable
    s = on.summary()
    assert s['headline_root_date'] == ref['shared_root_date'] and 'state' not in s['final_rate_refit']
    json.loads(on.to_json())
    assert 'final rate refit' in on.table()


def test_final_rate_refit_off_by_default_reproduces_old_output(refit_pair):
    b, off, on = refit_pair
    assert off.final_rate_refit is None
    s_off, s_on = off.summary(), on.summary()
    assert 'headline_root_date' not in s_off and 'final_rate_refit' not in off.config
    assert {k: v for k, v in s_on.items() if k in s_off} == s_off
    assert {k: v for k, v in on.config.items() if k in off.config} == off.config
    assert off.headline_root_date == off.best.best_run.root['root_date']
    assert off.headline_treetime() is off.best_treetime()
    assert 'final rate refit' not in off.table()


def test_final_rate_refit_without_states_and_close_window(refit_pair):
    b, off, on = refit_pair
    kw = dict(regression_k=2, fit=FIT1, candidate_sources=plain_candidate_sources(2))
    cold = rank_root_candidates(b, keep_states=False, final_rate_refit=True, final_rate_refit_window=50.0, **kw)
    ref = cold.final_rate_refit
    assert cold.best.candidate_id == on.best.candidate_id
    # states are kept internally for the warm start, then dropped as asked
    assert ref['state'] is None
    assert all(x.state is None for r in cold.results for x in r.runs)
    assert all(c['state'] is None for c in ref['close'])
    assert ref['refit_logL'] == on.final_rate_refit['refit_logL']  # same refit as with keep_states=True
    assert len(ref['close']) == len(cold.results) and ref['close'][0]['candidate_id'] == cold.best.candidate_id
    assert sorted(c['refit_rank'] for c in ref['close']) == list(range(1, len(ref['close']) + 1))
    assert isinstance(ref['close_order_changed'], bool) and ref['close_refit_best'] in [r.candidate_id for r in cold.results]
    assert cold.headline_root_date == ref['shared_root_date']  # the refits never change the headline
    json.loads(cold.to_json())


def test_final_rate_refit_continues_a_ranking_rate_fit(refit_pair):
    """With rate_rounds > 0 the refit continues from the rate-fitted winner; shared_* stay the shared-rate fit."""
    b, off, on = refit_pair
    res = rank_root_candidates(b, regression_k=2, fit=FIT1, candidate_sources=plain_candidate_sources(2),
                               rate_rounds=1, rate_fit_min=1, final_rate_refit=True)
    ref = res.final_rate_refit
    assert ref['ranking_rate_fitted'] and ref['ranking_logL'] == res.best.score
    assert ref['shared_clock_rate'] == b.clock_rate and ref['ranking_clock_rate'] != b.clock_rate
    assert ref['refit_logL'] >= ref['ranking_logL'] >= ref['shared_logL']
    assert ref['refit_logL'] > ref['ranking_logL'] and ref['refit_clock_rate'] != ref['ranking_clock_rate']


def test_rate_profile_fit_reaches_the_grid_maximum(refit_pair):
    """fit_rate_profile's answer is the maximum of the profile likelihood over the rate, checked against a coarse
    grid over the whole clamp range and a fine grid (2% steps) around the fitted rate."""
    b, off, on = refit_pair
    win, mu0 = off.best.best_run, b.clock_rate

    def ev(c, mode, rounds, warm=None, iterations=None):
        return evaluate_root_candidate(b, c, FitConfig(iterations, rate_rounds=rounds), clock_rate=warm[1],
                                       initial_state=warm[0])

    def point(mu):
        return profile_rate_point(ev, win.candidate, win.state, mu0, mu)

    f = fit_rate_profile(point, mu0)
    assert f['converged'] and f['n_evals'] <= 15
    coarse = [point(mu0 * m)[0] for m in np.exp(np.linspace(np.log(0.1), np.log(10), 9))]
    fine = [point(f['mu'] * m)[0] for m in np.exp(np.linspace(-0.2, 0.2, 21))]
    # within 0.1 logL of both grids' maxima (near its top the profile is flat: < 0.1 logL over ~10% in rate)
    assert f['logL'] >= max(coarse) - 0.1 and f['logL'] >= max(fine) - 0.1
    # the refit inside rank_root_candidates is this optimisation, from the same state
    assert on.final_rate_refit['refit_logL'] == pytest.approx(max(f['logL'], off.best.score), abs=1e-6)


def test_rescale_times_for_rate_keeps_divergences(refit_pair):
    b, off, on = refit_pair
    st, mu0 = off.best.best_run.state, b.clock_rate
    tt = rescale_times_for_rate(st, mu0, 0.5 * mu0)
    t_max = max(n.numdate for n in st.tree.get_terminals())
    for n0, n1 in zip(st.tree.find_clades(), tt.tree.find_clades()):
        if n0.is_terminal():
            assert n1.numdate == n0.numdate
        else:
            assert t_max - n1.numdate == pytest.approx(2 * (t_max - n0.numdate))
        if n0.up is not None and not n0.is_terminal():
            assert n1.branch_length == pytest.approx(n0.branch_length, rel=1e-9)  # internal edges: same divergence


# --------------------------------------------------------------------------
# duplicate rooted trees: edge fits that collapse onto a node
# --------------------------------------------------------------------------
def test_fitted_root_candidate_collapses_within_tolerance(b20):
    v = hub(b20)
    w = next(x for x in sorted(b20.adj[v]) if b20.toward_ref.get(x) == v)  # child of v away from the reference
    e = b20.edge(w, 0.5)
    a, b = b20.node_of[e.split], b20.toward_ref[b20.node_of[e.split]]
    assert b == v
    root = lambda la, lb: {'fitted_fraction': la / (la + lb), 'root_children': [a, b], 'root_child_lengths': [la, lb],
                           'n_sites': 1000}
    # 0.04 expected mutations from v: v's tree; 0.06: still the edge, at its fitted fraction
    assert fitted_root_candidate(b20, e, root(0.01, 0.04e-3)) == (NodeRoot(b20.split_of[v]), 1.0)
    cand, frac = fitted_root_candidate(b20, e, root(0.01, 0.06e-3))
    assert cand == EdgeRoot(e.split, frac) and 0.99 < frac < 1.0
    assert fitted_root_candidate(b20, e, root(0.01, 0.06e-3), tol=0.1)[0] == NodeRoot(b20.split_of[v])
    if not b20.is_leaf(a):
        assert fitted_root_candidate(b20, e, root(0.0, 0.01)) == (NodeRoot(b20.split_of[a]), 0.0)
    node = NodeRoot(b20.split_of[v])
    assert fitted_root_candidate(b20, node, {'fitted_fraction': None}) == (node, None)


@pytest.fixture(scope='module')
def dup_ranking():
    """poly20: node INT_0005 and its incident edge to L17, whose fit collapses onto INT_0005, plus node INT_0004."""
    b = make_baseline('poly20')
    explicit = [b.node('INT_0005'), b.edge('L17', 0.5, 'INT_0005'), b.node('INT_0004')]
    return b, rank_root_candidates(b, explicit=explicit, regression_k=0, candidate_sources=(), fit=FIT2)


def test_duplicate_roots_are_merged_and_margins_compare_distinct_trees(dup_ranking):
    b, res = dup_ranking
    by = {r.candidate_id: r for r in res.results}
    e, n5, n4 = by['edge:L17-INT_0005'], by['node:INT_0005'], by['node:INT_0004']
    assert e.root_id == n5.root_id == 'node:INT_0005' and n4.root_id == 'node:INT_0004'
    assert e.root_fraction == 1.0 and n5.root_fraction is None
    rep, dup = (e, n5) if e.score >= n5.score else (n5, e)
    assert abs(e.score - n5.score) < 0.05  # the same tree, fitted twice
    assert dup.duplicate_of == rep.candidate_id and rep.duplicate_of is None and dup.resolved_from_best is None
    assert rep.merged == [{'candidate_id': dup.candidate_id, 'source': dup.source, 'logL': dup.score}]
    assert [r.candidate_id for r in res.distinct] == [r.candidate_id for r in res.results if r is not dup]
    # the margin is between distinct trees, never the ~0 gap between the duplicates
    d = res.distinct
    assert res.margin == d[0].score - d[1].score and res.margin > 0.5
    assert dup.candidate_id not in res.unresolved_with_best
    s = res.summary()
    assert s['best_root'] == res.best.root_id and s['n_distinct'] == 2 and s['n_candidates'] == 3
    row = {r['candidate_id']: r for r in res.rows()}
    assert row[dup.candidate_id]['duplicate_of'] == rep.candidate_id
    assert row[rep.candidate_id]['merged_ids'] == [dup.candidate_id]
    assert '= ' + rep.candidate_id in res.table() and 'root_id' in res.to_tsv()


@pytest.mark.parametrize('fx', ['binary20', 'poly20'])
def test_duplicate_merge_never_changes_the_winner(fx):
    """Merging relabels and changes margins only: order, scores and the winning fitted tree are those
    without merging (collapse_tol < 0 disables it)."""
    b = make_baseline(fx)
    res = rank_root_candidates(b, fit=FIT1)
    plain = copy.copy(res)
    plain.results = [copy.copy(r) for r in res.results]
    merge_duplicate_roots(b, plain.results, tol=-1.0)
    assert [r.candidate_id for r in res.results] == [r.candidate_id for r in plain.results]
    assert [r.score for r in res.results] == [r.score for r in plain.results]
    assert res.best.best_run is plain.best.best_run and res.best.duplicate_of is None
    assert all(r.duplicate_of is None for r in plain.results)
    # every duplicate is the same fitted tree as its representative, within fit noise
    rep = {r.candidate_id: r for r in res.results}
    for r in res.results:
        if r.duplicate_of:
            assert rep[r.duplicate_of].root_id == r.root_id and rep[r.duplicate_of].score >= r.score
