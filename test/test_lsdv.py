"""lsdv (treetime.ls_dated_rate): covariation regression with branch variances
from least-squares-dated durations, and its opt-in uses in root search: the
shared rate (UnrootedBaseline(shared_rate='lsdv')), a candidate source
(LsdvSource) and the final rate report (rank_root_candidates(final_lsdv=)).
The defaults are checked to be unchanged."""
import copy

import numpy as np
import pytest
from Bio.Phylo.BaseTree import Clade, Tree

from test_root_candidates import load
from treetime import TreeTime
from treetime import ls_dated_rate as ld
from treetime.root_candidates import (EdgeRoot, FitConfig, LsdvSource, NodeRoot, UnrootedBaseline, candidate_key,
                                      default_candidate_sources, plain_candidate_sources, rank_root_candidates,
                                      regression_candidates)

L = 1000
FIT1 = FitConfig(iterations=1)


def rows(res):
    """Result rows without wall-clock times."""
    return [{k: v for k, v in r.items() if k != 'runtime'} for r in res.rows()]


# ---------------------------------------------------------------- simulated trees (as in report 88's tests)

def simulated_tree(n_tips=60, seed=1, internal_min=0, mu=2e-3, span=10.0):
    """Random binary tree with Poisson counts on its edges (k / L as length).
    ``internal_min`` counts are added to internal edges (1 keeps them uncollapsed)."""
    rng = np.random.default_rng(seed)
    nodes = [(Clade(name=f'T{i:03d}'), 2020.0 + span * rng.random()) for i in range(n_tips)]
    t = min(d for _, d in nodes)
    k = 0
    while len(nodes) > 1:
        t -= rng.exponential(0.5)
        i, j = sorted(rng.choice(len(nodes), 2, replace=False))
        (a, ta), (b, tb) = nodes[i], nodes[j]
        tn = min(t, ta - 0.01, tb - 0.01)
        c = Clade(clades=[a, b], name=f'N{k:03d}')
        k += 1
        a.branch_length = (rng.poisson(mu * L * (ta - tn)) + (internal_min if a.clades else 0)) / L
        b.branch_length = (rng.poisson(mu * L * (tb - tn)) + (internal_min if b.clades else 0)) / L
        nodes = [x for q, x in enumerate(nodes) if q not in (i, j)] + [(c, tn)]
    root = nodes[0][0]
    root.branch_length = 0.0
    return Tree(root=root, rooted=True)


def sim(n_tips=60, seed=1, internal_min=0):
    tree = simulated_tree(n_tips, seed, internal_min)
    rng = np.random.default_rng(seed)
    d = 2020.0 + 10.0 * rng.random(n_tips)
    return tree, {f'T{i:03d}': float(d[i]) for i in range(n_tips)}


def tt_from_tree(tree, dates):
    tt = TreeTime(tree=tree, aln=None, seq_len=L, dates=dates, verbose=0, gtr='JC69', rng_seed=7)
    for n in tt.tree.find_clades():
        n.mutation_length = n.branch_length
    return tt


def dense_ls_durations(f, mu):
    """Independent version of _ls_durations: dense least squares for the free node positions."""
    par, tip = np.asarray(f.par), np.asarray(f.tip)
    dates, ml = np.asarray(f.date, float), np.asarray(f.ml, float)
    n = len(par)
    fixed = tip & np.isfinite(dates)
    free = np.flatnonzero(~fixed)
    col = {v: i for i, v in enumerate(free)}
    A, y = np.zeros((n - 1, len(free))), np.zeros(n - 1)
    for r, v in enumerate(range(1, n)):  # ml_v = x_v - x_par
        y[r] = ml[v]
        for node, sgn in ((v, 1.0), (par[v], -1.0)):
            if fixed[node]:
                y[r] -= sgn * mu * dates[node]
            else:
                A[r, col[node]] += sgn
    x = np.where(fixed, mu * dates, 0.0)
    x[free] = np.linalg.lstsq(A, y, rcond=None)[0]
    t = x / mu
    eps = 1e-6 * np.ptp(dates[fixed])
    for v in range(n - 1, 0, -1):
        t[par[v]] = min(t[par[v]], t[v] - eps)
    return np.maximum(np.r_[0.0, mu * (t[1:] - t[par[1:]])], 0.0)


# ---------------------------------------------------------------- the estimator

def test_ls_durations_match_dense_least_squares():
    tree, dates = sim(40, seed=5, internal_min=1)
    f = ld._flatten(tt_from_tree(tree, dates).tree, 1.0 / L, collapse=False)
    mu = ld._plain_regression(f)[0]
    assert np.allclose(ld._ls_durations(f, mu), dense_ls_durations(f, mu), atol=1e-12)


@pytest.mark.parametrize('seed,n', [(3, 50), (8, 25), (11, 100)])
def test_lsdv_equals_treetime_covariation_regression_with_ls_clock_length(seed, n):
    tree, dates = sim(n, seed=seed, internal_min=1)
    tt = tt_from_tree(tree, dates)
    est = ld.ls_dated_variance_rate(tt, tip_slack=3.0, collapse=False)
    f = ld._flatten(tt.tree, 1.0 / L, collapse=False)
    by_name = dict(zip(f.name, dense_ls_durations(f, est['plain_rate'])))
    named = [x for x in tt.tree.find_clades() if x.up is not None]
    assert all(x.name in by_name for x in named)
    for x in named:
        x.clock_length = by_name[x.name]
    tt.tip_slack = 3.0
    r = tt.setup_TreeRegression(covariation=True).regression()
    assert est['rate'] == pytest.approx(r['slope'], rel=1e-9)
    assert est['intercept'] == pytest.approx(r['intercept'], rel=1e-9, abs=1e-12)
    assert est['chisq'] == pytest.approx(r['chisq'], rel=1e-7)


def test_lsdv_rounds_reference_rate_and_errors():
    tree, dates = sim(60, seed=2)
    tt = tt_from_tree(tree, dates)
    one = ld.ls_dated_variance_rate(tt)
    two = ld.ls_dated_variance_rate(tt, rounds=2)
    assert two['round_rates'][0] == pytest.approx(one['rate'], rel=1e-12)
    assert ld.ls_dated_variance_rate(tt, rate=one['rate'])['rate'] == pytest.approx(two['rate'], rel=1e-12)
    with pytest.raises(ValueError):
        ld.ls_dated_variance_rate(tt, rate=-1.0)
    with pytest.raises(ValueError):
        ld.ls_dated_variance_rate(tt, tip_slack=-1.0)


def test_lsdv_default_reference_rate_is_treetime_plain_regression():
    tree, dates = sim(50, seed=6)
    tt = tt_from_tree(tree, dates)
    est = ld.ls_dated_variance_rate(tt)
    plain = tt.setup_TreeRegression(covariation=False).regression()['slope']
    assert est['plain_rate'] == pytest.approx(plain, rel=1e-9)
    assert ld.ls_dated_variance_rate(tt, rate=plain)['rate'] == pytest.approx(est['rate'], rel=1e-12)


def test_lsdv_outlier_tip_carries_no_date_and_tree_is_untouched():
    tree, dates = sim(40, seed=4, internal_min=1)
    tt = tt_from_tree(copy.deepcopy(tree), dates)
    tip = tt.tree.get_terminals()[5]
    tip.bad_branch = True
    before = [(x.name, x.branch_length, x.mutation_length) for x in tt.tree.find_clades()]
    flagged = ld.ls_dated_variance_rate(tt)
    assert [(x.name, x.branch_length, x.mutation_length) for x in tt.tree.find_clades()] == before
    tt2 = tt_from_tree(copy.deepcopy(tree), {k: v for k, v in dates.items() if k != tip.name})
    assert flagged['rate'] == pytest.approx(ld.ls_dated_variance_rate(tt2)['rate'], rel=1e-12)


def test_lsdv_collapse_removes_zero_length_internal_edges():
    tree, dates = sim(60, seed=2)  # internal_min=0: zero-count internal edges exist
    tt = tt_from_tree(tree, dates)
    nc = ld.ls_dated_variance_rate(tt, collapse=False)
    co = ld.ls_dated_variance_rate(tt)
    n_zero = sum(1 for x in tt.tree.get_nonterminals() if x.up is not None and x.mutation_length == 0)
    assert n_zero > 0 and nc['n_nodes'] - co['n_nodes'] == n_zero


# ---------------------------------------------------------------- baseline: lsdv at a placement, shared rate

def baseline(fx, **kw):
    nwk, aln, dates, _ = load(fx)
    return UnrootedBaseline(nwk, aln, dates, gtr='JC69', **kw)


@pytest.fixture(scope='module')
def b60():
    return baseline('mixed60')


@pytest.fixture(scope='module')
def b20():
    return baseline('binary20')


@pytest.mark.parametrize('fx', ['binary20', 'mixed60', 'poly20'])
def test_baseline_lsdv_equals_lsdv_on_a_treetime_rooted_there(fx):
    b = baseline(fx)
    cands = [c for c, _ in regression_candidates(b, k=3, slope=None)]
    cands += [b.canonical(NodeRoot(b.split_of[b.internal_nodes()[0]]))]
    n_ok = 0
    for c in cands:
        tt = b.make_treetime(b.rooted_phylo(c), messages=False)
        try:
            x = b.lsdv(c)
        except ValueError:  # plain rate not positive here: both raise
            with pytest.raises(ValueError):
                ld.ls_dated_variance_rate(tt)
            continue
        n_ok += 1
        y = ld.ls_dated_variance_rate(tt)
        for k in ('rate', 'intercept', 'chisq', 'plain_rate'):
            assert x[k] == pytest.approx(y[k], rel=1e-8), (c, k)
    assert n_ok >= 3


def test_default_shared_rate_is_the_regression_slope(b60):
    best = b60.regression().find_best_root(force_positive=True)
    assert b60.clock_rate == float(best['slope'])
    assert b60.shared_rate_info is None


def test_shared_rate_argument_checks():
    with pytest.raises(ValueError):
        baseline('binary20', shared_rate='lsdv', clock_rate=1e-3)
    with pytest.raises(ValueError):
        baseline('binary20', shared_rate='datp')
    with pytest.raises(ValueError):
        baseline('binary20', shared_rate='lsdv', shared_rate_k=0)


def test_lsdv_shared_rate_at_the_regression_best_and_median_over_top_k(b60):
    b1 = baseline('mixed60', shared_rate='lsdv')
    top = regression_candidates(b60, k=1, slope=None)[0][0]
    assert b1.clock_rate == b60.lsdv(top)['rate']
    info = b1.shared_rate_info
    assert info['source'] == 'lsdv' and info['k'] == 1 and info['regression_rate'] == b60.clock_rate
    assert info['log_range'] == 0.0 and info['fallback'] is None
    b5 = baseline('mixed60', shared_rate='lsdv', shared_rate_k=5)
    rates = [r['lsdv_rate'] for r in b5.shared_rate_info['candidates']]
    assert len(rates) == 5 and b5.clock_rate == pytest.approx(np.median(rates), rel=1e-15)
    assert b5.shared_rate_info['log_range'] == pytest.approx(np.log(max(rates) / min(rates)))
    assert [r['candidate_id'] for r in b5.shared_rate_info['candidates']] == [
        b60.candidate_id(c) for c, _ in regression_candidates(b60, k=5, slope=None)]
    b3 = baseline('mixed60', shared_rate='lsdv', shared_rate_kwargs={'rounds': 2})
    assert b3.clock_rate == b60.lsdv(top, rounds=2)['rate'] != b1.clock_rate


def test_shared_rate_option_changes_only_the_rate_passed_to_ranking(b20):
    bl = baseline('binary20', shared_rate='lsdv')
    assert bl.clock_rate != b20.clock_rate
    bf = baseline('binary20', clock_rate=bl.clock_rate)  # default path at the lsdv rate
    assert bl.adj == bf.adj and bl.splits == bf.splits
    kw = dict(regression_k=2, fit=FIT1, keep_states=False)
    rl, rf = rank_root_candidates(bl, **kw), rank_root_candidates(bf, **kw)
    assert rows(rl) == rows(rf)
    sl, sf = rl.summary(), rf.summary()
    assert rl.config.pop('shared_rate_info') == bl.shared_rate_info
    assert rl.config == rf.config
    assert sl.pop('shared_rate_source') == 'lsdv' and sl.pop('shared_rate_log_range') == 0.0
    assert sl == sf


# ---------------------------------------------------------------- candidate source

def test_lsdv_source_returns_valid_ranked_candidates(b60):
    keys = {candidate_key(b60.canonical(c)) for c in b60.all_candidates()}
    for src in (LsdvSource(), LsdvSource(k=8, rerank=None), LsdvSource(k=None, rerank=5)):
        cands = src.candidates(b60)
        assert cands and (src.k is None or len(cands) <= src.k)
        assert all(b60.canonical(c) == c and candidate_key(c) in keys for c, _ in cands)
        assert len({candidate_key(c) for c, _ in cands}) == len(cands)
        chisq = [i['regression_chisq'] for _, i in cands]
        assert chisq == sorted(chisq) and [i['regression_rank'] for _, i in cands] == list(range(1, len(cands) + 1))
        for c, i in cands:
            x = b60.lsdv(c)
            assert i['lsdv_rate'] == x['rate'] > 0 and i['regression_chisq'] == x['chisq']
    assert len(LsdvSource(k=None, rerank=5).candidates(b60)) <= 5
    scan = LsdvSource(k=None, rerank=None).candidates(b60)
    rr = LsdvSource(k=None, rerank=20).candidates(b60)
    assert scan[0][1]['regression_chisq'] <= rr[0][1]['regression_chisq']
    assert any(i['base_rank'] is None for _, i in scan)


def test_lsdv_source_composes_with_the_default_sources(b20):
    srcs = default_candidate_sources(2) + (LsdvSource(k=3),)
    res = rank_root_candidates(b20, candidate_sources=srcs, fit=FIT1, keep_states=False)
    labels = set()
    for r in res.results:
        labels |= set(r.source_ranks)
        assert r.source == '+'.join(r.source_ranks)
    assert 'lsdv' in labels and 'plain' in labels
    lsdv_keys = {candidate_key(c) for c, _ in LsdvSource(k=3).candidates(b20)}
    assert lsdv_keys == {candidate_key(r.candidate) for r in res.results if 'lsdv' in r.source_ranks}
    assert res.config['candidate_sources'][-1]['name'] == 'lsdv'
    with pytest.raises(ValueError):  # names must stay distinct
        rank_root_candidates(b20, candidate_sources=(LsdvSource(name='plain'),) + plain_candidate_sources(2),
                             fit=FIT1)


# ---------------------------------------------------------------- final rate report

def test_final_lsdv_reports_without_changing_the_ranking(b20):
    kw = dict(regression_k=2, fit=FIT1, keep_states=False, candidate_sources=plain_candidate_sources(2))
    off, on = rank_root_candidates(b20, **kw), rank_root_candidates(b20, final_lsdv=True, **kw)
    assert off.final_lsdv is None and 'final_lsdv' not in off.summary() and 'final_lsdv' not in off.config
    assert rows(on) == rows(off)
    x = b20.lsdv(on.best.root_candidate)
    fl = on.final_lsdv
    assert fl['rate'] == x['rate'] and fl['root_date'] == x['root_date'] and fl['root_id'] == on.best.root_id
    assert on.summary()['final_lsdv']['rate'] == x['rate']
    two = rank_root_candidates(b20, final_lsdv={'rounds': 2}, **kw)
    assert two.final_lsdv['rate'] == b20.lsdv(two.best.root_candidate, rounds=2)['rate']


def test_default_output_has_no_new_fields(b20):
    res = rank_root_candidates(b20, regression_k=1, fit=FIT1, keep_states=False,
                               candidate_sources=plain_candidate_sources(1))
    assert 'shared_rate_info' not in res.config and 'final_lsdv' not in res.config
    assert not {'final_lsdv', 'shared_rate_source', 'shared_rate_log_range'} & set(res.summary())
    assert res.final_lsdv is None


def test_stage2_with_lsdv_shared_rate_and_lsdv_source():
    from treetime.root_candidates import outgroup_candidate_sources
    from treetime.root_messages import rank_with_root_messages
    b = baseline('binary20', shared_rate='lsdv')
    msg = {0: np.ones(len(b.gtr.alphabet))}  # one neutral site
    res = rank_with_root_messages(b, msg, regression_k=2, fit=FIT1, final_lsdv=True,
                                  candidate_sources=outgroup_candidate_sources(2) + (LsdvSource(k=2),))
    assert res.stage1.clock_rate == b.clock_rate == res.stage2b.clock_rate
    assert res.stage2b.config['shared_rate_info']['source'] == 'lsdv'
    assert res.stage2b.final_lsdv['rate'] > 0
    assert any('lsdv' in r.source_ranks for r in res.stage2b.results)
