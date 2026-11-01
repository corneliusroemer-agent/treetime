"""Opt-in lsdv variants (report 103): ladder resolution of polytomies, the iterated LSD2 outlier rule, an internal
smoothing constant, the model SE, the resampling interval and LSD2's first pass; their wiring into the shared rate
and the final report. Each option at its default leaves lsdv unchanged."""
import copy

import numpy as np
import pytest

from test_lsdv import FIT1, L, baseline, dense_ls_durations, sim, tt_from_tree
from treetime import ls_dated_rate as ld
from treetime.root_candidates import UnrootedBaseline, plain_candidate_sources, rank_root_candidates


def flat(n=80, seed=2, internal_min=0):
    tree, dates = sim(n, seed=seed, internal_min=internal_min)
    tt = tt_from_tree(tree, dates)
    return tt, ld._flatten(tt.tree, 1.0 / L)


def rtt(f):
    par, ml = np.asarray(f.par), np.asarray(f.ml, float)
    d = np.zeros(len(par))
    for v in range(1, len(par)):
        d[v] = d[par[v]] + ml[v]
    return {f.name[v]: d[v] for v in range(len(par)) if f.tip[v]}


def n_children(f):
    c = np.zeros(len(f.par), int)
    for p in f.par[1:]:
        c[p] += 1
    return c


CORE = ('rate', 'intercept', 'chisq', 'plain_rate', 'root_date', 'round_rates', 'n_nodes')


@pytest.mark.parametrize('seed', [2, 5])
def test_explicit_defaults_equal_the_default_fit(seed):
    tt, f = flat(seed=seed)
    a = ld.lsdv_flat(f, L)
    b = ld.lsdv_flat(f, L, **ld.LSDV_OPTIONS)
    c = ld.lsdv_flat(f, L, outlier_z=None, internal_b=0.0, resolve=None, first_pass='ls', resample=0)
    for k in CORE:
        assert a[k] == b[k] == c[k]
    assert 'outliers' not in b and 'resample' not in b
    assert ld.ls_dated_variance_rate(tt)['rate'] == a['rate']


def test_se_matches_treetime_covariance():
    tree, dates = sim(50, seed=3, internal_min=1)
    tt = tt_from_tree(tree, dates)
    est = ld.ls_dated_variance_rate(tt, collapse=False)
    f = ld._flatten(tt.tree, 1.0 / L, collapse=False)
    by_name = dict(zip(f.name, dense_ls_durations(f, est['plain_rate'])))
    for x in tt.tree.find_clades():
        if x.up is not None:
            x.clock_length = by_name[x.name]
    tt.tip_slack = 3.0
    r = tt.setup_TreeRegression(covariation=True).regression()
    assert est['rate_se'] == pytest.approx(np.sqrt(r['cov'][0, 0]), rel=1e-7)
    assert np.allclose(est['cov'], r['cov'], rtol=1e-7)
    h = 1.96 * 2 * est['rate_se'] / est['rate']
    assert est['rate_ci'] == pytest.approx([est['rate'] * np.exp(-h), est['rate'] * np.exp(h)])
    k1 = ld.ls_dated_variance_rate(tt, collapse=False, se_inflation=1.0)
    assert k1['rate_se'] == est['rate_se'] and k1['rate_ci'][1] < est['rate_ci'][1]
    assert est['root_date_se'] > 0


@pytest.mark.parametrize('seed', [2, 4, 9])
def test_ladder_makes_a_binary_tree_and_keeps_tips(seed):
    tt, f = flat(120, seed=seed)
    assert n_children(f).max() > 2
    mu = ld._plain_regression(f)[0]
    R = ld.ladder_resolve(f, mu)
    nc = n_children(R)
    assert nc.max() <= 2 and all(nc[v] == 2 for v in range(len(R.par)) if not R.tip[v])
    assert sum(R.tip) == sum(f.tip)
    a, b = rtt(f), rtt(R)
    assert set(a) == set(b) and all(abs(a[k] - b[k]) < 1e-12 for k in a)
    da = {f.name[v]: f.date[v] for v in range(len(f.par)) if f.tip[v]}
    db = {R.name[v]: R.date[v] for v in range(len(R.par)) if R.tip[v]}
    assert all(da[k] == db[k] or (np.isnan(da[k]) and np.isnan(db[k])) for k in da)
    res = ld.lsdv_flat(f, L, resolve='ladder')
    assert res['n_nodes'] == len(R.par) and res['n_nodes_unresolved'] == len(f.par)
    assert res['plain_rate'] == pytest.approx(ld.lsdv_flat(f, L)['plain_rate'], rel=1e-12)


def test_ladder_on_a_binary_tree_changes_nothing():
    tt, f = flat(60, seed=3, internal_min=1)
    assert n_children(f).max() <= 2
    a, b = ld.lsdv_flat(f, L), ld.lsdv_flat(f, L, resolve='ladder')
    assert b['rate'] == pytest.approx(a['rate'], rel=1e-12)


def test_ladder_xo_and_auto():
    tt, f = flat(120, seed=4)
    xo = ld.lsdv_flat(f, L, resolve='ladder_xo')
    assert len(xo['xo_rates']) == 4 and xo['rate'] == pytest.approx(np.exp(np.mean(np.log(xo['xo_rates']))))
    assert xo['rate_ci'][0] < xo['rate'] < xo['rate_ci'][1]
    assert ld.lsdv_flat(f, L, resolve='auto')['resolve'] == 'ladder'
    auto = ld.lsdv_flat(f, L, resolve='auto', resolve_auto_tips=50)
    assert auto['resolve'] == 'ladder_xo' and auto['rate'] == xo['rate']


def test_outlier_rule_off_changes_nothing_and_finds_a_planted_date_error():
    tt, f = flat(80, seed=6, internal_min=1)
    a = ld.lsdv_flat(f, L)
    assert ld.lsdv_flat(f, L, outlier_z=None, outlier_mode='flag')['rate'] == a['rate']
    date = np.asarray(f.date, float).copy()
    tips = [v for v in range(len(f.par)) if f.tip[v]]
    date[tips[5]] += 20.0
    g = ld._with(f, date=date)
    fl = ld.lsdv_flat(g, L, outlier_z=3.0, outlier_mode='flag')
    dr = ld.lsdv_flat(g, L, outlier_z=3.0)
    assert f.name[tips[5]] in fl['outliers'] and fl['outliers'] == dr['outliers']
    assert fl['rate'] == ld.lsdv_flat(g, L)['rate']
    dropped = date.copy()
    dropped[fl['outlier_nodes']] = np.nan
    assert dr['rate'] == ld.lsdv_flat(ld._with(f, date=dropped), L)['rate']
    assert abs(np.log(dr['rate'] / a['rate'])) < abs(np.log(fl['rate'] / a['rate']))
    huge = ld.lsdv_flat(g, L, outlier_z=50.0)
    assert huge['outliers'] == [] and huge['rate'] == ld.lsdv_flat(g, L)['rate']


def test_internal_b_zero_changes_nothing_and_b_matches_closed_form():
    tt, f = flat(80, seed=7)
    a = ld.lsdv_flat(f, L)
    assert ld.lsdv_flat(f, L, internal_b=0.0)['rate'] == a['rate']
    b = ld.lsdv_flat(f, L, internal_b=5.0)
    par, tip, date, y = ld._arrays(f)
    V = ld._ls_durations(f, a['plain_rate']) + np.where(tip, 9.0 / L, 5.0 / L)
    mu, _ = ld._wls_positions(par, tip, date, y, 1.0 / V)
    assert b['rate'] == pytest.approx(mu, rel=1e-7)
    assert b['rate'] != a['rate']


def test_first_pass_lsd2():
    tt, f = flat(80, seed=8)
    r = ld.lsdv_flat(f, L, first_pass='lsd2')
    par, tip, date, y = ld._arrays(f)
    mu1, x = ld._wls_positions(par, tip, date, y, 1.0 / (y + ld.lsd2_b(f, L)))
    assert r['first_pass_rate'] == pytest.approx(mu1)
    V = np.maximum(np.r_[0.0, x[1:] - x[par[1:]]], 0) + np.where(tip, 9.0 / L, 0.0)
    V[0] = 1.0
    assert r['rate'] == pytest.approx(ld._wls_positions(par, tip, date, y, 1.0 / V)[0], rel=1e-7)


def test_resampling_interval():
    tt, f = flat(60, seed=2)
    r = ld.lsdv_flat(f, L, resolve='ladder', resample=30, seed=3)
    s = r['resample']
    assert s['n'] == 30 and s['n_ok'] >= 25
    assert s['rate_ci'][0] <= r['rate'] <= s['rate_ci'][1]
    assert s['root_date_ci'][0] <= r['root_date'] <= s['root_date_ci'][1]
    assert ld.lsdv_flat(f, L, resolve='ladder', resample=30, seed=3)['resample'] == s


def test_option_checks():
    tt, f = flat(40, seed=2)
    for bad in (dict(resolve='x'), dict(outlier_z=-1), dict(outlier_mode='soft'), dict(internal_b=-1),
                dict(first_pass='x'), dict(se_inflation=0), dict(resample=-1), dict(bogus=1)):
        with pytest.raises(ValueError):
            ld.lsdv_flat(f, L, **bad)


@pytest.fixture(scope='module')
def b20():
    return baseline('binary20')


def test_shared_rate_and_final_lsdv_take_the_options(b20):
    opts = dict(resolve='ladder', outlier_z=3.0, internal_b=1.0)
    b = baseline('binary20', shared_rate='lsdv', shared_rate_kwargs=opts)
    row = b.shared_rate_info['candidates'][0]
    from treetime.root_candidates import regression_candidates
    cand = regression_candidates(b20, k=1, covariation=False, slope=None)[0][0]
    x = b20.lsdv(cand, **opts)
    assert b.clock_rate == x['rate'] == row['lsdv_rate'] and row['rate_se'] == x['rate_se']
    assert row['resolve'] == 'ladder' and 'outliers' in row
    with pytest.raises(ValueError):
        baseline('binary20', shared_rate='lsdv', shared_rate_kwargs=dict(resolve='bad'))
    kw = dict(regression_k=2, fit=FIT1, keep_states=False, candidate_sources=plain_candidate_sources(2))
    off = rank_root_candidates(b20, **kw)
    on = rank_root_candidates(b20, final_lsdv=dict(opts, se_inflation=1.5), **kw)
    assert [r['candidate_id'] for r in on.rows()] == [r['candidate_id'] for r in off.rows()]
    fl = on.final_lsdv
    y = b20.lsdv(on.best.root_candidate, se_inflation=1.5, **opts)
    assert fl['rate'] == y['rate'] and fl['rate_se'] == y['rate_se'] and fl['rate_ci'] == y['rate_ci']
    assert fl['se_inflation'] == 1.5 and on.summary()['final_lsdv']['rate_se'] == y['rate_se']
    with pytest.raises(ValueError):
        rank_root_candidates(b20, final_lsdv=dict(outlier_z=0), **kw)
