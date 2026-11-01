"""lsdv defaults of report 107: ladder resolution (cross-fitted ladder on large trees with a weak root-to-tip signal),
LSD2's outlier rule at z = 3 with the clade extension, SE inflation 2, one round, no internal b. Each feature can be
turned off, report 93's estimator is reachable with LSDV_REPORT93, and the pre-108 package's outputs are reproduced
(reference values from 4818c34ba, investigation report 108, 04-tests/old_refs.py)."""
import numpy as np
import pytest

from test_lsdv import FIT1, L, REPORT93_DEFAULTS, baseline, sim, simulated_tree, tt_from_tree
from treetime import ls_dated_rate as ld
from treetime.root_candidates import LsdvSource, plain_candidate_sources, rank_root_candidates, regression_candidates

R93 = ld.LSDV_REPORT93
REC103 = dict(resolve='auto', resolve_auto_tips=2000, resolve_auto_r2=None, outlier_z=3.5, outlier_clade_max=None)
# old option dict -> the same estimator under the report-107 defaults
OLD_TO_NEW = {'r93': R93, 'rec103': REC103,
              'ladder_z3': dict(resolve='ladder', outlier_clade_max=None),
              'xo': dict(resolve='ladder_xo', outlier_z=None),
              'z3': dict(resolve=None, outlier_clade_max=None)}
# (seed, n, internal_min, planted tip) -> (rate, intercept, chisq, rate_se, n outliers) from the 4818c34ba package
OLD = {
    (2, 80, 0, None, 'r93'): (0.0019693985331516468, -3.922934010996987, 25.212162998892882, 0.00027720219153957207, 0),
    (2, 80, 0, None, 'rec103'): (0.0019657537143121058, -3.915478176896223, 25.022973193533584, 0.0002776446361387844, 0),  # noqa: E501
    (2, 80, 0, None, 'ladder_z3'): (0.0019657537143121058, -3.915478176896223, 25.022973193533584, 0.0002776446361387844, 0),  # noqa: E501
    (2, 80, 0, None, 'xo'): (0.0019651453522371363, -3.9143061358745346, 25.058690382183993, 0.0002778201634405592, 0),
    (2, 80, 0, None, 'z3'): (0.0019693985331516468, -3.922934010996987, 25.212162998892882, 0.00027720219153957207, 0),
    (5, 80, 0, None, 'r93'): (0.001975652436856015, -3.920120583630016, 30.00215698894891, 0.00026853578841684757, 0),
    (5, 80, 0, None, 'rec103'): (0.001975197704109129, -3.9191447917725766, 29.712144648788904, 0.000269027662002481, 0),  # noqa: E501
    (5, 80, 0, None, 'ladder_z3'): (0.001975197704109129, -3.9191447917725766, 29.712144648788904, 0.000269027662002481, 0),  # noqa: E501
    (5, 80, 0, None, 'xo'): (0.001975729746788003, -3.9202322029750887, 29.728622612398112, 0.0002686176116587481, 0),
    (5, 80, 0, None, 'z3'): (0.001975652436856015, -3.920120583630016, 30.00215698894891, 0.00026853578841684757, 0),
    (6, 80, 1, 5, 'r93'): (0.0013888822687443813, -2.70746920638529, 42.07248060498298, 0.00023986980037713274, 0),
    (6, 80, 1, 5, 'rec103'): (0.002175893867997748, -4.3011041421861975, 31.534116171436544, 0.00029256558996085605, 1),
    (6, 80, 1, 5, 'ladder_z3'): (0.002146180388467085, -4.2407469994639895, 24.408849047218073, 0.0003023978959451196, 5),  # noqa: E501
    (6, 80, 1, 5, 'xo'): (0.0013888822687443806, -2.7074692063852885, 42.07248060498298, 0.00023986980037713263, 0),
    (6, 80, 1, 5, 'z3'): (0.002146180388467085, -4.2407469994639895, 24.408849047218073, 0.0003023978959451196, 5),
    (4, 120, 0, None, 'r93'): (0.0021014658603225557, -4.125128607128734, 49.80173216304399, 0.00026593807362443263, 0),
    (4, 120, 0, None, 'rec103'): (0.002198751241777331, -4.320831131695704, 42.06502095934006, 0.00027071785147384354, 2),  # noqa: E501
    (4, 120, 0, None, 'ladder_z3'): (0.002157412503828101, -4.237722136987799, 37.82691193251192, 0.00027118866772285987, 4),  # noqa: E501
    (4, 120, 0, None, 'xo'): (0.0020961189034650186, -4.113746530161512, 49.52541788985465, 0.00026709617430802063, 0),
    (4, 120, 0, None, 'z3'): (0.0021621150598289105, -4.2477802641571065, 38.0155882228264, 0.0002700011859658235, 4),
}


def flat(n=80, seed=2, internal_min=0, planted=None, years=20.0):
    tree, dates = sim(n, seed=seed, internal_min=internal_min)
    f = ld._flatten(tt_from_tree(tree, dates).tree, 1.0 / L)
    if planted is not None:
        date = np.asarray(f.date, float).copy()
        tips = [v for v in range(len(f.par)) if f.tip[v]]
        date[tips[planted]] += years
        f = ld._with(f, date=date)
    return f


def big_flat(n, seed, shuffle=False, span=100.0):
    """Simulated flat tree of n tips without a TreeTime object (fast at 1000+ tips); a long sampling span keeps the
    root-to-tip signal strong at this size."""
    tree = simulated_tree(n, seed, span=span)
    d = 2020.0 + span * np.random.default_rng(seed).random(n)  # the dates simulated_tree drew
    dates = {f'T{i:03d}': float(d[i]) for i in range(n)}
    if shuffle:
        names = sorted(dates)
        vals = np.random.default_rng(seed).permutation([dates[k] for k in names])
        dates = dict(zip(names, vals))
    return ld._flatten(tree, 1.0 / L, length=lambda c: c.branch_length, date=lambda c: dates.get(c.name, np.nan))


def cherries(f):
    ch = [[] for _ in f.par]
    for v in range(1, len(f.par)):
        ch[f.par[v]].append(v)
    return ch, [v for v in range(1, len(f.par)) if len(ch[v]) == 2 and all(f.tip[c] for c in ch[v])]


def test_defaults_are_the_report107_spec():
    o = ld.LSDV_OPTIONS
    assert (o['resolve'], o['resolve_auto_tips'], o['resolve_auto_r2']) == ('auto', 1000, 0.2)
    assert (o['outlier_z'], o['outlier_mode'], o['outlier_rounds']) == (3.0, 'drop', 5)
    assert (o['outlier_clade_max'], o['outlier_clade_z']) == (4, 3.0)
    assert (o['se_inflation'], o['internal_b'], o['first_pass'], o['resample']) == (2.0, 0.0, 'ls', 0)
    f = flat(120, seed=4)
    a = ld.lsdv_flat(f, L)
    b = ld.lsdv_flat(f, L, resolve='ladder', outlier_z=3.0, outlier_mode='drop', outlier_clade_max=4,
                     outlier_clade_z=3.0, se_inflation=2.0, internal_b=0.0, rounds=1)
    for k in ('rate', 'intercept', 'chisq', 'rate_se', 'outliers', 'n_nodes'):
        assert a[k] == b[k]
    assert a['resolve'] == 'ladder' and len(a['round_rates']) == 1 and a['internal_b'] == 0.0
    assert a['n_nodes'] > a['n_nodes_unresolved'] and len(a['outliers']) > 0
    tt = tt_from_tree(*sim(120, seed=4))
    assert ld.ls_dated_variance_rate(tt)['rate'] == a['rate']


@pytest.mark.parametrize('key', sorted(OLD))
def test_old_configurations_are_reproduced(key):
    seed, n, imin, planted, cfg = key
    r = ld.lsdv_flat(flat(n, seed, imin, planted), L, **OLD_TO_NEW[cfg])
    assert (r['rate'], r['intercept'], r['chisq'], r['rate_se'], len(r.get('outliers', []))) == OLD[key]


def test_each_feature_turns_off():
    f = flat(80, seed=6, internal_min=1, planted=5)
    full = ld.lsdv_flat(f, L)
    no_out = ld.lsdv_flat(f, L, outlier_z=None)
    assert 'outliers' not in no_out and no_out['rate'] == ld.lsdv_flat(f, L, resolve='ladder', outlier_z=None)['rate']
    no_res = ld.lsdv_flat(f, L, resolve=None)
    assert no_res['resolve'] is None and no_res['n_nodes'] == len(f.par) and no_res['outliers'] == full['outliers']
    flag = ld.lsdv_flat(f, L, outlier_mode='flag')
    assert flag['outliers'] == full['outliers'] and flag['rate'] == no_out['rate']
    k1 = ld.lsdv_flat(f, L, se_inflation=1.0)
    assert k1['rate_se'] == full['rate_se'] and k1['rate_ci'][1] < full['rate_ci'][1]
    assert ld.lsdv_flat(f, L, rounds=2)['round_rates'][0] == full['rate']
    assert ld.lsdv_flat(f, L, internal_b=1.0)['rate'] != full['rate']


def test_report93_options_equal_report93_defaults_on_every_entry_point(monkeypatch):
    """LSDV_REPORT93 passed explicitly gives what test_lsdv checks under report 93's defaults."""
    f = flat(80, seed=5)
    tt = tt_from_tree(*sim(80, seed=5))
    b = baseline('binary20', shared_rate='lsdv', shared_rate_kwargs=dict(R93))
    kw = dict(regression_k=2, fit=FIT1, keep_states=False, candidate_sources=plain_candidate_sources(2))
    b20 = baseline('binary20')
    cand = regression_candidates(b20, k=1, covariation=False, slope=None)[0][0]
    exp_flat = ld.lsdv_flat(f, L, **R93)
    exp_tt = ld.ls_dated_variance_rate(tt, **R93)
    exp_row = b.shared_rate_info['candidates'][0]
    exp_fl = rank_root_candidates(b20, final_lsdv=dict(R93), **kw).final_lsdv
    exp_b = b20.lsdv(cand, **R93)
    exp_src = LsdvSource(k=3).candidates(b20)
    monkeypatch.setattr(ld, 'LSDV_OPTIONS', REPORT93_DEFAULTS)
    assert ld.lsdv_flat(f, L)['rate'] == exp_flat['rate']
    assert ld.ls_dated_variance_rate(tt)['rate'] == exp_tt['rate']
    assert baseline('binary20', shared_rate='lsdv').shared_rate_info['candidates'][0] == exp_row
    assert rank_root_candidates(b20, final_lsdv=True, **kw).final_lsdv == exp_fl
    def strip(x):
        return {k: v for k, v in x.items() if k != 'seconds'}
    assert strip(b20.lsdv(cand)) == strip(exp_b)
    # LsdvSource ranks by report 93's lsdv whatever the defaults
    assert [(c, i) for c, i in LsdvSource(k=3).candidates(b20)] == exp_src
    assert set(exp_row) == {'candidate_id', 'regression_rank', 'lsdv_rate', 'plain_rate', 'root_date', 'error'}


def test_defaults_reach_the_shared_rate_and_final_lsdv():
    b20 = baseline('binary20')
    b = baseline('binary20', shared_rate='lsdv')
    cand = regression_candidates(b20, k=1, covariation=False, slope=None)[0][0]
    x = b20.lsdv(cand)
    row = b.shared_rate_info['candidates'][0]
    assert b.clock_rate == x['rate'] == row['lsdv_rate'] and row['resolve'] == x['resolve'] == 'ladder'
    assert row['rate_se'] == x['rate_se'] and row['se_inflation'] == 2.0 and 'outliers' in row
    kw = dict(regression_k=2, fit=FIT1, keep_states=False, candidate_sources=plain_candidate_sources(2))
    fl = rank_root_candidates(b20, final_lsdv=True, **kw).final_lsdv
    assert fl['resolve'] == 'ladder' and fl['rate_se'] > 0 and len(fl['rate_ci']) == 2 and 'outliers' in fl


def test_se_is_reported_by_default():
    f = flat(80, seed=2)
    r = ld.lsdv_flat(f, L)
    assert r['se_inflation'] == 2.0 and r['rate_se'] > 0 and r['root_date_se'] > 0
    h = 1.96 * 2.0 * r['rate_se'] / r['rate']
    assert r['rate_ci'] == pytest.approx([r['rate'] * np.exp(-h), r['rate'] * np.exp(h)])
    assert r['se_rel'] == r['rate_se'] / r['rate']


def test_auto_switches_on_size_and_r2():
    strong = big_flat(1001, seed=3)
    weak = big_flat(1001, seed=3, shuffle=True)
    a = ld.lsdv_flat(strong, L, outlier_z=None)
    assert a['auto_r2'] >= 0.2 and a['resolve'] == 'ladder'
    w = ld.lsdv_flat(weak, L, outlier_z=None)
    assert w['auto_r2'] < 0.2 and w['resolve'] == 'ladder_xo'
    assert w['rate'] == ld.lsdv_flat(weak, L, resolve='ladder_xo', outlier_z=None)['rate']
    # at <= 1000 tips the weak tree stays on the ladder, and R^2 is not computed
    small = big_flat(1000, seed=3, shuffle=True)
    s = ld.lsdv_flat(small, L, outlier_z=None)
    assert s['resolve'] == 'ladder' and 'auto_r2' not in s
    # resolve_auto_r2=None: size alone (report 103's rule)
    assert ld.lsdv_flat(strong, L, outlier_z=None, resolve_auto_r2=None)['resolve'] == 'ladder_xo'
    assert ld.lsdv_flat(strong, L, outlier_z=None, resolve_auto_tips=2000, resolve_auto_r2=None)['resolve'] == 'ladder'


def test_auto_r2_is_computed_after_the_outlier_dates_are_dropped():
    f = flat(120, seed=4, planted=7, years=60.0)
    raw_r2 = ld._plain_regression(f)[2] ** 2
    r = ld.lsdv_flat(f, L, resolve_auto_tips=50)
    date = np.asarray(f.date, float).copy()
    date[r['outlier_nodes']] = np.nan
    dropped_r2 = ld._plain_regression(ld._with(f, date=date))[2] ** 2
    assert r['auto_r2'] == dropped_r2 and dropped_r2 > raw_r2 + 0.1
    thr = (raw_r2 + dropped_r2) / 2  # raw R^2 would switch to ladder_xo, the R^2 after dropping does not
    assert ld.lsdv_flat(f, L, resolve_auto_tips=50, resolve_auto_r2=thr)['resolve'] == 'ladder'
    assert ld.lsdv_flat(f, L, resolve_auto_tips=50, resolve_auto_r2=thr, outlier_z=None)['resolve'] == 'ladder_xo'


@pytest.mark.parametrize('seed,v', [(2, 2), (7, 4)])
def test_clade_extension_catches_a_shared_shift(seed, v):
    f = flat(80, seed=seed, internal_min=1)
    ch, ch2 = cherries(f)
    assert v in ch2
    date = np.asarray(f.date, float).copy()
    date[ch[v]] += 8.0  # one shared error on a 2-tip clade
    g = ld._with(f, date=date)
    names = {f.name[c] for c in ch[v]}
    tip_only = ld.lsdv_flat(g, L, outlier_clade_max=None)
    full = ld.lsdv_flat(g, L)
    assert not names & set(tip_only['outliers'])
    assert names <= set(full['outliers']) and full['outlier_clades'] >= 1
    clean = ld.lsdv_flat(f, L)
    assert abs(np.log(full['rate'] / clean['rate'])) < abs(np.log(tip_only['rate'] / clean['rate']))
    # clade size cap: with outlier_clade_max=4 a 2-tip clade qualifies; the threshold can be raised
    assert not names <= set(ld.lsdv_flat(g, L, outlier_clade_z=50.0)['outliers'])


@pytest.mark.parametrize('seed', [2, 4, 5, 7, 8])
def test_clade_extension_leaves_clean_trees_alone(seed):
    f = flat(80, seed=seed, internal_min=1)
    full, tip_only = ld.lsdv_flat(f, L), ld.lsdv_flat(f, L, outlier_clade_max=None)
    assert full['outlier_clades'] == 0 and full['outliers'] == tip_only['outliers'] and full['rate'] == tip_only['rate']


def test_floor_holds_with_the_clade_extension():
    f = flat(80, seed=6, internal_min=1)
    n0 = int(np.sum(np.asarray(f.tip) & np.isfinite(np.asarray(f.date, float))))
    for z in (0.3, 0.7, 1.0):
        r = ld.lsdv_flat(f, L, outlier_z=z, outlier_clade_z=z)
        assert len(r['outliers']) <= n0 - max(3, n0 // 2) and r['rate'] > 0


def test_new_option_checks():
    f = flat(40, seed=2)
    for bad in (dict(outlier_clade_max=1), dict(outlier_clade_max=2.5), dict(outlier_clade_z=0),
                dict(resolve_auto_r2=1.5), dict(resolve_auto_r2=-0.1)):
        with pytest.raises(ValueError):
            ld.lsdv_flat(f, L, **bad)


@pytest.mark.parametrize('seed', [2, 4, 9])
def test_flat_regression_equals_treeregression_bit_for_bit(seed):
    f = flat(150, seed=seed)
    for g in (f, ld.ladder_resolve(f, ld._plain_regression(f)[0])):
        dur = ld._ls_durations(g, ld._plain_regression(g)[0])
        var = (dur + np.where(np.asarray(g.tip), 9.0 / L, 0.0)) / L
        var[0] = 1.0 / L ** 2
        ml = np.asarray(g.ml, float)
        a, b = ld._Regression(g).fit(ml, var), ld._PhyloRegression(g).fit(ml, var)
        for k in ('slope', 'intercept', 'chisq'):
            assert a[k] == b[k]
        assert np.array_equal(a['cov'], b['cov'])
