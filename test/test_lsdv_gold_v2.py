"""Gold defaults v2 (investigation report 116): lsdv outlier rule and clade extension at z = 4.5 (report 111)."""
import numpy as np
import pytest

from test_lsdv import L, sim, simulated_tree, tt_from_tree
from test_lsdv_defaults import big_flat, flat
from treetime import ls_dated_rate as ld


def test_outlier_thresholds_are_4_5():
    o = ld.LSDV_OPTIONS
    assert (o['outlier_z'], o['outlier_clade_z'], o['outlier_clade_max']) == (4.5, 4.5, 4)
    assert ld.LSDV_REPORT108 == dict(outlier_z=3.0, outlier_clade_z=3.0)
    f = flat(120, seed=4)
    a = ld.lsdv_flat(f, L)
    b = ld.lsdv_flat(f, L, outlier_z=4.5, outlier_clade_z=4.5)
    for k in ('rate', 'intercept', 'chisq', 'rate_se', 'outliers', 'n_nodes'):
        assert a[k] == b[k]
    tt = tt_from_tree(*sim(120, seed=4))
    assert ld.ls_dated_variance_rate(tt)['rate'] == a['rate']


def test_report108_options_reproduce_z3():
    f = flat(80, seed=6, internal_min=1, planted=5)
    a = ld.lsdv_flat(f, L, **ld.LSDV_REPORT108)
    b = ld.lsdv_flat(f, L, outlier_z=3.0, outlier_clade_z=3.0)
    assert (a['rate'], a['outliers']) == (b['rate'], b['outliers'])


def test_fewer_false_flags_and_gross_errors_still_caught():
    n4, n3 = 0, 0
    for seed in (2, 4, 5):
        f = big_flat(1000, seed)
        n4 += len(ld.lsdv_flat(f, L)['outliers'])
        n3 += len(ld.lsdv_flat(f, L, **ld.LSDV_REPORT108)['outliers'])
    assert n4 < n3 and n4 <= 0.005 * 3000
    g = flat(80, seed=6, internal_min=1, planted=5, years=60.0)
    tip = [v for v in range(len(g.par)) if g.tip[v]][5]
    assert g.name[tip] in ld.lsdv_flat(g, L)['outliers']


# ---- lsdv is the default shared rate and the headline -------------------------------------------------------------
def _b(fx='binary20', **kw):
    from test_root_candidates import load
    from treetime.root_candidates import UnrootedBaseline
    nwk, aln, dates, _ = load(fx)
    return UnrootedBaseline(nwk, aln, dates, gtr='JC69', **kw)


def test_default_shared_rate_is_lsdv():
    from treetime.root_candidates import regression_candidates
    b, br = _b(), _b(shared_rate='regression')
    assert b.shared_rate_info['source'] == 'lsdv' and br.shared_rate_info is None
    top = regression_candidates(br, k=1, covariation=False, slope=None)[0][0]
    assert b.clock_rate == br.lsdv(top)['rate'] != br.clock_rate
    assert br.clock_rate == float(br.regression().find_best_root(force_positive=True)['slope'])


def test_default_headline_is_final_lsdv():
    from test_lsdv import FIT1
    from treetime.root_candidates import plain_candidate_sources, rank_root_candidates
    b = _b()
    kw = dict(regression_k=2, fit=FIT1, keep_states=False, candidate_sources=plain_candidate_sources(2))
    on, off = rank_root_candidates(b, **kw), rank_root_candidates(b, final_lsdv=False, **kw)
    assert [r.score for r in on.results] == [r.score for r in off.results]
    fl = on.final_lsdv
    assert fl['rate'] == b.lsdv(on.best.root_candidate)['rate'] and fl['error'] is None
    assert on.headline_source == 'final_lsdv' and on.headline_clock_rate == fl['rate']
    assert on.headline_root_date == fl['root_date'] == on.summary()['headline_root_date']
    assert off.headline_source == 'fit' and off.final_lsdv is None and 'headline_root_date' not in off.summary()
    assert off.headline_root_date == off.best.best_run.root['root_date']


# ---- partial (range) dates ---------------------------------------------------------------------------------------
def _year_only(f, idx, true_shift=0.0):
    """Flat tree with tips ``idx`` (positions among the tips) turned into year-only dates [floor(d), floor(d) + 1],
    published as the midpoint; ``true_shift`` moves the true date first (a wrong date)."""
    par, tip, date, y = ld._arrays(f)
    date = date.copy()
    dv = np.zeros(len(par))
    tips = [v for v in range(len(par)) if tip[v]]
    for i in idx:
        v = tips[i]
        lo = np.floor(date[v] + true_shift)
        date[v], dv[v] = lo + 0.5, 1.0 / 12
    return ld._from_arrays(par, tip, date, y, f.name, dv), [tips[i] for i in idx]


def test_exact_dates_unchanged_by_the_range_handling():
    f = flat(80, seed=6, internal_min=1, planted=5)
    a, b = ld.lsdv_flat(f, L), ld.lsdv_flat(f, L, date_ranges='midpoint')
    a.pop('seconds'); b.pop('seconds')
    assert a == b


def test_range_variance_enters_the_tip_variance():
    f = flat(80, seed=2)
    g, vs = _year_only(f, [3, 10, 20])
    mid = ld.lsdv_flat(g, L, date_ranges='midpoint', **ld.LSDV_REPORT93)
    var = ld.lsdv_flat(g, L, **ld.LSDV_REPORT93)
    assert var['rate'] != mid['rate']
    # by hand: report 93's fit with mu^2 * width^2 / 12 added to the year-only tips' variances
    mu = ld._plain_regression(g)[0]
    om = 1.0 / L
    tip = np.asarray(g.tip)
    v = (ld._ls_durations(g, mu) + np.where(tip, 9.0 * om, 0.0)) * om + np.where(tip, mu * mu * ld._dvar(g), 0.0)
    v[0] = om * om
    r = ld._Regression(g).fit(np.asarray(g.ml, float), v)
    assert var['rate'] == pytest.approx(r['slope'], rel=1e-12)
    assert set(np.flatnonzero(ld._dvar(g) > 0)) == set(vs)


def test_flatten_reads_tree_time_date_ranges():
    tree, dates = sim(40, seed=3)
    name = sorted(dates)[0]
    dates = dict(dates)
    d = dates[name]
    dates[name] = [np.floor(d), np.floor(d) + 1.0]
    f = ld._flatten(tt_from_tree(tree, dates).tree, 1.0 / L)
    v = f.name.index(name)
    assert f.date[v] == np.floor(d) + 0.5 and f.dvar[v] == pytest.approx(1.0 / 12)
    assert sum(x > 0 for x in f.dvar) == 1


def test_baseline_passes_date_ranges():
    from test_lsdv import load
    from treetime.root_candidates import UnrootedBaseline, regression_candidates
    nwk, aln, dates, _ = load('binary20')
    dates = dict(dates)
    k = sorted(dates)[0]
    dates[k] = [np.floor(dates[k]), np.floor(dates[k]) + 1.0]
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    cand = regression_candidates(b, k=1, covariation=False, slope=None)[0][0]
    a, m = b.lsdv(cand, **ld.LSDV_REPORT93), b.lsdv(cand, date_ranges='midpoint', **ld.LSDV_REPORT93)
    assert a['rate'] != m['rate']


@pytest.mark.parametrize('seed', [2, 4, 5])
def test_correct_year_only_tips_are_flagged_less(seed):
    """Correct year-only dates sit up to half a year from their midpoints; judged against their ranges they are flagged
    no more often than read as exact midpoints (here at a low threshold, z = 2)."""
    f = flat(80, seed=seed, internal_min=1)
    tips = [v for v in range(len(f.par)) if f.tip[v]]
    g, vs = _year_only(f, list(range(0, len(tips), 4)))
    mid = ld.lsdv_flat(g, L, outlier_z=2.0, outlier_clade_max=None, date_ranges='midpoint')
    var = ld.lsdv_flat(g, L, outlier_z=2.0, outlier_clade_max=None)
    fm, fv = set(mid['outlier_nodes']) & set(vs), set(var['outlier_nodes']) & set(vs)
    assert len(fv) <= len(fm)


def _clocklike(seed, n=80, mu=0.05):
    """80-tip simulated tree at 50 substitutions per year (L = 1000), so that half a year is well above the noise."""
    tree = simulated_tree(n, seed, 1, mu=mu)
    d = 2020.0 + 10.0 * np.random.default_rng(seed).random(n)  # the dates simulated_tree drew
    dates = {f'T{i:03d}': float(d[i]) for i in range(n)}
    return ld._flatten(tree, 1.0 / L, length=lambda c: c.branch_length, date=lambda c: dates.get(c.name, np.nan))


@pytest.mark.parametrize('seed', [2, 3, 4])
def test_planted_year_only_tip(seed):
    """A correct year-only date whose true date lies 0.45 years from the published midpoint is flagged when read as an
    exact date at a threshold it passes when judged against its range; a year-only date 10 years off is flagged
    either way, and the default fit drops it but keeps the correct one."""
    f = _clocklike(seed)
    par, tip, date, y = ld._arrays(f)
    tips = [v for v in range(len(par)) if tip[v]]
    v0, v1 = tips[7], tips[30]
    date = date.copy()
    dv = np.zeros(len(par))
    date[v0] -= 0.45; dv[v0] = 1.0 / 12   # range [d - 0.95, d + 0.05] contains the true date d
    date[v1] += 10.0; dv[v1] = 1.0 / 12   # wrong by 10 years
    g = ld._from_arrays(par, tip, date, y, f.name, dv)
    g0 = ld._from_arrays(par, tip, np.where(np.arange(len(par)) == v1, date - 10.0, date), y, f.name,
                         np.where(np.arange(len(par)) == v1, 0.0, dv))  # the correct year-only tip alone
    zs = [z for z in np.arange(0.8, 6.0, 0.05) if v0 in ld.lsd2_outliers(g0, L, z, rounds=1)[1]]
    assert zs, 'the midpoint reading never flags the planted tip'
    assert v0 not in ld.lsd2_outliers(g0, L, max(zs), rounds=1, date_ranges=True)[1]
    for dr in (False, True):
        assert v1 in ld.lsd2_outliers(g, L, 4.5, date_ranges=dr)[1]
    full = ld.lsdv_flat(g, L)
    assert g.name[v1] in full['outliers'] and g.name[v0] not in full['outliers']


def test_date_ranges_option_checked():
    with pytest.raises(ValueError):
        ld.lsdv_flat(flat(40, seed=2), L, date_ranges='exact')
