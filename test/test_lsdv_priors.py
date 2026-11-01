"""Opt-in priors for lsdv. Rate prior (investigation report 113): rate_prior=(median rate, log-scale sd S) adds a
pseudo-observation on the slope to the GLS fit, weighted against the SE-inflated GLS curvature, and stabilises the
reference rate (plain slope combined with the prior). Off by default: outputs are unchanged without it."""
import numpy as np
import pytest

from test_lsdv_defaults import flat
from treetime import ls_dated_rate as ld


def test_default_unchanged():
    f = flat(80, 2)
    a, b = ld.lsdv_flat(f, 1000), ld.lsdv_flat(f, 1000, rate_prior=None)
    a.pop('seconds'); b.pop('seconds')
    assert a == b
    assert 'rate_prior' not in a


@pytest.mark.parametrize('opts', [dict(), ld.LSDV_REPORT93, dict(resolve='ladder_xo')])
def test_limits(opts):
    f = flat(80, 5)
    base = ld.lsdv_flat(f, 1000, **opts)['rate']
    wide = ld.lsdv_flat(f, 1000, rate_prior=(3 * base, 50.0), **opts)
    assert abs(np.log(wide['rate'] / base)) < 0.02
    tight = ld.lsdv_flat(f, 1000, rate_prior=(3 * base, 1e-4), **opts)
    assert abs(np.log(tight['rate'] / (3 * base))) < 1e-3
    mid = ld.lsdv_flat(f, 1000, rate_prior=(3 * base, np.log(2)), **opts)
    assert base < mid['rate'] < 3 * base and 0 < mid['prior_share'] < 1


def test_closed_form_matches_brute_force():
    """b* minimises the profiled GLS chisq/k^2 + the prior row; check against a grid over the slope."""
    f = flat(80, 2)
    mu0, S, k = 0.004, np.log(2), 2.0
    o = ld.lsdv_flat(f, 1000, rate_prior=(mu0, S), **ld.LSDV_REPORT93)
    # rebuild the final round's regression at the reference rate the core used
    ref = ld.plain_rate_with_prior(f, (mu0, S))[0]
    reg = ld._Regression(f)
    om = 1e-3
    var = (ld._ls_durations(f, ref) + np.where(np.asarray(f.tip), 9.0 * om, 0.0)) * om
    var[0] = om * om
    ml = np.asarray(f.ml, float)
    bs = np.linspace(0.5 * mu0 * 0.3, 1.5 * mu0, 4001)
    obj = [ld._Regression(f).fit(ml, var, slope=b)['chisq'] / k ** 2 + 0.5 * (b - mu0) ** 2 / (mu0 * S) ** 2 for b in bs]
    assert abs(bs[int(np.argmin(obj))] - o['rate']) < 2 * (bs[1] - bs[0]) + 1e-9
    assert reg is not None


def test_rescues_nonpositive_plain_rate():
    f = flat(80, 2)
    par, tip, date, y = ld._arrays(f)
    d = np.zeros(len(par))
    for v in range(1, len(par)):
        d[v] = d[par[v]] + y[v]
    tips = np.flatnonzero(tip)
    date = np.asarray(date, float).copy()
    date[tips] = 2030.0 - 1000.0 * d[tips]          # dates run backwards against divergence: negative slope
    g = ld._with(f, date=date)
    if ld._plain_regression(g)[0] > 0:
        pytest.skip('shuffled dates still give a positive plain slope')
    with pytest.raises(ValueError):
        ld.lsdv_flat(g, 1000, **ld.LSDV_REPORT93)
    o = ld.lsdv_flat(g, 1000, rate_prior=(0.002, np.log(2)), **ld.LSDV_REPORT93)   # runs: the prior supplies the
    assert np.isfinite(o['rate'])                                                   # reference rate; the slope
    o = ld.lsdv_flat(g, 1000, rate_prior=(0.002, 0.05), **ld.LSDV_REPORT93)        # itself follows the data
    assert o['rate'] > 0


def test_bad_prior_rejected():
    with pytest.raises(ValueError):
        ld.lsdv_flat(flat(80, 2), 1000, rate_prior=(-1.0, 0.5))


def test_posterior_covariance():
    """cov = inv(H + diag(wp, 0)): the slope variance shrinks, the root-date SE stays consistent with it."""
    f = flat(80, 2)
    ref = ld.lsdv_flat(f, 1000, **ld.LSDV_REPORT93)['rate']        # same reference rate -> same variances
    base = ld.lsdv_flat(f, 1000, rate=ref, **ld.LSDV_REPORT93, se_inflation=2.0)
    o = ld.lsdv_flat(f, 1000, rate=ref, rate_prior=(base['rate'], 0.2), **ld.LSDV_REPORT93)
    c0, c1 = np.asarray(base['cov']), np.asarray(o['cov'])
    assert c1[0, 0] < c0[0, 0]
    wp = 4.0 / (base['rate'] * 0.2) ** 2
    H = np.linalg.inv(c1) - np.linalg.inv(c0)
    assert abs(H[0, 0] - wp) / wp < 0.05 and abs(H[1, 1]) < 1e-6 * abs(np.linalg.inv(c0)[1, 1])


# ---- root-date prior (LogNormal on the root age, log scale) -------------------------------------------------------
R93 = ld.LSDV_REPORT93


def _tref(f):
    d = np.asarray(f.date, float)[np.asarray(f.tip, bool)]
    return float(np.nanmax(d))


def test_root_date_prior_limits():
    f = flat(80, 5)
    base = ld.lsdv_flat(f, 1000, **R93)
    t = _tref(f)
    wide = ld.lsdv_flat(f, 1000, root_date_prior=(base['root_date'] - 30.0, 50.0), **R93)
    assert abs(wide['root_date'] - base['root_date']) < 0.01 * (t - base['root_date'])
    assert abs(np.log(wide['rate'] / base['rate'])) < 0.01
    d0 = base['root_date'] - 15.0
    tight = ld.lsdv_flat(f, 1000, root_date_prior=(d0, 1e-4), **R93)
    assert abs(tight['root_date'] - d0) < 1e-3 * (t - d0) and tight['rate'] > 0
    mid = ld.lsdv_flat(f, 1000, root_date_prior=(d0, np.log(1.5)), **R93)
    assert d0 < mid['root_date'] < base['root_date'] and 0 < mid['root_date_prior_share'] < 1
    assert mid['root_date_se'] < base['root_date_se'] and mid['gls_root_date'] == pytest.approx(base['root_date'])


def test_root_date_prior_mode_matches_brute_force():
    """The mode minimises (theta - theta_gls)' H (theta - theta_gls) / 2 + k^2 (log A - log A0)^2 / (2 S^2), H the
    inverse GLS covariance; grid search over (log b, log A)."""
    f = flat(80, 2)
    k, S = 2.0, np.log(1.5)
    g = ld.lsdv_flat(f, 1000, **R93, se_inflation=k)
    t = _tref(f)
    d0 = g['root_date'] - 10.0
    o = ld.lsdv_flat(f, 1000, root_date_prior=(d0, S), **R93)
    H = np.linalg.inv(np.asarray(g['cov']))
    th = np.array([g['rate'], g['intercept']])
    lb = np.log(g['rate']) + np.linspace(-0.3, 0.3, 601)
    lA = np.log(t - g['root_date']) + np.linspace(-0.2, 0.6, 801)
    B, A = np.meshgrid(np.exp(lb), np.exp(lA), indexing='ij')
    d1, d2 = B - th[0], B * (A - t) - th[1]
    obj = 0.5 * (H[0, 0] * d1 ** 2 + 2 * H[0, 1] * d1 * d2 + H[1, 1] * d2 ** 2) \
        + 0.5 * k * k * (np.log(A) - np.log(t - d0)) ** 2 / S ** 2
    i, j = np.unravel_index(np.argmin(obj), obj.shape)
    assert abs(np.log(o['rate']) - lb[i]) <= 2 * (lb[1] - lb[0])
    assert abs(np.log(t - o['root_date']) - lA[j]) <= 2 * (lA[1] - lA[0])


def test_root_date_prior_keeps_the_rate_positive_where_a_linear_update_would_not():
    """Dates running against divergence: the GLS slope is negative; the log-scale root-age prior still gives a positive
    rate and a root date near the prior, with or without a rate prior."""
    f = flat(80, 2)
    par, tip, date, y = ld._arrays(f)
    d = np.zeros(len(par))
    for v in range(1, len(par)):
        d[v] = d[par[v]] + y[v]
    tips = np.flatnonzero(tip)
    date = np.asarray(date, float).copy()
    date[tips] = 2030.0 - 1000.0 * d[tips]
    g = ld._with(f, date=date)
    if ld._plain_regression(g)[0] > 0:
        pytest.skip('dates still give a positive plain slope')
    t = _tref(g)
    for rp in (None, (0.002, np.log(2))):
        o = ld.lsdv_flat(g, 1000, root_date_prior=(t - 30.0, np.log(1.5)), rate_prior=rp, **R93)
        assert o['rate'] > 0 and np.isfinite(o['root_date']) and o['root_date'] < t


def test_priors_compose():
    f = flat(80, 5)
    base = ld.lsdv_flat(f, 1000, **R93)
    d0, mu0 = base['root_date'] - 10.0, base['rate'] * 0.7
    both = ld.lsdv_flat(f, 1000, rate_prior=(mu0, 1e-4), root_date_prior=(d0, 1e-4), **R93)
    assert both['rate'] == pytest.approx(mu0, rel=1e-2) and both['root_date'] == pytest.approx(d0, abs=0.1)
    r_only = ld.lsdv_flat(f, 1000, rate_prior=(mu0, np.log(2)), **R93)
    j = ld.lsdv_flat(f, 1000, rate_prior=(mu0, np.log(2)), root_date_prior=(d0, np.log(1.5)), **R93)
    assert j['rate_prior'] == [mu0, np.log(2)] and j['root_date_prior'][0] == d0
    assert j['root_date'] < r_only['root_date']   # the root-date prior pulls the root earlier, towards d0
    # the joint path with an uninformative root-date prior reproduces the rate prior's pseudo-row fit
    w = ld.lsdv_flat(f, 1000, rate_prior=(mu0, np.log(2)), root_date_prior=(d0, 1e4), **R93)
    assert w['rate'] == pytest.approx(r_only['rate'], rel=1e-5)
    assert w['root_date'] == pytest.approx(r_only['root_date'], abs=1e-3)
    assert w['rate_se'] == pytest.approx(r_only['rate_se'], rel=1e-4)


@pytest.mark.parametrize('opts', [dict(), dict(resolve='ladder_xo'), dict(rounds=2)])
def test_root_date_prior_under_the_defaults(opts):
    f = flat(80, 6, internal_min=1, planted=5)
    base = ld.lsdv_flat(f, 1000, **opts)
    o = ld.lsdv_flat(f, 1000, root_date_prior=(base['root_date'] - 10.0, np.log(1.5)), **opts)
    assert o['rate'] > 0 and o['root_date'] < base['root_date'] and o['outliers'] == base['outliers']


def test_root_date_prior_checks():
    f = flat(40, 2)
    for bad in (dict(root_date_prior=(2100.0, 0.5)), dict(root_date_prior=(1990.0, 0.0)),
                dict(root_date_prior=(1990.0,)), dict(rate_prior=(0.0, 0.5))):
        with pytest.raises(ValueError):
            ld.lsdv_flat(f, 1000, **bad)


def test_priors_reach_the_shared_rate_and_final_lsdv():
    from test_lsdv import FIT1, baseline
    from treetime.root_candidates import plain_candidate_sources, rank_root_candidates
    b0 = baseline('binary20', shared_rate='lsdv')
    d0 = b0.shared_rate_info['candidates'][0]['root_date'] - 5.0
    kw = dict(root_date_prior=(d0, np.log(1.5)), rate_prior=(b0.clock_rate, np.log(2)))
    b = baseline('binary20', shared_rate='lsdv', shared_rate_kwargs=kw)
    row = b.shared_rate_info['candidates'][0]
    assert row['root_date_prior'] == [d0, np.log(1.5)] and b.clock_rate == row['lsdv_rate'] != b0.clock_rate
    res = rank_root_candidates(b, regression_k=2, fit=FIT1, keep_states=False,
                               candidate_sources=plain_candidate_sources(2), final_lsdv=kw)
    assert res.final_lsdv['root_date_prior'] == [d0, np.log(1.5)] and res.final_lsdv['prior_share'] > 0
