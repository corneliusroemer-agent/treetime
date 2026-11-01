"""Gold defaults v2 (investigation report 116): lsdv outlier rule and clade extension at z = 4.5 (report 111)."""
import numpy as np
import pytest

from test_lsdv import L, sim, tt_from_tree
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
