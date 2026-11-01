"""Clock rate at a fixed root from TreeTime's covariation regression with
dated-duration branch variances, where the durations come from one
least-squares node-time solve instead of a TreeTime dated fit ("lsdv").

At a given root:

1. Reference rate: plain root-to-tip regression (OLS of root-to-tip distance on
   tip date), TreeTime's ``covariation=False`` model.
2. Node times: one unweighted least-squares fit of the branch lengths with the
   dated tips fixed at mu x date, solved in O(n) on the tree
   (:func:`_node_positions`); each parent is then moved no later than its
   children. Durations are mu x (t_child - t_parent), clipped at 0.
3. Covariation regression with branch variance
   (duration + tip_slack^2 / L [terminal branches]) / L. This is TreeTime's
   ``setup_TreeRegression(covariation=True)`` with ``clock_length`` set to the
   durations, so the result equals that regression on the tree the estimator
   works on.

Internal edges shorter than 0.1 mutations are collapsed first (``collapse``,
default), their length pushed onto the children; every root-to-tip distance is
kept. With collapsing the equality above holds for TreeTime's regression on the
collapsed tree; TreeTime's regression on the uncollapsed tree (with LS durations
solved there) differs slightly, 0.1-0.2% in rate on simulated 80-tip trees.
The tree passed in is never modified.

By default (report 107) the polytomies left by the collapse are resolved by
attachment order (the ladder, or its cross-fitted version on large trees with a
weak root-to-tip signal) and LSD2's outlier rule (z = 4.5), with a clade extension,
drops outlying tip dates first; ``resolve=None, outlier_z=None``
(:data:`LSDV_REPORT93`) gives the estimator described above unchanged.

Cost is linear in the number of nodes: the regression runs TreeRegression's
postorder recursion on flat arrays (:class:`_Regression`), with results equal to
TreeRegression's to the bit. Report 93's estimator takes about 1 ms at 80 tips;
the defaults about 0.6 s at 10k tips and 5 s at 100k (the outlier rule
dominates; the ladder fit alone is 1 s at 100k).
"""
import time

import numpy as np
from Bio.Phylo.BaseTree import Clade, Tree

from .treeregression import TreeRegression, base_regression

COLLAPSE_MUTATIONS = 0.1  # internal edges shorter than this (in mutations) and without length are collapsed


class _Flat:
    """Rooted tree as parallel lists, parents before children after
    :meth:`preorder` (root = 0). Per node: par, tip, date (nan if none; the
    midpoint of a date range), ml (branch value, mutation_length), name, dvar
    (variance of a range date, width^2 / 12; 0 for an exact date)."""

    def __init__(self):
        self.par, self.tip, self.date, self.ml, self.name, self.dvar = [], [], [], [], [], []

    def add(self, par, tip, date, ml, name=None, dvar=0.0):
        self.par.append(par)
        self.tip.append(tip)
        self.date.append(date)
        self.ml.append(ml)
        self.name.append(name)
        self.dvar.append(dvar)
        return len(self.par) - 1

    def children(self):
        ch = [[] for _ in self.par]
        for i, p in enumerate(self.par):
            if p >= 0:
                ch[p].append(i)
        return ch

    def preorder(self):
        """Renumber so that every parent precedes its children (root = 0)."""
        ch = self.children()
        order, stack = [], [0]
        while stack:
            v = stack.pop()
            order.append(v)
            stack.extend(reversed(ch[v]))
        new = {v: i for i, v in enumerate(order)}
        out = _Flat()
        for v in order:
            out.add(new[self.par[v]] if self.par[v] >= 0 else -1, self.tip[v], self.date[v], self.ml[v], self.name[v],
                    self.dvar[v])
        return out


def _tip_date(n):
    """TreeTime's regression tip value: mean date of a dated, non-outlier tip."""
    if not n.is_terminal() or getattr(n, 'bad_branch', False):
        return np.nan
    d = getattr(n, 'raw_date_constraint', None)
    if d is None:
        return np.nan
    d = float(np.mean(d))
    return d if np.isfinite(d) else np.nan


def _range_var(d):
    """Variance width^2 / 12 of a date given as a range [lo, hi] (uniform on it); 0 for a single date."""
    if d is None or np.ndim(d) == 0 or len(d) != 2:
        return 0.0
    lo, hi = float(d[0]), float(d[1])
    return (hi - lo) ** 2 / 12.0 if np.isfinite(lo) and np.isfinite(hi) else 0.0


def _tip_date_var(n):
    """Variance of a tip's date range (``raw_date_constraint`` [lo, hi]); 0 for an exact date."""
    if not n.is_terminal() or getattr(n, 'bad_branch', False):
        return 0.0
    return _range_var(getattr(n, 'raw_date_constraint', None))


def _flatten(tree, om, collapse=True, length=None, date=None, date_var=None):
    """Flat copy of a rooted Bio.Phylo tree. Branch values are ``length(node)``
    (default ``mutation_length``), tip dates ``date(node)`` (default
    :func:`_tip_date`), date-range variances ``date_var(node)`` (default
    :func:`_tip_date_var`). With ``collapse``, internal edges of length below
    COLLAPSE_MUTATIONS mutations (``om`` = one mutation) are collapsed: their
    length is pushed onto the children, so every root-to-tip sum is kept."""
    length = length or (lambda n: n.mutation_length)
    date = date or _tip_date
    date_var = date_var or _tip_date_var
    f = _Flat()
    stack = [(tree.root, -1, 0.0)]
    while stack:
        n, p, add = stack.pop()
        ml = 0.0 if n is tree.root else max(0.0, float(length(n) or 0.0)) + add
        if collapse and n is not tree.root and not n.is_terminal() and p >= 0 and ml < COLLAPSE_MUTATIONS * om:
            for c in reversed(n.clades):  # children attach to p, inheriting this edge
                stack.append((c, p, ml))
            continue
        i = f.add(p, n.is_terminal(), date(n) if n.is_terminal() else np.nan, ml, n.name,
                  float(date_var(n)) if n.is_terminal() else 0.0)
        for c in reversed(n.clades):
            stack.append((c, i, 0.0))
    return f.preorder()


def _node_positions(par, tip, fixed_x, y, w):
    """Minimise sum_b w_b (y_b - x_child + x_parent)^2 over the free node
    positions x (every internal node, and tips with fixed_x = nan), with
    fixed_x for the others. Nodes in preorder (root 0). O(n) Gaussian
    elimination on the tree. Returns x (length n)."""
    n = len(par)
    a = np.zeros(n)  # curvature of the subtree's quadratic in x_v
    s = np.zeros(n)  # curvature x centre accumulated from the children
    fixed = np.isfinite(fixed_x)
    for v in range(n - 1, 0, -1):
        p = par[v]
        if fixed[v]:
            cv, mv = w[v], fixed_x[v] - y[v]
        else:
            av = a[v]
            if av <= 0:
                continue
            mv = s[v] / av - y[v]
            cv = av * w[v] / (av + w[v])
        a[p] += cv
        s[p] += cv * mv
    x = np.full(n, np.nan)
    if fixed[0]:
        x[0] = fixed_x[0]
    elif a[0] > 0:
        x[0] = s[0] / a[0]
    else:
        raise ValueError('ls_dated_rate: no dated tip below the root')
    for v in range(1, n):
        if fixed[v]:
            x[v] = fixed_x[v]
        else:
            av = a[v]
            m = s[v] / av if av > 0 else 0.0
            x[v] = (av * m + w[v] * (x[par[v]] + y[v])) / (av + w[v])
    return x


def _plain_regression(f):
    """OLS of root-to-tip distance on date (TreeTime's covariation=False model)."""
    par, ml = np.asarray(f.par), np.asarray(f.ml)
    d = np.zeros(len(par))
    for v in range(1, len(par)):
        d[v] = d[par[v]] + ml[v]
    dates = np.asarray(f.date)
    ok = np.asarray(f.tip) & np.isfinite(dates)
    t, y = dates[ok], d[ok]
    if ok.sum() < 2 or np.ptp(t) <= 0:
        raise ValueError('No variation in sampling dates! Please specify your clock rate explicitly.')
    slope, intercept = np.polyfit(t, y, 1)
    r = np.corrcoef(t, y)[0, 1]
    return float(slope), float(intercept), float(r)


def _postorder_Q(treg):
    """Root averages of a TreeRegression by its postorder recursion alone, O(n)."""
    for nd in treg.tree.get_nonterminals(order='postorder'):
        Q = np.zeros(6)
        for c in nd:
            Q += treg.propagate_averages(c, treg.tip_value(c), treg.branch_value(c), treg.branch_variance(c))
        nd.Q = Q
    return treg.tree.root.Q


class _PhyloRegression:
    """TreeRegression on a light Bio.Phylo copy of a flat tree (report 93's implementation). Kept as the reference
    that :class:`_Regression` reproduces bit for bit; Bio.Phylo's recursive traversal makes it O(n x depth), which
    dominates on deep (ladder-resolved) trees."""

    def __init__(self, f):
        self.f = f
        clades = []
        for v in range(len(f.par)):
            c = Clade(branch_length=f.ml[v], name=f.name[v])
            c._i = v
            clades.append(c)
            if f.par[v] >= 0:
                clades[f.par[v]].clades.append(c)
        self.tree = Tree(root=clades[0], rooted=True)
        dates = np.asarray(f.date, float)
        self.cur = {}
        tipval = lambda c: (dates[c._i] if np.isfinite(dates[c._i]) else None) if c.is_terminal() else None  # noqa: E731
        self.treg = TreeRegression(self.tree, tip_value=tipval,
                                   branch_value=lambda c: self.cur['val'][c._i],
                                   branch_variance=lambda c: self.cur['var'][c._i])

    def fit(self, val, var, slope=None):
        self.cur['val'], self.cur['var'] = val, var
        return base_regression(_postorder_Q(self.treg), slope=slope)


def _sq(a):
    """Elementwise square as numpy's scalar ``x ** 2`` computes it (C pow), which TreeRegression uses; an array's
    ``** 2`` is x * x and can differ in the last bit."""
    return np.fromiter((x ** 2 for x in a), float, len(a))


class _Regression:
    """TreeTime's covariation regression (TreeRegression's postorder averages, then ``base_regression``) on flat
    arrays. The node updates are TreeRegression.propagate_averages written elementwise over all nodes of one depth
    at a time, and children are added into their parent in the same order as Bio.Phylo's, so the root averages,
    and hence slope, intercept, chisq and cov, equal :class:`_PhyloRegression`'s exactly. O(n)."""

    def __init__(self, f):
        par = np.asarray(f.par)
        n = len(par)
        self.par = par
        self.tip = np.asarray(f.tip, bool)
        dates = np.asarray(f.date, float)
        self.tv = np.where(self.tip & np.isfinite(dates), dates, np.nan)
        depth = np.zeros(n, int)
        for v in range(1, n):
            depth[v] = depth[par[v]] + 1
        order = np.argsort(-depth, kind='stable')  # deepest first; within a depth by index (= Bio.Phylo child order)
        cuts = np.flatnonzero(np.diff(depth[order])) + 1
        self.levels = [lv for lv in np.split(order, cuts) if len(lv) and lv[0] != 0]

    def fit(self, val, var, slope=None):
        val = np.asarray(val, float)
        var = np.asarray(var, float)
        n = len(self.par)
        Q = np.zeros((n, 6))
        for lv in self.levels:
            res = np.zeros((len(lv), 6))
            t = self.tip[lv]
            if t.any():
                ti = lv[t]
                tv, bv, vr = self.tv[ti], val[ti], var[ti]
                ok = np.isfinite(tv)
                r = np.zeros((len(ti), 6))
                with np.errstate(divide='ignore', invalid='ignore'):
                    r[ok] = np.column_stack([tv[ok] / vr[ok], bv[ok] / vr[ok], _sq(tv[ok]) / vr[ok],
                                             bv[ok] * tv[ok] / vr[ok], _sq(bv[ok]) / vr[ok], 1.0 / vr[ok]])
                r[ok & (vr == 0)] = np.inf
                res[t] = r
            if (~t).any():
                ii = lv[~t]
                q = Q[ii]
                bv, vr = val[ii], var[ii]
                ta, da, ts, dt, ds, si = q.T
                denom = 1.0 / (1 + vr * si)
                res[~t] = np.column_stack([
                    ta * denom,
                    (da + bv * si) * denom,
                    ts - vr * _sq(ta) * denom,
                    dt + ta * bv - vr * ta * (da + bv * si) * denom,
                    ds + 2 * bv * da + _sq(bv) * si - vr * (_sq(da) + 2 * bv * da * si + _sq(bv) * _sq(si)) * denom,
                    si * denom])
            np.add.at(Q, self.par[lv], res)  # unbuffered, in index order: children summed as TreeRegression does
        return base_regression(Q[0], slope=slope)


def _ls_durations(f, mu):
    """Branch durations (in substitution units, mu x time) from one unweighted
    least-squares fit of the branch lengths with dated tips fixed at mu x date,
    parents then moved no later than their children. Element 0 (root) is 0."""
    par = np.asarray(f.par)
    tip = np.asarray(f.tip)
    dates = np.asarray(f.date, float)
    ml = np.asarray(f.ml, float)
    fixed = np.where(tip & np.isfinite(dates), mu * dates, np.nan)
    x = _node_positions(par, tip, fixed, ml, np.ones(len(par)))
    t = x / mu
    dated = dates[tip & np.isfinite(dates)]
    span = float(np.ptp(dated)) if len(dated) > 1 else 1.0
    eps = 1e-6 * max(span, 1e-12)
    for v in range(len(par) - 1, 0, -1):
        p = par[v]
        if t[p] > t[v] - eps:
            t[p] = t[v] - eps
    d = np.r_[0.0, mu * (t[1:] - t[par[1:]])]
    return np.maximum(d, 0.0)


LSDV_OPTIONS = dict(resolve='auto', resolve_auto_tips=1000, resolve_auto_r2=0.2,
                    outlier_z=4.5, outlier_mode='drop', outlier_rounds=5, outlier_clade_max=4, outlier_clade_z=4.5,
                    internal_b=0.0, first_pass='ls', se_inflation=2.0, resample=0, resample_q=0.2, seed=1,
                    date_ranges='variance', rate_prior=None, root_date_prior=None, root_date_ci_factor=1.5)
"""The lsdv options and their defaults (reports 107 and 111): ladder resolution, switching to the cross-fitted ladder
when there are more than 1000 tips and the plain root-to-tip R^2 (after outlier dates are dropped) is below 0.2; LSD2's
outlier rule at z = 4.5, iterated, in drop mode, with the clade extension (an internal branch with |z| > 4.5 above 2-4
dated tips drops all their dates); SE inflation 2; one round; no internal b; range dates (month or year only) carry
their uncertainty (``date_ranges='variance'``). z = 4.5 keeps the share of correctly
dated tips that are flagged at or below about 0.2% from 100 to 10,000 tips (z = 3 flags 1.5-3%) and still catches
gross date errors. Every option can be overridden: :data:`LSDV_REPORT108` restores report 108's z = 3, and
:data:`LSDV_REPORT93` turns the estimator back into report 93's."""

DATE_SE_REL_MAX = 0.5
"""``date_reliable`` is False when the rate's relative SE (``se_rel``, without SE inflation) exceeds this."""
DATE_R2_MIN = 0.2
"""``date_reliable`` is False when the plain root-to-tip R^2 (on the dates the fit uses) is below this."""

LSDV_REPORT108 = dict(outlier_z=3.0, outlier_clade_z=3.0)
"""Options that give report 108's defaults (outlier rule and clade extension at z = 3) on top of the current ones."""

LSDV_REPORT93 = dict(resolve=None, outlier_z=None)
"""Options that give report 93's lsdv (no resolution, no outlier rule): its rate, intercept, chisq and, through
:class:`~treetime.root_candidates.UnrootedBaseline`, its output keys, exactly. Report 103's recommended configuration
is ``dict(resolve='auto', resolve_auto_tips=2000, resolve_auto_r2=None, outlier_z=3.5, outlier_clade_max=None)``."""


def check_lsdv_options(opts):
    """Raise ValueError on unknown or invalid lsdv options (keys of LSDV_OPTIONS plus tip_slack, rate, rounds,
    collapse)."""
    bad = set(opts) - set(LSDV_OPTIONS) - {'tip_slack', 'rate', 'rounds', 'collapse'}
    if bad:
        raise ValueError('lsdv: unknown options %s' % sorted(bad))
    try:
        return _check(dict(LSDV_OPTIONS, **opts), opts)
    except TypeError as e:
        raise ValueError('lsdv: invalid option type (%s)' % e)


def _isint(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _check(o, opts):
    if o['resolve'] not in (None, 'ladder', 'ladder_xo', 'auto'):
        raise ValueError("lsdv: resolve must be None, 'ladder', 'ladder_xo' or 'auto'")
    if o['outlier_z'] is not None and not o['outlier_z'] > 0:
        raise ValueError('lsdv: outlier_z must be None or > 0')
    if o['outlier_clade_max'] is not None and not (_isint(o['outlier_clade_max']) and o['outlier_clade_max'] >= 2):
        raise ValueError('lsdv: outlier_clade_max must be None or an integer >= 2')
    if o['outlier_clade_z'] is not None and not o['outlier_clade_z'] > 0:
        raise ValueError('lsdv: outlier_clade_z must be None (= outlier_z) or > 0')
    if o['resolve_auto_r2'] is not None and not 0 <= o['resolve_auto_r2'] <= 1:
        raise ValueError('lsdv: resolve_auto_r2 must be None or in [0, 1]')
    if o['outlier_mode'] not in ('drop', 'flag'):
        raise ValueError("lsdv: outlier_mode must be 'drop' or 'flag'")
    if not (_isint(o['outlier_rounds']) and o['outlier_rounds'] >= 1):
        raise ValueError('lsdv: outlier_rounds must be an integer >= 1')
    if not (_isint(o['resolve_auto_tips']) and o['resolve_auto_tips'] >= 0):
        raise ValueError('lsdv: resolve_auto_tips must be an integer >= 0')
    if not _isint(o['seed']):
        raise ValueError('lsdv: seed must be an integer')
    if not o['internal_b'] >= 0:
        raise ValueError('lsdv: internal_b must be >= 0')
    for key in ('rate_prior', 'root_date_prior'):
        p = o[key]
        if p is not None and not (np.ndim(p) == 1 and len(p) == 2 and all(np.isfinite(float(x)) for x in p)
                                  and float(p[1]) > 0 and (key == 'root_date_prior' or float(p[0]) > 0)):
            raise ValueError('lsdv: %s must be None or a pair (%s, log-scale sd > 0)'
                             % (key, 'median rate > 0' if key == 'rate_prior' else 'median root date'))
    if not o['root_date_ci_factor'] > 0:
        raise ValueError('lsdv: root_date_ci_factor must be > 0')
    if o['date_ranges'] not in ('variance', 'midpoint'):
        raise ValueError("lsdv: date_ranges must be 'variance' or 'midpoint'")
    if o['first_pass'] not in ('ls', 'lsd2'):
        raise ValueError("lsdv: first_pass must be 'ls' or 'lsd2'")
    if not o['se_inflation'] > 0:
        raise ValueError('lsdv: se_inflation must be > 0')
    if not (_isint(o['resample']) and o['resample'] >= 0 and o['resample_q'] >= 0):
        raise ValueError('lsdv: resample must be an integer >= 0 and resample_q >= 0')
    if opts.get('tip_slack', 3.0) < 0 or opts.get('rounds', 1) < 1:
        raise ValueError('lsdv: tip_slack must be >= 0 and rounds >= 1')
    return o


def _arrays(f):
    return (np.asarray(f.par), np.asarray(f.tip, bool), np.asarray(f.date, float), np.asarray(f.ml, float))


def _dvar(f):
    """Date-range variances of a flat tree (0 for exact dates and internal nodes)."""
    return np.asarray(getattr(f, 'dvar', None) or np.zeros(len(f.par)), float)


def _from_arrays(par, tip, date, ml, name, dvar=None):
    out = _Flat()
    for i in range(len(par)):
        out.add(int(par[i]), bool(tip[i]), float(date[i]) if tip[i] else np.nan, float(ml[i]), name[i],
                float(dvar[i]) if dvar is not None and tip[i] else 0.0)
    return out


def _with(f, date=None, ml=None):
    """Copy of flat tree ``f`` with new tip dates and/or branch values (same topology and numbering)."""
    par, tip, d0, m0 = _arrays(f)
    return _from_arrays(par, tip, d0 if date is None else date, m0 if ml is None else ml, f.name, _dvar(f))


def _collapse_flat(f, thr):
    """Collapse internal non-root edges with value < thr (value pushed onto the children), as :func:`_flatten`."""
    par, tip, date, ml = _arrays(f)
    dv = _dvar(f)
    ch = [[] for _ in par]
    for v in range(1, len(par)):
        ch[par[v]].append(v)
    out = _Flat()
    stack = [(0, -1, 0.0)]
    while stack:
        v, p, add = stack.pop()
        y = (ml[v] if v else 0.0) + add
        if v and not tip[v] and p >= 0 and y < thr:
            for c in reversed(ch[v]):
                stack.append((c, p, y))
            continue
        i = out.add(p, bool(tip[v]), float(date[v]) if tip[v] else np.nan, y, f.name[v], float(dv[v]))
        for c in reversed(ch[v]):
            stack.append((c, i, 0.0))
    return out.preorder()


def _wls_positions(par, tip, date, y, w):
    """Closed-form branch-wise WLS clock fit (equal to the covariation regression with variance 1/w): node
    positions x(mu) = x0 + mu x1 from two O(n) solves, mu = sum w g (y - D x0) / sum w g^2 with g = D x1.
    Returns (mu, x at mu)."""
    dated = tip & np.isfinite(date)
    w = np.minimum(w, 1e9)
    x0 = np.nan_to_num(_node_positions(par, tip, np.where(dated, 0.0, np.nan), y, w))
    x1 = np.nan_to_num(_node_positions(par, tip, np.where(dated, date, np.nan), np.zeros(len(par)), w))
    g = np.r_[0.0, x1[1:] - x1[par[1:]]]
    a = np.r_[0.0, y[1:] - (x0[1:] - x0[par[1:]])]
    info = float(np.sum(w[1:] * g[1:] ** 2))
    mu = float(np.sum(w[1:] * g[1:] * a[1:]) / info) if info > 0 else np.nan
    return mu, x0 + mu * x1


def lsd2_b(f, L):
    """LSD2's default variance constant b = max(median branch value >= 0.5/L, 10/L), in substitutions per site."""
    y = np.asarray(f.ml, float)[1:]
    inf = y[y >= 0.5 / L]
    return max(float(np.median(inf)) if len(inf) else 0.0, 10.0 / L)


def _lsd2_first_pass(f, L):
    """LSD2's run 1 at a fixed root: joint WLS with V = B + b (no constraints). Returns durations (substitution
    units, negative clipped to 0) and the run-1 rate."""
    par, tip, date, y = _arrays(f)
    mu, x = _wls_positions(par, tip, date, y, 1.0 / (y + lsd2_b(f, L)))
    if not mu > 0:
        raise ValueError('ls_dated_rate: LSD2 first-pass rate not positive')
    return np.maximum(np.r_[0.0, x[1:] - x[par[1:]]], 0.0), mu


def _robust_rate(par, tip, date, y, rng, m=10):
    """LSD2's median rate (outliers.cpp median_rate): per dated tip, the median root-to-tip slope to m random other
    dated tips; positive medians kept; their median."""
    idx = np.flatnonzero(tip & np.isfinite(date))
    if len(idx) < 3:
        return np.nan
    d = np.zeros(len(par))
    for v in range(1, len(par)):
        d[v] = d[par[v]] + y[v]
    paths, dates = d[idx], date[idx]
    n = len(idx)
    m = min(m, n - 1)
    S = np.empty((n, m), int)
    for i in range(n):
        s = rng.choice(n - 1, size=m, replace=False)
        S[i] = s + (s >= i)
    dt = dates[S] - dates[:, None]
    ok = dt != 0
    sl = np.where(ok, (paths[S] - paths[:, None]) / np.where(ok, dt, 1.0), np.nan)
    good = ok.any(axis=1)
    med = np.full(n, np.nan)
    med[good] = np.nanmedian(sl[good], axis=1)
    med = med[np.isfinite(med) & (med > 0)]
    return float(np.median(med)) if len(med) else np.nan


def lsd2_outliers(f, L, z, rounds=5, seed=0, clade_max=None, clade_z=None, clades=None, date_ranges=False):
    """LSD2's ``-e z`` outlier rule at a fixed root, iterated (report 98, 101): robust median-of-slopes rate, LD node
    times at that rate with V = B + b, branch residuals (B - duration) / sqrt(V) standardised by their mean and SD
    over all branches; dated tips with |z-score| > z lose their date, and the rule is repeated on the remaining dated
    tips until it flags nothing new (at most ``rounds`` passes). A pass that would leave fewer than half of the
    originally dated tips (or fewer than 3) is not applied; it is recorded with a negative count.

    Clade extension (report 107; ``clade_max`` not None): in each pass, every non-root internal branch with
    |z-score| > ``clade_z`` (default ``z``) whose subtree holds 2 to ``clade_max`` currently dated tips also flags all
    of those tips. This catches a date error shared by a small clade, which lands on the branch above it rather than on
    the terminal branches. The floor applies to the whole pass. ``clades``: optional list that receives the triggering
    internal node indices.

    ``date_ranges``: a tip whose date is a range (``dvar`` > 0) is judged against the range rather than its midpoint:
    its terminal residual is the distance from the tip position its branch implies to the nearest end of the range (0
    inside it), so a month-only or year-only date is flagged only when it is wrong by more than the range.

    Returns (dates with flagged tips set to nan, flagged node indices in flag order, list of (robust rate, number
    flagged) per pass)."""
    par, tip, date, y = _arrays(f)
    date = date.copy()
    V = y + lsd2_b(f, L)
    half = np.sqrt(3.0 * _dvar(f)) if date_ranges else np.zeros(len(par))  # half-width of each range
    flagged, passes = [], []
    n0 = int(np.sum(tip & np.isfinite(date)))
    zc = z if clade_z is None else clade_z
    ch = None
    for _ in range(int(rounds)):
        rho = _robust_rate(par, tip, date, y, np.random.default_rng(seed))
        if not rho > 0:
            passes.append((rho, 0))
            break
        dated = tip & np.isfinite(date)
        x = _node_positions(par, tip, np.where(dated, rho * date, np.nan), y, np.minimum(1.0 / V, 1e9))
        xt = x
        rt = np.flatnonzero(dated & (half > 0))
        if len(rt):  # range tips: the tip position in rho x [lo, hi] nearest the one its branch implies
            xt = x.copy()
            xt[rt] = np.clip(x[par[rt]] + y[rt], rho * (date[rt] - half[rt]), rho * (date[rt] + half[rt]))
        r = (y - np.r_[0.0, xt[1:] - x[par[1:]]])[1:] / np.sqrt(V[1:])
        sd = r.std(ddof=1)
        if not sd > 0:
            passes.append((rho, 0))
            break
        zz = np.r_[0.0, (r - r.mean()) / sd]
        bad = np.flatnonzero((np.abs(zz) > z) & dated)
        hit = []
        if clade_max is not None:
            cnt = dated.astype(int)
            for v in range(len(par) - 1, 0, -1):
                cnt[par[v]] += cnt[v]
            hit = [int(v) for v in np.flatnonzero((np.abs(zz) > zc) & ~tip) if v and 2 <= cnt[v] <= clade_max]
            if hit:
                if ch is None:
                    ch = [[] for _ in par]
                    for v in range(1, len(par)):
                        ch[par[v]].append(v)
                extra = set(bad.tolist())
                for v in hit:
                    st = [v]
                    while st:
                        u = st.pop()
                        if dated[u]:
                            extra.add(u)
                        st.extend(ch[u])
                bad = np.array(sorted(extra), int)
        if len(bad) and int(dated.sum()) - len(bad) < max(3, n0 // 2):
            passes.append((rho, -len(bad)))  # floor: a pass that would leave < half the dated tips is not applied
            break
        passes.append((rho, len(bad)))
        if not len(bad):
            break
        if clades is not None:
            clades.extend(hit)
        date[bad] = np.nan
        flagged.extend(int(b) for b in bad)
    return date, flagged, passes


def ladder_resolve(f, mu, y_order=None):
    """Resolve every node with more than 2 children into a ladder (report 101): children sorted by attachment
    position x_c - y_c, x the unweighted least-squares node positions at rate ``mu`` (dated tips at mu x date,
    no parent clamp). The earliest child stays on the node; later ones hang off a chain of new zero-length
    internal edges. ``y_order``: branch values used only for the ordering (cross-fitting); the resolved tree keeps
    ``f``'s values. Root-to-tip sums are unchanged. O(sum k log k)."""
    par, tip, date, y = _arrays(f)
    yo = y if y_order is None else np.asarray(y_order, float)
    dated = tip & np.isfinite(date)
    x = _node_positions(par, tip, np.where(dated, mu * date, np.nan), yo, np.ones(len(par)))
    ch = [[] for _ in par]
    for v in range(1, len(par)):
        ch[par[v]].append(v)
    dvar = list(_dvar(f))
    par, tip, date, y, name = list(par), list(tip), list(date), list(y), list(f.name)
    for v in range(len(ch)):
        k = len(ch[v])
        if k <= 2:
            continue
        cs = sorted(ch[v], key=lambda c: x[c] - yo[c])
        cur = v
        for j, c in enumerate(cs[1:], start=1):
            if j < k - 1:
                h = len(par)
                par.append(cur); tip.append(False); date.append(np.nan); y.append(0.0); name.append(None)
                dvar.append(0.0)
                par[c] = h
                cur = h
            else:
                par[c] = cur
    return _from_arrays(par, tip, date, y, name, dvar).preorder()


def plain_rate_with_prior(f, rate_prior):
    """Plain root-to-tip slope combined with a rate prior (median mu0, log-scale sd S) as two Gaussian estimates on
    the linear scale (so a non-positive slope still combines): prior sd mu0*S, slope SE from the OLS residuals,
    sqrt(RSS/(n-2) / sum (t - tbar)^2), which ignores the shared-path covariance and so overstates the data's weight.
    Returns (combined rate, plain slope, plain SE, prior share)."""
    par, tip, date, y = _arrays(f)
    n = len(par)
    d = np.zeros(n)
    for v in range(1, n):  # flat trees are in preorder
        d[v] = d[par[v]] + y[v]
    m = tip & np.isfinite(date)
    t, yy = date[m], d[m]
    X = np.c_[np.ones(len(t)), t]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    rss = float(np.sum((yy - X @ beta) ** 2))
    sxx = float(np.sum((t - t.mean()) ** 2))
    se = np.sqrt(rss / max(len(t) - 2, 1) / sxx) if sxx > 0 else np.inf
    mu0, S = rate_prior
    wp = 1.0 / (mu0 * S) ** 2
    wd = 1.0 / se ** 2 if se > 0 else np.inf
    if not np.isfinite(wd):
        return float(beta[1]), float(beta[1]), float(se), 0.0
    b = (wd * beta[1] + wp * mu0) / (wd + wp)
    return float(b), float(beta[1]), float(se), float(wp / (wd + wp))


def _anchored_rate(f, date0):
    """Slope of the root-to-tip distance on (date - date0) with the line through (date0, 0): the rate a root at date0
    implies. Used as the reference rate when a root-date prior is given and the plain slope is not positive."""
    par, tip, date, y = _arrays(f)
    d = np.zeros(len(par))
    for v in range(1, len(par)):
        d[v] = d[par[v]] + y[v]
    m = tip & np.isfinite(date)
    dt = date[m] - date0
    den = float(np.sum(dt * dt))
    return float(np.sum(dt * d[m]) / den) if den > 0 else np.nan


def _reference_rate(f, rate, rate_prior=None, root_date_prior=None):
    """Reference rate of the durations and the ladder: ``rate`` if given; else the plain slope, combined with the rate
    prior (:func:`plain_rate_with_prior`, falling back to the prior median when that is not positive); else, with a
    root-date prior and a non-positive plain slope, :func:`_anchored_rate` at the prior median root date."""
    if rate is not None:
        return float(rate)
    mu = _plain_regression(f)[0]
    if rate_prior is not None:
        mu = plain_rate_with_prior(f, rate_prior)[0]
        if not mu > 0:
            mu = float(rate_prior[0])
    if root_date_prior is not None and not mu > 0:
        mu = _anchored_rate(f, float(root_date_prior[0]))
    return mu


def _apply_rate_prior(reg, ml, var, r, rate_prior, k):
    """Augmented-GLS pseudo-observation on the slope (report 113): minimise chisq(a, b) / k^2 + (b - mu0)^2 /
    (2 sigma^2), sigma = mu0 S (the log-normal prior linearised at its median). TreeTime's chisq is half the weighted
    residual sum of squares, so profiled over the intercept it is chisq_min + D (b - b_gls)^2 / 2 with D =
    1/cov[0,0], and b* = (D b_gls + w mu0) / (D + w), w = k^2 / sigma^2; the intercept is refitted at b* by the
    slope-fixed GLS. The k^2 (se_inflation) puts the prior on the same footing as the inflated SE that is reported.
    Returns the slope-fixed regression dict with the posterior covariance inv(inv(cov) + diag(w, 0)), and the prior's
    share of the slope precision."""
    cov = np.asarray(r['cov'], float)
    if not (cov[0, 0] > 0 and np.isfinite(cov[0, 0])):
        return r, None
    mu0, S = rate_prior
    D = 1.0 / cov[0, 0]
    wp = k * k / (mu0 * S) ** 2
    b = (D * float(r['slope']) + wp * mu0) / (D + wp)
    rf = reg.fit(ml, var, slope=b)
    post = np.linalg.inv(np.linalg.inv(cov) + np.diag([wp, 0.0]))
    return dict(rf, cov=post), float(wp / (D + wp))


def _apply_priors_joint(r, rate_prior, root_date_prior, t_ref, k, b_start=None):
    """Joint posterior mode of (slope b, intercept a) under the GLS likelihood, the rate prior of
    :func:`_apply_rate_prior` (if given) and a LogNormal prior on the root age A = t_ref - root date (median t_ref -
    date0, log-scale sd S_age). In chisq units the objective is
    chisq_min + (theta - theta_gls)' H (theta - theta_gls) / 2 + k^2 [(b - mu0)^2 / (2 sigma^2)
    + (log A - log A0)^2 / (2 S_age^2)], H = inv(cov), minimised over log b and log A (so the rate and the root age
    stay positive), started from the GLS fit, the prior medians and ``b_start`` (the round's reference rate). The
    covariance returned is the inverse Hessian of that objective in (b, a) at the mode (Gauss-Newton term of the age
    prior, which is positive semi-definite), in the same units as the GLS cov (so the SE inflation applies). Returns
    (regression-like dict with slope, intercept, chisq, cov; root-age prior share; rate prior share or None)."""
    from scipy.optimize import minimize
    cov = np.asarray(r['cov'], float)
    H = np.linalg.inv(cov)
    th0 = np.array([float(r['slope']), float(r['intercept'])])
    date0, S_age = float(root_date_prior[0]), float(root_date_prior[1])
    A0 = t_ref - date0
    k2 = k * k
    wr = k2 / (rate_prior[0] * rate_prior[1]) ** 2 if rate_prior is not None else 0.0
    mu0 = rate_prior[0] if rate_prior is not None else 0.0

    def obj(p):
        b, A = np.exp(p[0]), np.exp(p[1])
        d = np.array([b, b * (A - t_ref)]) - th0
        return 0.5 * d @ H @ d + 0.5 * wr * (b - mu0) ** 2 + 0.5 * k2 * (p[1] - np.log(A0)) ** 2 / S_age ** 2

    starts = []
    if th0[0] > 0 and t_ref + th0[1] / th0[0] > 0:
        starts.append((np.log(th0[0]), np.log(t_ref + th0[1] / th0[0])))  # the GLS fit
    v = np.array([1.0, -date0])
    b_anchor = float(v @ H @ th0 / (v @ H @ v))  # best line through the prior median root date
    if b_anchor > 0:
        starts.append((np.log(b_anchor), np.log(A0)))
    if mu0 > 0:
        starts.append((np.log(mu0), np.log(A0)))
    if b_start is not None and b_start > 0 and np.isfinite(b_start):
        starts.append((np.log(b_start), np.log(A0)))
    if not starts:
        raise ValueError('ls_dated_variance_rate: no positive starting rate for the root-date prior')
    best = None
    for lb, lA in starts:
        res = minimize(obj, np.array([lb, lA]), method='Nelder-Mead',
                       options=dict(xatol=1e-10, fatol=1e-12, maxiter=4000))
        res = minimize(obj, res.x, method='BFGS', options=dict(gtol=1e-10))
        if best is None or res.fun < best.fun:
            best = res
    b, A = float(np.exp(best.x[0])), float(np.exp(best.x[1]))
    a = b * (A - t_ref)
    d = np.array([b, a]) - th0
    # Hessian in (b, a): data + rate prior + Gauss-Newton term of the age prior; g = log A, A = t_ref + a / b
    grad_g = np.array([-a / b ** 2, 1.0 / b]) / A
    Hage = k2 * np.outer(grad_g, grad_g) / S_age ** 2
    Hpost = H + np.diag([wr, 0.0]) + Hage
    post = np.linalg.inv(Hpost)
    # prior shares: of the root-date precision (delta method on -a/b) and of the slope precision
    gT = np.array([a / b ** 2, -1.0 / b])
    prec_T_data = 1.0 / float(gT @ np.linalg.inv(H + np.diag([wr, 0.0])) @ gT)
    prec_T_post = 1.0 / float(gT @ post @ gT)
    age_share = float(1.0 - prec_T_data / prec_T_post) if prec_T_post > 0 else None
    rate_share = float(wr / (1.0 / post[0, 0])) if wr else None
    out = dict(r, slope=b, intercept=a, chisq=float(r['chisq'] + 0.5 * d @ H @ d), cov=post)
    return out, age_share, rate_share


def _lsdv_core(f, L, tip_slack, rate, rounds, internal_b=0.0, first_pass='ls', date_ranges=False, rate_prior=None,
               root_date_prior=None, t_ref=None, k=2.0):
    """lsdv's fit on flat tree ``f`` (report 93), plus the opt-in internal smoothing constant and first pass.
    ``date_ranges``: a range date's variance (width^2 / 12, ``dvar``) is added to its tip's variance, in distance
    units (times the round's reference rate squared). ``rate_prior`` / ``root_date_prior``: applied to every round's
    fit (:func:`_apply_rate_prior` alone, :func:`_apply_priors_joint` with a root-date prior), with the reference
    rate of :func:`_reference_rate`. Returns (result dict, final variances, fitted regression dict)."""
    om = 1.0 / L
    plain_rate, plain_icpt, r_val = _plain_regression(f)
    mu = _reference_rate(f, rate, rate_prior, root_date_prior)
    if not (mu > 0 and np.isfinite(mu)) and not (first_pass == 'lsd2' and int(rounds) <= 1):
        raise ValueError('ls_dated_variance_rate: reference rate not positive')
    tip = np.asarray(f.tip)
    ml = np.asarray(f.ml, float)
    reg = _Regression(f)
    extra = np.where(tip, float(tip_slack) ** 2 * om, 0.0)
    if internal_b:
        extra = extra + np.where(tip, 0.0, float(internal_b) * om)
    dv = _dvar(f) if date_ranges else None
    if dv is not None and not np.any(dv > 0):
        dv = None
    round_rates = []
    first_rate = None
    for i in range(max(1, int(rounds))):
        if i == 0 and first_pass == 'lsd2':
            dur, first_rate = _lsd2_first_pass(f, L)
            mu_dur = first_rate
        else:
            dur = _ls_durations(f, mu)
            mu_dur = mu
        var = (dur + extra) * om
        if dv is not None:  # date uncertainty of range tips, in root-to-tip distance units
            var = var + np.where(tip, mu_dur * mu_dur * dv, 0.0)
        var[0] = om * om
        r = reg.fit(ml, var)
        if root_date_prior is not None:
            gls = (float(r['slope']), float(r['intercept']))
            r, age_share, prior_share = _apply_priors_joint(r, rate_prior, root_date_prior, t_ref, k, b_start=mu_dur)
        elif rate_prior is not None:
            gls = (float(r['slope']), float(r['intercept']))
            r, prior_share = _apply_rate_prior(reg, ml, var, r, rate_prior, k)
        slope, icpt = float(r['slope']), float(r['intercept'])
        round_rates.append(slope)
        if not slope > 0:
            break
        mu = slope
    out = dict(rate=slope, intercept=icpt, root_date=(-icpt / slope) if slope > 0 else np.nan,
               chisq=float(r['chisq']), plain_rate=plain_rate, plain_intercept=plain_icpt, r_val=r_val,
               round_rates=round_rates, n_nodes=len(f.par))
    if first_rate is not None:
        out['first_pass_rate'] = first_rate
    if rate_prior is not None or root_date_prior is not None:
        out.update(gls_rate=gls[0], gls_root_date=(-gls[1] / gls[0]) if gls[0] > 0 else None)
    if rate_prior is not None:
        out.update(rate_prior=[float(x) for x in rate_prior], prior_share=prior_share)
    if root_date_prior is not None:
        out.update(root_date_prior=[float(x) for x in root_date_prior], root_date_prior_share=age_share,
                   root_date_prior_t_ref=float(t_ref))
    return out, var, r


def _se_fields(r, rate, icpt, k):
    """Model SE of the rate (TreeTime's GLS inverse curvature, sqrt(cov[0,0]), report 97), the root-date SE by the
    delta method, and the log-symmetric interval rate x exp(+-1.96 k SE / rate)."""
    cov = np.asarray(r['cov'], float)
    se = float(np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else np.nan
    out = dict(rate_se=se, se_rel=float(se / rate) if rate > 0 else None, cov=cov.tolist(), se_inflation=float(k),
               rate_ci=None, root_date_se=None)
    if rate > 0 and np.isfinite(se):
        h = 1.96 * k * se / rate
        out['rate_ci'] = [float(rate * np.exp(-h)), float(rate * np.exp(h))]
        g = np.array([icpt / rate ** 2, -1.0 / rate])  # d(-icpt/rate) / d(rate, icpt)
        out['root_date_se'] = float(np.sqrt(max(g @ cov @ g, 0.0)))
    return out


def _fit_resolved(f, L, tip_slack, rate, rounds, internal_b, first_pass, method, seed, xo_reps=2, date_ranges=False,
                  rate_prior=None, root_date_prior=None, t_ref=None, k=2.0):
    """Resolution (None / 'ladder' / 'ladder_xo') + core fit. Returns (result, fitted flat tree, variances, reg).
    With a prior, the reference rate of the ladder ordering is :func:`_reference_rate`'s, and in ``ladder_xo`` each
    member fit gets the prior (each member uses all the counts, so the prior is not counted twice in their geometric
    mean)."""
    pk = dict(rate_prior=rate_prior, root_date_prior=root_date_prior, t_ref=t_ref, k=k)
    if method is None:
        out, var, r = _lsdv_core(f, L, tip_slack, rate, rounds, internal_b, first_pass, date_ranges, **pk)
        return out, f, var, r
    mu0 = _reference_rate(f, rate, rate_prior, root_date_prior)
    if not (mu0 > 0 and np.isfinite(mu0)):
        raise ValueError('ls_dated_variance_rate: reference rate not positive')
    if method == 'ladder':
        R = ladder_resolve(f, mu0)
        out, var, r = _lsdv_core(R, L, tip_slack, rate, rounds, internal_b, first_pass, date_ranges, **pk)
        out.update(n_nodes_unresolved=len(f.par))
        return out, R, var, r
    # ladder_xo: order by a Poisson-thinned half, fit on all counts, both halves, xo_reps draws; geometric mean
    rng = np.random.default_rng(seed)
    y = np.asarray(f.ml, float)
    k = np.rint(np.maximum(y, 0) * L).astype(np.int64)
    fits = []
    for _ in range(xo_reps):
        k1 = rng.binomial(k, 0.5)
        for h in (2.0 * k1 / L, 2.0 * (k - k1) / L):
            mu = _reference_rate(_with(f, ml=h), rate, rate_prior, root_date_prior)
            if not mu > 0:
                continue
            R = ladder_resolve(f, mu, y_order=h)
            o, var, r = _lsdv_core(R, L, tip_slack, rate, rounds, internal_b, first_pass, date_ranges, **pk)
            if o['rate'] > 0:
                fits.append((o, R, var, r))
    if not fits:
        raise ValueError('ls_dated_variance_rate: no positive cross-fitted ladder rate')
    lr = np.log([o['rate'] for o, *_ in fits])
    rate_xo = float(np.exp(lr.mean()))
    # report the member fit closest to the geometric mean for the tree, variances and curvature; rate and root date
    # are the cross-fit averages
    j = int(np.argmin(np.abs(lr - lr.mean())))
    out = dict(fits[j][0])
    icpts = np.array([o['intercept'] for o, *_ in fits])
    roots = np.array([o['root_date'] for o, *_ in fits])
    out.update(rate=rate_xo, root_date=float(np.mean(roots)), intercept=float(-rate_xo * np.mean(roots)),
               xo_rates=[float(o['rate']) for o, *_ in fits], xo_root_dates=roots.tolist(),
               xo_intercepts=icpts.tolist(), n_nodes_unresolved=len(f.par))
    return out, fits[j][1], fits[j][2], fits[j][3]


def _resample_interval(R, L, var, out, refit, n, q, seed):
    """LSD2-style resampling interval (report 98's Poisson x lognormal, with report 101's handling of resolution):
    counts ~ Poisson(L x fitted duration) x lognormal(mean 1, sd q) on the fitted tree (the resolved tree when
    ``resolve`` is set), zero-count internal edges collapsed again (as on a fitted tree; report 98 kept the tree
    fixed), the same estimator refitted (resolution included); 2.5 / 97.5% quantiles of rate and root date, widened to
    include the point estimate. Skipped (no ``resample`` field) when the point rate is not positive."""
    par, tip, date, y = _arrays(R)
    rate = out['rate']
    dated = tip & np.isfinite(date)
    w = np.where(var > 0, 1.0 / np.where(var > 0, var, 1.0), 1e18)
    x = _node_positions(par, tip, np.where(dated, rate * date, np.nan), y, np.minimum(w, 1e18))
    B = np.maximum(np.r_[0.0, x[1:] - x[par[1:]]], 0.0)
    rng = np.random.default_rng(seed)
    s2 = np.log(q * q + 1.0)
    rates, roots = [], []
    for _ in range(int(n)):
        k = rng.poisson(B * L).astype(float)
        if q > 0:
            k *= rng.lognormal(-s2 / 2, np.sqrt(s2), size=len(k))
        try:
            o = refit(_collapse_flat(_with(R, ml=k / L), COLLAPSE_MUTATIONS / L))
            rates.append(o['rate']); roots.append(o['root_date'])
        except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
            rates.append(np.nan); roots.append(np.nan)
    rates, roots = np.array(rates), np.array(roots)
    ok = np.isfinite(rates) & (rates > 0) & np.isfinite(roots)
    res = dict(n=int(n), q=float(q), n_ok=int(ok.sum()), rate_ci=None, root_date_ci=None, sd_log_rate=None)
    if ok.sum() >= 3:
        rq = np.quantile(rates[ok], [0.025, 0.975]); dq = np.quantile(roots[ok], [0.025, 0.975])
        res.update(rate_ci=[float(min(rq[0], rate)), float(max(rq[1], rate))],
                   root_date_ci=[float(min(dq[0], out['root_date'])), float(max(dq[1], out['root_date']))],
                   sd_log_rate=float(np.std(np.log(rates[ok]))))
    return res


def _date_fields(out, factor):
    """Root-date interval and the don't-date flag of a fit (report 114): ``root_date_ci`` = root date +- 1.96 x
    ``factor`` x root_date_se; ``r2`` = plain root-to-tip R^2 on the dates the fit uses (after the outlier rule);
    ``date_reliable`` False when se_rel > DATE_SE_REL_MAX, r2 < DATE_R2_MIN or the rate is not positive, with the
    reasons in ``date_flags``."""
    r2 = float(out['r_val'] ** 2) if np.isfinite(out['r_val']) else float('nan')
    ok = out['rate'] > 0 and out.get('root_date_se') is not None and np.isfinite(out['root_date_se'])
    h = 1.96 * factor * out['root_date_se'] if ok else None
    flags = []
    if not out['rate'] > 0:
        flags.append('rate not positive')
    elif out.get('se_rel') is None or not out['se_rel'] <= DATE_SE_REL_MAX:
        flags.append('se_rel > %g' % DATE_SE_REL_MAX)
    if not r2 >= DATE_R2_MIN:
        flags.append('r2 < %g' % DATE_R2_MIN)
    return dict(root_date_ci=[out['root_date'] - h, out['root_date'] + h] if ok else None,
                root_date_ci_factor=float(factor), r2=r2, date_reliable=not flags, date_flags=flags)


def _report93_like(o):
    """True when the effective options ``o`` give report 93's estimator (no resolution, outlier rule, internal b,
    LSD2 first pass or resampling)."""
    return (o['resolve'] is None and o['outlier_z'] is None and not o['internal_b'] and o['first_pass'] == 'ls'
            and not o['resample'] and o.get('rate_prior') is None and o.get('root_date_prior') is None)


def lsdv_reports_extra(options):
    """Whether an lsdv fit with these options reports its option fields (``resolve``, ``outliers``, ...) and, in
    :mod:`treetime.root_candidates`, the SE alongside its rate. False only for report 93's estimator
    (:data:`LSDV_REPORT93`) without ``se_inflation`` passed, so report 93's outputs stay byte-identical."""
    return not _report93_like(dict(LSDV_OPTIONS, **options)) or 'se_inflation' in options


def lsdv_flat(f, L, tip_slack=3.0, rate=None, rounds=1, **options):
    """:func:`ls_dated_variance_rate` on a flat tree (:func:`_flatten`) with ``L`` alignment sites. Returns the
    same dict. ``options``: overrides of :data:`LSDV_OPTIONS` (see :func:`ls_dated_variance_rate`)."""
    t0 = time.time()
    if tip_slack < 0:
        raise ValueError('ls_dated_variance_rate: tip_slack must be >= 0')
    if 'collapse' in options or 'rate' in options:
        raise ValueError('lsdv_flat: collapse and rate are not options here (collapse when flattening)')
    o = check_lsdv_options(options) if options else LSDV_OPTIONS
    ranges = o['date_ranges'] == 'variance'
    t_ref = None
    if o['root_date_prior'] is not None:
        dd = np.asarray(f.date, float)[np.asarray(f.tip, bool)]
        dd = dd[np.isfinite(dd)]
        t_ref = float(dd.max()) if len(dd) else np.nan
        if not t_ref > float(o['root_date_prior'][0]):
            raise ValueError('lsdv: root_date_prior median must be earlier than the latest tip date')
    info = {}
    g = f
    if o['outlier_z'] is not None:
        clades = []
        dates, flagged, passes = lsd2_outliers(f, L, o['outlier_z'], rounds=o['outlier_rounds'], seed=0,
                                               clade_max=o['outlier_clade_max'], clade_z=o['outlier_clade_z'],
                                               clades=clades, date_ranges=ranges)
        info.update(outliers=[f.name[i] for i in flagged], outlier_nodes=flagged, outlier_mode=o['outlier_mode'],
                    outlier_passes=[[float(a), int(b)] for a, b in passes])
        if o['outlier_clade_max'] is not None:
            info['outlier_clades'] = len(clades)
        if o['outlier_mode'] == 'drop' and flagged:
            g = _with(f, date=dates)
    method = o['resolve']
    if method == 'auto':
        n_tips = int(np.sum(f.tip))
        big = n_tips > o['resolve_auto_tips']
        if big and o['resolve_auto_r2'] is not None:
            try:  # plain root-to-tip R^2 on the dates that remain after the outlier rule
                info['auto_r2'] = float(_plain_regression(g)[2] ** 2)
            except ValueError:
                info['auto_r2'] = float('nan')
            big = info['auto_r2'] < o['resolve_auto_r2']
        method = 'ladder_xo' if big else 'ladder'
    out, R, var, r = _fit_resolved(g, L, tip_slack, rate, rounds, o['internal_b'], o['first_pass'], method,
                                   o['seed'], date_ranges=ranges, rate_prior=o['rate_prior'],
                                   root_date_prior=o['root_date_prior'], t_ref=t_ref, k=o['se_inflation'])
    out.update(_se_fields(r, out['rate'], out['intercept'], o['se_inflation']))
    if 'xo_rates' in out and r['slope'] > 0 and np.isfinite(out['rate_se']):
        # cross-fit: keep the member fit's relative SE around the geometric-mean rate
        out['rate_se'] = float(out['rate_se'] * out['rate'] / r['slope'])
        out['se_rel'] = float(out['rate_se'] / out['rate'])
        h = 1.96 * o['se_inflation'] * out['rate_se'] / out['rate']
        out['rate_ci'] = [out['rate'] * float(np.exp(-h)), out['rate'] * float(np.exp(h))]
    out.update(_date_fields(out, o['root_date_ci_factor']))
    if lsdv_reports_extra(options):
        out.update(info, resolve=method, internal_b=float(o['internal_b']), first_pass=o['first_pass'])
        if o['resample'] and out['rate'] > 0:
            ro = dict(o, resample=0, outlier_z=None, resolve=method)
            refit = lambda h: lsdv_flat(h, L, tip_slack=tip_slack, rate=rate, rounds=rounds, **ro)  # noqa: E731
            out['resample'] = _resample_interval(R, L, var, out, refit, o['resample'], o['resample_q'], o['seed'])
    out['seconds'] = time.time() - t0
    return out


def ls_dated_variance_rate(tt, tip_slack=3.0, rate=None, rounds=1, collapse=True, **options):
    """Clock rate at the current root of TreeTime object ``tt`` from the
    covariation regression with least-squares-dated variances (lsdv; module
    docstring).

    Branch values are ``mutation_length``; branch variances are
    (duration + tip_slack^2 / L [terminal branches]) / L. Durations: one
    least-squares node-time solve at the reference rate (``rate``, default the
    plain root-to-tip rate at this root). ``rounds`` > 1 re-solves the node
    times at the previous round's rate. ``collapse``: zero-length internal edges
    are collapsed first. Outlier tips (``bad_branch``) and undated tips carry no
    date. Raises ``ValueError`` when the reference rate is not positive.

    Returns a dict: rate, intercept, root_date (nan unless rate > 0), chisq (the
    GLS residual of the fit), plain_rate, plain_intercept, r_val (of the plain
    regression), round_rates, n_nodes (after collapsing), seconds; and the model
    SE (report 97): rate_se = sqrt(cov[0,0]) of the GLS fit, se_rel = rate_se /
    rate (above ~0.5 the interval is uninformative), cov (2x2, slope and
    intercept), root_date_se (delta method), se_inflation and
    rate_ci = rate x exp(+-1.96 se_inflation rate_se / rate).

    ``options`` (defaults in :data:`LSDV_OPTIONS`, set by reports 107 and 111; each one
    can be overridden, and ``**LSDV_REPORT93`` = ``resolve=None,
    outlier_z=None`` gives report 93's estimator exactly):

    - ``resolve``: 'auto' (default) = 'ladder', or 'ladder_xo' when there are
      more than ``resolve_auto_tips`` (1000) tips and the plain root-to-tip R^2,
      computed after the outlier dates are dropped (in 'flag' mode on all
      dates), is below
      ``resolve_auto_r2`` (0.2; None = switch on size alone, report 103's
      rule with ``resolve_auto_tips=2000``). 'ladder' resolves every polytomy
      of the collapsed tree by attachment order before the fit (report 101);
      'ladder_xo' orders by a Poisson-thinned half of the counts and averages
      4 fits on the log scale; None = no resolution. With 'auto' and the size
      condition met, ``auto_r2`` reports the R^2 used.
    - ``outlier_z``: threshold of LSD2's ``-e`` rule (4.5), iterated up to
      ``outlier_rounds`` (5) passes (report 98/101); None = off. A pass that
      would leave fewer than half the dated tips is not applied.
      ``outlier_mode`` 'drop' (default: flagged tips lose their date before
      the fit) or 'flag' (reported only). ``outlier_clade_max`` (4) and
      ``outlier_clade_z`` (4.5; None = ``outlier_z``): the clade extension, an
      internal branch with |z| > ``outlier_clade_z`` above 2 to
      ``outlier_clade_max`` dated tips flags all of them (report 107);
      ``outlier_clade_max=None`` turns it off (``outlier_clade_z`` is then
      ignored). Reported: ``outliers`` (names),
      ``outlier_nodes``, ``outlier_passes`` ([robust rate, number flagged] per
      pass) and, with the clade extension on, ``outlier_clades`` (number of
      internal branches that triggered).
    - ``internal_b``: LSD2-style constant added to every internal branch's
      variance, in mutations (b x L in LSD2 units; LSD2's default is 10). 0 = off
      (default).
    - ``first_pass``: 'ls' (default; lsdv's unweighted LS node times) or 'lsd2'
      (LSD2's run 1: joint WLS with V = B + b) for the first round's durations;
      with 'lsd2' round 1 uses neither the plain rate nor ``rate``, but the
      ladder still needs a positive reference rate, so without one pass
      ``resolve=None`` as well.
    - ``se_inflation``: factor on the SE in ``rate_ci`` (2, reports 97/101/107).
    - ``rate_prior``: None (default) or (mu0, S), a LogNormal prior on the
      rate with median mu0 and log-scale sd S, entered as a pseudo-observation
      on the slope of the GLS fit, linearised at the median and weighted
      against the curvature of the SE-inflated fit (report 113). It also
      stabilises the reference rate (plain slope combined with the prior),
      so a non-positive plain slope no longer stops the fit. The outlier rule
      runs before it and never sees it. Reported: ``rate_prior``,
      ``prior_share`` (the prior's share of the slope precision),
      ``gls_rate`` and ``gls_root_date`` (the fit without priors); ``cov``,
      ``rate_se`` and ``root_date_se`` are posterior.
    - ``root_date_prior``: None (default) or (date0, S_age), a LogNormal
      prior on the root age A = t_ref - root date, t_ref the latest tip date
      (before the outlier rule), with median t_ref - date0 and log-scale sd
      S_age. Applied on the log scale, jointly with ``rate_prior`` when both
      are given: the posterior mode over log rate and log A of the GLS
      likelihood times the priors, so neither the rate nor the root age can
      become negative; the covariance is the inverse Hessian at the mode
      (Gauss-Newton term for the age prior). Reported:
      ``root_date_prior``, ``root_date_prior_share`` (the prior's share of the
      root-date precision), ``root_date_prior_t_ref``.
    - ``root_date_ci_factor``: factor k of the root-date interval
      ``root_date_ci`` = root_date +- 1.96 x k x root_date_se (1.5). The
      delta-method SE alone under-covers; report 114 found that 95% coverage
      at the true root needs k of about 1.4 at 80 tips, 1.55-1.6 at 100-1000
      tips and about 2.2 at 10,000 tips, so 1.5 under-covers on large trees.
      ``root_date_se`` itself is reported unscaled.
    - ``date_ranges``: how a tip date given as a range [lo, hi] (a month-only
      or year-only date, TreeTime's ``raw_date_constraint``) enters.
      'variance' (default): the regression uses the midpoint and adds the
      range's variance, width^2 / 12 times the reference rate squared, to that
      tip's variance, so the tip weighs less; the outlier rule judges the tip
      by its distance to the nearest end of the range, so a correct partial
      date is not flagged. 'midpoint': the midpoint as an exact date
      (before report 116).
    - ``resample``: number of LSD2-style resampling replicates (0 = off) for
      ``resample['rate_ci']`` and ``resample['root_date_ci']``; ``resample_q``
      the lognormal SD (0.2); ``seed`` for the resampling and the cross-fit.

    Every fit reports ``root_date_ci``, ``root_date_ci_factor``, ``r2`` (plain
    root-to-tip R^2 on the dates used) and the don't-date flag
    ``date_reliable``: False when ``se_rel`` > 0.5 (DATE_SE_REL_MAX), ``r2`` <
    0.2 (DATE_R2_MIN) or the rate is not positive, with the reasons in
    ``date_flags`` (report 114: these flags pick out the large weak-signal
    trees whose dates are hopeless, but cannot catch a narrow interval around
    a wrong date; with a prior, ``se_rel`` is the posterior one, so prior
    information can make a date reliable).

    Every fit reports the option fields above plus ``resolve`` (the method
    used), ``internal_b`` and ``first_pass``, except report 93's estimator
    without ``se_inflation`` passed (:func:`lsdv_reports_extra`).
    """
    L = int(tt.data.full_length)
    return lsdv_flat(_flatten(tt.tree, 1.0 / L, collapse=collapse), L, tip_slack=tip_slack, rate=rate, rounds=rounds,
                     **options)
