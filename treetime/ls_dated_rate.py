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

Cost is linear in the number of nodes (the postorder half of TreeRegression's
recursion only), about 1.5 ms at 80 tips and 15 ms at 1000 tips.
"""
import time

import numpy as np
from Bio.Phylo.BaseTree import Clade, Tree

from .treeregression import TreeRegression, base_regression

COLLAPSE_MUTATIONS = 0.1  # internal edges shorter than this (in mutations) and without length are collapsed


class _Flat:
    """Rooted tree as parallel lists, parents before children after
    :meth:`preorder` (root = 0). Per node: par, tip, date (nan if none), ml
    (branch value, mutation_length), name."""

    def __init__(self):
        self.par, self.tip, self.date, self.ml, self.name = [], [], [], [], []

    def add(self, par, tip, date, ml, name=None):
        self.par.append(par)
        self.tip.append(tip)
        self.date.append(date)
        self.ml.append(ml)
        self.name.append(name)
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
            out.add(new[self.par[v]] if self.par[v] >= 0 else -1, self.tip[v], self.date[v], self.ml[v], self.name[v])
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


def _flatten(tree, om, collapse=True, length=None, date=None):
    """Flat copy of a rooted Bio.Phylo tree. Branch values are ``length(node)``
    (default ``mutation_length``), tip dates ``date(node)`` (default
    :func:`_tip_date`). With ``collapse``, internal edges of length below
    COLLAPSE_MUTATIONS mutations (``om`` = one mutation) are collapsed: their
    length is pushed onto the children, so every root-to-tip sum is kept."""
    length = length or (lambda n: n.mutation_length)
    date = date or _tip_date
    f = _Flat()
    stack = [(tree.root, -1, 0.0)]
    while stack:
        n, p, add = stack.pop()
        ml = 0.0 if n is tree.root else max(0.0, float(length(n) or 0.0)) + add
        if collapse and n is not tree.root and not n.is_terminal() and p >= 0 and ml < COLLAPSE_MUTATIONS * om:
            for c in reversed(n.clades):  # children attach to p, inheriting this edge
                stack.append((c, p, ml))
            continue
        i = f.add(p, n.is_terminal(), date(n) if n.is_terminal() else np.nan, ml, n.name)
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


class _Regression:
    """TreeRegression on a light Bio.Phylo copy of a flat tree, with branch
    values and variances supplied as arrays per fit. Only the postorder half of
    TreeRegression's recursion is run (the root's averages are all a slope and
    intercept need)."""

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


LSDV_OPTIONS = dict(resolve=None, resolve_auto_tips=2000, outlier_z=None, outlier_mode='drop', outlier_rounds=5,
                    internal_b=0.0, first_pass='ls', se_inflation=2.0, resample=0, resample_q=0.2, seed=1)
"""The opt-in lsdv options (report 103) and their defaults. With every option at its default, lsdv is the estimator
of report 93 and its rate, intercept and chisq are unchanged."""


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


def _from_arrays(par, tip, date, ml, name):
    out = _Flat()
    for i in range(len(par)):
        out.add(int(par[i]), bool(tip[i]), float(date[i]) if tip[i] else np.nan, float(ml[i]), name[i])
    return out


def _with(f, date=None, ml=None):
    """Copy of flat tree ``f`` with new tip dates and/or branch values (same topology and numbering)."""
    par, tip, d0, m0 = _arrays(f)
    return _from_arrays(par, tip, d0 if date is None else date, m0 if ml is None else ml, f.name)


def _collapse_flat(f, thr):
    """Collapse internal non-root edges with value < thr (value pushed onto the children), as :func:`_flatten`."""
    par, tip, date, ml = _arrays(f)
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
        i = out.add(p, bool(tip[v]), float(date[v]) if tip[v] else np.nan, y, f.name[v])
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


def lsd2_outliers(f, L, z, rounds=5, seed=0):
    """LSD2's ``-e z`` outlier rule at a fixed root, iterated (report 98, 101): robust median-of-slopes rate, LD node
    times at that rate with V = B + b, branch residuals (B - duration) / sqrt(V) standardised by their mean and SD
    over all branches; dated tips with |z-score| > z lose their date, and the rule is repeated on the remaining dated
    tips until it flags nothing new (at most ``rounds`` passes). A pass that would leave fewer than half of the
    originally dated tips (or fewer than 3) is not applied; it is recorded with a negative count. Returns (dates with
    flagged tips set to nan, flagged node indices in flag order, list of (robust rate, number flagged) per pass)."""
    par, tip, date, y = _arrays(f)
    date = date.copy()
    V = y + lsd2_b(f, L)
    flagged, passes = [], []
    n0 = int(np.sum(tip & np.isfinite(date)))
    for _ in range(int(rounds)):
        rho = _robust_rate(par, tip, date, y, np.random.default_rng(seed))
        if not rho > 0:
            passes.append((rho, 0))
            break
        dated = tip & np.isfinite(date)
        x = _node_positions(par, tip, np.where(dated, rho * date, np.nan), y, np.minimum(1.0 / V, 1e9))
        r = (y - np.r_[0.0, x[1:] - x[par[1:]]])[1:] / np.sqrt(V[1:])
        sd = r.std(ddof=1)
        if not sd > 0:
            passes.append((rho, 0))
            break
        zz = np.r_[0.0, (r - r.mean()) / sd]
        bad = np.flatnonzero((np.abs(zz) > z) & dated)
        if len(bad) and int(dated.sum()) - len(bad) < max(3, n0 // 2):
            passes.append((rho, -len(bad)))  # floor: a pass that would leave < half the dated tips is not applied
            break
        passes.append((rho, len(bad)))
        if not len(bad):
            break
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
                par[c] = h
                cur = h
            else:
                par[c] = cur
    return _from_arrays(par, tip, date, y, name).preorder()


def _lsdv_core(f, L, tip_slack, rate, rounds, internal_b=0.0, first_pass='ls'):
    """lsdv's fit on flat tree ``f`` (report 93), plus the opt-in internal smoothing constant and first pass.
    Returns (result dict, final variances, fitted regression dict)."""
    om = 1.0 / L
    plain_rate, plain_icpt, r_val = _plain_regression(f)
    mu = plain_rate if rate is None else float(rate)
    if not (mu > 0 and np.isfinite(mu)) and not (first_pass == 'lsd2' and int(rounds) <= 1):
        raise ValueError('ls_dated_variance_rate: reference rate not positive')
    tip = np.asarray(f.tip)
    ml = np.asarray(f.ml, float)
    reg = _Regression(f)
    extra = np.where(tip, float(tip_slack) ** 2 * om, 0.0)
    if internal_b:
        extra = extra + np.where(tip, 0.0, float(internal_b) * om)
    round_rates = []
    first_rate = None
    for i in range(max(1, int(rounds))):
        if i == 0 and first_pass == 'lsd2':
            dur, first_rate = _lsd2_first_pass(f, L)
        else:
            dur = _ls_durations(f, mu)
        var = (dur + extra) * om
        var[0] = om * om
        r = reg.fit(ml, var)
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


def _fit_resolved(f, L, tip_slack, rate, rounds, internal_b, first_pass, method, seed, xo_reps=2):
    """Resolution (None / 'ladder' / 'ladder_xo') + core fit. Returns (result, fitted flat tree, variances, reg)."""
    if method is None:
        out, var, r = _lsdv_core(f, L, tip_slack, rate, rounds, internal_b, first_pass)
        return out, f, var, r
    plain_rate = _plain_regression(f)[0]
    mu0 = plain_rate if rate is None else float(rate)
    if not (mu0 > 0 and np.isfinite(mu0)):
        raise ValueError('ls_dated_variance_rate: reference rate not positive')
    if method == 'ladder':
        R = ladder_resolve(f, mu0)
        out, var, r = _lsdv_core(R, L, tip_slack, rate, rounds, internal_b, first_pass)
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
            mu = _plain_regression(_with(f, ml=h))[0] if rate is None else float(rate)
            if not mu > 0:
                continue
            R = ladder_resolve(f, mu, y_order=h)
            o, var, r = _lsdv_core(R, L, tip_slack, rate, rounds, internal_b, first_pass)
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


def lsdv_flat(f, L, tip_slack=3.0, rate=None, rounds=1, **options):
    """:func:`ls_dated_variance_rate` on a flat tree (:func:`_flatten`) with ``L`` alignment sites. Returns the
    same dict. ``options``: the opt-in variants of :data:`LSDV_OPTIONS` (see :func:`ls_dated_variance_rate`)."""
    t0 = time.time()
    if tip_slack < 0:
        raise ValueError('ls_dated_variance_rate: tip_slack must be >= 0')
    if 'collapse' in options or 'rate' in options:
        raise ValueError('lsdv_flat: collapse and rate are not options here (collapse when flattening)')
    o = check_lsdv_options(options) if options else LSDV_OPTIONS
    info = {}
    g = f
    if o['outlier_z'] is not None:
        dates, flagged, passes = lsd2_outliers(f, L, o['outlier_z'], rounds=o['outlier_rounds'], seed=0)
        info.update(outliers=[f.name[i] for i in flagged], outlier_nodes=flagged, outlier_mode=o['outlier_mode'],
                    outlier_passes=[[float(a), int(b)] for a, b in passes])
        if o['outlier_mode'] == 'drop' and flagged:
            g = _with(f, date=dates)
    method = o['resolve']
    if method == 'auto':
        method = 'ladder_xo' if int(np.sum(f.tip)) > o['resolve_auto_tips'] else 'ladder'
    out, R, var, r = _fit_resolved(g, L, tip_slack, rate, rounds, o['internal_b'], o['first_pass'], method,
                                   o['seed'])
    out.update(_se_fields(r, out['rate'], out['intercept'], o['se_inflation']))
    if 'xo_rates' in out and r['slope'] > 0 and np.isfinite(out['rate_se']):
        # cross-fit: keep the member fit's relative SE around the geometric-mean rate
        out['rate_se'] = float(out['rate_se'] * out['rate'] / r['slope'])
        out['se_rel'] = float(out['rate_se'] / out['rate'])
        h = 1.96 * o['se_inflation'] * out['rate_se'] / out['rate']
        out['rate_ci'] = [out['rate'] * float(np.exp(-h)), out['rate'] * float(np.exp(h))]
    if options:
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

    Opt-in ``options`` (report 103; defaults in :data:`LSDV_OPTIONS`, with all of
    them at their defaults the estimator is unchanged):

    - ``resolve``: None; 'ladder' resolves every polytomy of the collapsed tree
      by attachment order before the fit (report 101), 'ladder_xo' orders by a
      Poisson-thinned half of the counts and averages 4 fits on the log scale,
      'auto' = 'ladder_xo' above ``resolve_auto_tips`` (2000) tips, else 'ladder'.
    - ``outlier_z``: None (off), or the threshold of LSD2's ``-e`` rule, iterated
      up to ``outlier_rounds`` (5) passes (report 98/101). ``outlier_mode``
      'drop' (flagged tips lose their date before the fit) or 'flag' (reported in
      ``outliers`` only). Reported: ``outliers`` (names), ``outlier_nodes``,
      ``outlier_passes`` ([robust rate, number flagged] per pass).
    - ``internal_b``: LSD2-style constant added to every internal branch's
      variance, in mutations (b x L in LSD2 units; LSD2's default is 10). 0 = off.
    - ``first_pass``: 'ls' (lsdv's unweighted LS node times) or 'lsd2' (LSD2's run
      1: joint WLS with V = B + b) for the first round's durations; with 'lsd2'
      round 1 uses neither the plain rate nor ``rate``.
    - ``se_inflation``: factor on the SE in ``rate_ci`` (2, reports 97/101).
    - ``resample``: number of LSD2-style resampling replicates (0 = off) for
      ``resample['rate_ci']`` and ``resample['root_date_ci']``; ``resample_q``
      the lognormal SD (0.2); ``seed`` for the resampling and the cross-fit.
    """
    L = int(tt.data.full_length)
    return lsdv_flat(_flatten(tt.tree, 1.0 / L, collapse=collapse), L, tip_slack=tip_slack, rate=rate, rounds=rounds,
                     **options)
