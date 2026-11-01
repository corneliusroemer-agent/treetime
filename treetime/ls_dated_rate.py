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


def lsdv_flat(f, L, tip_slack=3.0, rate=None, rounds=1):
    """:func:`ls_dated_variance_rate` on a flat tree (:func:`_flatten`) with
    ``L`` alignment sites. Returns the same dict."""
    t0 = time.time()
    if tip_slack < 0:
        raise ValueError('ls_dated_variance_rate: tip_slack must be >= 0')
    om = 1.0 / L
    plain_rate, plain_icpt, r_val = _plain_regression(f)
    mu = plain_rate if rate is None else float(rate)
    if not (mu > 0 and np.isfinite(mu)):
        raise ValueError('ls_dated_variance_rate: reference rate not positive')
    tip = np.asarray(f.tip)
    ml = np.asarray(f.ml, float)
    reg = _Regression(f)
    extra = np.where(tip, float(tip_slack) ** 2 * om, 0.0)
    round_rates = []
    for _ in range(max(1, int(rounds))):
        var = (_ls_durations(f, mu) + extra) * om
        var[0] = om * om
        r = reg.fit(ml, var)
        slope, icpt = float(r['slope']), float(r['intercept'])
        round_rates.append(slope)
        if not slope > 0:
            break
        mu = slope
    return dict(rate=slope, intercept=icpt, root_date=(-icpt / slope) if slope > 0 else np.nan,
                chisq=float(r['chisq']), plain_rate=plain_rate, plain_intercept=plain_icpt, r_val=r_val,
                round_rates=round_rates, n_nodes=len(f.par), seconds=time.time() - t0)


def ls_dated_variance_rate(tt, tip_slack=3.0, rate=None, rounds=1, collapse=True):
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
    regression), round_rates, n_nodes (after collapsing), seconds.
    """
    L = int(tt.data.full_length)
    return lsdv_flat(_flatten(tt.tree, 1.0 / L, collapse=collapse), L, tip_slack=tip_slack, rate=rate, rounds=rounds)
