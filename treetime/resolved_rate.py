"""Clock rate at a fixed root from a date-resolved copy of the tree with
cross-fitted branch weights (``clock_rate_method='date_resolved_xfit'``).

Collapsed zero-mutation edges bias the covariation regression low: the edges
removed are exactly those whose counts are low, so the covariance depends on
the counts. This estimator works on a private copy of the rooted tree and
leaves the tree it was given untouched:

1. Reference rate: plain root-to-tip regression (OLS of root-to-tip distance on
   tip date). It does not depend on the topology.
2. Collapse internal edges without mutations (length < 0.1 mutations), then
   resolve every polytomy by dates. Node times come from one unweighted least
   squares fit of the branch lengths at the reference rate, with parents then
   moved no later than their children. At each polytomy with parent time t_p
   the pair (i, j) with the largest gain
       mu L (t_n - t_p) + sum_{c in i,j} k_c log((t_c - t_n) / (t_c - t_p)),
   maximised over t_n in [t_p, min(t_i, t_j)], is joined under a new node at
   t_n, until no pair has a gain above 1e-9 (k_c = mutation_length x L). Above
   ``max_exact_polytomy`` children only pairs that are close in attachment time
   t_c - k_c / (mu L) are scored (see ``_resolve_restricted``).
3. Covariation regression (root-to-tip GLS, TreeRegression's recursion) on the
   resolved copy with cross-fitted weights. Alignment columns are split at
   random into halves A and B. On half A the regression is fitted and its
   predicted branch lengths l_b give branch variances (l_b L + floor) / L^2;
   half B's values are regressed with those variances, and vice versa. The
   rate is the mean slope over 2 x ``n_splits`` half fits. The root date is
   refitted once on all values at that rate.

Per-half branch values split each branch's mutation_length in proportion to
its mutations in each half (positions from ancestral reconstruction), scaled to
full length: (L_A v_A + L_B v_B) / L is the full-data value exactly. A branch
with length but no mutation keeps its length in both halves (its noise is then
shared by the halves). Without
mutation positions (no alignment or no reconstruction yet) the per-half counts
are binomial thinnings of round(mutation_length x L).
"""
import math
import time

import numpy as np
from Bio.Phylo.BaseTree import Clade, Tree

from .treeregression import TreeRegression, base_regression

COLLAPSE_MUTATIONS = 0.1  # internal edges shorter than this (in mutations) and without mutations are collapsed


class DateResolvedRate(dict):
    """Result of :func:`date_resolved_xfit_rate`: a dict with the fields
    documented there, plus :meth:`clock_model`."""

    def clock_model(self):
        """TreeTime clock_model dict at the cross-fitted rate."""
        cm = {
            'slope': self['rate'],
            'intercept': self['intercept'],
            'chisq': self['chisq'],
            'cov': self['cov'],
            'valid_confidence': bool(np.all(np.isfinite(self['cov']))),
            'r_val': self['r_val'],
            'date_resolved_xfit': {k: v for k, v in self.items() if k not in ('cov',)},
        }
        return cm


# --------------------------------------------------------------------------------------------------------------
# flat tree


class _Flat:
    """Rooted tree as parallel lists (any order with parents before children
    after :meth:`preorder`). Fields per node: par, tip, date (nan if none),
    ml (mutation_length), pos (mutated columns, or None), k (count used by
    the resolver gain), t (time, resolver only)."""

    def __init__(self):
        self.par, self.tip, self.date, self.ml, self.pos, self.name = [], [], [], [], [], []

    def add(self, par, tip, date, ml, pos, name=None):
        self.par.append(par)
        self.tip.append(tip)
        self.date.append(date)
        self.ml.append(ml)
        self.pos.append(pos)
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
            out.add(new[self.par[v]] if self.par[v] >= 0 else -1, self.tip[v], self.date[v], self.ml[v], self.pos[v],
                    self.name[v])
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


def _flatten(tree, om, positions, collapse=True):
    """Flat copy of a Bio.Phylo tree. With ``collapse``, internal edges with no
    mutation and length < COLLAPSE_MUTATIONS mutations are collapsed (their
    length and mutations pushed onto the children, so every root-to-tip sum is
    kept): TreeTime's own resolved but empty edges, or near-zero edges of the
    input, are re-resolved by dates. ``positions``: id(node) -> mutated
    columns, None for a node without them (thinned), or None overall."""
    f = _Flat()
    stack = [(tree.root, -1, 0.0, None)]
    while stack:
        n, p, add_ml, add_pos = stack.pop()
        if n is tree.root:
            ml, pos = 0.0, (np.zeros(0, int) if positions is not None else None)
        else:
            ml = max(0.0, float(n.mutation_length)) + add_ml
            pos = positions.get(id(n)) if positions is not None else None
            if pos is not None and add_pos is not None and len(add_pos):
                pos = np.concatenate([pos, add_pos])
        if collapse and n is not tree.root and not n.is_terminal() and p >= 0 and ml < COLLAPSE_MUTATIONS * om and (
                pos is None or len(pos) == 0):
            for c in reversed(n.clades):  # collapse: children attach to p, inheriting this edge
                stack.append((c, p, ml, pos))
            continue
        i = f.add(p, n.is_terminal(), _tip_date(n), ml, pos, n.name)
        for c in reversed(n.clades):
            stack.append((c, i, 0.0, None))
    return f.preorder()


# --------------------------------------------------------------------------------------------------------------
# linear-time least squares on the flat tree


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
        raise ValueError('date_resolved_xfit: no dated tip below the root')
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


# --------------------------------------------------------------------------------------------------------------
# date resolution


def _pair_gain(A, tp, ta, tb, ka, kb, tiny):
    """Best gain of joining two children (times ta, tb >= tp, counts ka, kb)
    under parent time tp, and the new node's time. Vectorised over arrays.
    The objective A (u - tp) + ka log(ta - u) + kb log(tb - u) is concave in
    u; its stationary point below min(ta, tb) is the smaller root of
    A u^2 - (A (ta + tb) - ka - kb) u + A ta tb - ka tb - kb ta = 0."""
    ta, tb, ka, kb = np.broadcast_arrays(*(np.asarray(z, float) for z in (ta, tb, ka, kb)))
    tm = np.minimum(ta, tb) - tiny
    B = A * (ta + tb) - ka - kb
    C = A * ta * tb - ka * tb - kb * ta
    disc = np.maximum(B * B - 4 * A * C, 0.0)
    u = (B - np.sqrt(disc)) / (2 * A)
    u = np.clip(u, tp, np.maximum(tm, tp))
    with np.errstate(divide='ignore', invalid='ignore'):
        ga = np.where(ka > 0, ka * np.log((ta - u) / (ta - tp)), 0.0)
        gb = np.where(kb > 0, kb * np.log((tb - u) / (tb - tp)), 0.0)
    g = A * (u - tp) + ga + gb
    g = np.where((tm > tp) & np.isfinite(g), g, 0.0)
    return g, u


def _resolve_exact(A, tp, T, K, tiny, thr):
    """Greedy join at one polytomy with every pair scored (report 85's
    resolve_dates). T, K: times and counts of the children. Returns the joins
    as (i, j, t_new) over growing indices: the new node of the q-th join has
    index len(T) + q. Gains of untouched pairs do not change when other
    children are joined, so each join rescores one row: O(m^2) gains and
    O(m^3) cheap argmax work."""
    m = len(T)
    Ts, Ks = np.array(T, float), np.array(K, float)
    G, U = _pair_gain(A, tp, Ts[:, None], Ts[None, :], Ks[:, None], Ks[None, :], tiny)
    np.fill_diagonal(G, -np.inf)
    idx = list(range(m))  # slot -> node index
    alive = np.ones(m, bool)
    joins = []
    while alive.sum() > 2:
        a, b = np.unravel_index(np.argmax(G), G.shape)
        if not G[a, b] > thr:
            break
        tn = float(U[a, b])
        joins.append((idx[a], idx[b], tn))
        idx[a] = m + len(joins) - 1  # slot a now holds the new node
        Ts[a], Ks[a] = tn, 0.0
        alive[b] = False
        g, u = _pair_gain(A, tp, tn, Ts, 0.0, Ks, tiny)
        g = np.where(alive, g, -np.inf)
        g[a] = -np.inf
        G[a, :], G[:, a], U[a, :], U[:, a] = g, g, u, u
        G[b, :] = G[:, b] = -np.inf
    return joins


def _pair_gain1(A, tp, ta, tb, ka, kb, tiny):
    """Scalar :func:`_pair_gain` (plain floats; the restricted join calls it per pair)."""
    tm = min(ta, tb) - tiny
    if not tm > tp:
        return 0.0, tp
    B = A * (ta + tb) - ka - kb
    C = A * ta * tb - ka * tb - kb * ta
    u = (B - math.sqrt(max(B * B - 4 * A * C, 0.0))) / (2 * A)
    u = min(max(u, tp), tm)
    g = A * (u - tp)
    if ka > 0:
        g += ka * math.log((ta - u) / (ta - tp))
    if kb > 0:
        g += kb * math.log((tb - u) / (tb - tp))
    return (g, u) if math.isfinite(g) else (0.0, tp)


def _resolve_restricted(A, tp, T, K, tiny, thr, neighbours):
    """Greedy join for large polytomies, scoring only pairs that are within
    ``neighbours`` places of each other in attachment-time order
    (t_c - k_c / A, when a child's lineage is expected to start). The best
    candidate pair is joined first, as in the exact greedy. The new node
    (count 0) enters the order at its own time and is paired with its
    neighbours; across the gap a removed node leaves, the nodes within the
    window on either side are paired. O(m neighbours^2 log m). On simulated
    polytomies of 100-1000 children it reaches 98-102% of the exact greedy's
    total gain (the greedy is not optimal either)."""
    import bisect
    import heapq
    T, K = [float(z) for z in T], [float(z) for z in K]
    m = len(T)
    key = lambda i: T[i] - K[i] / A
    order = sorted(range(m), key=key)
    keys = [key(i) for i in order]
    alive = [True] * m
    heap = []

    def push(i, j):
        g, u = _pair_gain1(A, tp, T[i], T[j], K[i], K[j], tiny)
        if g > thr:
            heapq.heappush(heap, (-g, min(i, j), max(i, j), u))

    for a in range(m):
        for b in range(a + 1, min(m, a + neighbours + 1)):
            push(order[a], order[b])

    def remove(i):
        p = bisect.bisect_left(keys, key(i))
        while order[p] != i:
            p += 1
        del order[p], keys[p]
        return p

    def across(p):
        """pair the nodes within the window left of position p with those right of it"""
        for a in range(max(0, p - neighbours), p):
            for b in range(p, min(len(order), a + neighbours + 1)):
                push(order[a], order[b])

    joins = []
    n_alive = m
    while n_alive > 2 and heap:
        g, i, j, u = heapq.heappop(heap)
        if not (alive[i] and alive[j]):
            continue
        h = len(T)
        T.append(u)
        K.append(0.0)
        alive.append(True)
        alive[i] = alive[j] = False
        joins.append((i, j, u))
        n_alive -= 1
        across(remove(i))
        across(remove(j))
        kh = key(h)
        p = bisect.bisect_left(keys, kh)
        order.insert(p, h)
        keys.insert(p, kh)
        for q in range(max(0, p - neighbours), min(len(order), p + neighbours + 1)):
            if q != p:
                push(h, order[q])
    return joins


def _resolve(f, mu, L, max_exact, neighbours):
    """Resolve every polytomy of the flat tree f by dates at rate mu.
    Returns (new flat tree in preorder, info dict)."""
    par = np.asarray(f.par)
    n = len(par)
    tip = np.asarray(f.tip)
    dates = np.asarray(f.date, float)
    ml = np.asarray(f.ml, float)
    fixed = np.where(tip & np.isfinite(dates), mu * dates, np.nan)
    x = _node_positions(par, tip, fixed, ml, np.ones(n))
    t = x / mu
    dated = dates[tip & np.isfinite(dates)]
    span = float(np.ptp(dated)) if len(dated) > 1 else 1.0
    eps = 1e-6 * max(span, 1e-12)
    for v in range(n - 1, 0, -1):  # parents no later than their children
        p = par[v]
        if t[p] > t[v] - eps:
            t[p] = t[v] - eps
    A = mu * L
    tiny = 1e-9 * max(span, 1e-12)
    ch = f.children()
    g = _Flat()
    for v in range(n):
        g.add(f.par[v], f.tip[v], f.date[v], f.ml[v], f.pos[v], f.name[v])
    times = list(t)
    info = dict(n_polytomies=0, n_polytomies_left=0, n_new_nodes=0, n_restricted=0, max_polytomy=0)
    empty = None if f.pos[0] is None else np.zeros(0, int)
    for v in range(n):
        kids = ch[v]
        if len(kids) <= 2:
            continue
        info['n_polytomies'] += 1
        info['max_polytomy'] = max(info['max_polytomy'], len(kids))
        T = [times[c] for c in kids]
        K = [ml[c] * L for c in kids]
        if len(kids) <= max_exact:
            joins = _resolve_exact(A, times[v], T, K, tiny, 1e-9)
        else:
            info['n_restricted'] += 1
            joins = _resolve_restricted(A, times[v], T, K, tiny, 1e-9, neighbours)
        local = list(kids)  # local index -> flat index
        for i, j, tn in joins:
            h = g.add(v, False, np.nan, 0.0, empty)
            times.append(tn)
            g.par[local[i]] = h
            g.par[local[j]] = h
            local.append(h)
            info['n_new_nodes'] += 1
        if len(kids) - len(joins) > 2:
            info['n_polytomies_left'] += 1
    return g.preorder(), info


# --------------------------------------------------------------------------------------------------------------
# regression on the copy


def _postorder_Q(treg):
    """Root averages of a TreeRegression by its postorder recursion alone, O(n)."""
    for nd in treg.tree.get_nonterminals(order='postorder'):
        Q = np.zeros(6)
        for c in nd:
            Q += treg.propagate_averages(c, treg.tip_value(c), treg.branch_value(c), treg.branch_variance(c))
        nd.Q = Q
    return treg.tree.root.Q


class _Regression:
    """TreeRegression on a light Bio.Phylo copy of a flat tree, with the
    branch values and variances supplied as arrays per fit. Only the
    postorder half of TreeRegression's recursion is run (the root's averages
    are all a slope and intercept need)."""

    def __init__(self, f):
        self.f = f
        n = len(f.par)
        clades = []
        for v in range(n):
            c = Clade(branch_length=f.ml[v], name=f.name[v])
            c._i = v
            clades.append(c)
            if f.par[v] >= 0:
                clades[f.par[v]].clades.append(c)
        self.tree = Tree(root=clades[0], rooted=True)
        dates = np.asarray(f.date, float)
        self.cur = {}
        tipval = lambda c: (dates[c._i] if np.isfinite(dates[c._i]) else None) if c.is_terminal() else None
        self.treg = TreeRegression(self.tree, tip_value=tipval,
                                   branch_value=lambda c: self.cur['val'][c._i],
                                   branch_variance=lambda c: self.cur['var'][c._i])

    def averages(self, val, var):
        self.cur['val'], self.cur['var'] = val, var
        return _postorder_Q(self.treg)

    def fit(self, val, var, slope=None):
        return base_regression(self.averages(val, var), slope=slope)


def _half_values(ml, k, kh, Lh, L):
    """Branch values of one half: mutation_length split in proportion to the
    half's share of the branch's mutations, scaled to full length (k = 0:
    the branch keeps its length in both halves)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.where(k > 0, ml * kh / k * (L / Lh), ml)
    return v


def _xfit(reg, ml, k, splits, L, floor, weight_iterations, tip_extra=0.0):
    """Cross-fitted slopes on the flat tree behind ``reg``. ``splits``: list of
    (kA, LA), the per-branch counts in half A and its number of columns.
    For each split and direction the weight half is fitted
    ``weight_iterations`` times (first with constant branch variances plus the
    tip term, then with the variances
    from the previous fit's predicted branch lengths l_b: (l_b L + floor)/L^2),
    and the other half's values are regressed with the final variances.
    ``tip_extra`` (mutations) is added to the variance of terminal branches.
    Returns (2R slopes, mean variance vector over the 2R weight fits)."""
    f = reg.f
    par, tip = np.asarray(f.par), np.asarray(f.tip)
    dates = np.asarray(f.date, float)
    n, om = len(par), 1.0 / L
    dated = tip & np.isfinite(dates)
    extra = np.where(tip, tip_extra, 0.0)

    def predicted_variance(val, var):
        mu = reg.fit(val, var)['slope']
        x = _node_positions(par, tip, np.where(dated, mu * dates, np.nan), val, 1.0 / var)
        lhat = np.r_[0.0, x[1:] - x[par[1:]]]
        return (np.maximum(lhat, 0.0) * L + floor + extra) * om * om

    slopes, var_sum = [], np.zeros(n)
    for kA, LA in splits:
        halves = (_half_values(ml, k, kA, LA, L), _half_values(ml, k, k - kA, L - LA, L))
        for src, dst in ((0, 1), (1, 0)):
            var = np.full(n, om * om) + extra * om * om
            for _ in range(weight_iterations):
                var = predicted_variance(halves[src], var)
            slopes.append(float(reg.fit(halves[dst], var)['slope']))
            var_sum += var
    var_bar = var_sum / len(slopes)
    var_bar[0] = om * om
    return slopes, var_bar


def date_resolved_xfit_rate(tt, floor=0.5, n_splits=5, seed=0, weight_iterations=3, resolve=True,
                            max_exact_polytomy=1000, neighbours=8, counts=None, tip_slack=0.0, collapse=True):
    """Clock rate at the current root of ``tt`` (a ClockTree/TreeTime with
    branch ``mutation_length`` set): covariation regression on a date-resolved
    copy of the tree with cross-fitted branch weights. ``tt.tree`` is not
    modified apart from the attributes TreeTime's own regression sets.

    Parameters
    ----------
    floor : float
        Variance floor in mutations added to every branch, (l_b L + floor) / L^2.
        Must be positive.
    n_splits : int
        Random column splits R; the rate is the mean over 2R half fits.
    seed : int
        Seed of the column splits (and of the thinning without mutations).
    weight_iterations : int
        Fits on the weight half, each with the previous fit's variances
        (the first is unweighted branch least squares).
    resolve : bool
        False skips the date resolution (cross-fitted weights on the
        collapsed tree; a diagnostic).
    max_exact_polytomy : int
        Polytomies with more children use the restricted greedy join. The
        exact greedy keeps two m x m matrices (16 MB at m = 1000) and takes
        about 0.3 s at m = 800.
    neighbours : int
        Width of the attachment-time window of the restricted join.
    tip_slack : float
        Extra variance tip_slack^2 (mutations) on terminal branches in every
        weight, as TreeTime's covariation regression has (default 0: none).
    counts : {None, 'mutations', 'thinned'}
        Source of the per-half counts. None: mutation positions when ``tt``
        has an alignment and a reconstruction (branches whose node has no
        reconstructed sequence are thinned, with a warning), else thinning.
    collapse : bool
        Collapse internal edges with no mutation and length < 0.1 mutations
        on the copy before resolving (default). Without it such edges keep
        their length and are not re-resolved.

    Returns
    -------
    DateResolvedRate
        rate, rate_se (model-based GLS standard error at the averaged
        cross-fitted variances, root and topology fixed; ``se_kind``; it
        ignores overdispersion, tip-date errors and the cross-fitting, and
        undercovers on misspecified data), split_rates and
        split_sd (spread over the 2R half fits, split noise only), intercept and
        root_date (refitted on all values at ``rate``), cov, chisq, r_val,
        plain_rate / plain_intercept (reference), covariation_rate /
        covariation_intercept (TreeTime's own covariation regression on the
        tree as given, with its tip_slack and branch_variance_floor), counts
        ('mutations', 'mixed' or 'thinned'), n_thinned_branches, resolution info (n_polytomies,
        n_polytomies_left, n_new_nodes, n_restricted, max_polytomy,
        resolution_skipped), seconds.
    """
    t0 = time.time()
    if not floor > 0:
        raise ValueError('date_resolved_xfit: floor must be positive')
    if n_splits < 1:
        raise ValueError('date_resolved_xfit: n_splits must be >= 1')
    L = int(tt.data.full_length)
    om = 1.0 / L
    rng = np.random.default_rng(seed)

    if counts not in (None, 'mutations', 'thinned'):
        raise ValueError("date_resolved_xfit: counts must be None, 'mutations' or 'thinned'")
    if tip_slack < 0:
        raise ValueError('date_resolved_xfit: tip_slack must be >= 0')
    positions, n_thinned = None, 0
    if counts != 'thinned' and tt.aln is not None:
        # nodes created after the reconstruction (a new root edge from rerooting, TreeTime's own polytomy
        # resolution) have no sequence yet: those branches alone are thinned (pos None). TreeTime flags the
        # whole reconstruction stale after a reroot (sequence_reconstruction = False), but the branches the
        # reroot did not touch keep their mutations, so the flag is not used here.
        positions = {}
        for nd in tt.tree.find_clades():
            if nd.up is None:
                continue
            try:
                p = np.array([m[1] for m in nd.mutations], dtype=int)
                positions[id(nd)] = p[(p >= 0) & (p < L)]
            except (AttributeError, TypeError, KeyError, ValueError):
                positions[id(nd)] = None
                n_thinned += 1
        if n_thinned == len(positions):
            positions, n_thinned = None, 0
        elif n_thinned:
            tt.logger('date_resolved_xfit: %d branches without reconstructed mutations; their per-half counts are '
                      'binomial thinnings' % n_thinned, 1, warn=True)
    if positions is None and counts == 'mutations':
        raise ValueError('date_resolved_xfit: no mutation positions (run ancestral reconstruction first)')

    # TreeTime's own covariation regression on the tree as given (diagnostic). Only the postorder half of the
    # recursion: TreeRegression.regression() also runs a preorder pass that is quadratic in polytomy size.
    try:
        treg = tt.setup_TreeRegression(covariation=True)
        cov_model = base_regression(_postorder_Q(treg))
        cov_rate, cov_icpt = float(cov_model['slope']), float(cov_model['intercept'])
    except ValueError:
        cov_rate = cov_icpt = np.nan

    f = _flatten(tt.tree, om, positions, collapse=collapse)
    plain_rate, plain_icpt, r_val = _plain_regression(f)
    info = dict(n_polytomies=0, n_polytomies_left=0, n_new_nodes=0, n_restricted=0, max_polytomy=0)
    skipped = None
    if resolve:
        if plain_rate > 0 and np.isfinite(plain_rate):
            f, info = _resolve(f, plain_rate, L, max_exact_polytomy, neighbours)
        else:
            skipped = 'plain regression rate not positive'
    if skipped or not resolve:
        info['n_polytomies'] = info['n_polytomies_left'] = sum(len(c) > 2 for c in f.children())
    info['resolution_skipped'] = skipped
    info['n_thinned_branches'] = int(n_thinned)

    n = len(f.par)
    ml = np.asarray(f.ml, float)
    reg = _Regression(f)

    # full counts and per-split half counts
    has_pos = np.array([positions is not None and v > 0 and f.pos[v] is not None for v in range(n)])
    k = np.round(ml * L)
    for v in np.flatnonzero(has_pos):
        k[v] = float(len(f.pos[v]))
    k[0] = 0.0
    source = 'thinned' if not has_pos.any() else ('mutations' if has_pos[1:].all() else 'mixed')

    splits = []
    for _ in range(n_splits):
        inA = np.zeros(L, bool)
        inA[rng.permutation(L)[: L // 2]] = True
        LA = float(inA.sum())
        kA = np.zeros(n) if has_pos[1:].all() else rng.binomial(k.astype(int), LA / L).astype(float)
        for v in np.flatnonzero(has_pos):
            kA[v] = float(inA[f.pos[v]].sum()) if len(f.pos[v]) else 0.0
        splits.append((kA, LA))
    slopes, var_bar = _xfit(reg, ml, k, splits, L, floor, weight_iterations, tip_extra=float(tip_slack) ** 2)
    rate = float(np.mean(slopes))
    full_free = reg.fit(ml, var_bar)
    at_rate = reg.fit(ml, var_bar, slope=rate)
    cov = np.asarray(full_free['cov'], float)
    out = DateResolvedRate(
        rate=rate,
        rate_se=float(np.sqrt(cov[0, 0])),
        se_kind='GLS at the averaged cross-fitted variances (Poisson + floor), root and topology fixed',
        split_rates=slopes,
        split_sd=float(np.std(slopes)),
        intercept=float(at_rate['intercept']),
        root_date=float(-at_rate['intercept'] / rate) if rate != 0 else np.nan,
        chisq=float(at_rate['chisq']),
        cov=cov,
        r_val=r_val,
        plain_rate=plain_rate,
        plain_intercept=plain_icpt,
        covariation_rate=cov_rate,
        covariation_intercept=cov_icpt,
        counts=source,
        floor=float(floor),
        tip_slack=float(tip_slack),
        n_splits=int(n_splits),
        seed=seed,
        n_nodes=int(n),
        **info,
    )
    out['seconds'] = time.time() - t0
    return out


# --------------------------------------------------------------------------------------------------------------
# least-squares-dated variance (no TreeTime dated fit)


def _ls_durations(f, mu):
    """Branch durations (in substitution units, mu x time) from one unweighted
    least-squares fit of the branch lengths with dated tips fixed at mu x date,
    parents then moved no later than their children (the node times
    ``_resolve`` uses). Element 0 (root) is 0."""
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


def ls_dated_variance_rate(tt, tip_slack=3.0, rate=None, rounds=1, resolve=False, collapse=True,
                           max_exact_polytomy=1000, neighbours=8):
    """Clock rate at the current root of ``tt`` from TreeTime's covariation
    regression with dated-duration variances, where the durations come from a
    least-squares node-time solve instead of a dated fit (no ``make_time_tree``).

    Branch values are ``mutation_length``; branch variances are
    (duration + tip_slack^2 / L [terminal branches]) / L, TreeTime's
    ``clock_length`` variance with its tip slack. Durations: ``_ls_durations``
    at the reference rate (``rate``, default the plain root-to-tip rate), on the
    flat copy (``collapse``: zero-mutation internal edges collapsed first;
    ``resolve``: polytomies date-resolved at the reference rate first, as
    ``date_resolved_xfit_rate`` does). ``rounds`` > 1 re-solves the node times at
    the previous round's rate. Outlier tips (``bad_branch``) carry no date.

    Returns a dict: rate, intercept, root_date, plain_rate, r_val, round_rates,
    n_nodes, n_new_nodes, seconds.
    """
    t0 = time.time()
    if tip_slack < 0:
        raise ValueError('ls_dated_variance_rate: tip_slack must be >= 0')
    L = int(tt.data.full_length)
    om = 1.0 / L
    f = _flatten(tt.tree, om, None, collapse=collapse)
    plain_rate, plain_icpt, r_val = _plain_regression(f)
    mu = plain_rate if rate is None else float(rate)
    if not (mu > 0 and np.isfinite(mu)):
        raise ValueError('ls_dated_variance_rate: reference rate not positive')
    n_new = 0
    if resolve:
        f, info = _resolve(f, mu, L, max_exact_polytomy, neighbours)
        n_new = info['n_new_nodes']
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
                plain_rate=plain_rate, r_val=r_val, round_rates=round_rates, n_nodes=len(f.par), n_new_nodes=n_new,
                seconds=time.time() - t0)
