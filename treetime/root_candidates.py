"""Likelihood-based selection among candidate root placements.

Each candidate root on a fixed unrooted topology is fitted independently under a
strict clock with a fixed rate and a fixed GTR model, using TreeTime's own
alternation of marginal ancestral inference and time-tree inference, and is
scored by the full marginal sequence log-likelihood recomputed at the fitted
(clock-constrained) branch lengths.

Layout
------
* :class:`UnrootedBaseline` -- the unrooted topology with branch lengths, the
  alignment, tip dates, GTR model and clock rate shared by every candidate.
  An edge is identified by the set of leaves on its side away from a fixed
  reference leaf, which is stable under rerooting.
* :class:`NodeRoot` / :class:`EdgeRoot` -- candidate root placements.
* :func:`regression_candidates` -- the best placements of the existing
  root-to-tip regression (:meth:`TreeRegression.rank_roots`).
* :class:`LsdvSource` -- opt-in candidate source ranked by the lsdv regression
  (:mod:`treetime.ls_dated_rate`); ``UnrootedBaseline(shared_rate='lsdv')``
  takes the shared rate from it.
* :func:`evaluate_root_candidate` -- one candidate from one initialisation: a
  new ``TreeTime`` object, a fixed number of fitting iterations without the
  ``ndiff`` early stop, a fresh likelihood after every time tree, and a copy of
  the best-scoring fitted object.
* :func:`rank_root_candidates` -- evaluates explicit and regression candidates,
  re-fits close candidates from alternative initialisations and reports
  whether the best candidate is resolved from the others.
* :func:`plot_root_candidates` -- the tree with every candidate coloured by its
  likelihood deficit from the best.
"""
import copy
import io
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree

from . import TreeTimeError
from . import config as ttconf
from .gtr import GTR
from .treetime import TreeTime

EDGE_ROOT_NAME = 'EDGE_ROOT'
ENDPOINT_TOL = 1e-5  # same tolerance as TreeRegression.optimal_reroot
COLLAPSE_TOL = 0.05  # expected mutations: a fitted edge root this close to an endpoint is that node's tree
FIT_ERRORS = (TreeTimeError, ValueError, ZeroDivisionError, FloatingPointError)


# --------------------------------------------------------------------------
# candidate types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeRoot:
    """Root exactly at an existing internal node; its degree is preserved.

    ``split`` is the node's key: the leaf set below the node when the tree is
    hung from the reference leaf.
    """

    split: frozenset

    @property
    def kind(self):
        return 'node'


@dataclass(frozen=True)
class EdgeRoot:
    """Root inside an edge, at a new bifurcating root node.

    ``fraction`` is the initial position measured from the endpoint away from
    the reference leaf (0 = at that endpoint, 1 = at the reference-side
    endpoint), the convention of TreeRegression's ``split``. It only initialises
    the fit: the dated fit decides where on the edge the root ends up, so edge
    roots differing only in ``fraction`` are the same candidate.
    """

    split: frozenset
    fraction: float = 0.5

    @property
    def kind(self):
        return 'edge'


def candidate_key(cand):
    """Identity of a candidate: the node, or the edge (not the fraction)."""
    return (cand.kind, cand.split)


def stable_id(cand):
    """Identifier that depends only on the leaf bipartition, not on node names or
    on the input tree's rooting: kind plus a short hash of the sorted leaf set."""
    import hashlib
    return '%s:%s' % (cand.kind, hashlib.sha1(','.join(sorted(cand.split)).encode()).hexdigest()[:10])


# --------------------------------------------------------------------------
# baseline: unrooted topology + shared model
# --------------------------------------------------------------------------
class UnrootedBaseline:
    """Fixed unrooted topology, branch lengths and model shared by all candidates.

    Parameters
    ----------
    tree : Bio.Phylo tree or newick path
        Any rooting; the root is discarded. A bifurcating root and other
        degree-2 nodes are suppressed. Internal nodes are renamed ``INT_nnnn``
        (tree-builder labels such as support values need not be unique). The
        input tree is not modified.
    aln : alignment accepted by TreeAnc
    dates : dict
        leaf name -> numeric date
    gtr : GTR or str
        Fixed substitution model, never re-inferred; deep-copied per candidate.
    clock_rate : float, optional
        Fixed positive rate shared by all candidates. ``None`` takes it from
        ``shared_rate``.
    shared_rate : {'regression', 'lsdv'}
        Source of the shared rate when ``clock_rate`` is None.
        ``'regression'`` (default): the slope of the best free-slope plain
        root-to-tip regression on the baseline. ``'lsdv'``: the lsdv rate
        (:func:`treetime.ls_dated_rate.lsdv_flat`, covariation regression with
        least-squares-dated variances) at the regression's best placement, or
        with ``shared_rate_k`` > 1 the median of the lsdv rates at the top
        ``shared_rate_k`` distinct placements of that regression. Falls back to
        the regression slope (``shared_rate_info['fallback']``) when no
        placement gives a positive lsdv rate.
    shared_rate_k : int
        Number of regression placements for ``shared_rate='lsdv'``.
    shared_rate_kwargs : dict, optional
        Options of the lsdv rate (``tip_slack``, default 3; ``rounds``,
        default 1; ``collapse``, default True), plus the opt-in variants of
        :data:`treetime.ls_dated_rate.LSDV_OPTIONS` (``resolve``,
        ``outlier_z``, ``outlier_mode``, ``internal_b``, ``first_pass``,
        ``se_inflation``, ``resample``, ...). When any of these is passed,
        each per-placement row of ``shared_rate_info['candidates']`` also
        carries the SE (``rate_se``, ``se_rel``, ``rate_ci``) and the variant's
        report fields (e.g. ``outliers``); a ``resample`` option is dropped for
        the shared rate, which needs only the rate.

    The shared rate also sets the slope of the plain regression source
    (``RegressionSource(slope='fixed')``, in the default Stage 1 shortlist),
    so with ``shared_rate='lsdv'`` that list can differ too; the covariation
    source fits its own slope and is unaffected.
    optimize_baseline : bool
        Re-estimate branch lengths once with TreeTime's marginal optimiser on
        the unrooted tree, as ``TreeTime.run`` does before rerooting. These
        lengths drive the regression and start every candidate.
    rng_seed : int
        Seed of every per-candidate TreeTime, so the RNG cannot carry state
        from one candidate to the next.
    tt_kwargs : dict, optional
        Extra TreeTime constructor arguments (e.g. ``precision``).

    Attributes: ``clock_rate``; ``shared_rate_info`` (None for the default
    regression rate or a given ``clock_rate``; for ``'lsdv'`` a dict with the
    rate used, the regression slope, the per-placement lsdv rates and, with
    ``shared_rate_k`` > 1, their spread ``log_range`` = log(max / min), a
    diagnostic of how strongly the rate depends on the root; None at k = 1).
    """

    def __init__(self, tree, aln, dates, gtr='JC69', clock_rate=None, optimize_baseline=True,
                 rng_seed=1234, tt_kwargs=None, shared_rate='regression', shared_rate_k=1, shared_rate_kwargs=None):
        tree = Phylo.read(tree, 'newick') if isinstance(tree, str) else copy.deepcopy(tree)
        self.aln = aln
        self.dates = dict(dates)
        self.gtr = gtr if isinstance(gtr, GTR) else GTR.standard(gtr)
        self.rng_seed = rng_seed
        self.tt_kwargs = dict(tt_kwargs or {})
        self.root_state_messages = None  # external root-state likelihoods, see with_root_state_messages
        self._from_phylo(tree)
        if optimize_baseline:
            tt = self.make_treetime(self.rooted_phylo(None), messages=False)
            tt.optimize_tree(infer_gtr=False, max_iter=1, method_anc='probabilistic', **_seq_kwargs())
            self._from_phylo(tt.tree, keep_names=True)
        if shared_rate not in ('regression', 'lsdv'):
            raise ValueError("UnrootedBaseline: shared_rate must be 'regression' or 'lsdv'")
        if shared_rate != 'regression' and clock_rate is not None:
            raise ValueError('UnrootedBaseline: pass either clock_rate or shared_rate, not both')
        self.shared_rate_info = None
        if clock_rate is None:
            best = self.regression().find_best_root(force_positive=True)
            if best is None or not best['slope'] > 0:
                raise ValueError('UnrootedBaseline: no positive clock rate from regression; pass clock_rate')
            clock_rate = float(best['slope'])
            if shared_rate == 'lsdv':
                self.shared_rate_info = self._lsdv_shared_rate(clock_rate, shared_rate_k, shared_rate_kwargs or {})
                clock_rate = self.shared_rate_info['rate']
        if not clock_rate > 0:
            raise ValueError('UnrootedBaseline: clock_rate must be positive')
        self.clock_rate = float(clock_rate)

    # -- construction --------------------------------------------------------
    def _from_phylo(self, tree, keep_names=False, allow_edge_root=False):
        leaf_names = set(c.name for c in tree.get_terminals())
        counter = 0
        for c in tree.get_nonterminals(order='preorder'):
            if keep_names and c.name:
                continue
            while 'INT_%04d' % counter in leaf_names:
                counter += 1
            c.name = 'INT_%04d' % counter
            counter += 1
        all_names = [c.name for c in tree.find_clades()]
        if None in all_names or len(set(all_names)) != len(all_names):
            raise ValueError('UnrootedBaseline: leaf names must be present and unique')
        adj = {}
        for c in tree.find_clades(order='preorder'):
            adj.setdefault(c.name, {})
            for ch in c.clades:
                bl = max(0.0, float(ch.branch_length or 0.0))
                adj[c.name][ch.name] = bl
                adj.setdefault(ch.name, {})[c.name] = bl
        self.leaves = sorted(leaf_names)
        if EDGE_ROOT_NAME in adj and not allow_edge_root:
            raise ValueError('UnrootedBaseline: node name %s is reserved' % EDGE_ROOT_NAME)
        # suppress degree-2 nodes (bifurcating root, unary nodes)
        changed = True
        while changed:
            changed = False
            for nm in list(adj):
                if nm in leaf_names or len(adj[nm]) != 2:
                    continue
                (a, la), (b, lb) = adj[nm].items()
                del adj[a][nm], adj[b][nm], adj[nm]
                adj[a][b] = adj[b][a] = la + lb
                changed = True
        self.adj = adj
        self.ref_leaf = self.leaves[0]
        self.all_leaves = frozenset(self.leaves)
        self.center = next(iter(adj[self.ref_leaf]))  # internal neighbour of the reference leaf
        # hang from the reference leaf: split_of[v] = leaves below v, toward_ref[v] = parent
        self.split_of, self.node_of, self.toward_ref = {}, {}, {}
        order, stack = [], [(self.ref_leaf, None)]
        while stack:
            v, p = stack.pop()
            order.append((v, p))
            for w in sorted(adj[v]):
                if w != p:
                    stack.append((w, v))
        below = {}
        for v, p in reversed(order):
            if v in leaf_names:
                below[v] = frozenset([v])
            else:
                below[v] = frozenset().union(*(below[w] for w in adj[v] if w != p))
            if p is not None:
                self.split_of[v] = below[v]
                self.node_of[below[v]] = v
                self.toward_ref[v] = p

    # -- edge / node lookup --------------------------------------------------
    @property
    def splits(self):
        """All edge keys, one per unrooted edge."""
        return set(self.node_of)

    def edge_length(self, split):
        a = self.node_of[split]
        return self.adj[a][self.toward_ref[a]]

    def is_leaf(self, name):
        return name in self.all_leaves

    def internal_nodes(self):
        return sorted(v for v in self.adj if not self.is_leaf(v))

    def node(self, name):
        """Candidate rooted exactly at node ``name`` (canonicalised)."""
        if name == self.ref_leaf:
            return self.canonical(EdgeRoot(self.split_of[self.center], 1.0))
        return self.canonical(NodeRoot(self.split_of[name]))

    def edge(self, name, fraction=0.5, other=None):
        """Edge between ``name`` and ``other`` (default: its neighbour toward the
        reference leaf), initial root at ``fraction`` of the edge from ``name``."""
        if other is None or other == self.toward_ref.get(name):
            return self.canonical(EdgeRoot(self.split_of[name], float(fraction)))
        if self.toward_ref.get(other) == name:
            return self.canonical(EdgeRoot(self.split_of[other], 1.0 - float(fraction)))
        raise ValueError('UnrootedBaseline.edge: %s and %s are not adjacent' % (name, other))

    def canonical(self, cand):
        """Deduplicate placements: an edge root at an internal endpoint is that
        node's NodeRoot; a node root at a leaf is the edge root at the leaf end
        of its edge (a leaf cannot be the root)."""
        if isinstance(cand, NodeRoot):
            if self.is_leaf(self.node_of[cand.split]):
                return EdgeRoot(cand.split, 0.0)
            return cand
        x = min(1.0, max(0.0, float(cand.fraction)))
        a = self.node_of[cand.split]
        if x < ENDPOINT_TOL:
            return NodeRoot(cand.split) if not self.is_leaf(a) else EdgeRoot(cand.split, 0.0)
        if x > 1.0 - ENDPOINT_TOL:
            b = self.toward_ref[a]
            return NodeRoot(self.split_of[b]) if not self.is_leaf(b) else EdgeRoot(cand.split, 1.0)
        return EdgeRoot(cand.split, x)

    def candidate_id(self, cand):
        a = self.node_of[cand.split]
        if isinstance(cand, NodeRoot):
            return 'node:%s' % a
        return 'edge:%s-%s' % (a, self.toward_ref[a])

    def all_candidates(self, fraction=0.5):
        """Exhaustive set: every internal node plus every edge at ``fraction``."""
        out = [NodeRoot(self.split_of[v]) for v in self.internal_nodes()]
        out += [EdgeRoot(s, fraction) for s in sorted(self.splits, key=lambda s: self.node_of[s])]
        return out

    # -- tree building -------------------------------------------------------
    def rooted_phylo(self, cand):
        """New Bio.Phylo tree rooted at ``cand``; ``None`` roots at the reference
        leaf's neighbour (used for the baseline itself)."""
        def build(v, p, bl):
            c = Clade(name=v, branch_length=bl)
            for w in sorted(self.adj[v]):
                if w != p:
                    c.clades.append(build(w, v, self.adj[v][w]))
            return c

        if cand is None:
            root = build(self.center, None, 0.0)
        elif isinstance(cand, NodeRoot):
            root = build(self.node_of[cand.split], None, 0.0)
        else:
            a = self.node_of[cand.split]
            b = self.toward_ref[a]
            w = self.adj[a][b]
            root = Clade(name=EDGE_ROOT_NAME, branch_length=0.0)
            root.clades = [build(a, b, w * cand.fraction), build(b, a, w * (1.0 - cand.fraction))]
        return Tree(root=root, rooted=True)

    def make_treetime(self, tree, messages=True):
        """New TreeTime object on ``tree``: same alignment, dates, model and seed,
        carrying the baseline's root-state messages unless ``messages=False``."""
        kw = dict(verbose=0, branch_length_mode='marginal', use_covariation=False)
        kw.update(self.tt_kwargs)
        tt = TreeTime(tree=tree, aln=self.aln, dates=self.dates, gtr=copy.deepcopy(self.gtr),
                      rng_seed=self.rng_seed, **kw)
        if messages and self.root_state_messages is not None:
            tt.set_root_state_messages(self.root_state_messages)
        return tt

    def make_treeanc(self, tree, messages=True):
        """New TreeAnc on ``tree`` for scoring at fixed branch lengths: same
        alignment, model and seed, the data-defining constructor settings of
        ``tt_kwargs`` (e.g. ``seq_len`` for added constant sites) and, unless
        ``messages=False``, the baseline's root-state messages."""
        import inspect
        from .treeanc import TreeAnc
        allowed = set(inspect.signature(TreeAnc.__init__).parameters) - {'self', 'tree', 'aln', 'gtr', 'kwargs'}
        kw = {k: v for k, v in self.tt_kwargs.items() if k in allowed}
        kw.setdefault('verbose', 0)
        kw.setdefault('rng_seed', self.rng_seed)
        ta = TreeAnc(tree=tree, aln=self.aln, gtr=copy.deepcopy(self.gtr), **kw)
        if messages and self.root_state_messages is not None:
            ta.set_root_state_messages(self.root_state_messages)
        return ta

    def with_root_state_messages(self, messages):
        """Copy of this baseline (same topology, branch lengths, model and clock
        rate) whose candidate fits carry external root-state likelihoods
        (:meth:`TreeAnc.set_root_state_messages`; ``None`` removes them). The
        baseline branch lengths and the regression do not depend on the rooting
        and stay the ingroup ones."""
        other = copy.copy(self)
        other.root_state_messages = messages
        return other

    def regression(self, covariation=False, tip_slack=None):
        """TreeRegression on the baseline branch lengths. ``tip_slack`` sets the
        extra tip variance of the covariation model (TreeTime's default if None)."""
        tt = self.make_treetime(self.rooted_phylo(None), messages=False)
        if tip_slack is not None:
            tt.tip_slack = tip_slack
        treg = tt.setup_TreeRegression(covariation=covariation)
        treg._tt = tt  # keep the node owner alive
        return treg

    @property
    def full_length(self):
        """Number of alignment sites as TreeTime counts them (``data.full_length``)."""
        if getattr(self, '_full_length', None) is None:
            self._full_length = int(self.make_treetime(self.rooted_phylo(None), messages=False).data.full_length)
        return self._full_length

    def _tip_date(self, leaf):
        d = self.dates.get(leaf.name)
        if d is None:
            return np.nan
        d = float(np.mean(d))
        return d if np.isfinite(d) else np.nan

    def lsdv(self, cand, tip_slack=3.0, rounds=1, collapse=True, rate=None, **options):
        """lsdv rate at root placement ``cand`` on the baseline branch lengths:
        :func:`treetime.ls_dated_rate.lsdv_flat` on the rooted tree, equal to
        :func:`~treetime.ls_dated_rate.ls_dated_variance_rate` on a TreeTime
        object rooted there. ``options``: the opt-in lsdv variants
        (``resolve``, ``outlier_z``, ``internal_b``, ``se_inflation``,
        ``resample``, ...; :data:`treetime.ls_dated_rate.LSDV_OPTIONS`). Raises
        ``ValueError`` when the plain root-to-tip rate at ``cand`` (the
        reference rate unless ``rate`` is given) is not positive."""
        from .ls_dated_rate import _flatten, lsdv_flat
        L = self.full_length
        f = _flatten(self.rooted_phylo(self.canonical(cand)), 1.0 / L, collapse=collapse,
                     length=lambda n: n.branch_length, date=self._tip_date)
        return lsdv_flat(f, L, tip_slack=tip_slack, rate=rate, rounds=rounds, **options)

    def _lsdv_shared_rate(self, regression_rate, k, kwargs):
        """Median lsdv rate over the top ``k`` free-slope plain regression placements."""
        t0 = time.time()
        if not (isinstance(k, int) and k >= 1):
            raise ValueError('UnrootedBaseline: shared_rate_k must be a positive integer')
        from .ls_dated_rate import check_lsdv_options
        try:
            if 'rate' in kwargs:
                raise ValueError('rate')
            check_lsdv_options(kwargs)
        except ValueError:
            raise ValueError('UnrootedBaseline: invalid shared_rate_kwargs %s' % kwargs)
        rows = []
        for cand, info in regression_candidates(self, k=k, covariation=False, slope=None):
            row = {'candidate_id': self.candidate_id(cand), 'regression_rank': info['regression_rank'],
                   'lsdv_rate': None, 'plain_rate': None, 'root_date': None, 'error': None}
            try:
                r = self.lsdv(cand, **{k: v for k, v in kwargs.items() if k != 'resample'})
                row.update(lsdv_rate=r['rate'], plain_rate=r['plain_rate'],
                           root_date=r['root_date'] if r['rate'] > 0 else None)
                row.update(_lsdv_extra(r, kwargs))
                if not r['rate'] > 0:
                    row['error'] = 'lsdv rate not positive'
            except (ValueError, FloatingPointError, ZeroDivisionError) as e:
                row['error'] = str(e)
            rows.append(row)
        ok = np.array([r['lsdv_rate'] for r in rows if r['error'] is None], float)
        info = {'source': 'lsdv', 'k': k, 'kwargs': dict(kwargs), 'regression_rate': float(regression_rate),
                'candidates': rows, 'n_ok': len(ok), 'fallback': None}
        if len(ok):
            info.update(rate=float(np.median(ok)), min=float(ok.min()), max=float(ok.max()),
                        log_range=float(np.log(ok.max() / ok.min())) if k > 1 else None)
        else:
            info.update(rate=float(regression_rate), min=None, max=None, log_range=None,
                        fallback='no positive lsdv rate at the regression placements; regression slope used')
        info['seconds'] = time.time() - t0
        return info

    def topology_signature(self, tree):
        """(set of non-trivial splits, sorted internal degrees) of ``tree`` read as
        unrooted, degree-2 nodes suppressed."""
        other = UnrootedBaseline.__new__(UnrootedBaseline)
        other._from_phylo(_strip_tree(tree), keep_names=True, allow_edge_root=True)
        nontriv = {s for s in other.splits if 1 < len(s) < len(other.all_leaves) - 1}
        degrees = sorted(len(other.adj[v]) for v in other.internal_nodes())
        return nontriv, degrees


def _strip_tree(tree):
    """Plain copy of a tree with names and branch lengths only."""
    def cp(c):
        return Clade(name=c.name, branch_length=c.branch_length, clades=[cp(ch) for ch in c.clades])
    return Tree(root=cp(tree.root), rooted=True)


# --------------------------------------------------------------------------
# fitting and scoring
# --------------------------------------------------------------------------
def _seq_kwargs(branch_length_mode='marginal'):
    """Sequence-inference arguments as TreeTime._run builds them for the
    canonical configuration (marginal, prune_short=False)."""
    return {
        'marginal_sequences': branch_length_mode == 'marginal',
        'branch_length_mode': branch_length_mode,
        'sample_from_profile': 'root',
        'prune_short': False,
        'reconstruct_tip_states': False,
    }


def fresh_marginal_score(tt):
    """Full marginal sequence log-likelihood at the tree's current branch lengths.

    Recomputed (postorder pass and root combination), never the cached value.
    Does not draw from the RNG or reassign sequences, but overwrites the cached
    ``sequence_marginal_LH``/``sequence_LH`` and the root profile, so read any
    cached value before calling this.
    """
    tt.postorder_traversal_marginal()
    tt.total_LH_and_root_sequence(assign_sequence=False)
    return float(tt.tree.total_sequence_LH)


def fitted_newick(tt):
    """Newick of the fitted tree at full precision (clock-constrained lengths)."""
    t = _strip_tree(tt.tree)
    t.root.branch_length = None
    buf = io.StringIO()
    Phylo.write(t, buf, 'newick', format_branch_length='%.17g')
    return buf.getvalue().strip()


def score_newick(baseline, newick):
    """Marginal log-likelihood of the baseline alignment on a tree given as a
    newick string, at exactly its branch lengths, with a new TreeAnc object
    (:meth:`UnrootedBaseline.make_treeanc`: the baseline's data settings and its
    root-state messages, if any)."""
    tree = Phylo.read(io.StringIO(newick), 'newick')
    return fresh_marginal_score(baseline.make_treeanc(tree))


def classify_trajectory(scores, tol=1e-6):
    """'fixed_point' if the last two scores agree within tol; 'cycling' if the
    last score revisits an earlier, non-adjacent one; 'oscillating' if the last
    three steps alternate in sign and the score returns closer to where it was
    two steps before than to the previous step; otherwise 'drifting'."""
    if len(scores) < 2:
        return 'single'
    if abs(scores[-1] - scores[-2]) <= tol:
        return 'fixed_point'
    if any(abs(scores[-1] - s) <= tol for s in scores[:-2]):
        return 'cycling'
    if len(scores) >= 4:
        d = np.diff(scores[-4:])
        if d[0] * d[1] < 0 and d[1] * d[2] < 0 and abs(scores[-1] - scores[-3]) < 0.5 * abs(d[2]):
            return 'oscillating'
    return 'drifting'


@dataclass
class FitConfig:
    iterations: int = 4             # fitting iterations after the initial time tree
    initial_mode: str = 'marginal'  # 'joint': alternative first branch-length round
    rate_rounds: int = 0            # per-candidate rate: max rounds of (rate step, ancestral inference, time tree)
    rate_span: float = 4.0          # each rate step searches [mu/span, mu*span] around the current rate
    rate_clamp: float = 10.0        # overall the rate stays within [mu0/clamp, mu0*clamp]
    rate_xatol: float = 1e-3        # tolerance of the rate step on log(mu); also the early-stop threshold


@dataclass
class FitRun:
    """One fitting trajectory of one candidate from one initialisation.

    Per-step lists have one entry per time tree (index 0 = the initial one).
    ``scores`` (fresh marginal logL, used for ranking), ``cached_LH`` (TreeTime's
    stored value, stale) and ``positional_LH`` (dating objective) are separate
    quantities and are never differenced against each other.
    """

    candidate: object
    candidate_id: str
    initial_fraction: Optional[float]
    initial_mode: str
    scores: list = field(default_factory=list)
    cached_LH: list = field(default_factory=list)
    positional_LH: list = field(default_factory=list)
    clock_rates: list = field(default_factory=list)  # clock rate of the time tree at each step
    rate_fitted: bool = False  # the rate was fitted (fit.rate_rounds > 0)
    rate_rounds_performed: int = 0
    ndiff: list = field(default_factory=list)  # TreeTime's ndiff per iteration (None at index 0)
    root_state_changes: list = field(default_factory=list)  # root sequence positions changed
    iterations_performed: int = 0
    treetime_stop_iteration: Optional[int] = None  # where run() would have stopped on ndiff == 0
    best_index: Optional[int] = None
    best_score: float = -np.inf
    clock_rate: Optional[float] = None  # rate of the best-scoring state
    trajectory_class: str = ''
    tail_fluctuation: float = 0.0  # max - min of the second half of the trajectory
    root: dict = field(default_factory=dict)  # root summary at the best step
    fitted_newick: str = ''
    runtime: float = 0.0
    error: Optional[str] = None
    warnings: list = field(default_factory=list)
    state: object = None  # deep copy of the TreeTime at the best step
    extra: dict = field(default_factory=dict)  # evaluator-specific diagnostics (e.g. Stage 3 outgroup lengths)


def _root_summary(baseline, cand, tt):
    root = tt.tree.root
    kids = sorted(root.clades, key=lambda c: c.name)
    bls = [float(c.branch_length) for c in kids]
    frac = None
    if isinstance(cand, EdgeRoot) and sum(bls) > 0:
        frac = bls[[c.name for c in kids].index(baseline.node_of[cand.split])] / sum(bls)
    return {
        'root_date': float(root.numdate),
        'root_degree': len(kids),
        'root_children': [c.name for c in kids],
        'root_child_lengths': bls,
        'fitted_fraction': frac,
        'n_sites': int(tt.data.full_length),
    }


def fitted_root_candidate(baseline, cand, root, tol=COLLAPSE_TOL):
    """The rooted tree a fit actually reached: ``(candidate, fitted fraction)``.

    A node candidate is its own tree (fraction ``None``). An edge candidate whose
    fitted root lies within ``tol`` expected mutations (branch length x number of
    sites) of an endpoint is the tree rooted at that endpoint, the same rooted
    tree as that node's candidate and as every other incident edge that
    collapsed there. Otherwise it stays the edge root, at its fitted fraction
    (measured from the endpoint away from the reference leaf, as
    :class:`EdgeRoot`'s ``fraction``)."""
    if not isinstance(cand, EdgeRoot) or not root or root.get('fitted_fraction') is None:
        return cand, None
    frac = float(root['fitted_fraction'])
    lengths = dict(zip(root['root_children'], root['root_child_lengths']))
    a = baseline.node_of[cand.split]
    la = lengths.get(a)
    lb = sum(v for k, v in lengths.items() if k != a)
    n = root.get('n_sites')
    if la is not None and n:
        if la * n <= tol and la <= lb:
            return baseline.canonical(EdgeRoot(cand.split, 0.0)), 0.0
        if lb * n <= tol:
            return baseline.canonical(EdgeRoot(cand.split, 1.0)), 1.0
    return EdgeRoot(cand.split, frac), frac


def _rescale_rate(tt, mu):
    """Set every branch length to mu x (its calendar time span), keeping all node
    dates: the strict-clock state (same dates, rate mu). Used inside the rate
    step only; the next time tree rebuilds the clock model at mu."""
    for n in tt.tree.find_clades(order='preorder'):
        if n.up is not None:
            n.branch_length = mu * (n.numdate - n.up.numdate)


def evaluate_root_candidate(baseline, cand, fit=None, keep_state=True, catch_errors=False, clock_rate=None,
                            initial_state=None):
    """Fit one candidate from a clean baseline; returns a :class:`FitRun`.

    The steps mirror ``TreeTime.run(root=None)`` on the pre-rooted tree in the
    canonical configuration (marginal mode, fixed GTR and clock rate, no
    coalescent, no polytomy resolution, prune_short=False), without the
    ``ndiff`` early stop: two marginal branch-length rounds, the initial time
    tree, then exactly ``fit.iterations`` x (marginal ancestral inference, time
    tree). After every time tree the full marginal likelihood is recomputed;
    the best-scoring complete TreeTime object is kept (``keep_state``), so the
    result never depends on which step happened to be last.

    An edge root on a zero-length edge (shorter than TreeTime's
    MIN_BRANCH_LENGTH floor) gives two zero-length root-incident edges, on which
    TreeTime's marginal branch-length step divides by zero; their initial
    lengths are then set to one mutation in total, split equally, and a warning
    is recorded. This is an initialisation only: the dated fit sets the final
    lengths. Node roots are never altered.

    With ``fit.rate_rounds`` > 0 the clock rate is fitted per candidate, jointly
    with the dates, by coordinate ascent starting from ``clock_rate`` (default
    the baseline rate mu0). Each round maximises the fresh likelihood over the
    rate at fixed node dates (every branch length = rate x time span; bounded
    Brent on log rate within a factor ``rate_span`` of the current rate and
    ``rate_clamp`` of mu0), then runs one more (marginal ancestral inference,
    time tree) at the new rate and records that state; it stops early when the
    rate moves by less than ``rate_xatol`` in log. States are recorded only after
    a time tree, so each retained state is a consistent TreeTime fit at its rate
    (``FitRun.clock_rates``, ``FitRun.clock_rate`` for the best step). A rate
    within 1% of the clamp is reported as a warning.

    ``initial_state`` (a fitted TreeTime of this candidate, e.g. a retained
    best state) starts the trajectory from that state instead of the baseline
    (``initial_mode`` 'warm'): it is deep-copied, given the baseline's root-state
    messages, scored at its own branch lengths (step 0), then iterated as above.
    Pass its clock rate as ``clock_rate``.

    Errors propagate unless ``catch_errors`` is set; a failed fit then has
    score -inf and the error in ``error``.
    """
    fit = fit or FitConfig()
    cand = baseline.canonical(cand)
    run = FitRun(candidate=cand, candidate_id=baseline.candidate_id(cand),
                 initial_fraction=None if initial_state is not None else getattr(cand, 'fraction', None),
                 initial_mode='warm' if initial_state is not None else fit.initial_mode,
                 rate_fitted=fit.rate_rounds > 0)
    t0 = time.time()
    try:
        seq_kwargs = _seq_kwargs()
        mu = float(clock_rate if clock_rate is not None else baseline.clock_rate)
        if not mu > 0:
            raise ValueError('clock_rate must be positive')
        if initial_state is not None:
            tt = copy.deepcopy(initial_state)
            if baseline.root_state_messages is not None or tt.root_state_message is not None:
                tt.set_root_state_messages(baseline.root_state_messages)
        else:
            tt = baseline.make_treetime(baseline.rooted_phylo(cand))
            kids = tt.tree.root.clades
            if len(kids) == 2 and kids[0].branch_length + kids[1].branch_length < ttconf.MIN_BRANCH_LENGTH * tt.one_mutation:
                for c in kids:
                    c.branch_length = c.mutation_length = 0.5 * tt.one_mutation
                run.warnings.append('zero-length root edge: initial length set to one mutation')
            tt._set_branch_length_mode('marginal')
            tt.optimize_tree(infer_gtr=False, max_iter=1, method_anc='probabilistic', **_seq_kwargs(fit.initial_mode))
            tt.optimize_tree(max_iter=1, method_anc='probabilistic', **seq_kwargs)
            tt.make_time_tree(clock_rate=mu, time_marginal=False)
        _record(baseline, run, tt, mu, None, None, keep_state)

        def iterate():
            # no root sequence yet on a warm start (its messages reset the reconstruction)
            prev_root = np.array(tt.tree.root._cseq, copy=True) if hasattr(tt.tree.root, '_cseq') else None
            ndiff = tt.infer_ancestral_sequences('ml', marginal=True, **seq_kwargs)
            root_changes = None if prev_root is None else int(np.sum(prev_root != tt.tree.root._cseq))
            tt.make_time_tree(clock_rate=mu, time_marginal=False)
            run.iterations_performed += 1
            if ndiff == 0 and run.treetime_stop_iteration is None:
                run.treetime_stop_iteration = run.iterations_performed
            _record(baseline, run, tt, mu, int(ndiff), root_changes, keep_state)

        for _ in range(fit.iterations):
            iterate()
        if fit.rate_rounds:
            from scipy.optimize import minimize_scalar

            def neg(x):
                _rescale_rate(tt, float(np.exp(x)))
                return -fresh_marginal_score(tt)
            glo, ghi = np.log(mu / fit.rate_clamp), np.log(mu * fit.rate_clamp)
        for _ in range(fit.rate_rounds):
            x0 = np.log(mu)
            lo, hi = max(glo, x0 - np.log(fit.rate_span)), min(ghi, x0 + np.log(fit.rate_span))
            x = float(minimize_scalar(neg, bounds=(lo, hi), method='bounded', options={'xatol': fit.rate_xatol}).x)
            mu = float(np.exp(x))
            _rescale_rate(tt, mu)
            if min(x - glo, ghi - x) < np.log(1.01) and 'rate at bound' not in run.warnings:
                run.warnings.append('rate at bound')
            iterate()
            run.rate_rounds_performed += 1
            if abs(x - x0) < fit.rate_xatol:
                break
        # oscillation measure: second half of the last phase (the rate rounds when the
        # rate is fitted, whose earlier steps are the climb towards the fitted rate)
        phase = run.scores[fit.iterations + 1:] if run.rate_rounds_performed >= 2 else run.scores
        tail = phase[len(phase) // 2:]
        run.tail_fluctuation = float(max(tail) - min(tail))
        run.trajectory_class = classify_trajectory(run.scores)
    except FIT_ERRORS as exc:
        if not catch_errors:
            raise
        run.error = '%s: %s' % (type(exc).__name__, exc)
        run.best_score = -np.inf
    run.runtime = time.time() - t0
    return run


def _record(baseline, run, tt, mu, ndiff, root_changes, keep_state):
    """Record the state after a time tree (never after a bare rate rescale)."""
    run.cached_LH.append(float(getattr(tt.tree, 'sequence_marginal_LH', np.nan)))  # before the refresh
    run.positional_LH.append(float(tt.tree.positional_LH))
    score = fresh_marginal_score(tt)
    run.scores.append(score)
    run.clock_rates.append(mu)
    run.ndiff.append(ndiff)
    run.root_state_changes.append(root_changes)
    if score > run.best_score:
        run.best_score = score
        run.best_index = len(run.scores) - 1
        run.clock_rate = mu
        run.root = _root_summary(baseline, run.candidate, tt)
        run.fitted_newick = fitted_newick(tt)
        if keep_state:
            run.state = copy.deepcopy(tt)  # node back-references point into the copy


def restore_best_fitted_state(run):
    """The TreeTime object retained at a fit's best-scoring step (a
    :class:`FitRun` or :class:`CandidateResult`): rooted tree, dates, branch
    lengths, clock model, profiles, GTR and RNG state as they were when that
    score was computed."""
    if isinstance(run, CandidateResult):
        run = run.best_run
    if run is None or run.state is None:
        raise ValueError('no retained state (keep_state=False or the fit failed)')
    return run.state


# --------------------------------------------------------------------------
# candidate generation
# --------------------------------------------------------------------------
def _regression_edges(baseline, covariation=False, tip_slack=None, slope='fixed'):
    """Every edge the regression ranks (finite objective, positive slope), best
    first, as ``(edge split, fraction, rank_roots entry, objective at the edge
    midpoint)``. ``fraction`` follows :class:`EdgeRoot`'s convention."""
    from .treeregression import base_regression
    treg = baseline.regression(covariation=covariation, tip_slack=tip_slack)
    s = baseline.clock_rate if slope == 'fixed' else slope
    out = []
    for p in treg.rank_roots(force_positive=True, slope=s):
        nd = p['node']
        if nd.name == baseline.ref_leaf:
            split, frac = baseline.split_of[baseline.center], 1.0 - p['split']
        else:
            split, frac = baseline.split_of[nd.name], p['split']
        tv, bv, var = treg.tip_value(nd), treg.branch_value(nd), treg.branch_variance(nd)
        q = (treg.propagate_averages(nd, tv, bv * 0.5, var * 0.5)
             + treg.propagate_averages(nd, tv, bv * 0.5, var * 0.5, outgroup=True))
        out.append((split, frac, p, float(base_regression(q, slope=s)['chisq'])))
    return out


def regression_candidates(baseline, k=5, covariation=False, tip_slack=None, slope='fixed'):
    """Best ``k`` distinct placements of the root-to-tip regression as canonical
    candidates, best first, each as ``(candidate, info)`` with the regression
    rank, objective and split. ``slope='fixed'`` uses the baseline clock rate;
    ``None`` lets the regression fit its own slope."""
    return _distinct_placements(baseline, _regression_edges(baseline, covariation, tip_slack, slope), k)


def _distinct_placements(baseline, edges, k):
    out, seen = [], set()
    for split, frac, p, _ in edges:
        cand = baseline.canonical(EdgeRoot(split, frac))
        if candidate_key(cand) in seen:
            continue
        seen.add(candidate_key(cand))
        out.append((cand, {'regression_rank': len(out) + 1, 'regression_chisq': float(p['edge_chisq']),
                           'regression_split': float(p['split'])}))
        if k is not None and len(out) >= k:
            break
    return out


def node_incident_edges(baseline, node, edges, n=None):
    """Up to ``n`` (``None``: all) edges incident to the internal node of
    ``node`` (a :class:`NodeRoot`), as midpoint :class:`EdgeRoot` candidates,
    best first by the regression objective at the edge midpoint (from
    ``edges``, :func:`_regression_edges`; edges it does not rank come last).
    Zero-length edges are skipped: a root on one is the node itself.

    A regression optimum at an edge endpoint is rounded to that node, and a
    node candidate's fit never leaves the node, so without this the edges next
    to a node optimum are never fitted."""
    mid = {split: m for split, _, _, m in edges}
    v = baseline.node_of[node.split]
    out = []
    for w, length in baseline.adj[v].items():
        if length > 0:
            split = baseline.split_of[v] if baseline.toward_ref.get(v) == w else baseline.split_of[w]
            out.append((mid.get(split, np.inf), sorted(split), split))
    out.sort(key=lambda e: e[:2])
    return [baseline.canonical(EdgeRoot(split, 0.5)) for _, _, split in out[:n]]


@dataclass(frozen=True)
class RegressionSource:
    """One regression shortlist: the best ``k`` placements of
    :func:`regression_candidates` with these arguments, labelled ``name``
    (``k=None``: every placement).

    ``node_edges`` expands each node candidate (ranked at most
    ``node_edges_max_rank``) into its incident edges (:func:`node_incident_edges`):
    0 = off, ``n`` = the ``n`` best by the midpoint objective, ``None`` = all.
    A regression optimum at an edge endpoint is rounded to the node, and the
    regression cannot tell which incident edge holds the root (in report 46 the
    midpoint objective picked the true edge about as often as chance), so
    ``None`` is the useful setting; it roughly doubles the shortlist, hence
    off by default."""

    name: str
    k: Optional[int] = 5
    covariation: bool = False
    tip_slack: Optional[float] = None
    slope: object = 'fixed'
    node_edges: Optional[int] = 0  # per node candidate, add this many incident edges (None: all); label '<name>_adj'
    node_edges_max_rank: Optional[int] = None  # expand only node candidates ranked <= this (None: every one)

    def candidates(self, baseline):
        return self.candidates_and_expansions(baseline)[0]

    def candidates_and_expansions(self, baseline):
        """``(candidates, expansions)``: the source's ``(candidate, info)`` list,
        and ``[(edge candidate, rank of the node it expands)]`` from
        :func:`node_incident_edges` for each node candidate (empty when
        ``node_edges`` is 0)."""
        edges = _regression_edges(baseline, self.covariation, self.tip_slack, self.slope)
        cands = _distinct_placements(baseline, edges, self.k)
        exp = []
        if self.node_edges is None or self.node_edges > 0:
            for c, info in cands:
                if isinstance(c, NodeRoot) and (self.node_edges_max_rank is None
                                                or info['regression_rank'] <= self.node_edges_max_rank):
                    exp += [(e, info['regression_rank']) for e in node_incident_edges(baseline, c, edges,
                                                                                      self.node_edges)]
        return cands, exp


DEFAULT_NODE_EDGES = 0


def default_candidate_sources(k=5, node_edges=DEFAULT_NODE_EDGES):
    """The default shortlist without an outgroup: plain root-to-tip regression
    at the baseline's fixed rate, top ``k``, and covariation regression
    (tip_slack 3, free slope), top ``k``. On simulated benchmarks the true root
    was often off the plain list alone and on the covariation list
    (investigation reports 35, 45). ``node_edges``: see :class:`RegressionSource`."""
    return (RegressionSource('plain', k, node_edges=node_edges),
            RegressionSource('cov', k, covariation=True, tip_slack=3.0, slope=None, node_edges=node_edges))


def outgroup_candidate_sources(k=5, node_edges=DEFAULT_NODE_EDGES):
    """The regression part of the default shortlist when outgroup-attachment
    candidates are added (Stage 2): covariation regression (tip_slack 3, free
    slope), top ``k``; with the attachment top 3 the plain list added no
    recall (investigation report 45)."""
    return (RegressionSource('cov', k, covariation=True, tip_slack=3.0, slope=None, node_edges=node_edges),)


def plain_candidate_sources(k=5):
    """The shortlist before the covariation source was added: plain regression
    at the baseline's fixed rate, top ``k``."""
    return (RegressionSource('plain', k),)


@dataclass(frozen=True)
class LsdvSource:
    """Candidate source ranked by the lsdv regression (opt-in): the best ``k``
    placements by the GLS residual (chisq) of the lsdv fit at each placement,
    each at its own lsdv rate (:meth:`UnrootedBaseline.lsdv`). It composes with
    the regression sources and ranked sources like a :class:`RegressionSource`,
    e.g. ``candidate_sources=default_candidate_sources(5) + (LsdvSource(),)``.

    The placements scored:

    * ``rerank`` = N (default 20): the top N distinct placements of the
      covariation regression (tip slack ``base_tip_slack``, free slope), at its
      fractions;
    * ``rerank=None``: every placement, i.e. every distinct placement the
      covariation regression ranks (at its fractions) plus the midpoint of every
      edge and every internal node it does not rank. This costs one lsdv fit per
      edge, O(n^2) in total (about 0.3 s at 80 tips).

    A placement whose plain root-to-tip rate or lsdv rate is not positive is
    skipped. Each candidate's info carries ``regression_rank`` (rank by lsdv
    chisq), ``regression_chisq``, ``regression_split`` (fraction scored),
    ``lsdv_rate``, ``lsdv_root_date`` and ``base_rank`` (rank in the covariation
    regression, None for added placements). The chisq values of different roots
    use different variances (durations at each root's own plain rate); whether
    this ranking places roots better than cov3 is an open benchmark question."""

    name: str = 'lsdv'
    k: Optional[int] = 5
    rerank: Optional[int] = 20
    tip_slack: float = 3.0
    rounds: int = 1
    base_tip_slack: float = 3.0
    node_edges: Optional[int] = 0  # unused; kept so callers reading RegressionSource fields can read this too

    def candidates(self, baseline):
        return self.candidates_and_expansions(baseline)[0]

    def candidates_and_expansions(self, baseline):
        if self.rerank is not None and self.rerank < 1:
            raise ValueError('LsdvSource: rerank must be >= 1 or None')
        if self.k is not None and self.k < 1:
            return [], []
        edges = _regression_edges(baseline, True, self.base_tip_slack, None)
        base = _distinct_placements(baseline, edges, self.rerank)
        pool = [(c, info['regression_rank']) for c, info in base]
        if self.rerank is None:
            seen = {candidate_key(c) for c, _ in pool}
            for c in baseline.all_candidates():
                c = baseline.canonical(c)
                if candidate_key(c) not in seen:
                    seen.add(candidate_key(c))
                    pool.append((c, None))
        scored = []
        for c, base_rank in pool:
            try:
                r = baseline.lsdv(c, tip_slack=self.tip_slack, rounds=self.rounds)
            except (ValueError, FloatingPointError, ZeroDivisionError):
                continue
            if r['rate'] > 0 and np.isfinite(r['chisq']):
                scored.append((r['chisq'], len(scored), c, base_rank, r))
        scored.sort(key=lambda x: x[:2])
        out = []
        for chisq, _, c, base_rank, r in scored[:self.k]:
            out.append((c, {'regression_rank': len(out) + 1, 'regression_chisq': float(chisq),
                            'regression_split': getattr(c, 'fraction', None), 'lsdv_rate': r['rate'],
                            'lsdv_root_date': r['root_date'], 'base_rank': base_rank}))
        return out, []


@dataclass
class ShortlistEntry:
    """One distinct candidate (node, or edge regardless of fraction) of a shortlist."""

    inits: list  # distinct initialisations, the first regression source's placement first
    source_ranks: dict  # {source name: rank within that source (1 = best)}, in the order the sources proposed it
    regression_info: dict  # {regression source name: rank, chisq and split from regression_candidates}

    @property
    def candidate(self):
        return self.inits[0]

    @property
    def source(self):
        return '+'.join(self.source_ranks)


def candidate_shortlist(baseline, explicit=(), sources=None, ranked_sources=None):
    """Merge candidates from several sources into distinct candidates, keyed by
    :func:`candidate_key` (bipartition identity), in first-proposed order.

    Sources are applied in this order: ``explicit`` (label 'explicit', rank =
    list position), ``ranked_sources`` (``{name: [candidate, ...]}``, best
    first, e.g. outgroup-attachment placements), then the regression
    ``sources`` (:class:`RegressionSource`; ``None`` = none). Explicit and ranked
    placements each add an initialisation; of the regression sources, only the
    first one to propose a candidate adds its placement, as the first
    initialisation, so a candidate on several regression lists is fitted once.
    Last come the node expansions of every regression source
    (``RegressionSource.node_edges``, label '<name>_adj', rank = the rank of
    the node they expand); they start at the edge midpoint and add no
    initialisation to a candidate that is already listed.
    Returns ``{candidate_key: ShortlistEntry}``.
    """
    merged = {}

    def entry(c):
        return merged.setdefault(candidate_key(c), ShortlistEntry(inits=[], source_ranks={}, regression_info={}))

    def add_listed(name, cands):
        for i, c in enumerate(cands):
            c = baseline.canonical(c)
            e = entry(c)
            e.source_ranks.setdefault(name, i + 1)
            if c not in e.inits:
                e.inits.append(c)

    add_listed('explicit', explicit)
    for name, cands in (ranked_sources or {}).items():
        if name == 'explicit':
            raise ValueError("candidate source name 'explicit' is reserved")
        add_listed(name, cands)
    names = [s.name for s in sources or ()]
    names += [n + '_adj' for n in names]
    if len(set(names)) != len(names) or set(names) & set(ranked_sources or {}):
        raise ValueError('candidate source names must be distinct: %s' % names)
    expanded = [(src, src.candidates_and_expansions(baseline)) for src in sources or ()]
    for src, (cands, _) in expanded:
        for c, info in cands:
            e = entry(c)
            if not e.regression_info:
                if c in e.inits:
                    e.inits.remove(c)
                e.inits.insert(0, c)
            e.source_ranks.setdefault(src.name, info['regression_rank'])
            e.regression_info[src.name] = info
    for src, (_, expansions) in expanded:  # after every list, so an edge already listed keeps its own start
        for c, node_rank in expansions:
            e = entry(c)
            e.source_ranks.setdefault(src.name + '_adj', node_rank)
            if not e.inits:
                e.inits.append(c)
    return merged


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
@dataclass
class CandidateResult:
    """All fits of one candidate; its score is the best over initialisations."""

    candidate: object
    candidate_id: str
    source: str
    regression_rank: Optional[int] = None
    regression_chisq: Optional[float] = None
    source_ranks: dict = field(default_factory=dict)  # {source name: rank within that source}
    runs: list = field(default_factory=list)
    delta_logL: Optional[float] = None
    resolved_from_best: Optional[bool] = None
    rate_fitted: bool = False
    root_candidate: object = None  # the rooted tree the best fit reached (fitted_root_candidate)
    root_id: Optional[str] = None
    root_fraction: Optional[float] = None  # fitted fraction on the edge; None for a node root
    duplicate_of: Optional[str] = None  # candidate_id of the better-scoring result with the same rooted tree
    merged: list = field(default_factory=list)  # on a representative: [{candidate_id, source, logL}] of its duplicates

    @property
    def best_run(self):
        ok = [r for r in self.runs if r.error is None]
        return max(ok, key=lambda r: r.best_score) if ok else (self.runs[0] if self.runs else None)

    @property
    def score(self):
        b = self.best_run
        return b.best_score if b is not None and b.error is None else -np.inf

    @property
    def init_spread(self):
        """max - min of the best scores reached from different initialisations
        (among runs with the same rate setting as the best run)."""
        b = self.best_run
        sc = [r.best_score for r in self.runs if r.error is None and r.rate_fitted == b.rate_fitted]
        return float(max(sc) - min(sc)) if len(sc) > 1 else None

    @property
    def uncertainty(self):
        """Optimiser uncertainty: the larger of the spread across initialisations
        and the tail fluctuation of the best run's trajectory."""
        b = self.best_run
        return max(self.init_spread or 0.0, b.tail_fluctuation if b is not None else 0.0)

    def row(self, clock_rate):
        b = self.best_run
        ok = b is not None and b.error is None
        return {
            'candidate_id': self.candidate_id,
            'stable_id': stable_id(self.candidate),
            'split': sorted(self.candidate.split),
            'kind': self.candidate.kind,
            'source': self.source,
            'source_ranks': format_source_ranks(self.source_ranks),
            'regression_rank': self.regression_rank,
            'regression_chisq': self.regression_chisq,
            'logL': self.score if ok else None,
            'delta_logL': self.delta_logL,
            'positional_LH': b.positional_LH[b.best_index] if ok else None,
            'cached_LH': b.cached_LH[b.best_index] if ok else None,
            'clock_rate': b.clock_rate if ok else clock_rate,
            'rate_fitted': self.rate_fitted,
            'initial_fraction': b.initial_fraction if b else None,
            'initial_mode': b.initial_mode if b else None,
            'fitted_fraction': b.root.get('fitted_fraction') if ok else None,
            'root_date': b.root.get('root_date') if ok else None,
            'root_degree': b.root.get('root_degree') if ok else None,
            'root_child_lengths': b.root.get('root_child_lengths') if ok else None,
            'iterations_performed': b.iterations_performed if b else None,
            'best_iteration': b.best_index if b else None,
            'treetime_stop_iteration': b.treetime_stop_iteration if b else None,
            'trajectory_class': b.trajectory_class if b else None,
            'tail_fluctuation': b.tail_fluctuation if ok else None,
            'n_initialisations': len(self.runs),
            'init_spread': self.init_spread,
            'uncertainty': self.uncertainty if ok else None,
            'resolved_from_best': self.resolved_from_best,
            'root_id': self.root_id,
            'root_fraction': self.root_fraction,
            'duplicate_of': self.duplicate_of,
            'merged_ids': [m['candidate_id'] for m in self.merged] or None,
            'merged_sources': [m['source'] for m in self.merged] or None,
            'runtime': sum(r.runtime for r in self.runs),
            'error': '; '.join(r.error for r in self.runs if r.error) or None,
            'warnings': '; '.join(sorted({w for r in self.runs for w in r.warnings})) or None,
            'scores': b.scores if b else None,
            'fitted_newick': b.fitted_newick if ok else None,
        }


def format_source_ranks(ranks):
    """``{'plain': 1, 'cov': 3}`` -> ``'plain:1,cov:3'``."""
    return ','.join('%s:%d' % (k, v) for k, v in ranks.items())


TSV_COLS = ['candidate_id', 'stable_id', 'kind', 'source', 'source_ranks', 'regression_rank', 'regression_chisq', 'logL', 'delta_logL',
            'positional_LH', 'cached_LH', 'clock_rate', 'rate_fitted', 'initial_fraction', 'initial_mode', 'fitted_fraction',
            'root_date', 'root_degree', 'root_child_lengths', 'iterations_performed', 'best_iteration',
            'treetime_stop_iteration', 'trajectory_class', 'tail_fluctuation', 'n_initialisations',
            'init_spread', 'uncertainty', 'resolved_from_best', 'root_id', 'root_fraction', 'duplicate_of', 'merged_ids',
            'runtime', 'error', 'warnings', 'scores']


@dataclass
class RootSearchResult:
    results: list  # CandidateResult, best first
    clock_rate: float
    min_margin: float
    config: dict
    final_rate_refit: Optional[dict] = None  # see rank_root_candidates(final_rate_refit=); None when off
    final_lsdv: Optional[dict] = None  # see rank_root_candidates(final_lsdv=); None when off

    @property
    def best(self):
        return self.results[0]

    @property
    def distinct(self):
        """Results with distinct fitted rooted trees, best first: duplicates (an
        edge candidate whose fit collapsed onto a node another candidate also
        reached) are left out; their labels and sources are in the
        representative's ``merged``."""
        return [r for r in self.results if r.duplicate_of is None]

    @property
    def margin(self):
        """logL gap between the best and the second distinct rooted tree."""
        d = self.distinct
        if len(d) < 2 or not np.isfinite(d[1].score):
            return None
        return d[0].score - d[1].score

    @property
    def unresolved_with_best(self):
        return [r.candidate_id for r in self.distinct[1:] if not r.resolved_from_best]

    @property
    def best_is_resolved(self):
        return not self.unresolved_with_best

    @property
    def regression_best(self):
        return next((r for r in self.results if r.regression_rank == 1), None)

    def best_treetime(self):
        """The winner's retained state from ranking (at the rate it was ranked with)."""
        return restore_best_fitted_state(self.best)

    def headline_treetime(self):
        """The state behind the headline root date: the winner's final rate refit
        when one was run and kept its state, else :meth:`best_treetime`."""
        ref = self.final_rate_refit
        if ref is not None and ref.get('headline') == 'refit' and ref['state'] is not None:
            return ref['state']
        return self.best_treetime()

    @property
    def headline_root_date(self):
        """Root date of the winner: the final rate refit's when it ran, else the ranking fit's."""
        ref = self.final_rate_refit
        if ref is not None and ref.get('headline') == 'refit':
            return ref['refit_root_date']
        b = self.best.best_run
        return b.root.get('root_date') if b is not None and b.error is None else None

    @property
    def headline_clock_rate(self):
        ref = self.final_rate_refit
        if ref is not None and ref.get('headline') == 'refit':
            return ref['refit_clock_rate']
        b = self.best.best_run
        return b.clock_rate if b is not None and b.error is None else self.clock_rate

    def summary(self):
        reg = self.regression_best
        out = {
            'best': self.best.candidate_id,
            'best_root': self.best.root_id,
            'best_root_fraction': self.best.root_fraction,
            'best_logL': self.best.score,
            'margin': self.margin,
            'best_is_resolved': self.best_is_resolved,
            'unresolved_with_best': self.unresolved_with_best,
            'regression_best': reg.candidate_id if reg else None,
            'regression_best_delta_logL': reg.delta_logL if reg else None,
            'clock_rate': self.clock_rate,
            'min_margin': self.min_margin,
            'n_candidates': len(self.results),
            'n_distinct': len(self.distinct),
            'n_failed': sum(not np.isfinite(r.score) for r in self.results),
        }
        ref = self.final_rate_refit
        if ref is not None:
            out.update({'headline_root_date': self.headline_root_date, 'headline_clock_rate': self.headline_clock_rate,
                        'final_rate_refit': final_rate_refit_summary(ref)})
        if self.final_lsdv is not None:
            out['final_lsdv'] = dict(self.final_lsdv)
        sri = self.config.get('shared_rate_info')
        if sri is not None:
            out['shared_rate_source'] = sri['source'] if not sri.get('fallback') else 'regression (lsdv fallback)'
            out['shared_rate_log_range'] = sri.get('log_range')
        return out

    def rows(self):
        return [r.row(self.clock_rate) for r in self.results]

    def to_tsv(self, path=None, cols=TSV_COLS):
        lines = ['\t'.join(cols)] + ['\t'.join(_fmt(r.get(c)) for c in cols) for r in self.rows()]
        s = '\n'.join(lines) + '\n'
        if path:
            with open(path, 'w') as fh:
                fh.write(s)
        return s

    def to_json(self, path=None):
        s = json.dumps({'summary': self.summary(), 'config': self.config, 'candidates': self.rows()},
                       indent=1, default=_json_default)
        if path:
            with open(path, 'w') as fh:
                fh.write(s)
        return s

    def table(self, n=None):
        """Concise comparison table."""
        out = ['%4s %-28s %-19s %4s %12s %8s %6s %10s %3s %3s %-12s %s'
               % ('rank', 'candidate', 'source', 'reg', 'logL', 'dlogL', 'u', 'root_date', 'deg', 'it',
                  'trajectory', 'resolved')]
        for i, r in enumerate(self.rows()[:n]):
            out.append('%4d %-28s %-19s %4s %12s %8s %6s %10s %3s %3s %-12s %s' % (
                i + 1, r['candidate_id'][:28], r['source'][:19], _fmt(r['regression_rank']), _fmt(r['logL'], 3),
                _fmt(r['delta_logL'], 3), _fmt(r['uncertainty'], 2), _fmt(r['root_date'], 2),
                _fmt(r['root_degree']), _fmt(r['best_iteration']), r['trajectory_class'] or '',
                '' if i == 0 else ('= ' + r['duplicate_of']) if r['duplicate_of'] else _fmt(r['resolved_from_best'])))
        s = self.summary()
        out.append('best %s (root %s), margin %s, %s; regression best %s (dlogL %s)' % (
            s['best'], s['best_root'], _fmt(s['margin'], 3), 'resolved' if s['best_is_resolved'] else
            'unresolved with ' + ', '.join(s['unresolved_with_best']), s['regression_best'],
            _fmt(s['regression_best_delta_logL'], 3)))
        ref = self.final_rate_refit
        if ref is not None:
            out.append('final rate refit (%s): root date %s -> %s, rate %s -> %s, logL %s -> %s; headline: %s' % (
                ref['candidate_id'], _fmt(ref['shared_root_date'], 2), _fmt(ref['refit_root_date'], 2),
                _fmt(ref['shared_clock_rate'], 6), _fmt(ref['refit_clock_rate'], 6), _fmt(ref['shared_logL'], 3),
                _fmt(ref['refit_logL'], 3), ref['headline']))
        return '\n'.join(out)


def _fmt(v, nd=6):
    if v is None:
        return ''
    if isinstance(v, (float, np.floating)):
        return ('%.' + str(nd) + 'f') % v if np.isfinite(v) else str(v)
    if isinstance(v, (list, tuple)):
        return json.dumps(v, default=_json_default)
    return str(v)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def _alternatives(cand, tried_fractions, fractions):
    """Alternative initialisations of a close candidate: other initial fractions
    for an edge, and a joint-mode first branch-length round for every candidate
    (the only perturbation of a node root, and of an edge root on a zero-length
    edge, where all fractions start alike)."""
    alts = []
    if isinstance(cand, EdgeRoot):
        alts += [(EdgeRoot(cand.split, f), 'marginal') for f in fractions
                 if all(abs(f - t) > 0.05 for t in tried_fractions)]
    alts.append((cand, 'joint'))
    return alts


def resolve_candidate_sources(regression_k=5, candidate_sources=None, covariation=None, tip_slack=None):
    """The regression sources :func:`rank_root_candidates` uses (see there)."""
    if covariation is not None or tip_slack is not None:
        if candidate_sources is not None:
            raise ValueError('pass either candidate_sources or covariation/tip_slack, not both')
        if not regression_k:
            return ()
        name = 'cov_fixed_slope' if covariation else 'plain'
        return (RegressionSource(name, regression_k, covariation=bool(covariation), tip_slack=tip_slack),)
    if candidate_sources is not None:
        return tuple(candidate_sources)
    return default_candidate_sources(regression_k) if regression_k else ()


FINAL_RATE_EVALS = 15         # profile evaluations (dated fits at fixed rate) per final refit
FINAL_RATE_XATOL = 0.01       # tolerance on log(rate): 1% in the rate
FINAL_RATE_ITERATIONS = 2     # (ancestral inference, time tree) iterations per block of a profile evaluation
FINAL_RATE_MAX_ITERATIONS = 8  # cap per profile evaluation: blocks continue while a block gains >= the tolerance
FINAL_RATE_POINT_TOL = 0.01    # logL gain below which a profile evaluation stops
FAILED_POINT_PENALTY = 1e100   # finite stand-in for a failed evaluation, so Brent keeps bracketing
FALLBACK_GRID = 9              # after a failed evaluation: log-rate grid over the clamp range, then Brent in its best cell


def rescale_times_for_rate(state, mu0, mu):
    """Copy of a fitted TreeTime ``state`` (a strict-clock fit at rate ``mu0``)
    moved to rate ``mu`` at (about) fixed divergences: every internal node's
    height below the latest tip is scaled by mu0/mu, tips keep their dates, and
    every branch = mu x its new time span. A node is kept slightly older than
    its oldest child (``ttconf.MIN_BRANCH_LENGTH`` mutations at ``mu``). This is the
    starting state of a dated fit at ``mu``: a rate change at fixed divergences
    mostly rescales times homogeneously."""
    tt = copy.deepcopy(state)
    t_max = max(float(n.numdate) for n in tt.tree.get_terminals())
    f = float(mu0) / float(mu)
    eps = ttconf.MIN_BRANCH_LENGTH * tt.one_mutation / float(mu)
    for n in tt.tree.find_clades(order='postorder'):
        if n.is_terminal():
            continue
        t = t_max - (t_max - float(n.numdate)) * f
        n.numdate = min(t, min(float(c.numdate) for c in n.clades) - eps)
    _rescale_rate(tt, float(mu))
    return tt


def fit_rate_profile(point, mu0, clamp=10.0, xatol=FINAL_RATE_XATOL, max_evals=FINAL_RATE_EVALS):
    """Maximise a profile likelihood over the clock rate, ``point(mu) ->
    (logL, payload)`` (``logL`` -inf or a payload with ``error`` set marks a
    failed evaluation), over log(rate) in [mu0/clamp, mu0*clamp] by bounded
    Brent (scipy ``minimize_scalar``, 'bounded'), at most ``max_evals``
    evaluations, tolerance ``xatol`` on log(rate).

    A failed evaluation is given a large finite penalty (an infinite one
    makes Brent drift towards the bound and still report success) and is
    listed in ``failures`` as ``(mu, error)``. Because the penalty is flat,
    any failure triggers a fallback (``grid_fallback``): a ``FALLBACK_GRID``
    point grid on log(rate) over the range, then Brent within the cells next
    to the best grid point (each stage at most ``max_evals`` evaluations). ``converged``: Brent met the
    tolerance within the budget, the maximum is not at a bound (within
    max(1%, 2 xatol) in the rate) and no failed evaluation neighbours it.
    With every evaluation failed, ``mu``/``logL``/``payload`` are None and
    ``error`` says so. Returns a dict with the best point, every evaluated
    ``(mu, logL)`` (failures as -inf) and the flags."""
    from scipy.optimize import minimize_scalar
    cache = {}

    def ok(v):
        return np.isfinite(v[0]) and getattr(v[1], 'error', None) is None

    def neg(x):
        x = float(x)
        if x not in cache:
            cache[x] = point(float(np.exp(x)))
        return -cache[x][0] if ok(cache[x]) else FAILED_POINT_PENALTY

    lo, hi = np.log(mu0 / clamp), np.log(mu0 * clamp)
    res = minimize_scalar(neg, bounds=(lo, hi), method='bounded',
                          options={'xatol': xatol, 'maxiter': max_evals})
    fallback = False
    if any(not ok(v) for v in cache.values()):
        # failures make the penalised objective flat there: scan a grid, then Brent inside the best grid cell
        fallback = True
        grid = np.linspace(lo, hi, FALLBACK_GRID)
        for x in grid:
            neg(x)
        good = [x for x in grid if ok(cache[float(x)])]
        if good:
            k = int(np.searchsorted(grid, max(good, key=lambda x: cache[float(x)][0])))
            a, c = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
            res = minimize_scalar(neg, bounds=(a, c), method='bounded', options={'xatol': xatol, 'maxiter': max_evals})
    xs = sorted(cache)
    failures = [(float(np.exp(x)), getattr(cache[x][1], 'error', None) or 'logL not finite')
                for x in xs if not ok(cache[x])]
    points = [(float(np.exp(x)), cache[x][0] if ok(cache[x]) else -np.inf) for x in xs]
    good = [x for x in xs if ok(cache[x])]
    out = {'points': points, 'n_evals': len(cache), 'brent_converged': bool(res.success), 'failures': failures,
           'grid_fallback': fallback}
    if not good:
        out.update(mu=None, logL=None, payload=None, at_bound=None, converged=False,
                   error='all %d profile evaluations failed: %s' % (len(xs), failures[0][1] if failures else '?'))
        return out
    xb = max(good, key=lambda x: cache[x][0])
    at_bound = min(xb - lo, hi - xb) < max(np.log(1.01), 2 * xatol)
    k = xs.index(xb)
    failed_next = any(0 <= n < len(xs) and not ok(cache[xs[n]]) for n in (k - 1, k + 1))
    out.update(mu=float(np.exp(xb)), logL=cache[xb][0], payload=cache[xb][1], at_bound=bool(at_bound),
               converged=bool(res.success and not at_bound and not failed_next), error=None,
               failure_next_to_best=bool(failed_next))
    return out


def profile_rate_point(evaluate, cand, state, mu0, mu, iterations=FINAL_RATE_ITERATIONS,
                       max_iterations=FINAL_RATE_MAX_ITERATIONS, tol=FINAL_RATE_POINT_TOL):
    """One point of the profile likelihood over the clock rate: the dated fit of
    candidate ``cand`` at fixed rate ``mu`` (``evaluate(cand, mode, rounds,
    warm=(state, mu), iterations=)``, i.e. the ranking's evaluator, so Stage 2
    messages and Stage 3 outgroup subtrees are included), warm-started from
    :func:`rescale_times_for_rate` (``state`` at rate ``mu0``, moved to ``mu``).
    The fit runs in blocks of ``iterations``, each warm from the previous
    block's best state, while a block gains at least ``tol`` and up to
    ``max_iterations`` in all: two iterations alone underfit well above the
    profile maximum, where the rescaled start is far from the fixed point.
    Returns ``(logL, FitRun)``; the FitRun's ``extra`` gains
    ``profile_iterations`` and ``profile_last_gain``. A failed fit gives -inf
    and the run with ``error`` set."""
    run = evaluate(cand, None, 0, warm=(rescale_times_for_rate(state, mu0, mu), float(mu)), iterations=iterations)
    done, gain = iterations, None
    while run.error is None and done + iterations <= max_iterations:
        nxt = evaluate(cand, None, 0, warm=(run.state, float(mu)), iterations=iterations)
        done += iterations
        if nxt.error is not None:
            break
        gain = nxt.best_score - run.best_score
        if gain > 0:
            run = nxt
        if gain < tol:
            break
    if run.error is None:
        run.extra = dict(run.extra or {}, profile_iterations=done, profile_last_gain=gain)
    return (run.best_score if run.error is None else -np.inf), run


def _final_rate_refit_one(r, evaluate, clamp, xatol, max_evals, iterations, catch_errors):
    """Rate refit of one ranked candidate: :func:`fit_rate_profile` from its
    best run's retained state. The ranking state itself counts as a point, so
    the refit logL is never below the ranking logL. Returns a record dict; the
    ranking runs are untouched.

    ``shared_*`` are the candidate's best fit at the shared rate (its runs
    without a rate fit); ``ranking_*`` its ranking score and rate, which differ
    from ``shared_*`` only when ranking already fitted rates (``rate_rounds`` > 0)."""
    b = r.best_run
    shared = [x for x in r.runs if x.error is None and not x.rate_fitted]
    s = max(shared, key=lambda x: x.best_score) if shared else None
    rec = {'candidate_id': r.candidate_id,
           'shared_logL': s.best_score if s else None,
           'shared_root_date': s.root.get('root_date') if s else None,
           'shared_clock_rate': s.clock_rate if s else None,
           'ranking_logL': r.score, 'ranking_clock_rate': b.clock_rate, 'ranking_rate_fitted': b.rate_fitted,
           'error': None, 'warnings': [], 'state': None, 'refit_logL': None, 'refit_root_date': None,
           'refit_clock_rate': None, 'n_evals': 0, 'converged': None, 'at_bound': None, 'profile_points': [],
           'refit_is_ranking_state': None, 'refit_extra': {}, 'profile_failures': []}
    if b.state is None:
        raise ValueError('final rate refit: the winner kept no state')
    t0 = time.time()
    f = None

    def point(mu):
        try:
            return profile_rate_point(evaluate, b.candidate, b.state, b.clock_rate, mu, iterations)
        except FIT_ERRORS as exc:
            if not catch_errors:
                raise
            return -np.inf, FitRun(candidate=b.candidate, candidate_id=r.candidate_id, initial_fraction=None,
                                   initial_mode='warm',
                                   error='%s: %s' % (type(exc).__name__, exc))
    try:
        f = fit_rate_profile(point, b.clock_rate, clamp=clamp, xatol=xatol, max_evals=max_evals)
    except FIT_ERRORS as exc:
        if not catch_errors:
            raise
        rec.update(error='%s: %s' % (type(exc).__name__, exc), converged=False)
    else:
        rec.update(n_evals=f['n_evals'], profile_points=f['points'], profile_failures=f['failures'])
        if f['failures']:
            rec['warnings'].append('%d of %d profile evaluations failed' % (len(f['failures']), f['n_evals']))
    if f is not None and f['error'] is not None:
        rec.update(error=f['error'], converged=False)
    elif f is not None:
        keep_ranking = r.score >= f['logL']
        best = b if keep_ranking else f['payload']
        st = best.state
        rec.update({'state': st, 'refit_logL': max(r.score, f['logL']),
                    'refit_root_date': float(st.tree.root.numdate),
                    'refit_clock_rate': b.clock_rate if keep_ranking else f['mu'],
                    'n_evals': f['n_evals'], 'converged': f['converged'], 'at_bound': f['at_bound'],
                    'profile_points': f['points'], 'refit_is_ranking_state': bool(keep_ranking),
                    'refit_extra': dict(best.extra) if getattr(best, 'extra', None) else {}})
        if f['at_bound']:
            rec['warnings'].append('rate at bound')
        elif f.get('failure_next_to_best'):
            rec['warnings'].append('failed profile evaluation next to the best rate')
        elif not f['converged']:
            rec['warnings'].append('rate refit not converged')
    rec['runtime'] = time.time() - t0
    return rec


def final_rate_refit_summary(ref):
    """JSON-friendly view of :attr:`RootSearchResult.final_rate_refit` (no FitRun objects)."""
    out = {k: v for k, v in ref.items() if k not in ('state', 'close')}
    if ref.get('close') is not None:
        out['close'] = [{k: v for k, v in c.items() if k != 'state'} for c in ref['close']]
    return out



def merge_duplicate_roots(baseline, results, tol=COLLAPSE_TOL):
    """Label each result (sorted best first) by its fitted rooted tree and mark
    duplicates in place. The first (best-scoring) result reaching a tree
    represents it; later ones get ``duplicate_of`` and are listed in the
    representative's ``merged``. Failed fits keep their own label and are
    never merged."""
    reps = {}
    for r in results:
        r.duplicate_of, r.merged = None, []
        b = r.best_run
        ok = b is not None and b.error is None and np.isfinite(r.score)
        root, frac = fitted_root_candidate(baseline, r.candidate, b.root if ok else None, tol)
        r.root_candidate, r.root_fraction = root, frac
        r.root_id = baseline.candidate_id(root)
        if not ok:
            continue
        key = candidate_key(root)
        if key in reps:
            rep = reps[key]
            r.duplicate_of = rep.candidate_id
            rep.merged.append({'candidate_id': r.candidate_id, 'source': r.source, 'logL': r.score})
        else:
            reps[key] = r
    return results


def rank_root_candidates(baseline, explicit=(), regression_k=5, fit=None, sensitivity_window=None,
                         sensitivity_fractions=(0.1, 0.5, 0.9), min_margin=1.0, covariation=None,
                         tip_slack=None, keep_states=True, catch_errors=False, rate_rounds=0, rate_fit_window=5.0,
                         rate_fit_min=3, rate_fit_max=15, rate_fit_extra=(), warm_starts=None,
                         candidate_sources=None, ranked_sources=None, evaluator=None, final_rate_refit=False,
                         final_rate_evals=FINAL_RATE_EVALS, final_rate_xatol=FINAL_RATE_XATOL,
                         final_rate_iterations=FINAL_RATE_ITERATIONS, final_rate_refit_window=None,
                         collapse_tol=COLLAPSE_TOL, final_lsdv=False):
    """Evaluate explicit and regression-ranked candidates and rank them by fitted
    full marginal log-likelihood.

    **Candidates** (:func:`candidate_shortlist`): ``explicit`` placements,
    ``ranked_sources`` (``{name: [candidate, ...]}``, best first; Stage 2 passes
    the outgroup-attachment placements as ``'attach'``) and the regression
    sources, deduplicated by bipartition. Each result records every source that
    proposed it and its rank there (``source``, ``source_ranks``). Several
    placements on one edge are one candidate with several initialisations, the
    first regression source's fraction first. Regression scores are kept for
    diagnostics only.

    Regression sources (:class:`RegressionSource`):

    * default (``candidate_sources=None``): :func:`default_candidate_sources`,
      plain regression at the baseline's rate, top ``regression_k``, and
      covariation regression (tip_slack 3, free slope), top ``regression_k``;
    * ``candidate_sources=plain_candidate_sources(k)``: the plain top ``k`` alone,
      the default before the covariation source was added;
    * any sequence of :class:`RegressionSource` (``regression_k`` is then unused);
    * ``covariation``/``tip_slack`` (legacy, exclusive with ``candidate_sources``):
      one regression source with that model at the baseline's fixed rate, top
      ``regression_k``;
    * ``regression_k=0``: none.

    ``regression_rank``/``regression_chisq`` (and :attr:`RootSearchResult.regression_best`,
    and the rate-fit extra below) refer to the *first* regression source, plain
    by default.

    A candidate's optimiser uncertainty ``u`` is the tail fluctuation of its best
    trajectory, or, where alternative initialisations were run, the larger of
    that and their spread. A candidate is *resolved* from the best when
    ΔlogL > max(min_margin, u_best + u_candidate). "Resolved" is a statement about
    the optimiser, not about the true root: under clock misspecification the
    likelihood can prefer a wrong root by many log units.

    Opt-in extras (off by default; on simulated benchmarks they changed almost
    no answer and cost several fits per candidate):

    * ``sensitivity_window``: every candidate within this many log units of the
      best is re-fitted from alternative initialisations (:func:`_alternatives`:
      other initial fractions and a joint-mode start); its score is its best
      over initialisations.
    * ``rate_rounds`` > 0: per-candidate clock rate, below.

    ``min_margin`` (default 1 log unit) is calibrated on simulations, where
    optimiser spread alone left most wrong answers looking resolved; it only
    affects the verdict, never the ranking. ``min_margin=0`` disables it.

    **Clock rate.** Every candidate is first fitted at the baseline's shared rate.
    With ``rate_rounds`` > 0, every candidate within
    ``rate_fit_window`` of the best (at least ``rate_fit_min``, at most
    ``rate_fit_max``, best first), plus the regression's rank-1 candidate and any
    ``rate_fit_extra`` candidates, is then re-fitted with the rate fitted per
    candidate jointly with the dates
    (:func:`evaluate_root_candidate`, ``FitConfig.rate_rounds``), starting from the
    shared rate; their score is the maximum over rate and dates. A shared-rate fit
    is the special case of one fixed rate, so a candidate's score never drops.
    Candidates not re-fitted keep their shared-rate score, a lower bound
    (``rate_fitted`` column). Their sensitivity re-fits use the same rate setting.
    ``rate_rounds=0`` uses the shared rate throughout.

    **Final rate refit** (``final_rate_refit=True``, opt-in). After ranking, the
    winner alone gets a clock rate of its own: :func:`fit_rate_profile`
    maximises the profile likelihood over the rate (bounded Brent on log rate
    within ``fit.rate_clamp`` of the winner's rate, at most ``final_rate_evals``
    evaluations, tolerance ``final_rate_xatol`` on log rate). Each evaluation is
    the winner's dated fit at that fixed rate through the ranking's evaluator
    (so ``evaluator=``, e.g. Stage 3's outgroup subtree, applies), warm-started
    from its retained best state with node times rescaled to that rate
    (:func:`rescale_times_for_rate`; ``final_rate_iterations`` iterations; states are kept internally
    for this even with ``keep_states=False`` and dropped afterwards). The
    ranking state counts as a point, so the refit logL is never below the
    winner's ranking logL. Selection is unchanged: ranks, scores, verdicts and
    rows stay those of the ranking. The refit is reported in
    :attr:`RootSearchResult.final_rate_refit` (dict: shared, ranking and refit
    root date, clock rate and logL, ``converged``, ``at_bound``, ``n_evals``,
    ``profile_points``, ``headline``) as a diagnostic: the headline root date
    and rate (:attr:`RootSearchResult.headline_root_date`, ``headline_clock_rate``,
    :meth:`RootSearchResult.headline_treetime`) stay those of the shared-rate
    ranking (``headline`` 'shared'). The reason: with a shared regression rate the root date
    inherits the regression's rate error, which dominates root-date error under
    weak temporal signal. ``converged`` is False (warning 'rate refit not
    converged', or 'rate at bound') when Brent did not meet the tolerance within
    the budget or the maximum lies at the clamp. ``final_rate_refit=False``
    gives exactly the output without it (``final_rate_refit`` None, summary
    without the headline keys). It is off by default: on simulated weak-signal
    data the rate profile is flat towards low rates and its maximum lies far
    from the true rate more often than the shared regression rate does, and
    the dated fit at a fixed rate can reach different fixed points from
    different starting states (investigation report 49).

    ``final_rate_refit_window`` (opt-in, report only; e.g. ``CLOSE_ODDS_WINDOW``
    = log 3, 3:1 odds): every other candidate within this many log units of the
    winner is rate-refitted the same way, and ``final_rate_refit['close']`` lists
    them with ``close_refit_best`` and ``close_order_changed`` (whether the
    refitted logLs would order them differently). The winner and the headline
    never change because of it.

    ``warm_starts``: ``{candidate_key: (TreeTime, clock_rate)}``; each listed
    candidate gets one more trajectory started from that fitted state
    (``initial_mode`` 'warm', see :func:`evaluate_root_candidate`), so its score
    is at least that state's score under this baseline. Stage 2b uses it to
    continue from the Stage 1 best state.

    ``keep_states=False`` keeps no TreeTime copies; otherwise only the best run
    of each candidate keeps its copy.

    **Duplicate rooted trees** (:func:`merge_duplicate_roots`). After fitting,
    each candidate is labelled by the rooted tree its best fit reached
    (``root_id``, ``root_fraction``): an edge root within ``collapse_tol``
    expected mutations of an endpoint is that node's tree. Results that reached
    the same tree are duplicates: the best-scoring one represents it and lists
    the others in ``merged``; the others get ``duplicate_of`` and no verdict.
    ``margin``, ``resolved_from_best`` and ``unresolved_with_best`` compare
    distinct trees only. The order, the scores and the winner are unchanged.

    ``evaluator``: replaces :func:`evaluate_root_candidate` (same signature and
    return type) for every fit; Stage 3 passes an
    :class:`~treetime.outgroup_subtree.OutgroupEvaluator`, whose messages depend
    on the candidate. Selection, rate fits and margins are unchanged.

    ``final_lsdv`` (opt-in, report only; ``True`` or a dict of
    :meth:`UnrootedBaseline.lsdv` options): the lsdv rate and root date at the
    winner's fitted root (``root_candidate``, on the baseline branch lengths),
    in :attr:`RootSearchResult.final_lsdv` and the summary. Ranking, scores and
    the headline are unchanged. ``False`` gives exactly the output without it.
    """
    fit = fit or FitConfig()
    sources = resolve_candidate_sources(regression_k, candidate_sources, covariation, tip_slack)
    merged = candidate_shortlist(baseline, explicit=explicit, sources=sources, ranked_sources=ranked_sources)
    primary = sources[0].name if sources else None

    warm_starts = warm_starts or {}
    keep_internal = keep_states or bool(final_rate_refit)  # the final refit warm-starts from the winner's state

    def evaluate(c, mode, rounds=0, warm=None, iterations=None):
        if mode == 'warm':  # a warm run's rate fit restarts from the same state
            warm, mode = warm_starts[candidate_key(baseline.canonical(c))], fit.initial_mode
        cfg = FitConfig(fit.iterations if iterations is None else iterations, mode or fit.initial_mode,
                        rate_rounds=rounds, rate_span=fit.rate_span, rate_clamp=fit.rate_clamp,
                        rate_xatol=fit.rate_xatol)
        state, mu = warm if warm is not None else (None, None)
        return (evaluator or evaluate_root_candidate)(baseline, c, cfg, keep_state=keep_internal,
                                                      catch_errors=catch_errors, clock_rate=mu, initial_state=state)

    results = []
    for key, d in merged.items():
        cand = d.candidate
        info = d.regression_info.get(primary, {})
        res = CandidateResult(candidate=cand, candidate_id=baseline.candidate_id(cand), source=d.source,
                              regression_rank=info.get('regression_rank'),
                              regression_chisq=info.get('regression_chisq'), source_ranks=dict(d.source_ranks))
        for c in dict.fromkeys(d.inits):
            res.runs.append(evaluate(c, fit.initial_mode))
        if key in warm_starts:
            res.runs.append(evaluate(cand, fit.initial_mode, warm=warm_starts[key]))
        results.append(res)

    results.sort(key=lambda r: -r.score)
    if rate_rounds and results and np.isfinite(results[0].score):
        top = results[0].score
        ok = [r for r in results if np.isfinite(r.score)]
        chosen = [r for i, r in enumerate(ok) if i < rate_fit_min or top - r.score <= rate_fit_window][:rate_fit_max]
        extra = {candidate_key(baseline.canonical(c)) for c in rate_fit_extra}
        chosen += [r for r in ok if r not in chosen and (r.regression_rank == 1 or candidate_key(r.candidate) in extra)]
        for r in chosen:
            b = r.best_run  # rate fit from the best shared-rate initialisation
            r.runs.append(evaluate(b.candidate, b.initial_mode, rate_rounds))
            r.rate_fitted = True
        results.sort(key=lambda r: -r.score)
    if sensitivity_window is not None and results and np.isfinite(results[0].score):
        top = results[0].score
        for r in results:
            if np.isfinite(r.score) and top - r.score <= sensitivity_window:
                tried = [x.initial_fraction for x in r.runs if x.initial_fraction is not None]
                for alt, mode in _alternatives(r.candidate, tried, sensitivity_fractions):
                    r.runs.append(evaluate(alt, mode, rate_rounds if r.rate_fitted else 0))
        results.sort(key=lambda r: -r.score)

    if keep_internal:
        for r in results:
            b = r.best_run
            for x in r.runs:
                if x is not b:
                    x.state = None
    merge_duplicate_roots(baseline, results, collapse_tol)
    refit = None
    if final_rate_refit and results and np.isfinite(results[0].score):
        rargs = (evaluate, fit.rate_clamp, final_rate_xatol, final_rate_evals, final_rate_iterations, catch_errors)
        refit = _final_rate_refit_one(results[0], *rargs)
        refit['headline'] = 'shared'  # diagnostic only: the ML rate is biased low on flat profiles (report 49b)
        if final_rate_refit_window is not None:
            top = results[0].score
            close = [refit] + [_final_rate_refit_one(r, *rargs) for r in results[1:]
                               if r.duplicate_of is None and np.isfinite(r.score)
                               and top - r.score <= final_rate_refit_window]
            order = sorted(close, key=lambda c: -(c['refit_logL'] if c['refit_logL'] is not None else -np.inf))
            refit['close'] = [dict(c, refit_rank=order.index(c) + 1) for c in close]
            refit['close_refit_best'] = order[0]['candidate_id']
            refit['close_order_changed'] = [c['candidate_id'] for c in order] != [c['candidate_id'] for c in close]
    if not keep_states and keep_internal:
        for r in results:
            for x in r.runs:
                x.state = None
        if refit is not None:
            for c in [refit] + refit.get('close', []):
                c['state'] = None
    if results:
        top = results[0]
        for r in results:
            r.delta_logL = r.score - top.score if np.isfinite(r.score) else None
            r.resolved_from_best = None if r.duplicate_of else True if r is top else bool(
                np.isfinite(r.score) and -r.delta_logL > max(min_margin, top.uncertainty + r.uncertainty))
    config = {
        'branch_length_mode': 'marginal', 'Tc': None, 'resolve_polytomies': False, 'prune_short': False,
        'infer_gtr': False, 'time_marginal': False, 'clock_rate': baseline.clock_rate,
        'iterations': fit.iterations, 'initial_mode': fit.initial_mode, 'rng_seed': baseline.rng_seed,
        'regression_k': regression_k, 'candidate_sources': [vars(x).copy() for x in sources],
        'ranked_sources': {n: len(c) for n, c in (ranked_sources or {}).items()},
        'sensitivity_window': sensitivity_window, 'sensitivity_fractions': list(sensitivity_fractions),
        'min_margin': min_margin, 'rate_rounds': rate_rounds, 'rate_fit_window': rate_fit_window,
        'rate_fit_min': rate_fit_min, 'rate_fit_max': rate_fit_max, 'rate_span': fit.rate_span,
        'rate_clamp': fit.rate_clamp, 'warm_starts': len(warm_starts), 'collapse_tol': collapse_tol,
    }
    if final_rate_refit:
        config.update({'final_rate_refit': True, 'final_rate_evals': final_rate_evals,
                       'final_rate_xatol': final_rate_xatol, 'final_rate_iterations': final_rate_iterations,
                       'final_rate_refit_window': final_rate_refit_window})
    if getattr(baseline, 'shared_rate_info', None) is not None:
        config['shared_rate_info'] = baseline.shared_rate_info
    lsdv_out = None
    if final_lsdv:
        lkw = dict(final_lsdv) if isinstance(final_lsdv, dict) else {}
        from .ls_dated_rate import check_lsdv_options
        check_lsdv_options(lkw)
        config['final_lsdv'] = lkw
        if results:
            lsdv_out = _final_lsdv(baseline, results[0], lkw)
    return RootSearchResult(results=results, clock_rate=baseline.clock_rate, min_margin=min_margin, config=config,
                            final_rate_refit=refit, final_lsdv=lsdv_out)


_LSDV_EXTRA = ('rate_se', 'se_rel', 'rate_ci', 'se_inflation', 'root_date_se', 'outliers', 'outlier_mode',
               'outlier_passes', 'resolve', 'internal_b', 'first_pass', 'n_nodes', 'n_nodes_unresolved', 'xo_rates',
               'first_pass_rate', 'resample')
_LSDV_BASE_KEYS = {'tip_slack', 'rounds', 'collapse'}


def _lsdv_extra(x, kwargs):
    """The SE and opt-in fields of an lsdv result (ls_dated_rate.lsdv_flat) reported alongside its rate. Only when an
    option beyond report 93's (tip_slack, rounds, collapse) was passed, so report 93's outputs stay byte-identical;
    pass e.g. ``se_inflation=2`` to get the SE alone."""
    if not set(kwargs) - _LSDV_BASE_KEYS:
        return {}
    return {k: x[k] for k in _LSDV_EXTRA if k in x}


def _final_lsdv(baseline, r, kwargs):
    """lsdv rate and root date at a result's fitted root (its candidate if the fit failed)."""
    cand = getattr(r, 'root_candidate', None) or r.candidate
    out = {'candidate_id': r.candidate_id, 'root_id': baseline.candidate_id(cand),
           'fraction': getattr(cand, 'fraction', None), 'rate': None, 'root_date': None, 'plain_rate': None,
           'intercept': None, 'chisq': None, 'error': None}
    try:
        x = baseline.lsdv(cand, **kwargs)
        out.update(rate=x['rate'], root_date=x['root_date'] if x['rate'] > 0 else None,
                   plain_rate=x['plain_rate'], intercept=x['intercept'], chisq=x['chisq'])
        out.update(_lsdv_extra(x, kwargs))
        if not x['rate'] > 0:
            out['error'] = 'lsdv rate not positive'
    except (ValueError, FloatingPointError, ZeroDivisionError) as e:
        out['error'] = str(e)
    return out


# --------------------------------------------------------------------------
# plots (matplotlib imported lazily)
# --------------------------------------------------------------------------
def _edge_split(baseline, a, b):
    return baseline.split_of[a] if baseline.toward_ref.get(a) == b else baseline.split_of[b]


def _placement_xy(baseline, tree, coords, cand):
    """Plot position of a placement in a tree drawn by plot_root_candidates."""
    cand = baseline.canonical(cand)
    if isinstance(cand, NodeRoot):
        return coords[baseline.node_of[cand.split]]
    for c in tree.find_clades():
        for ch in c.clades:
            pa = c.name
            if pa == EDGE_ROOT_NAME:  # the root splits one unrooted edge into two halves
                pa = [x.name for x in c.clades if x is not ch][0]
            if _edge_split(baseline, ch.name, pa) == cand.split:
                (x0, _), (x1, y1) = coords[c.name], coords[ch.name]
                return (0.5 * (x0 + x1), y1)
    raise ValueError('placement not found')


def plot_root_candidates(result, baseline, true_candidate=None, regression_candidate=None, ax=None,
                         root_at=None, vmax=10.0, cmap='viridis', label_leaves=True):
    """Draw the tree with every evaluated root placement coloured by its likelihood
    deficit ΔlogL from the best (edges as coloured branches and midpoint squares,
    node roots as coloured dots; placements not evaluated in grey).

    The tree is drawn with the baseline branch lengths, rooted at ``root_at``
    (default: the likelihood-best placement). Markers: likelihood best (star),
    regression best (``regression_candidate`` or the result's regression rank 1;
    triangle) and ``true_candidate`` if given (open circle). Returns the axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(4, 0.16 * len(baseline.leaves))))
    best = result.best.candidate
    if regression_candidate is None and result.regression_best is not None:
        regression_candidate = result.regression_best.candidate
    tree = baseline.rooted_phylo(baseline.canonical(root_at or best))
    delta = {candidate_key(r.candidate): r.delta_logL for r in result.results if r.delta_logL is not None}
    norm = colors.Normalize(vmin=-vmax, vmax=0.0)
    cmap = plt.get_cmap(cmap)

    def col(key):
        d = delta.get(key)
        return cmap(norm(max(-vmax, d))) if d is not None else (0.75, 0.75, 0.75, 1.0)

    coords, y = {}, [0]

    def layout(c, x):
        if c.is_terminal():
            coords[c.name] = (x, y[0])
            y[0] += 1
        else:
            ys = [layout(ch, x + (ch.branch_length or 0.0)) for ch in c.clades]
            coords[c.name] = (x, 0.5 * (min(ys) + max(ys)))
        return coords[c.name][1]
    layout(tree.root, 0.0)

    for c in tree.find_clades():
        x0, y0 = coords[c.name]
        if c.clades:
            ys = [coords[ch.name][1] for ch in c.clades]
            ax.plot([x0, x0], [min(ys), max(ys)], color='0.5', lw=0.8, zorder=1)
        for ch in c.clades:
            pa = c.name if c.name != EDGE_ROOT_NAME else [x.name for x in c.clades if x is not ch][0]
            key = ('edge', _edge_split(baseline, ch.name, pa))
            x1, y1 = coords[ch.name]
            ax.plot([x0, x1], [y1, y1], color=col(key), lw=2.5 if key in delta else 0.8, zorder=2,
                    solid_capstyle='butt')
            if key in delta:
                ax.scatter([0.5 * (x0 + x1)], [y1], marker='s', s=14, color=col(key), zorder=3, linewidths=0)
        if c.clades and c.name != EDGE_ROOT_NAME:
            key = ('node', baseline.split_of[c.name]) if c.name in baseline.split_of else None
            if key in delta:
                ax.scatter([x0], [y0], s=30, color=col(key), edgecolors='k', linewidths=0.4, zorder=4)
        if label_leaves and c.is_terminal():
            ax.text(x0, y0, ' ' + c.name, va='center', fontsize=6)

    marks = [(best, '*', 'likelihood best', 160, 'tab:red'),
             (regression_candidate, '^', 'regression best', 70, 'tab:orange'),
             (true_candidate, 'o', 'true root', 120, 'k')]
    for cand, m, lab, s, c in marks:
        if cand is None:
            continue
        x, yy = _placement_xy(baseline, tree, coords, cand)
        if m == 'o':
            ax.scatter([x], [yy], marker=m, s=s, facecolors='none', edgecolors=c, linewidths=1.5, zorder=6,
                       label=lab)
        else:
            ax.scatter([x], [yy], marker=m, s=s, color=c, edgecolors='k', linewidths=0.5, zorder=5, label=lab)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    ax.figure.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label='ΔlogL from best (clipped at -%g)' % vmax)
    ax.set_yticks([])
    ax.set_ylim(y[0], -1)
    ax.set_xlabel('substitutions per site (baseline branch lengths)')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.legend(loc='lower left', fontsize=7, frameon=False)
    return ax


def plot_root_to_tip(baseline, cand, ax=None, label=None):
    """Root-to-tip distance (baseline branch lengths) against tip date with the
    tree rooted at ``cand``, with the least-squares line; the regression view of
    one root placement. Returns the axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    tree = baseline.rooted_phylo(baseline.canonical(cand))
    depths = tree.depths()
    leaves = [c for c in tree.get_terminals() if c.name in baseline.dates]
    x = np.array([baseline.dates[c.name] for c in leaves])
    y = np.array([depths[c] for c in leaves])
    slope, icpt = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2 if len(x) > 2 else np.nan
    ax.scatter(x, y, s=12, color='tab:blue')
    xx = np.array([min(x.min(), -icpt / slope if slope > 0 else x.min()), x.max()])
    ax.plot(xx, slope * xx + icpt, color='k', lw=1)
    ax.set_xlabel('date')
    ax.set_ylabel('root-to-tip distance')
    ax.set_title('%s\nrate %.2e, root %.1f, R² %.2f' % (label or baseline.candidate_id(baseline.canonical(cand)),
                                                         slope, -icpt / slope if slope else np.nan, r2), fontsize=8)
    return ax
