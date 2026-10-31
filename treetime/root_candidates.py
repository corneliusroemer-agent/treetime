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
        Fixed positive rate shared by all candidates. ``None`` takes the slope
        of the best free-slope root-to-tip regression on the baseline.
    optimize_baseline : bool
        Re-estimate branch lengths once with TreeTime's marginal optimiser on
        the unrooted tree, as ``TreeTime.run`` does before rerooting. These
        lengths drive the regression and start every candidate.
    rng_seed : int
        Seed of every per-candidate TreeTime, so the RNG cannot carry state
        from one candidate to the next.
    tt_kwargs : dict, optional
        Extra TreeTime constructor arguments (e.g. ``precision``).
    """

    def __init__(self, tree, aln, dates, gtr='JC69', clock_rate=None, optimize_baseline=True,
                 rng_seed=1234, tt_kwargs=None):
        tree = Phylo.read(tree, 'newick') if isinstance(tree, str) else copy.deepcopy(tree)
        self.aln = aln
        self.dates = dict(dates)
        self.gtr = gtr if isinstance(gtr, GTR) else GTR.standard(gtr)
        self.rng_seed = rng_seed
        self.tt_kwargs = dict(tt_kwargs or {})
        self._from_phylo(tree)
        if optimize_baseline:
            tt = self.make_treetime(self.rooted_phylo(None))
            tt.optimize_tree(infer_gtr=False, max_iter=1, method_anc='probabilistic', **_seq_kwargs())
            self._from_phylo(tt.tree, keep_names=True)
        if clock_rate is None:
            best = self.regression().find_best_root(force_positive=True)
            if best is None or not best['slope'] > 0:
                raise ValueError('UnrootedBaseline: no positive clock rate from regression; pass clock_rate')
            clock_rate = float(best['slope'])
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

    def make_treetime(self, tree):
        """New TreeTime object on ``tree``: same alignment, dates, model and seed."""
        kw = dict(verbose=0, branch_length_mode='marginal', use_covariation=False)
        kw.update(self.tt_kwargs)
        return TreeTime(tree=tree, aln=self.aln, dates=self.dates, gtr=copy.deepcopy(self.gtr),
                        rng_seed=self.rng_seed, **kw)

    def regression(self, covariation=False, tip_slack=None):
        """TreeRegression on the baseline branch lengths. ``tip_slack`` sets the
        extra tip variance of the covariation model (TreeTime's default if None)."""
        tt = self.make_treetime(self.rooted_phylo(None))
        if tip_slack is not None:
            tt.tip_slack = tip_slack
        treg = tt.setup_TreeRegression(covariation=covariation)
        treg._tt = tt  # keep the node owner alive
        return treg

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
    newick string, at exactly its branch lengths, with a new TreeAnc object."""
    from .treeanc import TreeAnc
    tree = Phylo.read(io.StringIO(newick), 'newick')
    ta = TreeAnc(tree=tree, aln=baseline.aln, gtr=copy.deepcopy(baseline.gtr), verbose=0,
                 rng_seed=baseline.rng_seed)
    return fresh_marginal_score(ta)


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
    ndiff: list = field(default_factory=list)  # TreeTime's ndiff per iteration (None at index 0)
    root_state_changes: list = field(default_factory=list)  # root sequence positions changed
    iterations_performed: int = 0
    treetime_stop_iteration: Optional[int] = None  # where run() would have stopped on ndiff == 0
    best_index: Optional[int] = None
    best_score: float = -np.inf
    trajectory_class: str = ''
    tail_fluctuation: float = 0.0  # max - min of the second half of the trajectory
    root: dict = field(default_factory=dict)  # root summary at the best step
    fitted_newick: str = ''
    runtime: float = 0.0
    error: Optional[str] = None
    warnings: list = field(default_factory=list)
    state: object = None  # deep copy of the TreeTime at the best step


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
    }


def evaluate_root_candidate(baseline, cand, fit=None, keep_state=True, catch_errors=False):
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

    Errors propagate unless ``catch_errors`` is set; a failed fit then has
    score -inf and the error in ``error``.
    """
    fit = fit or FitConfig()
    cand = baseline.canonical(cand)
    run = FitRun(candidate=cand, candidate_id=baseline.candidate_id(cand),
                 initial_fraction=getattr(cand, 'fraction', None), initial_mode=fit.initial_mode)
    t0 = time.time()
    try:
        tt = baseline.make_treetime(baseline.rooted_phylo(cand))
        kids = tt.tree.root.clades
        if len(kids) == 2 and kids[0].branch_length + kids[1].branch_length < ttconf.MIN_BRANCH_LENGTH * tt.one_mutation:
            for c in kids:
                c.branch_length = c.mutation_length = 0.5 * tt.one_mutation
            run.warnings.append('zero-length root edge: initial length set to one mutation')
        tt._set_branch_length_mode('marginal')
        seq_kwargs = _seq_kwargs()
        tt_kwargs = {'clock_rate': baseline.clock_rate, 'time_marginal': False}
        tt.optimize_tree(infer_gtr=False, max_iter=1, method_anc='probabilistic', **_seq_kwargs(fit.initial_mode))
        tt.optimize_tree(max_iter=1, method_anc='probabilistic', **seq_kwargs)
        tt.make_time_tree(**tt_kwargs)
        _record(baseline, run, tt, None, None, keep_state)
        for it in range(1, fit.iterations + 1):
            prev_root = np.array(tt.tree.root._cseq, copy=True)
            ndiff = tt.infer_ancestral_sequences('ml', marginal=True, **seq_kwargs)
            root_changes = int(np.sum(prev_root != tt.tree.root._cseq))
            tt.make_time_tree(**tt_kwargs)
            run.iterations_performed = it
            if ndiff == 0 and run.treetime_stop_iteration is None:
                run.treetime_stop_iteration = it
            _record(baseline, run, tt, int(ndiff), root_changes, keep_state)
        tail = run.scores[len(run.scores) // 2:]
        run.tail_fluctuation = float(max(tail) - min(tail))
        run.trajectory_class = classify_trajectory(run.scores)
    except FIT_ERRORS as exc:
        if not catch_errors:
            raise
        run.error = '%s: %s' % (type(exc).__name__, exc)
        run.best_score = -np.inf
    run.runtime = time.time() - t0
    return run


def _record(baseline, run, tt, ndiff, root_changes, keep_state):
    run.cached_LH.append(float(getattr(tt.tree, 'sequence_marginal_LH', np.nan)))  # before the refresh
    run.positional_LH.append(float(tt.tree.positional_LH))
    score = fresh_marginal_score(tt)
    run.scores.append(score)
    run.ndiff.append(ndiff)
    run.root_state_changes.append(root_changes)
    if score > run.best_score:
        run.best_score = score
        run.best_index = len(run.scores) - 1
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
def regression_candidates(baseline, k=5, covariation=False, tip_slack=None, slope='fixed'):
    """Best ``k`` distinct placements of the root-to-tip regression as canonical
    candidates, best first, each as ``(candidate, info)`` with the regression
    rank, objective and split. ``slope='fixed'`` uses the baseline clock rate;
    ``None`` lets the regression fit its own slope."""
    treg = baseline.regression(covariation=covariation, tip_slack=tip_slack)
    s = baseline.clock_rate if slope == 'fixed' else slope
    out, seen = [], set()
    for p in treg.rank_roots(force_positive=True, slope=s):
        n = p['node'].name
        if n == baseline.ref_leaf:
            cand = EdgeRoot(baseline.split_of[baseline.center], 1.0 - p['split'])
        else:
            cand = EdgeRoot(baseline.split_of[n], p['split'])
        cand = baseline.canonical(cand)
        if candidate_key(cand) in seen:
            continue
        seen.add(candidate_key(cand))
        out.append((cand, {'regression_rank': len(out) + 1, 'regression_chisq': float(p['edge_chisq']),
                           'regression_split': float(p['split'])}))
        if k is not None and len(out) >= k:
            break
    return out


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
    runs: list = field(default_factory=list)
    delta_logL: Optional[float] = None
    resolved_from_best: Optional[bool] = None

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
        """max - min of the best scores reached from different initialisations."""
        sc = [r.best_score for r in self.runs if r.error is None]
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
            'kind': self.candidate.kind,
            'source': self.source,
            'regression_rank': self.regression_rank,
            'regression_chisq': self.regression_chisq,
            'logL': self.score if ok else None,
            'delta_logL': self.delta_logL,
            'positional_LH': b.positional_LH[b.best_index] if ok else None,
            'cached_LH': b.cached_LH[b.best_index] if ok else None,
            'clock_rate': clock_rate,
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
            'runtime': sum(r.runtime for r in self.runs),
            'error': '; '.join(r.error for r in self.runs if r.error) or None,
            'warnings': '; '.join(sorted({w for r in self.runs for w in r.warnings})) or None,
            'scores': b.scores if b else None,
            'fitted_newick': b.fitted_newick if ok else None,
        }


TSV_COLS = ['candidate_id', 'kind', 'source', 'regression_rank', 'regression_chisq', 'logL', 'delta_logL',
            'positional_LH', 'cached_LH', 'clock_rate', 'initial_fraction', 'initial_mode', 'fitted_fraction',
            'root_date', 'root_degree', 'root_child_lengths', 'iterations_performed', 'best_iteration',
            'treetime_stop_iteration', 'trajectory_class', 'tail_fluctuation', 'n_initialisations',
            'init_spread', 'uncertainty', 'resolved_from_best', 'runtime', 'error', 'warnings', 'scores']


@dataclass
class RootSearchResult:
    results: list  # CandidateResult, best first
    clock_rate: float
    min_margin: float
    config: dict

    @property
    def best(self):
        return self.results[0]

    @property
    def margin(self):
        """logL gap between the best and the second candidate."""
        if len(self.results) < 2 or not np.isfinite(self.results[1].score):
            return None
        return self.results[0].score - self.results[1].score

    @property
    def unresolved_with_best(self):
        return [r.candidate_id for r in self.results[1:] if not r.resolved_from_best]

    @property
    def best_is_resolved(self):
        return not self.unresolved_with_best

    @property
    def regression_best(self):
        return next((r for r in self.results if r.regression_rank == 1), None)

    def best_treetime(self):
        return restore_best_fitted_state(self.best)

    def summary(self):
        reg = self.regression_best
        return {
            'best': self.best.candidate_id,
            'best_logL': self.best.score,
            'margin': self.margin,
            'best_is_resolved': self.best_is_resolved,
            'unresolved_with_best': self.unresolved_with_best,
            'regression_best': reg.candidate_id if reg else None,
            'regression_best_delta_logL': reg.delta_logL if reg else None,
            'clock_rate': self.clock_rate,
            'min_margin': self.min_margin,
            'n_candidates': len(self.results),
            'n_failed': sum(not np.isfinite(r.score) for r in self.results),
        }

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
                '' if i == 0 else _fmt(r['resolved_from_best'])))
        s = self.summary()
        out.append('best %s, margin %s, %s; regression best %s (dlogL %s)' % (
            s['best'], _fmt(s['margin'], 3), 'resolved' if s['best_is_resolved'] else
            'unresolved with ' + ', '.join(s['unresolved_with_best']), s['regression_best'],
            _fmt(s['regression_best_delta_logL'], 3)))
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


def rank_root_candidates(baseline, explicit=(), regression_k=5, fit=None, sensitivity_window=3.0,
                         sensitivity_fractions=(0.1, 0.5, 0.9), min_margin=1.0, covariation=False,
                         tip_slack=None, keep_states=True, catch_errors=False):
    """Evaluate explicit and regression-ranked candidates and rank them by fitted
    full marginal log-likelihood.

    Several placements on one edge (explicit and/or regression) are one
    candidate with several initialisations, the regression's fraction first.
    The regression candidates are the best ``regression_k`` placements of
    :func:`regression_candidates` (``covariation``/``tip_slack`` choose its
    model); its score is kept for diagnostics only.

    Every candidate within ``sensitivity_window`` log units of the best is
    re-fitted from alternative initialisations (:func:`_alternatives`). A
    candidate's score is its best over initialisations, and its optimiser
    uncertainty ``u`` the larger of the spread across initialisations and the
    tail fluctuation of its best trajectory. A candidate is *resolved* from the
    best when ΔlogL > max(min_margin, u_best + u_candidate).

    ``min_margin`` (default 1 log unit) is calibrated on simulations, where
    optimiser spread alone left most wrong answers looking resolved; it only
    affects the verdict, never the ranking. ``min_margin=0`` disables it.

    ``keep_states=False`` keeps no TreeTime copies; otherwise only the best run
    of each candidate keeps its copy.
    """
    fit = fit or FitConfig()
    merged = {}

    def add(c, source, info=None):
        c = baseline.canonical(c)
        d = merged.setdefault(candidate_key(c), {'sources': [], 'inits': [], 'info': {}})
        if source not in d['sources']:
            d['sources'].append(source)
        if info:
            d['info'] = info
            d['inits'].insert(0, c)
        else:
            d['inits'].append(c)

    for c in explicit:
        add(c, 'explicit')
    if regression_k:
        for c, info in regression_candidates(baseline, k=regression_k, covariation=covariation,
                                             tip_slack=tip_slack):
            add(c, 'regression', info)

    def evaluate(c, mode):
        return evaluate_root_candidate(baseline, c, FitConfig(fit.iterations, mode), keep_state=keep_states,
                                       catch_errors=catch_errors)

    results = []
    for key, d in merged.items():
        cand = d['inits'][0]
        res = CandidateResult(candidate=cand, candidate_id=baseline.candidate_id(cand),
                              source='+'.join(d['sources']), regression_rank=d['info'].get('regression_rank'),
                              regression_chisq=d['info'].get('regression_chisq'))
        for c in dict.fromkeys(d['inits']):
            res.runs.append(evaluate(c, fit.initial_mode))
        results.append(res)

    results.sort(key=lambda r: -r.score)
    if sensitivity_window is not None and results and np.isfinite(results[0].score):
        top = results[0].score
        for r in results:
            if np.isfinite(r.score) and top - r.score <= sensitivity_window:
                tried = [x.initial_fraction for x in r.runs if x.initial_fraction is not None]
                for alt, mode in _alternatives(r.candidate, tried, sensitivity_fractions):
                    r.runs.append(evaluate(alt, mode))
        results.sort(key=lambda r: -r.score)

    if keep_states:
        for r in results:
            b = r.best_run
            for x in r.runs:
                if x is not b:
                    x.state = None
    if results:
        top = results[0]
        for r in results:
            r.delta_logL = r.score - top.score if np.isfinite(r.score) else None
            r.resolved_from_best = True if r is top else bool(
                np.isfinite(r.score) and -r.delta_logL > max(min_margin, top.uncertainty + r.uncertainty))
    config = {
        'branch_length_mode': 'marginal', 'Tc': None, 'resolve_polytomies': False, 'prune_short': False,
        'infer_gtr': False, 'time_marginal': False, 'clock_rate': baseline.clock_rate,
        'iterations': fit.iterations, 'initial_mode': fit.initial_mode, 'rng_seed': baseline.rng_seed,
        'regression_k': regression_k, 'covariation': covariation, 'tip_slack': tip_slack,
        'sensitivity_window': sensitivity_window, 'sensitivity_fractions': list(sensitivity_fractions),
        'min_margin': min_margin,
    }
    return RootSearchResult(results=results, clock_rate=baseline.clock_rate, min_margin=min_margin, config=config)


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
