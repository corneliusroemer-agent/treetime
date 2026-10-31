"""
Root selection by the fitted marginal sequence likelihood under a strict clock.

Each candidate root placement on a fixed unrooted topology is fitted
independently with TreeTime's own alternation of marginal ancestral inference
and time-tree inference, at a fixed GTR model and a fixed clock rate, and is
scored by the full marginal sequence log-likelihood recomputed at the fitted
(clock-constrained) branch lengths.

Layout:

* candidate identity -- :class:`NodeRoot`, :class:`EdgeRoot` and the split
  helpers. Nodes and edges are identified by leaf bipartitions, which are
  stable under rerooting; node names are not.
* candidate generation -- :func:`regression_candidates` ranks every edge by
  the root-to-tip regression (:py:meth:`TreeRegression.rank_roots`).
* candidate evaluation -- :class:`RootCandidateEvaluator`, which builds a fresh
  ``TreeTime`` for every candidate from the same baseline, runs a fixed number
  of fitting iterations without the ``ndiff`` early stop, scores after every
  completed fit and keeps a deep copy of the best-scoring fitted ``TreeTime``.
"""

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from Bio import Phylo

from . import TreeTime, TreeTimeError

# root placements closer than this fraction to an edge endpoint are that endpoint
ENDPOINT_EPS = 1e-5


# ---------------------------------------------------------------------------
# tree helpers
# ---------------------------------------------------------------------------


def leaf_names(tree):
    return frozenset(n.name for n in tree.get_terminals())


def _leafsets(tree):
    """map id(clade) -> frozenset of leaf names below it"""
    sets = {}
    for n in tree.find_clades(order='postorder'):
        if n.is_terminal():
            sets[id(n)] = frozenset([n.name])
        else:
            sets[id(n)] = frozenset().union(*(sets[id(c)] for c in n.clades))
    return sets


def canonical_split(side, all_leaves):
    """The side of a bipartition that does not contain the reference leaf
    (the alphabetically first leaf name)."""
    ref = min(all_leaves)
    side = frozenset(side)
    return all_leaves - side if ref in side else side


def unrooted_splits(tree):
    """Set of canonical leaf bipartitions of all edges of the tree, read as
    unrooted. The two edges below a bifurcating root are one unrooted edge."""
    all_leaves = leaf_names(tree)
    sets = _leafsets(tree)
    out = set()
    for n in tree.find_clades():
        if n is tree.root:
            continue
        s = canonical_split(sets[id(n)], all_leaves)
        if 0 < len(s) < len(all_leaves):
            out.add(s)
    return out


def unrooted_degrees(tree):
    """Sorted list of the unrooted degrees of the internal nodes, ignoring a
    bifurcating root (which does not exist in the unrooted tree)."""
    degs = []
    for n in tree.get_nonterminals():
        d = len(n.clades) + (0 if n is tree.root else 1)
        if n is tree.root and d == 2:
            continue
        degs.append(d)
    return sorted(degs)


def node_key(clade, tree, sets=None):
    """Canonical identity of a node: frozenset of the canonical splits of its
    incident edges."""
    all_leaves = leaf_names(tree)
    sets = sets or _leafsets(tree)
    parts = [sets[id(c)] for c in clade.clades]
    if clade is not tree.root:
        parts.append(all_leaves - sets[id(clade)])
    return frozenset(canonical_split(p, all_leaves) for p in parts)


def deroot(tree):
    """Remove a bifurcating (or undated single-child) root in place, so that
    every unrooted edge is the edge above exactly one non-root node. The
    removed root's two edges are merged into one edge with their summed
    length. Returns the tree."""
    while len(tree.root.clades) == 1:
        tree.root = tree.root.clades[0]
    if len(tree.root.clades) == 2:
        a, b = tree.root.clades
        keep, other = (a, b) if not a.is_terminal() else (b, a)
        if keep.is_terminal():
            raise ValueError('deroot: tree with two leaves')
        other.branch_length = (other.branch_length or 0.0) + (keep.branch_length or 0.0)
        keep.clades.append(other)
        keep.branch_length = 0.0
        tree.root = keep
    return tree


def name_internal_nodes(tree, prefix='BASE_'):
    """Give unnamed internal nodes deterministic labels (preorder). Labels are
    for display only; identity is by leaf bipartition."""
    names = {n.name for n in tree.find_clades() if n.name}
    i = 0
    for n in tree.get_nonterminals(order='preorder'):
        if not n.name:
            while f'{prefix}{i:04d}' in names:
                i += 1
            n.name = f'{prefix}{i:04d}'
            names.add(n.name)
            i += 1
    return tree


def _short_hash(obj):
    if isinstance(obj, frozenset):
        rep = '|'.join(sorted(','.join(sorted(x)) if isinstance(x, frozenset) else x for x in obj))
    else:
        rep = str(obj)
    return hashlib.sha1(rep.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# candidate types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeRoot:
    """Root exactly at an internal node, keeping its degree (a polytomy stays a
    polytomy at the root). `key` is the frozenset of canonical splits of the
    node's incident edges; use :meth:`at` to build one from a baseline tree."""

    key: frozenset
    label: str = ''

    @property
    def kind(self):
        return 'node'

    @property
    def id(self):
        return 'node:' + _short_hash(self.key)

    @classmethod
    def at(cls, tree, node):
        """`node`: a clade of `tree`, an internal node name, or a list of leaf
        names whose MRCA in `tree` is the node."""
        clade = _resolve_clade(tree, node)
        if clade.is_terminal():
            # a leaf cannot be the root: rooting at it is the leaf edge at its leaf end
            return EdgeRoot.above(tree, clade, fraction_from_node=0.0)
        return cls(node_key(clade, tree), label=clade.name or '')


@dataclass(frozen=True)
class EdgeRoot:
    """Root inside an edge. The edge is the canonical leaf split `split`;
    `fraction` is the initial position of the root along the edge, measured
    from the endpoint adjacent to the leaves in `split` (0) to the other
    endpoint (1). The fraction initialises the fit only: the fitted dates
    decide where on the edge the root ends up, so candidates on the same edge
    with different fractions are the same candidate."""

    split: frozenset
    fraction: float = 0.5
    label: str = ''

    @property
    def kind(self):
        return 'edge'

    @property
    def id(self):
        return 'edge:' + _short_hash(self.split)

    @classmethod
    def above(cls, tree, node, fraction_from_node=0.5):
        """Edge above `node` (clade, name or leaf-name list) in `tree`, with
        the initial root at `fraction_from_node` of the edge length from
        `node`. An internal endpoint (fraction 0 or 1) gives a NodeRoot."""
        clade = _resolve_clade(tree, node)
        if clade is tree.root:
            raise ValueError('EdgeRoot.above: the root has no edge above it')
        all_leaves = leaf_names(tree)
        sets = _leafsets(tree)
        below = sets[id(clade)]
        split = canonical_split(below, all_leaves)
        frac = float(fraction_from_node) if split == below else 1.0 - float(fraction_from_node)
        parent = _parent_map(tree)[id(clade)]
        if parent is tree.root and len(tree.root.clades) == 2:
            raise ValueError('EdgeRoot.above: deroot the tree first (bifurcating root)')
        label = (clade.name or '') + '^'
        return cls(split, frac, label=label).snapped(tree)

    def snapped(self, tree):
        """Map an endpoint placement on an internal node to a NodeRoot."""
        if ENDPOINT_EPS < self.fraction < 1 - ENDPOINT_EPS:
            return self
        clade, parent, child_is_split_side = _find_edge(tree, self.split)
        near = clade if child_is_split_side else parent
        far = parent if child_is_split_side else clade
        end = near if self.fraction <= ENDPOINT_EPS else far
        if end.is_terminal():
            frac = 0.0 if end is near else 1.0
            return EdgeRoot(self.split, frac, label=self.label)
        return NodeRoot(node_key(end, tree), label=end.name or '')


def _parent_map(tree):
    return {id(c): n for n in tree.find_clades() for c in n.clades}


def _resolve_clade(tree, node):
    if isinstance(node, Phylo.BaseTree.Clade):
        return node
    if isinstance(node, (list, tuple, set, frozenset)):
        return tree.common_ancestor([n for n in tree.get_terminals() if n.name in set(node)])
    hits = [n for n in tree.find_clades() if n.name == node]
    if len(hits) != 1:
        raise ValueError(f'node {node!r} not found uniquely')
    return hits[0]


def _find_edge(tree, split):
    """Return (child, parent, child_is_split_side) for the edge with canonical
    split `split` in a tree without a bifurcating root."""
    all_leaves = leaf_names(tree)
    sets = _leafsets(tree)
    parents = _parent_map(tree)
    for n in tree.find_clades():
        if n is tree.root:
            continue
        below = sets[id(n)]
        if below == split or all_leaves - below == split:
            return n, parents[id(n)], below == split
    raise ValueError('edge not found in tree')


def _find_node(tree, key):
    sets = _leafsets(tree)
    for n in tree.get_nonterminals():
        if node_key(n, tree, sets) == key:
            return n
    raise ValueError('node not found in tree')


def candidate_key(c):
    """Deduplication key: one per node, one per edge (fractions are
    initialisations of the same edge candidate)."""
    return ('node', c.key) if isinstance(c, NodeRoot) else ('edge', c.split)


def dedupe_candidates(cands):
    seen, out = set(), []
    for c in cands:
        k = candidate_key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def apply_candidate_root(baseline_tree, candidate, fraction=None):
    """Return a deep copy of the (derooted) baseline Bio.Phylo tree rooted at
    `candidate`. `fraction` overrides the EdgeRoot initial fraction."""
    tree = copy.deepcopy(baseline_tree)
    if isinstance(candidate, NodeRoot):
        node = _find_node(tree, candidate.key)
        if node is not tree.root:
            tree.root_with_outgroup(node)  # internal node, no length: node becomes the root
    else:
        f = candidate.fraction if fraction is None else fraction
        clade, parent, child_is_split_side = _find_edge(tree, candidate.split)
        w = clade.branch_length or 0.0
        d_child = f * w if child_is_split_side else (1.0 - f) * w
        # insert a degree-2 node into the edge at distance d_child from `clade` and
        # root there. Not root_with_outgroup(clade, outgroup_branch_length=...):
        # when `clade` is a child of a multifurcating root, Biopython keeps the
        # full edge length on the other side as well.
        mid = Phylo.BaseTree.Clade(branch_length=w - d_child, clades=[clade])
        parent.clades[parent.clades.index(clade)] = mid
        clade.branch_length = d_child
        tree.root_with_outgroup(mid)  # internal node, no length: it becomes the root
    tree.root.branch_length = 0.0
    return tree


# ---------------------------------------------------------------------------
# configuration and results
# ---------------------------------------------------------------------------


def treetime_seq_kwargs(branch_length_mode='marginal'):
    """Sequence-inference arguments exactly as TreeTime._run builds them for
    the canonical configuration (prune_short=False)."""
    return {
        'marginal_sequences': True,
        'branch_length_mode': branch_length_mode,
        'sample_from_profile': 'root',
        'prune_short': False,
        'reconstruct_tip_states': False,
    }


def fresh_marginal_score(tt):
    """Full marginal sequence log-likelihood at the tree's current branch
    lengths. Does not draw from the RNG or reassign sequences. Overwrites the
    cached tree.sequence_LH / total_sequence_LH / sequence_marginal_LH and the
    root's marginal_profile, so read any cached value before calling this."""
    tt.postorder_traversal_marginal()
    tt.total_LH_and_root_sequence(assign_sequence=False)
    return float(tt.tree.total_sequence_LH)


def root_summary(tt):
    root = tt.tree.root
    kids = root.clades
    bl = [float(c.branch_length) for c in kids]
    ratio = bl[0] / (bl[0] + bl[1]) if len(kids) == 2 and (bl[0] + bl[1]) > 0 else None
    return {
        'root_degree': len(kids),
        'root_child_branch_lengths': bl,
        'root_child_mutation_lengths': [float(c.mutation_length) for c in kids],
        'root_child_leafsets': [sorted(n.name for n in c.get_terminals()) for c in kids],
        'split_ratio': ratio,
        'root_date': float(root.numdate),
    }


def classify_trajectory(scores, tol=1e-6):
    """'stable' if the last two scores agree within tol; 'cycling' if the last
    score repeats an earlier, non-adjacent one; otherwise 'drifting'."""
    if len(scores) < 2:
        return 'single'
    if abs(scores[-1] - scores[-2]) <= tol:
        return 'stable'
    for s in scores[:-2]:
        if abs(scores[-1] - s) <= tol:
            return 'cycling'
    return 'drifting'


@dataclass
class FitRun:
    """One fitting trajectory of one candidate from one initialisation."""

    init: dict
    scores: list = field(default_factory=list)  # fresh marginal logL after each fit (index 0 = initial round)
    cached_LH: list = field(default_factory=list)  # tree.sequence_marginal_LH before rescoring
    positional_LH: list = field(default_factory=list)  # time-tree (dating) objective
    ndiff: list = field(default_factory=list)  # TreeTime's ndiff per iteration (None for round 0)
    root_state_changes: list = field(default_factory=list)  # root sequence changes per iteration
    fits_performed: int = 0  # make_time_tree calls
    best_index: Optional[int] = None
    best_score: float = -np.inf
    trajectory: str = ''
    treetime_stop_iteration: Optional[int] = None  # where run() would have stopped on ndiff == 0
    runtime: float = 0.0
    error: Optional[str] = None
    warnings: list = field(default_factory=list)
    state: object = None  # deep copy of the TreeTime at the best score


@dataclass
class CandidateResult:
    candidate: object
    source: str
    regression_rank: Optional[int] = None
    regression_chisq: Optional[float] = None
    regression_split: Optional[float] = None
    runs: list = field(default_factory=list)
    best_run: Optional[FitRun] = None
    delta_vs_best: Optional[float] = None
    optimiser_spread: Optional[float] = None
    root: dict = field(default_factory=dict)

    @property
    def score(self):
        return self.best_run.best_score if self.best_run and self.best_run.error is None else -np.inf

    def row(self, clock_rate):
        b = self.best_run
        c = self.candidate
        return {
            'id': c.id,
            'type': c.kind,
            'label': c.label,
            'source': self.source,
            'regression_rank': self.regression_rank,
            'regression_chisq': self.regression_chisq,
            'initial_fraction': getattr(c, 'fraction', None),
            'logL': self.score if np.isfinite(self.score) else None,
            'delta_logL': self.delta_vs_best,
            'positional_LH': b.positional_LH[b.best_index] if b and b.best_index is not None else None,
            'cached_LH': b.cached_LH[b.best_index] if b and b.best_index is not None else None,
            'clock_rate': clock_rate,
            'best_iteration': b.best_index if b else None,
            'fits_performed': b.fits_performed if b else None,
            'trajectory': b.trajectory if b else None,
            'treetime_stop_iteration': b.treetime_stop_iteration if b else None,
            'scores': b.scores if b else None,
            'n_inits': len(self.runs),
            'optimiser_spread': self.optimiser_spread,
            'runtime': sum(r.runtime for r in self.runs),
            'error': '; '.join(r.error for r in self.runs if r.error) or None,
            'warnings': '; '.join(sorted({w for r in self.runs for w in r.warnings})) or None,
            **self.root,
        }


@dataclass
class RootSearchResult:
    results: list
    clock_rate: float
    verdict: str
    margin: Optional[float]
    config: dict

    @property
    def best(self):
        return self.results[0]

    def best_treetime(self):
        """The retained TreeTime at the best candidate's best-scoring fit."""
        return restore_best_fitted_state(self.best)

    def rows(self):
        return [r.row(self.clock_rate) for r in self.results]

    def to_json(self, path=None):
        payload = {
            'clock_rate': self.clock_rate,
            'verdict': self.verdict,
            'margin': self.margin,
            'config': self.config,
            'candidates': self.rows(),
        }
        s = json.dumps(payload, indent=1, default=_json_default)
        if path:
            with open(path, 'w') as fh:
                fh.write(s)
        return s

    def to_tsv(self, path=None):
        cols = [
            'id', 'type', 'label', 'source', 'regression_rank', 'regression_chisq', 'initial_fraction',
            'logL', 'delta_logL', 'positional_LH', 'cached_LH', 'clock_rate', 'root_date', 'root_degree',
            'split_ratio', 'root_child_branch_lengths', 'best_iteration', 'fits_performed', 'trajectory',
            'treetime_stop_iteration', 'n_inits', 'optimiser_spread', 'runtime', 'error', 'warnings', 'scores',
        ]
        lines = ['\t'.join(cols)]
        for r in self.rows():
            lines.append('\t'.join(_fmt(r.get(c)) for c in cols))
        s = '\n'.join(lines) + '\n'
        if path:
            with open(path, 'w') as fh:
                fh.write(s)
        return s

    def table(self):
        """Concise comparison table."""
        hdr = f'{"rank":>4} {"id":<14} {"label":<14} {"src":<4} {"reg":>4} {"logL":>12} {"dlogL":>8} {"spread":>7} {"root_date":>10} {"deg":>3} {"ratio":>6} {"it":>3} {"traj":<9}'
        out = [hdr]
        for i, r in enumerate(self.rows()):
            out.append(
                f'{i + 1:>4} {r["id"]:<14} {str(r["label"])[:14]:<14} {r["source"][:4]:<4} '
                f'{_fmt(r["regression_rank"]):>4} {_fmt(r["logL"], 3):>12} {_fmt(r["delta_logL"], 3):>8} '
                f'{_fmt(r["optimiser_spread"], 3):>7} {_fmt(r.get("root_date"), 3):>10} {_fmt(r.get("root_degree")):>3} '
                f'{_fmt(r.get("split_ratio"), 3):>6} {_fmt(r["best_iteration"]):>3} {str(r["trajectory"]):<9}'
            )
        out.append(f'verdict: {self.verdict} (margin {_fmt(self.margin, 4)})')
        return '\n'.join(out)


def _fmt(v, nd=6):
    if v is None:
        return ''
    if isinstance(v, float):
        return f'{v:.{nd}f}' if np.isfinite(v) else str(v)
    if isinstance(v, (list, tuple)):
        return json.dumps(v, default=_json_default)
    return str(v)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def restore_best_fitted_state(result):
    """Return the TreeTime object retained at a candidate's best-scoring fit.
    It holds the complete fitted state (rooted tree, dates, branch lengths,
    clock model, profiles, GTR, RNG)."""
    run = result.best_run if isinstance(result, CandidateResult) else result
    if run is None or run.state is None:
        raise ValueError('no retained state (keep_states=False or the fit failed)')
    return run.state


# ---------------------------------------------------------------------------
# evaluator
# ---------------------------------------------------------------------------


class RootCandidateEvaluator:
    """
    Evaluate root candidates on a fixed unrooted topology.

    Parameters
    ----------
    tree : str or Bio.Phylo tree
        Starting tree. Its root is discarded (the tree is read as unrooted);
        its topology is never changed except by rerooting.
    aln : str or alignment accepted by TreeTime
    dates : dict
        leaf name -> date
    gtr : GTR or str
        Fixed substitution model, shared by all candidates (deep-copied per
        candidate). Not re-inferred.
    clock_rate : float, optional
        Fixed positive clock rate shared by all candidates. If None, the slope
        of the best root-to-tip regression root is used (and must be > 0).
    iterations : int
        Fitting iterations after the initial round, each = marginal ancestral
        inference followed by time-tree inference. Always performed in full
        (no ndiff stop).
    rng_seed : int
        Seed for every per-candidate TreeTime, so evaluation order cannot leak
        through the RNG.
    keep_states : bool
        Retain a deep copy of each candidate's best-scoring TreeTime.
    treetime_kwargs : dict
        Extra TreeTime constructor arguments (e.g. precision, use_fft).
    """

    def __init__(self, tree, aln, dates, gtr='JC69', clock_rate=None, iterations=4, rng_seed=1234,
                 keep_states=True, treetime_kwargs=None, verbose=0):
        if isinstance(tree, str):
            tree = Phylo.read(tree, 'newick')
        self.baseline = name_internal_nodes(deroot(copy.deepcopy(tree)))
        for n in self.baseline.find_clades():
            if n.branch_length is None or n.branch_length < 0:
                n.branch_length = 0.0
        self.aln = aln
        self.dates = dates
        self.gtr = gtr
        self.iterations = int(iterations)
        self.rng_seed = rng_seed
        self.keep_states = keep_states
        self.verbose = verbose
        self.treetime_kwargs = dict(treetime_kwargs or {})
        self._regression = None
        if clock_rate is None:
            # free-slope regression, not cached: candidate ranking later uses the fixed rate
            ranked = self.regression_ranking(slope=None)
            if not ranked:
                raise ValueError('no root with a positive regression slope; pass clock_rate')
            clock_rate = float(ranked[0]['slope'])
        if not clock_rate > 0:
            raise ValueError('clock_rate must be positive')
        self.clock_rate = float(clock_rate)

    # -- construction --------------------------------------------------------

    def new_treetime(self, tree):
        gtr = copy.deepcopy(self.gtr) if not isinstance(self.gtr, str) else self.gtr
        kw = dict(verbose=self.verbose, rng_seed=self.rng_seed)
        kw.update(self.treetime_kwargs)
        return TreeTime(tree=tree, aln=self.aln, dates=self.dates, gtr=gtr, **kw)

    def config(self):
        return {
            'branch_length_mode': 'marginal',
            'Tc': None,
            'resolve_polytomies': False,
            'prune_short': False,
            'infer_gtr': False,
            'time_marginal': False,
            'clock_rate': self.clock_rate,
            'iterations': self.iterations,
            'rng_seed': self.rng_seed,
            'gtr': self.gtr if isinstance(self.gtr, str) else repr(self.gtr),
            'treetime_kwargs': self.treetime_kwargs,
        }

    # -- candidate generation ------------------------------------------------

    def regression_ranking(self, slope='fixed', covariation=False):
        """Rank every baseline edge by the root-to-tip regression, on the
        branch lengths TreeTime's run() would use for rerooting (one round of
        marginal branch-length optimisation of the input tree)."""
        if self._regression is not None and slope == 'fixed':
            return self._regression
        tt = self.new_treetime(copy.deepcopy(self.baseline))
        tt._set_branch_length_mode('marginal')
        tt.optimize_tree(infer_gtr=False, max_iter=1, method_anc='probabilistic', **treetime_seq_kwargs())
        for n in tt.tree.find_clades():
            n.branch_length = n.mutation_length
        s = self.clock_rate if slope == 'fixed' else slope
        treg = tt.setup_TreeRegression(covariation=covariation)
        ranked = treg.rank_roots(force_positive=True, slope=s)
        # translate to baseline candidates (tt.tree has the baseline topology and root)
        for r in ranked:
            r['candidate'] = EdgeRoot.above(tt.tree, r['node'], fraction_from_node=r['split'])
            r['candidate_key'] = candidate_key(r['candidate'])
            r.pop('node')
        if slope == 'fixed':
            self._regression = ranked
        return ranked

    def regression_candidates(self, k=5):
        """Best k distinct placements by regression, best first, as
        (candidate, rank, chisq, split)."""
        out, seen = [], set()
        for r in self.regression_ranking():
            if r['candidate_key'] in seen:
                continue
            seen.add(r['candidate_key'])
            out.append((r['candidate'], len(out) + 1, float(r['edge_chisq']), float(r['split'])))
            if len(out) >= k:
                break
        return out

    def all_candidates(self):
        """Every distinct root placement: each internal node of degree >= 3
        and each edge (at fraction 0.5, or 0 at a leaf end is not used)."""
        cands = []
        sets = _leafsets(self.baseline)
        for n in self.baseline.find_clades():
            if not n.is_terminal():
                cands.append(NodeRoot(node_key(n, self.baseline, sets), label=n.name or ''))
            if n is not self.baseline.root:
                cands.append(EdgeRoot.above(self.baseline, n, 0.5))
        return dedupe_candidates(cands)

    # -- evaluation ------------------------------------------------------------

    def fit(self, candidate, fraction=None, init_mode='marginal', iterations=None):
        """Fit one candidate from one initialisation; returns a FitRun.

        The steps reproduce TreeTime._run(root=None) for the rerooted tree in
        the canonical configuration: one round of branch-length optimisation
        (twice, as run() does around the reroot), the initial time tree, then
        `iterations` x (marginal ancestral inference, time tree). After every
        time tree the full marginal likelihood is recomputed."""
        iterations = self.iterations if iterations is None else iterations
        init = {'fraction': fraction if fraction is not None else getattr(candidate, 'fraction', None),
                'init_mode': init_mode}
        run = FitRun(init=init)
        t0 = time.time()
        try:
            tree = apply_candidate_root(self.baseline, candidate, fraction=fraction)
            tt = self.new_treetime(tree)
            kids = tt.tree.root.clades
            if len(kids) == 2 and kids[0].branch_length + kids[1].branch_length <= 0:
                # TreeTime's marginal branch-length step keeps the ratio of the two
                # root-incident edges and divides by their sum; give a zero-length
                # root edge one mutation, split equally (the dated fit moves it anyway)
                f = 0.5
                for c, share in zip(kids, (f, 1 - f)):
                    c.branch_length = c.mutation_length = share * tt.one_mutation
                run.warnings.append('zero-length root edge: initial length set to one mutation')
            seq_kwargs = treetime_seq_kwargs()
            tt_kwargs = {'clock_rate': self.clock_rate, 'time_marginal': False}
            tt.use_covariation = False
            tt._set_branch_length_mode('marginal')
            # init_mode='joint' replaces only the first branch-length round, as an
            # alternative initialisation; the second round is always marginal
            init_kwargs = treetime_seq_kwargs(init_mode)
            init_kwargs['marginal_sequences'] = init_mode == 'marginal'
            tt.optimize_tree(infer_gtr=False, max_iter=1, method_anc='probabilistic', **init_kwargs)
            tt.optimize_tree(max_iter=1, method_anc='probabilistic', **seq_kwargs)
            tt.make_time_tree(**tt_kwargs)
            run.fits_performed = 1
            self._record(tt, run, ndiff=None, root_changes=None)
            for it in range(iterations):
                prev_root = np.array(tt.tree.root._cseq, copy=True)
                ndiff = tt.infer_ancestral_sequences('ml', marginal=True, **seq_kwargs)
                root_changes = int(np.sum(prev_root != tt.tree.root._cseq))
                tt.make_time_tree(**tt_kwargs)
                run.fits_performed += 1
                if ndiff == 0 and run.treetime_stop_iteration is None:
                    run.treetime_stop_iteration = it + 1
                self._record(tt, run, ndiff=int(ndiff), root_changes=root_changes)
            run.trajectory = classify_trajectory(run.scores)
        except (TreeTimeError, ValueError, ZeroDivisionError, FloatingPointError) as e:
            run.error = f'{type(e).__name__}: {e}'
        run.runtime = time.time() - t0
        return run

    def _record(self, tt, run, ndiff, root_changes):
        run.cached_LH.append(float(getattr(tt.tree, 'sequence_marginal_LH', np.nan)))
        run.positional_LH.append(float(tt.tree.positional_LH))
        score = fresh_marginal_score(tt)
        run.scores.append(score)
        run.ndiff.append(ndiff)
        run.root_state_changes.append(root_changes)
        if score > run.best_score:
            run.best_score = score
            run.best_index = len(run.scores) - 1
            if self.keep_states:
                run.state = _snapshot(tt)
                run.state_root = root_summary(tt)
            else:
                run.state_root = root_summary(tt)

    def evaluate(self, candidate, source='explicit', regression=None):
        res = CandidateResult(candidate=candidate, source=source)
        if regression:
            res.regression_rank, res.regression_chisq, res.regression_split = regression
        self._add_run(res, self.fit(candidate))
        return res

    def _add_run(self, res, run):
        res.runs.append(run)
        ok = [r for r in res.runs if r.error is None]
        if ok:
            res.best_run = max(ok, key=lambda r: r.best_score)
            res.root = dict(res.best_run.state_root)
            res.optimiser_spread = (max(r.best_score for r in ok) - min(r.best_score for r in ok)) if len(ok) > 1 else None
        else:
            res.best_run = run

    def sensitivity(self, res, fractions=(0.1, 0.5, 0.9), joint_init=True):
        """Re-fit a candidate from alternative initialisations: other initial
        fractions for an edge, and a joint-mode branch-length initialisation."""
        c = res.candidate
        if isinstance(c, EdgeRoot):
            for f in fractions:
                if all(abs(f - (r.init['fraction'] or 0)) > 0.05 for r in res.runs):
                    self._add_run(res, self.fit(c, fraction=f))
        if joint_init:
            self._add_run(res, self.fit(c, init_mode='joint'))
        return res

    def run(self, candidates=(), n_regression=5, close_delta=2.0, sensitivity_fractions=(0.1, 0.5, 0.9)):
        """Evaluate explicit candidates plus the best `n_regression` regression
        candidates, rank by fitted marginal logL, and re-fit every candidate
        within `close_delta` log units of the best from alternative
        initialisations. `close_delta` only decides which candidates get the
        extra diagnostics; it is not a resolution threshold."""
        jobs = []  # (candidate, source, regression info)
        for c in candidates:
            jobs.append((c.snapped(self.baseline) if isinstance(c, EdgeRoot) else c, 'explicit', None))
        if n_regression:
            for c, rank, chisq, split in self.regression_candidates(n_regression):
                jobs.append((c, 'regression', (rank, chisq, split)))
        merged = {}
        for c, src, reg in jobs:
            k = candidate_key(c)
            if k in merged:
                prev = merged[k]
                merged[k] = (prev[0], prev[1] if prev[1] == src else 'explicit+regression', prev[2] or reg)
            else:
                merged[k] = (c, src, reg)
        results = [self.evaluate(c, src, reg) for c, src, reg in merged.values()]
        return self.finish(results, close_delta, sensitivity_fractions)

    def finish(self, results, close_delta=2.0, sensitivity_fractions=(0.1, 0.5, 0.9)):
        results.sort(key=lambda r: -r.score)
        if close_delta is not None and results and np.isfinite(results[0].score):
            top = results[0].score
            for r in results:
                if np.isfinite(r.score) and top - r.score <= close_delta:
                    self.sensitivity(r, fractions=sensitivity_fractions)
            results.sort(key=lambda r: -r.score)
        verdict, margin = 'single candidate', None
        if results and np.isfinite(results[0].score):
            top = results[0].score
            for r in results:
                r.delta_vs_best = r.score - top if np.isfinite(r.score) else None
            if len(results) > 1 and np.isfinite(results[1].score):
                margin = top - results[1].score
                spreads = [x.optimiser_spread for x in results[:2] if x.optimiser_spread is not None]
                if not spreads:
                    verdict = 'untested (no alternative initialisations)'
                elif margin > max(spreads):
                    verdict = 'resolved (margin exceeds optimiser spread)'
                else:
                    verdict = 'unresolved (margin within optimiser spread)'
        return RootSearchResult(results=results, clock_rate=self.clock_rate, verdict=verdict, margin=margin,
                                config=self.config())

    def exhaustive(self, close_delta=None):
        """Evaluate every distinct root placement on the baseline tree."""
        reg = {r['candidate_key']: r for r in reversed(self.regression_ranking())}
        ranks = {}
        for c, rank, chisq, split in self.regression_candidates(k=10 ** 9):
            ranks[candidate_key(c)] = (rank, chisq, split)
        results = [self.evaluate(c, 'exhaustive', ranks.get(candidate_key(c))) for c in self.all_candidates()]
        return self.finish(results, close_delta=close_delta)


def _snapshot(tt):
    """Deep copy of a TreeTime. Nodes carry a back-reference to the TreeTime;
    deep-copying the object copies it once with consistent references."""
    return copy.deepcopy(tt)
