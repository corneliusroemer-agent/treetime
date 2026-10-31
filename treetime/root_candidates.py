"""Likelihood-based selection among candidate root placements (Stage 1).

Each candidate root is fitted independently under a strict clock with a fixed
rate and a fixed GTR model, and scored by the full marginal sequence
log-likelihood recomputed at the fitted (clock-constrained) branch lengths.

Layout
------
* :class:`UnrootedBaseline` -- the fixed unrooted topology with branch lengths,
  alignment, tip dates, GTR model and clock rate shared by every candidate.
  Edges are identified by the leaf set on the side away from a fixed reference
  leaf, which is stable under rerooting.
* :class:`NodeRoot` / :class:`EdgeRoot` -- candidate root placements.
* :func:`regression_candidates` -- the best K placements of the existing
  root-to-tip regression (:meth:`TreeRegression.rank_roots`).
* :func:`evaluate_root_candidate` -- one candidate, fresh ``TreeTime`` object,
  bounded alternation of marginal ancestral inference and dating, fresh
  likelihood after every step, best complete state retained.
* :func:`rank_root_candidates` -- evaluates a candidate list, adds
  initialisation-sensitivity runs for close candidates and returns a table.
"""
import copy
import json
import time
from dataclasses import dataclass, field

import numpy as np
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree

from . import config as ttconf
from .gtr import GTR
from .treetime import TreeTime

EDGE_ROOT_NAME = 'EDGE_ROOT'
ENDPOINT_TOL = 1e-5  # same tolerance as TreeRegression.optimal_reroot


# --------------------------------------------------------------------------
# candidate types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeRoot:
    """Root exactly at an existing internal node (its degree is preserved).

    ``split`` is the canonical edge key of the node: the leaf set below the
    node when the tree is hung from the reference leaf.
    """

    split: frozenset

    @property
    def kind(self):
        return 'node'


@dataclass(frozen=True)
class EdgeRoot:
    """Root inside an edge, a new bifurcating root node.

    ``fraction`` is the initial position measured from the endpoint away from
    the reference leaf (0 = at that endpoint, 1 = at the reference-side
    endpoint), the same convention as TreeRegression's ``split``.
    """

    split: frozenset
    fraction: float = 0.5

    @property
    def kind(self):
        return 'edge'


# --------------------------------------------------------------------------
# baseline: unrooted topology + shared model
# --------------------------------------------------------------------------
class UnrootedBaseline:
    """Fixed unrooted topology, branch lengths and model shared by all candidates.

    Parameters
    ----------
    tree : Bio.Phylo tree or newick path
        Any rooting; a bifurcating root and other degree-2 nodes are suppressed.
    aln : dict or alignment accepted by TreeAnc
    dates : dict leaf -> numeric date
    gtr : GTR instance or model name (fixed, never inferred)
    clock_rate : float or None
        Fixed positive rate shared by all candidates. ``None`` takes the slope
        of the best free-slope root-to-tip regression on the baseline.
    optimize_baseline : bool
        Re-estimate branch lengths once with TreeTime's marginal optimiser on
        the unrooted tree (as ``TreeTime.run`` does before rerooting). These
        lengths drive the regression ranking and start every candidate.
    """

    def __init__(self, tree, aln, dates, gtr='JC69', clock_rate=None, optimize_baseline=True,
                 rng_seed=1234, covariation=False, tt_kwargs=None):
        if isinstance(tree, str):
            tree = Phylo.read(tree, 'newick')
        self.aln = aln
        self.dates = dict(dates)
        self.gtr = gtr if isinstance(gtr, GTR) else GTR.standard(gtr)
        self.rng_seed = rng_seed
        self.covariation = covariation
        self.tt_kwargs = dict(tt_kwargs or {})
        self._from_phylo(tree)
        if optimize_baseline:
            tt = self.make_treetime(self.rooted_phylo(None))
            tt.optimize_tree(infer_gtr=False, max_iter=1, method_anc='probabilistic', **_seq_kwargs())
            self._from_phylo(tt.tree, keep_names=True)
        self.clock_rate = clock_rate
        if self.clock_rate is None:
            treg = self.regression()
            best = treg.find_best_root(force_positive=True)
            if best is None or not best['slope'] > 0:
                raise ValueError('UnrootedBaseline: no positive clock rate from regression; pass clock_rate')
            self.clock_rate = float(best['slope'])
        if not self.clock_rate > 0:
            raise ValueError('UnrootedBaseline: clock_rate must be positive')

    # -- construction --------------------------------------------------------
    def _from_phylo(self, tree, keep_names=False, allow_edge_root=False):
        adj = {}
        counter = [0]
        names = set(c.name for c in tree.find_clades() if c.name)

        def name_of(c):
            if c.name is None or (not keep_names and not c.is_terminal() and c.name in ('', None)):
                while True:
                    nm = 'INT_%04d' % counter[0]
                    counter[0] += 1
                    if nm not in names:
                        break
                c.name = nm
                names.add(nm)
            return c.name

        for c in tree.find_clades(order='preorder'):
            nm = name_of(c)
            adj.setdefault(nm, {})
            for ch in c.clades:
                cn = name_of(ch)
                bl = float(ch.branch_length or 0.0)
                adj[nm][cn] = bl
                adj.setdefault(cn, {})[nm] = bl
        self.leaves = sorted(c.name for c in tree.get_terminals())
        if len(set(self.leaves)) != len(self.leaves):
            raise ValueError('UnrootedBaseline: duplicate leaf names')
        if EDGE_ROOT_NAME in adj and not allow_edge_root:
            raise ValueError('UnrootedBaseline: node name %s is reserved' % EDGE_ROOT_NAME)
        # suppress degree-2 nodes (bifurcating root, unary nodes)
        changed = True
        while changed:
            changed = False
            for nm in list(adj):
                if nm in self.leaves or len(adj[nm]) != 2:
                    continue
                (a, la), (b, lb) = adj[nm].items()
                del adj[a][nm], adj[b][nm], adj[nm]
                adj[a][b] = adj[b][a] = la + lb
                changed = True
        self.adj = adj
        self.ref_leaf = self.leaves[0]
        self.all_leaves = frozenset(self.leaves)
        self.center = next(iter(adj[self.ref_leaf]))  # internal neighbour of the reference leaf
        # hang from the reference leaf: away endpoint -> split, parent toward ref
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
            if v in self.leaves:
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
        """All edge keys (one per unrooted edge)."""
        return set(self.node_of)

    def edge_length(self, split):
        a = self.node_of[split]
        return self.adj[a][self.toward_ref[a]]

    def is_leaf(self, name):
        return name in self.leaves

    def node(self, name):
        """Candidate rooted exactly at node ``name`` (canonicalised)."""
        if name == self.ref_leaf:
            return self.canonical(EdgeRoot(self.split_of[self.center], 1.0))
        return self.canonical(NodeRoot(self.split_of[name]))

    def edge(self, name, fraction=0.5, other=None):
        """Edge between ``name`` and ``other`` (default: its neighbour toward the
        reference leaf); ``fraction`` measured from ``name``."""
        if other is None or other == self.toward_ref.get(name):
            return self.canonical(EdgeRoot(self.split_of[name], float(fraction)))
        if self.toward_ref.get(other) == name:
            return self.canonical(EdgeRoot(self.split_of[other], 1.0 - float(fraction)))
        raise ValueError('UnrootedBaseline.edge: %s and %s are not adjacent' % (name, other))

    def canonical(self, cand):
        """Deduplicate: an edge root at an internal endpoint is that node's NodeRoot;
        a node root at a leaf is the zero-length edge root above the leaf."""
        if isinstance(cand, NodeRoot):
            nm = self.node_of[cand.split]
            if self.is_leaf(nm):
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
        """Exhaustive set: every internal node plus the midpoint of every edge."""
        out = [NodeRoot(self.split_of[v]) for v in sorted(self.split_of) if not self.is_leaf(v)]
        out += [EdgeRoot(s, fraction) for s in sorted(self.splits, key=lambda s: self.node_of[s])]
        # the reference leaf's neighbour is a node too but has no edge toward ref of its own
        # (its split is all leaves minus ref): it is covered by the loop above.
        return out

    # -- tree building -------------------------------------------------------
    def rooted_phylo(self, cand):
        """New Bio.Phylo tree rooted at ``cand`` (None: at the reference leaf's
        neighbour, used for the baseline itself)."""
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
        """Fresh TreeTime object on ``tree``: same alignment, dates, model and seed."""
        kw = dict(verbose=0, branch_length_mode='marginal', use_covariation=False)
        kw.update(self.tt_kwargs)
        return TreeTime(tree=tree, aln=self.aln, dates=self.dates, gtr=copy.deepcopy(self.gtr),
                        rng_seed=self.rng_seed, **kw)

    def regression(self):
        """TreeRegression on the baseline lengths (rooted at the reference leaf's neighbour)."""
        tt = self.make_treetime(self.rooted_phylo(None))
        treg = tt.setup_TreeRegression(covariation=self.covariation)
        treg._tt = tt  # keep the node owner alive
        return treg

    # -- topology summary of any tree -----------------------------------------
    def topology_signature(self, tree):
        """(set of non-trivial splits, sorted internal degree list) of ``tree``
        viewed as unrooted, with degree-2 nodes suppressed."""
        other = UnrootedBaseline.__new__(UnrootedBaseline)
        other._from_phylo(_strip_tree(tree), keep_names=True, allow_edge_root=True)
        nontriv = {s for s in other.splits if 1 < len(s) < len(other.all_leaves) - 1}
        degrees = sorted(len(v) for k, v in other.adj.items() if k not in other.leaves)
        return nontriv, degrees


def _strip_tree(tree):
    """Plain copy of a tree with names and branch lengths only."""
    def cp(c):
        return Clade(name=c.name, branch_length=c.branch_length, clades=[cp(ch) for ch in c.clades])
    return Tree(root=cp(tree.root), rooted=True)


# --------------------------------------------------------------------------
# fitting and scoring
# --------------------------------------------------------------------------
def _seq_kwargs():
    return {
        'marginal_sequences': True,
        'branch_length_mode': 'marginal',
        'sample_from_profile': 'root',
        'prune_short': False,
        'reconstruct_tip_states': False,
    }


def fresh_marginal_score(tt):
    """Full marginal sequence log-likelihood at the tree's current branch lengths.

    Recomputed (upward pass + root combination); never the cached
    ``sequence_LH()``. Does not draw from the RNG or rewrite sequences.
    """
    tt.postorder_traversal_marginal()
    tt.total_LH_and_root_sequence(assign_sequence=False)
    return float(tt.tree.total_sequence_LH)


def snapshot_full_state(tt):
    """Complete fitted state: deep copy of the tree with all node attributes
    (dates, branch lengths, interpolators, profiles), the clock model, date
    conversion and RNG state. Shared immutable inputs are not copied."""
    memo = {id(tt): tt, id(tt.gtr): tt.gtr, id(tt.data): tt.data}
    return {
        'tree': copy.deepcopy(tt.tree, memo),
        'clock_model': copy.deepcopy(tt.clock_model),
        'date2dist': copy.deepcopy(tt._date2dist),
        'rng_state': copy.deepcopy(tt.rng.bit_generator.state),
        'sequence_reconstruction': tt.sequence_reconstruction,
    }


def restore_best_fitted_state(tt, state):
    """Install a snapshot into ``tt`` (bypasses the tree setter, which would reset
    branch lengths)."""
    memo = {id(tt): tt, id(tt.gtr): tt.gtr, id(tt.data): tt.data}
    tt._tree = copy.deepcopy(state['tree'], memo)
    tt._prepare_nodes()
    tt._leaves_lookup = {n.name: n for n in tt.tree.get_terminals()}
    tt.clock_model = copy.deepcopy(state['clock_model'])
    tt._date2dist = copy.deepcopy(state['date2dist'])
    tt.rng.bit_generator.state = copy.deepcopy(state['rng_state'])
    tt.sequence_reconstruction = state['sequence_reconstruction']


def fitted_newick(tt):
    """Newick of the fitted tree at full precision (clock-constrained lengths)."""
    import io
    t = _strip_tree(tt.tree)
    t.root.branch_length = None
    buf = io.StringIO()
    Phylo.write(t, buf, 'newick', format_branch_length='%.17g')
    return buf.getvalue().strip()


def score_newick(baseline, newick):
    """Marginal log-likelihood of ``baseline``'s alignment on a tree given as
    newick, at exactly its branch lengths, with a new TreeAnc object."""
    import io
    from .treeanc import TreeAnc
    tree = Phylo.read(io.StringIO(newick), 'newick')
    ta = TreeAnc(tree=tree, aln=baseline.aln, gtr=copy.deepcopy(baseline.gtr), verbose=0, rng_seed=baseline.rng_seed)
    return fresh_marginal_score(ta)


@dataclass
class FitConfig:
    iterations: int = 4             # fitting updates after the initial time tree
    initial_mode: str = 'marginal'  # 'joint' gives an alternative initial trajectory
    cycle_tol: float = 1e-6


def _root_summary(tt):
    root = tt.tree.root
    kids = sorted(root.clades, key=lambda c: c.name)
    bls = [float(c.branch_length) for c in kids]
    return {
        'root_date': float(root.numdate) if hasattr(root, 'numdate') else None,
        'root_children': [c.name for c in kids],
        'root_child_lengths': bls,
        'root_degree': len(kids),
    }


def _effective_fraction(baseline, cand, entry):
    """Fitted fraction of the root edge on the away-from-reference side."""
    if not isinstance(cand, EdgeRoot):
        return None
    a = baseline.node_of[cand.split]
    total = sum(entry['root_child_lengths'])
    if total <= 0:
        return None
    return entry['root_child_lengths'][entry['root_children'].index(a)] / total


def classify_trajectory(scores, tol=1e-6):
    """'fixed_point' if the last two scores agree within tol; 'cycling' if the
    last score revisits an earlier (non-adjacent) one within tol; 'oscillating'
    if the last three steps alternate in sign and the score returns closer to
    where it was two steps before than to the previous step; else 'drifting'."""
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


def evaluate_root_candidate(baseline, cand, fit=None, keep_tt=False, catch_errors=False):
    """Fit one candidate from a clean baseline and return its result dict.

    Sequence of updates (mirrors ``TreeTime.run(root=None)`` on a pre-rooted tree
    with ``max_iter=fit.iterations`` but without the ``ndiff`` early stop):
    two marginal branch-length optimisations, an initial time tree, then
    ``fit.iterations`` rounds of (marginal ancestral inference, time tree).
    The fresh marginal likelihood is recorded after every time tree; the
    best-scoring complete state is restored at the end.

    Errors propagate unless ``catch_errors`` is set, in which case the
    candidate is marked failed with score -inf and the error in ``warnings``.
    """
    fit = fit or FitConfig()
    cand = baseline.canonical(cand)
    t0 = time.time()
    res = {
        'candidate_id': baseline.candidate_id(cand),
        'kind': cand.kind,
        'split': sorted(cand.split),
        'initial_fraction': getattr(cand, 'fraction', None),
        'clock_rate': baseline.clock_rate,
        'initial_mode': fit.initial_mode,
        'warnings': [],
        'failed': False,
    }
    try:
        tt = baseline.make_treetime(baseline.rooted_phylo(cand))
        tt._set_branch_length_mode('marginal')
        seq_kwargs = _seq_kwargs()
        tt_kwargs = {'clock_rate': baseline.clock_rate, 'time_marginal': False, 'prune_short': False}
        if fit.initial_mode == 'joint':
            jk = dict(seq_kwargs, branch_length_mode='joint', marginal_sequences=False)
            tt.optimize_tree(infer_gtr=False, max_iter=1, method_anc='probabilistic', **jk)
        else:
            tt.optimize_tree(infer_gtr=False, max_iter=1, method_anc='probabilistic', **seq_kwargs)
        tt.optimize_tree(max_iter=1, method_anc='probabilistic', **seq_kwargs)
        tt.make_time_tree(**tt_kwargs)

        history = []
        best = None

        def record(it, ndiff):
            nonlocal best
            cached = float(getattr(tt.tree, 'sequence_marginal_LH', np.nan))  # read before refreshing
            score = fresh_marginal_score(tt)
            entry = {
                'iteration': it,
                'ndiff': ndiff,
                'fresh_marginal_LH': score,
                'cached_marginal_LH': cached,
                'positional_LH': float(tt.tree.positional_LH),
            }
            entry.update(_root_summary(tt))
            history.append(entry)
            if best is None or score > best['score']:
                best = {'score': score, 'iteration': it, 'state': snapshot_full_state(tt), 'entry': entry}

        record(0, None)
        for it in range(1, fit.iterations + 1):
            ndiff = tt.infer_ancestral_sequences('ml', marginal=True, **seq_kwargs)
            tt.make_time_tree(**tt_kwargs)
            record(it, int(ndiff))

        restore_best_fitted_state(tt, best['state'])
        scores = [h['fresh_marginal_LH'] for h in history]
        tail = scores[len(scores) // 2:]
        e = best['entry']
        bls = e['root_child_lengths']
        res.update({
            'fresh_marginal_LH': best['score'],
            'best_iteration': best['iteration'],
            'iterations_performed': len(history) - 1,
            'trajectory': scores,
            'trajectory_class': classify_trajectory(scores, fit.cycle_tol),
            'tail_fluctuation': float(max(tail) - min(tail)),
            'positional_LH': e['positional_LH'],
            'cached_marginal_LH': e['cached_marginal_LH'],
            'root_date': e['root_date'],
            'root_degree': e['root_degree'],
            'root_child_lengths': bls,
            'effective_fraction': _effective_fraction(baseline, cand, e),
            'fitted_newick': fitted_newick(tt),
            'history': history,
        })
        if res['trajectory_class'] != 'fixed_point':
            res['warnings'].append('trajectory %s' % res['trajectory_class'])
        if keep_tt:
            res['tt'] = tt
            res['best_state'] = best['state']
    except Exception as exc:
        if not catch_errors:
            raise
        res['failed'] = True
        res['warnings'].append('%s: %s' % (type(exc).__name__, exc))
        res['fresh_marginal_LH'] = -np.inf
    res['runtime_s'] = time.time() - t0
    return res


# --------------------------------------------------------------------------
# candidate generation
# --------------------------------------------------------------------------
def regression_candidates(baseline, k=5, force_positive=True):
    """Best ``k`` placements of the root-to-tip regression (fixed slope = the
    baseline clock rate), as canonical candidates with their rank and chi^2."""
    treg = baseline.regression()
    placements = treg.rank_roots(force_positive=force_positive, slope=baseline.clock_rate)
    out = []
    for p in placements:
        n = p['node'].name
        if n == baseline.ref_leaf:
            cand = EdgeRoot(baseline.split_of[baseline.center], 1.0 - p['split'])
        else:
            cand = EdgeRoot(baseline.split_of[n], p['split'])
        cand = baseline.canonical(cand)
        if any(c == cand for c, _ in out):
            continue
        out.append((cand, {'regression_rank': len(out) + 1, 'regression_chisq': float(p['chisq']),
                           'regression_split': float(p['split'])}))
        if k is not None and len(out) >= k:
            break
    return out


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def _alternatives(cand, tried):
    """Alternative initialisations for the sensitivity check."""
    if isinstance(cand, EdgeRoot):
        return [(EdgeRoot(cand.split, f), 'marginal') for f in (0.1, 0.5, 0.9)
                if all(abs(f - t) > 0.05 for t in tried)]
    return [(cand, 'joint')]


_FORK_JOBS = None  # job list inherited by forked workers (the baseline need not be picklable)


def _eval_job(i):
    baseline, cand, fit, catch_errors = _FORK_JOBS[i]
    return evaluate_root_candidate(baseline, cand, fit, catch_errors=catch_errors)


def _map_eval(baseline, jobs, n_jobs, catch_errors):
    """Evaluate (candidate, FitConfig) pairs, optionally in forked worker processes.
    Each evaluation builds its own TreeTime object, so results do not depend on
    which process or in which order a candidate is evaluated."""
    global _FORK_JOBS
    _FORK_JOBS = [(baseline, c, f, catch_errors) for c, f in jobs]
    try:
        if n_jobs and n_jobs > 1 and len(_FORK_JOBS) > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(n_jobs, len(_FORK_JOBS)),
                                     mp_context=mp.get_context('fork')) as ex:
                return list(ex.map(_eval_job, range(len(_FORK_JOBS))))
        return [_eval_job(i) for i in range(len(_FORK_JOBS))]
    finally:
        _FORK_JOBS = None


def candidate_key(cand):
    """Identity of a candidate: the node, or the edge (the fraction is only an
    initialisation)."""
    return (cand.kind, cand.split)


def rank_root_candidates(baseline, explicit=(), regression_k=5, fit=None, sensitivity_window=3.0,
                         run_sensitivity=True, n_jobs=1, catch_errors=False):
    """Evaluate explicit and regression-ranked candidates and rank them by fitted
    full marginal log-likelihood.

    Candidates are identified by node or edge; several initial fractions on the
    same edge (explicit and/or regression) are one candidate with several
    initialisations, the regression's fraction being the primary one.

    Candidates within ``sensitivity_window`` log units of the best are re-fitted
    from alternative initialisations (other fractions for edge roots, a joint-mode
    initial branch-length fit for node roots). A candidate's score is the best
    over its initialisations; its optimiser uncertainty ``u`` is the larger of the
    spread across initialisations and the tail fluctuation of its trajectory.
    A candidate is *resolved* from the best when ΔlogL > u_best + u_candidate.
    The window only decides where extra fits are spent; it is not a resolution
    floor.
    """
    fit = fit or FitConfig()
    merged = {}

    def add(c, source, info=None):
        c = baseline.canonical(c)
        d = merged.setdefault(candidate_key(c), {'sources': [], 'inits': []})
        if source not in d['sources']:
            d['sources'].append(source)
        if info:
            d.update(info)
            d['inits'].insert(0, c)  # regression fraction is the primary initialisation
        elif c not in d['inits']:
            d['inits'].append(c)

    for c in explicit:
        add(c, 'explicit')
    if regression_k:
        for c, info in regression_candidates(baseline, k=regression_k):
            add(c, 'regression', info)
    for d in merged.values():
        d['inits'] = list(dict.fromkeys(d['inits']))

    jobs = [(key, c, fit.initial_mode) for key, d in merged.items() for c in d['inits']]
    evaluated = _map_eval(baseline, [(c, FitConfig(fit.iterations, m, fit.cycle_tol)) for _, c, m in jobs],
                          n_jobs, catch_errors)
    results = {}
    for (key, c, mode), r in zip(jobs, evaluated):
        if key not in results:
            info = merged[key]
            r.update({'source': '+'.join(info['sources']),
                      'regression_rank': info.get('regression_rank'),
                      'regression_chisq': info.get('regression_chisq'),
                      'initialisations': [], '_cand': c})
            results[key] = r
        _fold(results[key], r, mode)
    results = list(results.values())

    def finalize():
        best = max(r['fresh_marginal_LH'] for r in results)
        for r in results:
            r['delta_LH'] = r['fresh_marginal_LH'] - best
        return best

    finalize()
    if run_sensitivity:
        todo = []
        for r in results:
            if r['failed'] or r['delta_LH'] < -sensitivity_window:
                continue
            tried = [i['fraction'] for i in r['initialisations'] if i['fraction'] is not None]
            todo += [(r, alt, mode) for alt, mode in _alternatives(r['_cand'], tried)]
        alts = _map_eval(baseline, [(alt, FitConfig(fit.iterations, mode, fit.cycle_tol)) for _, alt, mode in todo],
                         n_jobs, catch_errors)
        for (r, alt, mode), a in zip(todo, alts):
            _fold(r, a, mode)
        finalize()
    for r in results:
        sc = [i['fresh_marginal_LH'] for i in r['initialisations'] if np.isfinite(i['fresh_marginal_LH'])]
        r['n_initialisations'] = len(r['initialisations'])
        r['init_spread'] = float(max(sc) - min(sc)) if len(sc) > 1 else None
        r['uncertainty'] = max(r.get('tail_fluctuation') or 0.0, r['init_spread'] or 0.0)
    results.sort(key=lambda r: -r['fresh_marginal_LH'])
    top = results[0]
    for r in results:
        r['resolved_from_best'] = (r is top) or (-r['delta_LH'] > top['uncertainty'] + r['uncertainty'])
        r['same_fitted_tree_as_best'] = (r is not top) and r.get('fitted_newick') == top.get('fitted_newick')
        r.pop('_cand', None)
    summary = {
        'best': top['candidate_id'],
        'best_is_resolved': all(r['resolved_from_best'] for r in results[1:]),
        'unresolved_with_best': [r['candidate_id'] for r in results[1:] if not r['resolved_from_best']],
        'clock_rate': baseline.clock_rate,
        'regression_best': next((r['candidate_id'] for r in results if r.get('regression_rank') == 1), None),
        'n_failed': sum(r['failed'] for r in results),
    }
    return results, summary


def _fold(r, a, mode):
    """Record evaluation ``a`` as an initialisation of candidate result ``r``; if it
    scores higher, ``r`` takes over its fitted state (identity fields are kept)."""
    r['initialisations'].append({'fraction': a['initial_fraction'], 'mode': mode,
                                 'fresh_marginal_LH': a['fresh_marginal_LH'], 'failed': a['failed']})
    if a is r:
        r['best_initialisation'] = {'fraction': a['initial_fraction'], 'mode': mode}
        return
    if a['fresh_marginal_LH'] > r['fresh_marginal_LH']:
        keep = {k: r[k] for k in ('source', 'regression_rank', 'regression_chisq', 'initialisations', '_cand',
                                  'initial_fraction', 'candidate_id')}
        keep['warnings'] = r['warnings'] + [w for w in a['warnings'] if w not in r['warnings']]
        r.update(a)
        r.update(keep)
        r['best_initialisation'] = {'fraction': a['initial_fraction'], 'mode': mode}


TABLE_COLS = ['candidate_id', 'source', 'regression_rank', 'fresh_marginal_LH', 'delta_LH', 'positional_LH',
              'root_date', 'root_degree', 'initial_fraction', 'effective_fraction', 'iterations_performed',
              'best_iteration', 'trajectory_class', 'tail_fluctuation', 'n_initialisations', 'init_spread',
              'uncertainty', 'resolved_from_best', 'same_fitted_tree_as_best', 'runtime_s']


def results_table(results, cols=TABLE_COLS):
    def fmt(v):
        if isinstance(v, float):
            return '%.4f' % v
        return '' if v is None else str(v)
    lines = ['\t'.join(cols)]
    for r in results:
        lines.append('\t'.join(fmt(r.get(c)) for c in cols))
    return '\n'.join(lines)


def results_json(results, summary):
    clean = [{k: v for k, v in r.items() if k not in ('tt', 'best_state')} for r in results]
    return json.dumps({'summary': summary, 'candidates': clean}, indent=1, default=float)
