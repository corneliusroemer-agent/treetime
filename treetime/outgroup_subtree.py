"""Undated outgroup subtree as external root-state evidence (likelihood rooting, Stage 3).

One or more outgroup leaves with a topology are attached to the candidate root R
of the clock-constrained ingroup tree by a stem of length d. Their joint
conditional likelihood given the state of R, computed by Felsenstein pruning,
is the Stage 2 root-state message

    m_i(a; theta) = sum_b P(b | a, d) L_X,i(b),

where X is the outgroup end of the stem and L_X,i the pruned likelihood of the
outgroup columns below X. A single leaf with X = the leaf gives
m_i(a) = [e^{Qd}]_{a, o_i}, exactly the Stage 2 message. The outgroup branch
lengths theta (stem included) are free parameters, optimised per candidate; the
outgroup is never part of the TreeTime tree, so it does not enter the ingroup's
clock, dates or coalescent prior.

Compression: messages depend only on the outgroup column pattern, so they are
computed once per distinct outgroup pattern and passed as
:class:`~treetime.root_messages.PatternMessages`; the TreeAnc refines its
compressed patterns by (ingroup pattern, outgroup pattern), which is the grouped
form sum_{g,o} n_{g,o} log sum_a pi_a l_g(a) m_o(a). The full alignment is never
expanded.

Site-wise mixture ``eps`` (default 0, off): m'_i(a) = (1 - eps) m_i(a) + eps c_i
with c_i = sum_b pi_b L_X,i(b), the d -> infinity limit of m_i. It is a proper
likelihood, but **not a cap on the outgroup's pull**: any per-site mixture whose
second component does not depend on the root state is a stem change plus a
candidate-independent constant. Under F81/JC, e^{Qt} = e^{-mu t} I +
(1 - e^{-mu t}) 1 pi^T is affine in e^{-mu t}, so the mixture is exactly a
longer stem; under GTR approximately. With the stem fitted per candidate, eps
therefore acts only as a minimum effective stem length.

Attachment mixture (post hoc, :func:`attachment_mixture_scores`): the outgroup
attaches at the candidate root with probability 1 - w and, with probability w,
uniformly at one of the neighbouring attachment points. This is the
robustness lever against an outgroup that joins one edge off the true root.
"""
import copy
import io
from dataclasses import dataclass, field

import numpy as np
from Bio import Phylo

from .root_candidates import (CandidateResult, FitConfig, candidate_key, evaluate_root_candidate,
                              finalise_ranking, format_source_ranks, fresh_marginal_score, rank_root_candidates,
                              restore_best_fitted_state, outgroup_candidate_sources)
from .root_messages import PatternMessages, outgroup_leaf_messages


def _read_tree(tree):
    if tree is None or hasattr(tree, 'root'):
        return copy.deepcopy(tree)
    if isinstance(tree, str) and (tree.strip().endswith(';') or tree.strip().startswith('(')):
        return Phylo.read(io.StringIO(tree), 'newick')
    return Phylo.read(tree, 'newick')


class OutgroupSubtree:
    """Outgroup leaves, their topology and branch lengths, and the stem to the
    ingroup root.

    Parameters
    ----------
    gtr : GTR
        The ingroup model (Q, pi), shared and fixed.
    sequences : dict
        outgroup name -> aligned sequence (same columns as the ingroup
        alignment). Ambiguous characters use ``gtr.profile_map`` (indicator of
        the admissible states); a column where every outgroup is fully
        ambiguous gives an exactly neutral message.
    tree : Bio.Phylo tree, newick string or path, optional
        Outgroup topology with initial branch lengths (missing lengths start at
        ``init_length``). Without a tree: one leaf is attached directly
        (X = the leaf), several leaves form a star around X.
    attach : iterable of leaf names, optional
        Where the stem joins the outgroup tree. ``None``: at the root of
        ``tree`` as given (a bifurcating root keeps its two edges separate).
        Otherwise the tree is read unrooted and the stem joins the edge that
        separates these leaves from the others, ``attach_length`` from the
        edge's end on the side of these leaves (default: the midpoint). The
        same unrooted tree and attachment give the same messages whatever the
        rooting of the input.
    stem : float, optional
        Initial stem length d (default ``init_length``).
    """

    def __init__(self, gtr, sequences, tree=None, attach=None, attach_length=None, stem=None, init_length=0.01,
                 bounds=(1e-8, 5.0)):
        self.gtr = gtr
        self.sequences = {k: str(v).upper() for k, v in sequences.items()}
        if not self.sequences:
            raise ValueError('OutgroupSubtree: no outgroup sequences')
        self.bounds = tuple(float(x) for x in bounds)
        self.init_length = float(init_length)
        self._build_topology(_read_tree(tree), attach, attach_length)
        if stem is None:
            stem = getattr(self, '_stem_extra', None) or self.init_length
        self.lengths = np.array([float(stem)] +
                                [self.init_length if t is None else float(t) for t in self._edge_init[1:]])
        self._build_patterns()

    # -- topology ------------------------------------------------------------
    def _build_topology(self, tree, attach, attach_length):
        names = list(self.sequences)
        adj = {}  # node id -> {neighbour id: length or None}
        leaf_of = {}

        def link(a, b, t):
            adj.setdefault(a, {})[b] = t
            adj.setdefault(b, {})[a] = t

        if tree is None:
            if attach is not None:
                raise ValueError('OutgroupSubtree: attach needs a tree')
            if len(names) == 1:
                adj[0], leaf_of[0], x = {}, names[0], 0
            else:
                x = 0
                adj[0] = {}
                for i, n in enumerate(names, 1):
                    link(0, i, None)
                    leaf_of[i] = n
        else:
            ids = {}
            for i, c in enumerate(tree.find_clades(order='preorder')):
                ids[id(c)] = i
                adj.setdefault(i, {})
                if c.is_terminal():
                    leaf_of[i] = c.name
            for c in tree.find_clades(order='preorder'):
                for ch in c.clades:
                    link(ids[id(c)], ids[id(ch)], ch.branch_length)
            if sorted(leaf_of.values()) != sorted(names):
                raise ValueError('OutgroupSubtree: tree leaves %s differ from the sequences %s'
                                 % (sorted(leaf_of.values()), sorted(names)))
            x = ids[id(tree.root)]
            if attach is None:
                # a root with one child is in series with the stem: move X down
                while len(adj[x]) == 1 and x not in leaf_of:
                    (y, t), = adj[x].items()
                    self._stem_extra = (t or 0.0)
                    del adj[y][x]
                    del adj[x]
                    x = y
            else:
                # unrooted reading: suppress a degree-2 root, then put X on the attachment edge
                r = ids[id(tree.root)]
                if len(adj[r]) == 2 and r not in leaf_of:
                    (a, ta), (b, tb) = adj[r].items()
                    del adj[a][r], adj[b][r], adj[r]
                    link(a, b, None if ta is None and tb is None else (ta or 0.0) + (tb or 0.0))
                side = frozenset(attach)
                if not side or not side < set(names):
                    raise ValueError('OutgroupSubtree: attach must be a proper, non-empty subset of the outgroups')
                edge = self._find_edge(adj, leaf_of, side)
                if edge is None:
                    raise ValueError('OutgroupSubtree: no edge separates %s from the other outgroups' % sorted(side))
                a, b = edge  # a on the side of `side`
                t = adj[a][b]
                ta = (0.5 * t if attach_length is None else float(attach_length)) if t is not None else attach_length
                if t is not None and not 0 <= ta <= t:
                    raise ValueError('OutgroupSubtree: attach_length outside the edge (length %g)' % t)
                tb = None if t is None else t - ta
                del adj[a][b], adj[b][a]
                x = max(adj) + 1
                adj[x] = {}
                link(x, a, ta)
                link(x, b, tb)
        # rooted at X: preorder, parents, edge lengths; theta[0] = stem
        order, parent, edge_init = [x], {x: None}, {x: None}
        for v in order:
            for w, t in sorted(adj[v].items()):
                if w not in parent:
                    parent[w] = v
                    edge_init[w] = t
                    order.append(w)
        index = {v: i for i, v in enumerate(order)}
        self.parent = np.array([-1] + [index[parent[v]] for v in order[1:]])
        self.children = [[index[w] for w in order if parent[w] == v] for v in order]
        self.leaf_name = {index[v]: n for v, n in leaf_of.items()}
        self._edge_init = [None] + [edge_init[v] for v in order[1:]]
        self.n_nodes = len(order)

    @staticmethod
    def _find_edge(adj, leaf_of, side):
        def leaves_below(v, p):
            out, stack = set(), [(v, p)]
            while stack:
                u, q = stack.pop()
                if u in leaf_of:
                    out.add(leaf_of[u])
                stack.extend((w, u) for w in adj[u] if w != q)
            return out
        for a in adj:
            for b in adj[a]:
                if frozenset(leaves_below(a, b)) == side:
                    return a, b
        return None

    @property
    def edge_names(self):
        """Label per entry of ``lengths``: 'stem', then the leaf name or
        'node<i>' of the node below each edge."""
        return ['stem'] + [self.leaf_name.get(i, 'node%d' % i) for i in range(1, self.n_nodes)]

    # -- patterns ------------------------------------------------------------
    def _build_patterns(self):
        leaves = sorted(self.leaf_name)
        seqs = [self.sequences[self.leaf_name[i]] for i in leaves]
        L = len(seqs[0])
        if any(len(s) != L for s in seqs):
            raise ValueError('OutgroupSubtree: outgroup sequences differ in length')
        cols = [''.join(s[i] for s in seqs) for i in range(L)] if len(seqs) > 1 else list(seqs[0])
        uniq, inv = np.unique(np.array(cols), return_inverse=True)
        pm = self.gtr.profile_map
        try:
            prof = {leaf: np.array([pm[u[j]] for u in uniq], dtype=float) for j, leaf in enumerate(leaves)}
        except KeyError as exc:
            raise ValueError('OutgroupSubtree: character %s not in the GTR profile map' % exc)
        self.columns = uniq
        self.pattern = inv.astype(int)  # outgroup pattern per alignment position
        self.leaf_profiles = prof
        self.neutral = np.all([np.all(p == 1.0, axis=1) for p in prof.values()], axis=0)
        self.length = L

    @property
    def n_patterns(self):
        return len(self.columns)

    # -- likelihoods ---------------------------------------------------------
    def _check(self, lengths):
        th = np.asarray(self.lengths if lengths is None else lengths, dtype=float)
        if th.shape != (self.n_nodes,) or not np.all(th > 0) or not np.all(np.isfinite(th)):
            raise ValueError('OutgroupSubtree: need %d positive finite lengths' % self.n_nodes)
        return th

    def conditional(self, lengths=None):
        """L_X: (n_patterns, n_states) pruned likelihood of each outgroup
        pattern given the state of X."""
        th = self._check(lengths)
        lik = [None] * self.n_nodes
        for v in range(self.n_nodes - 1, -1, -1):  # children come after parents in preorder
            if v in self.leaf_name:
                lik[v] = self.leaf_profiles[v]
                continue
            acc = np.ones((self.n_patterns, len(self.gtr.alphabet)))
            for w in self.children[v]:
                f = lik[w] @ self.gtr.expQt(th[w])
                f[np.all(lik[w] == 1.0, axis=1)] = 1.0  # an unobserved subtree is exactly neutral
                acc = acc * f
            lik[v] = acc
        return lik[0]

    def message_table(self, lengths=None, eps=0.0):
        """Messages per outgroup pattern, (n_patterns, n_states), absolute
        scale; neutral patterns are exactly 1. ``eps``: site-wise mixture
        with the uninformative message c = sum_b pi_b L_X(b), equivalent to a
        longer stem (see the module docstring), default off."""
        th = self._check(lengths)
        lx = self.conditional(th)
        m = lx @ self.gtr.expQt(th[0])  # single leaf at X: the expression of outgroup_leaf_messages
        if eps:
            if not 0 < eps < 1:
                raise ValueError('OutgroupSubtree: eps must be in [0, 1)')
            c = lx @ np.asarray(self.gtr.Pi, dtype=float)
            m = (1.0 - eps) * m + eps * c[:, None]
        m[self.neutral] = 1.0
        return m

    def uninformative_table(self, lengths=None):
        """c per pattern (the d -> infinity limit of the message, constant in the root state)."""
        lx = self.conditional(self._check(lengths))
        c = lx @ np.asarray(self.gtr.Pi, dtype=float)
        c[self.neutral] = 1.0
        return np.repeat(c[:, None], len(self.gtr.alphabet), axis=1)

    def root_messages(self, lengths=None, eps=0.0, full_length=None):
        """:class:`PatternMessages` for :meth:`TreeAnc.set_root_state_messages`;
        positions beyond the outgroup alignment (``full_length`` longer) are neutral."""
        pat = self.pattern
        if full_length is not None and full_length != self.length:
            if full_length < self.length:
                raise ValueError('OutgroupSubtree: outgroup alignment longer than the ingroup one')
            pat = np.concatenate([pat, -np.ones(full_length - self.length, dtype=int)])
        return PatternMessages(pattern=pat, table=self.message_table(lengths, eps))

    def full_messages(self, lengths=None, eps=0.0):
        """Uncompressed (L, n_states) message array (tests and diagnostics)."""
        return self.message_table(lengths, eps)[self.pattern]

    # -- fit at a fixed ingroup state ----------------------------------------
    def state_groups(self, tt):
        """Sufficient statistics of a fitted ingroup TreeAnc/TreeTime for the
        outgroup lengths: fresh likelihood pass, then (ingroup root posterior
        w_g = pi l_g / sum, outgroup pattern, count) over the
        (compressed pattern, outgroup pattern) groups, and the ingroup score."""
        if np.ndim(tt.gtr.Pi) != 1:
            raise NotImplementedError('OutgroupSubtree: site-specific equilibrium frequencies')
        fresh_marginal_score(tt)
        ingroup = float(tt.tree.total_sequence_LH_ingroup)
        w = tt.gtr.Pi[None, :] * tt.tree.root.marginal_subtree_LH
        w = w / w.sum(axis=1, keepdims=True)
        full = tt.data.full_length
        pos = np.arange(min(full, self.length))
        pos = pos[~self.neutral[self.pattern[pos]]]
        cpat = tt.data.full_to_compressed_sequence_map[pos]
        groups, counts = np.unique(np.stack([cpat, self.pattern[pos]]), axis=1, return_counts=True)
        return {'w': w[groups[0]], 'og': groups[1], 'n': counts.astype(float), 'ingroup': ingroup}

    def combined_at_state(self, groups, lengths=None, eps=0.0):
        """Combined log-likelihood (ingroup + outgroup) at a fixed ingroup state."""
        m = self.message_table(lengths, eps)
        return groups['ingroup'] + float(np.sum(groups['n'] * np.log(np.sum(groups['w'] * m[groups['og']], axis=1))))

    def optimise_lengths(self, tt_or_groups, lengths=None, eps=0.0, max_sweeps=30, xatol=1e-7, ftol=1e-10):
        """Maximise the combined likelihood over the outgroup branch lengths at a
        fixed ingroup state: coordinate ascent, each step a bounded Brent search
        on log length. Returns (lengths, combined, sweeps)."""
        from scipy.optimize import minimize_scalar
        g = tt_or_groups if isinstance(tt_or_groups, dict) else self.state_groups(tt_or_groups)
        th = self._check(lengths).copy()
        lo, hi = np.log(self.bounds[0]), np.log(self.bounds[1])
        th = np.clip(th, *self.bounds)
        cur = self.combined_at_state(g, th, eps)
        sweeps = 0
        for sweeps in range(1, max_sweeps + 1):
            moved, start = 0.0, cur
            for j in range(self.n_nodes):
                def neg(x, j=j):
                    t = th.copy()
                    t[j] = np.exp(x)
                    return -self.combined_at_state(g, t, eps)
                r = minimize_scalar(neg, bounds=(lo, hi), method='bounded', options={'xatol': xatol})
                if -r.fun > cur:
                    moved = max(moved, abs(r.x - np.log(th[j])))
                    th[j] = float(np.exp(r.x))
                    cur = -float(r.fun)
            if self.n_nodes == 1 or (moved < 10 * xatol and cur - start < ftol):
                break
        return th, cur, sweeps


# --------------------------------------------------------------------------
# per-candidate fit and ranking
# --------------------------------------------------------------------------
class OutgroupCandidateResult(CandidateResult):
    """CandidateResult whose runs belong to successive outgroup-length rounds;
    the initialisation spread is taken over the round-0 runs only (later
    rounds are refinements, not alternative starts)."""

    @property
    def init_spread(self):
        sc = [r.best_score for r in self.runs if r.error is None and getattr(r, 'outgroup_round', 0) == 0]
        return float(max(sc) - min(sc)) if len(sc) > 1 else None


def fit_outgroup_candidate(baseline, stage1_result, outgroup, eps=0.0, fit=None, optimise_lengths=True, rounds=3,
                           cold_start=False, lengths=None, keep_state=True, catch_errors=False, tol=1e-6):
    """Stage 3 fit of one candidate, starting from its Stage 1 result.

    Alternating optimisation with bounded iterations and best-state retention:

    0. outgroup lengths theta_0 (``lengths`` or ``outgroup.lengths``),
       optimised at the candidate's Stage 1 best state unless
       ``optimise_lengths=False``; messages m(theta_0).
    1. ingroup refit with those messages: a warm trajectory from the Stage 1
       best state (step 0 = that state under m(theta_0)), plus, with
       ``cold_start``, one cold trajectory per Stage 1 initialisation.
    2. for up to ``rounds - 1`` more rounds: optimise theta at the best state so
       far; stop when that gains less than ``tol``; otherwise refit warm from
       that state with the new messages.

    Every retained state is scored by a fresh likelihood under its own theta
    (absolute message scale), so the best over all runs is the candidate's
    profile likelihood over the outgroup lengths. Each run carries
    ``outgroup_lengths`` and ``outgroup_round``. Returns an
    :class:`OutgroupCandidateResult` and a dict of diagnostics.
    """
    fit = fit or FitConfig()
    r1 = stage1_result
    b1 = r1.best_run
    if b1 is None or b1.error is not None or b1.state is None:
        raise ValueError('fit_outgroup_candidate: the Stage 1 result has no retained best state')
    res = OutgroupCandidateResult(candidate=r1.candidate, candidate_id=r1.candidate_id, source=r1.source,
                                  regression_rank=r1.regression_rank, regression_chisq=r1.regression_chisq,
                                  source_ranks=dict(r1.source_ranks))
    full = int(b1.state.data.full_length)
    th = np.array(outgroup.lengths if lengths is None else lengths, dtype=float)
    diag = {'theta_in': th.tolist(), 'posthoc': None, 'round_scores': [], 'round_thetas': [], 'stop': None}
    if optimise_lengths:
        th, val, _ = outgroup.optimise_lengths(copy.deepcopy(b1.state), th, eps)
    else:
        val = outgroup.combined_at_state(outgroup.state_groups(copy.deepcopy(b1.state)), th, eps)
    diag['posthoc'] = val  # Stage 1 fit, outgroup lengths fitted, no propagation ("3a")

    def run_with(theta, rnd, initial_state=None, mu=None, cand=None, mode=None):
        b = baseline.with_root_state_messages(outgroup.root_messages(theta, eps, full))
        cfg = fit if mode is None else FitConfig(fit.iterations, mode)
        x = evaluate_root_candidate(b, cand if cand is not None else r1.candidate, cfg, keep_state=keep_state,
                                    catch_errors=catch_errors, clock_rate=mu, initial_state=initial_state)
        x.outgroup_lengths = np.array(theta, dtype=float)
        x.outgroup_round = rnd
        res.runs.append(x)
        return x

    if cold_start:
        for x in r1.runs:
            if x.initial_mode != 'warm' and not x.rate_fitted:
                run_with(th, 0, cand=x.candidate, mode=x.initial_mode)
    run_with(th, 0, initial_state=b1.state, mu=b1.clock_rate)
    best = res.best_run
    diag['round_scores'].append(best.best_score)
    diag['round_thetas'].append(best.outgroup_lengths.tolist())
    for rnd in range(1, rounds if optimise_lengths else 1):
        if best.error is not None or best.state is None:
            diag['stop'] = 'no state'
            break
        th_new, val, _ = outgroup.optimise_lengths(copy.deepcopy(best.state), best.outgroup_lengths, eps)
        if val - best.best_score < tol:
            diag['stop'] = 'converged'
            break
        run_with(th_new, rnd, initial_state=best.state, mu=best.clock_rate)
        best = res.best_run
        diag['round_scores'].append(best.best_score)
        diag['round_thetas'].append(best.outgroup_lengths.tolist())
    else:
        diag['stop'] = 'max rounds' if optimise_lengths and rounds > 1 else 'fixed lengths'
    diag['rounds'] = len(diag['round_scores'])
    if keep_state:
        for x in res.runs:
            if x is not best:
                x.state = None
    return res, diag


@dataclass
class Stage3Result:
    stage1: object                  # RootSearchResult, ingroup only
    stage3: object                  # RootSearchResult of OutgroupCandidateResult
    rows: list = field(default_factory=list)
    extra: list = field(default_factory=list)       # OutgroupCandidateResult of extra candidates (not ranked)
    extra_rows: list = field(default_factory=list)
    extra_stage1: object = None
    eps: float = 0.0

    def summary(self):
        return {'stage1_best': self.stage1.best.candidate_id, 'stage1_margin': self.stage1.margin,
                'stage3_best': self.stage3.best.candidate_id, 'stage3_margin': self.stage3.margin,
                'stage3_resolved': self.stage3.best_is_resolved, 'eps': self.eps}

    def to_tsv(self, path):
        cols = list(self.rows[0].keys()) if self.rows else []
        with open(path, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in self.rows + self.extra_rows:
                fh.write('\t'.join('' if r.get(c) is None else (repr(r[c]) if isinstance(r[c], float) else str(r[c]))
                                   for c in cols) + '\n')


def _row(outgroup, r1, r3, diag, extra=False):
    b3 = r3.best_run
    ok = b3 is not None and b3.error is None
    row = {'candidate_id': r3.candidate_id, 'source': 'extra' if extra else r3.source,
           'source_ranks': format_source_ranks(r3.source_ranks), 'ingroup_1': r1.score,
           'posthoc_3a': diag.get('posthoc'), 'combined_3': r3.score if ok else None,
           'rounds': diag.get('rounds'), 'stop': diag.get('stop'),
           'round_scores': [round(s, 6) for s in diag.get('round_scores', [])],
           'run_3': (b3.initial_mode + ':%d' % b3.outgroup_round) if ok else None}
    if ok:
        row['outgroup_lengths'] = dict(zip(outgroup.edge_names, np.round(b3.outgroup_lengths, 8).tolist()))
        row['stem'] = float(b3.outgroup_lengths[0])
        row['root_date_1'] = r1.best_run.root.get('root_date')
        row['root_date_3'] = b3.root.get('root_date')
        if b3.state is not None:
            tt = b3.state
            row['ingroup_3'] = float(tt.tree.total_sequence_LH_ingroup)
            row['correction_3'] = float(tt.tree.total_root_message_LH)
    return row


def rank_with_outgroup(baseline, outgroup, eps=0.0, explicit=(), regression_k=5, attachment=(), attach_k=3,
                       optimise_lengths=True, rounds=3, cold_start=False, lengths=None, stage1=None,
                       extra_candidates=(), fit=None, min_margin=1.0, catch_errors=False, keep_states=True,
                       **rank_kwargs):
    """Stage 3: rank candidates by the combined likelihood of the clock-constrained
    ingroup and an undated outgroup subtree attached at each candidate root.

    Candidates as in :func:`~treetime.root_messages.rank_with_root_messages`
    (``explicit``, the first ``attach_k`` of ``attachment`` labelled 'attach',
    regression sources; covariation top ``regression_k`` when attachment
    candidates are given). Stage 1 (:func:`rank_root_candidates`, or the given
    ``stage1``, which must have kept its states) supplies each candidate's
    initialisations and best ingroup state; every Stage 1 candidate is then fitted
    with :func:`fit_outgroup_candidate`. The Stage 3 candidate set is exactly the
    Stage 1 one.

    ``extra_candidates`` (e.g. the true root, as an oracle) get their own Stage 1
    run and the same Stage 3 fit, and are reported in ``extra``/``extra_rows``
    only: they never enter the ranking, the margin or the verdict.

    With one outgroup leaf, ``optimise_lengths=False``, ``lengths=[d]``,
    ``cold_start=True`` and ``eps=0`` this is Stage 2b at fixed d.
    """
    fit = fit or FitConfig()
    if rank_kwargs.get('rate_rounds') or rank_kwargs.get('sensitivity_window') is not None:
        raise NotImplementedError('rank_with_outgroup: rate fits and sensitivity refits are not supported')
    if 'ranked_sources' in rank_kwargs:
        raise ValueError('rank_with_outgroup: pass attachment candidates as attachment=')
    attach = list(attachment)[:attach_k] if attach_k else []
    rank_kwargs['ranked_sources'] = {'attach': attach} if attach else None
    if attach and all(rank_kwargs.get(a) is None for a in ('candidate_sources', 'covariation', 'tip_slack')):
        rank_kwargs['candidate_sources'] = outgroup_candidate_sources(regression_k) if regression_k else ()
    if stage1 is None:
        stage1 = rank_root_candidates(baseline, explicit=explicit, regression_k=regression_k, fit=fit,
                                      min_margin=min_margin, catch_errors=catch_errors, keep_states=True,
                                      **rank_kwargs)
    kw = dict(eps=eps, fit=fit, optimise_lengths=optimise_lengths, rounds=rounds, cold_start=cold_start,
              lengths=lengths, keep_state=keep_states, catch_errors=catch_errors)
    results, rows = [], []
    for r1 in stage1.results:
        if r1.best_run is None or r1.best_run.error is not None or r1.best_run.state is None:
            if r1.best_run is not None and r1.best_run.error is None:
                raise ValueError('rank_with_outgroup: the Stage 1 result kept no states')
            continue  # a failed Stage 1 fit (catch_errors) has nothing to start from
        r3, diag = fit_outgroup_candidate(baseline, r1, outgroup, **kw)
        results.append(r3)
        rows.append(_row(outgroup, r1, r3, diag))
    s3 = finalise_ranking(results, min_margin, config={'eps': eps, 'rounds': rounds, 'optimise_lengths':
                                                       optimise_lengths, 'cold_start': cold_start,
                                                       'iterations': fit.iterations, 'n_outgroups':
                                                       len(outgroup.sequences)}, clock_rate=baseline.clock_rate)
    out = Stage3Result(stage1=stage1, stage3=s3, rows=rows, eps=eps)
    if extra_candidates:
        s1x = rank_root_candidates(baseline, explicit=list(extra_candidates), regression_k=0, fit=fit,
                                   candidate_sources=(), min_margin=min_margin, catch_errors=catch_errors,
                                   keep_states=True)
        out.extra_stage1 = s1x
        for r1 in s1x.results:
            if r1.best_run is None or r1.best_run.error is not None:
                continue
            r3, diag = fit_outgroup_candidate(baseline, r1, outgroup, **kw)
            out.extra.append(r3)
            out.extra_rows.append(_row(outgroup, r1, r3, diag, extra=True))
    return out


# --------------------------------------------------------------------------
# attachment mixture (post hoc)
# --------------------------------------------------------------------------
def _unrooted_from_state(tt):
    """Adjacency {node_id: {node_id: length}} of a fitted TreeAnc/TreeTime at the
    lengths it propagates with; a bifurcating root is suppressed (its two edges
    merged). Returns (adj, names, root_neighbour_edges)."""
    nodes = list(tt.tree.find_clades(order='preorder'))
    idx = {id(n): i for i, n in enumerate(nodes)}
    adj = {i: {} for i in range(len(nodes))}
    for n in nodes:
        for c in n.clades:
            t = float(tt._branch_length_to_gtr(c))
            adj[idx[id(n)]][idx[id(c)]] = t
            adj[idx[id(c)]][idx[id(n)]] = t
    names = {idx[id(n)]: n.name for n in nodes if n.is_terminal()}
    r = 0
    kids = list(adj[r])
    if len(kids) == 2:  # root on an edge (u, v): neighbours are the other edges at u and at v
        u, v = kids
        t = adj[r][u] + adj[r][v]
        del adj[u][r], adj[v][r], adj[r]
        adj[u][v] = adj[v][u] = t
        nb = [(u, w) for w in adj[u] if w != v] + [(v, w) for w in adj[v] if w != u]
    else:  # root at a node: neighbours are its incident edges
        nb = [(r, w) for w in kids]
    return adj, names, nb


def _rooted_on_edge(adj, names, a, b, frac=0.5):
    from Bio.Phylo.BaseTree import Clade, Tree

    def build(v, p, bl):
        return Clade(name=names.get(v), branch_length=bl,
                     clades=[build(w, v, adj[v][w]) for w in sorted(adj[v]) if w != p])
    t = adj[a][b]
    root = Clade(name='ATTACH_ROOT', branch_length=0.0, clades=[build(a, b, t * frac), build(b, a, t * (1 - frac))])
    return Tree(root=root, rooted=True)


def attachment_mixture_scores(baseline, tt, outgroup, ws=(0.0, 0.1, 0.3, 0.5), eps=0.0, lengths=None):
    """Post-hoc attachment mixture at a fitted ingroup state ``tt`` rooted at the
    candidate R (no messages of its own needed; its ingroup part is used).

    L_w = (1 - w) L(attach at R) + w mean_e L(attach at the midpoint of e), over
    the neighbouring edges e: for a root inside an edge (u, v), every other edge
    incident to u or v (one edge away); for a root at a node, its incident edges.
    Each term is the joint likelihood of ingroup and outgroup at ``tt``'s branch
    lengths with the outgroup lengths fitted for that attachment (by the pulley
    identity the ingroup tree rerooted at the attachment point, with the outgroup
    message there). Returns ({w: log L_w}, details)."""
    from scipy.special import logsumexp
    base = copy.deepcopy(tt)
    if base.root_state_message is not None:
        base.set_root_state_messages(None)
    th0 = outgroup.lengths if lengths is None else lengths
    th_r, s_r, _ = outgroup.optimise_lengths(base, th0, eps)
    adj, names, nb = _unrooted_from_state(base)
    s_nb = []
    for a, b in nb:
        ta = baseline.make_treeanc(_rooted_on_edge(adj, names, a, b), messages=False)
        s_nb.append(outgroup.optimise_lengths(ta, th_r, eps)[1])
    out = {}
    for w in ws:
        if w == 0 or not s_nb:
            out[w] = s_r
        else:
            out[w] = float(logsumexp([np.log1p(-w) + s_r] + [np.log(w / len(s_nb)) + x for x in s_nb]))
    return out, {'at_root': s_r, 'neighbours': s_nb, 'n_neighbours': len(s_nb)}
