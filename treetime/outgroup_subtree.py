"""Undated outgroup subtree as root-state evidence (likelihood rooting, Stage 3).

One or more outgroup leaves with a topology hang off the root of the
clock-constrained ingroup tree. Their joint conditional likelihood at the
attachment point,

    m_i(a; t) = P(outgroup column i | state a at the ingroup root, outgroup lengths t),

comes from Felsenstein pruning over the outgroup subtree and is passed to the
Stage 2 machinery as root-state messages (:meth:`TreeAnc.set_root_state_messages`).
The outgroup lengths are free parameters: they never enter the ingroup clock
model, the dates or the rate, and they are optimised per candidate root
(:class:`OutgroupEvaluator`) by alternating with the ingroup dated fit.

Representation (:class:`OutgroupSubtree`): a rooted tree whose root ``A`` is
the attachment point (the ingroup root). ``A`` has one child, reached by the
stem; the stem's far end is the outgroup leaf itself (one outgroup) or the
outgroup subtree's root. Under a reversible stationary model the likelihood
depends only on the unrooted tree (outgroup leaves + the attachment), and any
input rooting naming the attachment explicitly maps to the same canonical form.

Attachment mixture (:func:`attachment_mixture_scores`, post hoc): the
outgroup joins at the candidate root with probability 1 - w and at one of the
neighbouring edges with probability w, which bounds how far an outgroup that
joins one edge off can override the clock.

Compression: the messages depend on an alignment column only through its
outgroup pattern, so they are computed once per distinct outgroup pattern and
indexed back by position. Positions with equal outgroup patterns get
byte-identical messages, so the Stage 2 pattern refinement groups ingroup-
constant columns by outgroup pattern with their multiplicities, as plan.md asks;
the length fit uses the same (ingroup pattern, outgroup pattern) groups.

"""
import copy
import time

import numpy as np
from Bio import Phylo

from . import config as ttconf

from .root_candidates import (FIT_ERRORS, FitRun, classify_trajectory, evaluate_root_candidate)
from .seq_utils import seq2prof

STEM = 'stem'


def _read_tree(tree):
    if tree is None or hasattr(tree, 'root'):
        return tree
    import io
    tree = str(tree)
    return Phylo.read(io.StringIO(tree) if tree.strip().startswith('(') else tree, 'newick')


class OutgroupSubtree:
    """Outgroup leaves, their topology and branch lengths, and the model.

    Parameters
    ----------
    gtr : GTR
        Substitution model shared with the ingroup (fixed; not re-estimated).
    sequences : dict
        outgroup name -> aligned sequence (same columns as the ingroup
        alignment). Characters are read with ``gtr.profile_map``: ambiguous and
        missing characters give the indicator of the admissible states.
    tree : Bio.Phylo tree, newick string or path, optional
        Topology (and optionally initial lengths) on the outgroup leaves. If it
        contains a leaf named ``attach_name``, that leaf is the attachment point
        and the tree's own rooting is irrelevant. Otherwise the stem attaches at
        the tree's root (the root's own branch length, if given, initialises the
        stem). Default: one leaf -> a single branch; several -> a star below the
        stem (for two leaves, the cherry).
    attach_name : str
        Leaf name marking the attachment point in ``tree``.
    stem : float, optional
        Initial stem length (default: the tree's, else ``default_length``).
    default_length : float
        Initial length of edges without one.
    bounds : (float, float)
        Bounds of every outgroup branch length in the length fit.
    """

    def __init__(self, gtr, sequences, tree=None, attach_name='ATTACH', stem=None, default_length=0.01,
                 bounds=(1e-7, 10.0)):
        if not sequences:
            raise ValueError('OutgroupSubtree: no outgroup sequences')
        if len(np.asarray(gtr.Pi).shape) != 1:
            raise NotImplementedError('OutgroupSubtree: site-specific models are not supported')
        self.gtr = gtr
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.names = list(sequences)
        self._build_topology(_read_tree(tree), attach_name, stem, default_length)
        self._build_patterns(sequences)

    # -- topology --------------------------------------------------------------
    def _build_topology(self, tree, attach_name, stem, default_length):
        """Canonical form: node 0 = attachment A; node k > 0 has parent[k] and
        the edge above it has index k - 1 (edge 0 = the stem, A's only child)."""
        names = set(self.names)
        adj = {}  # undirected: node id -> {neighbour id: length or None}
        label = {}
        if tree is None:
            if len(self.names) == 1:
                adj = {0: {1: stem}, 1: {0: stem}}
                label = {0: attach_name, 1: self.names[0]}
            else:
                adj = {0: {1: stem}, 1: {0: stem}}
                label = {0: attach_name, 1: None}
                for i, n in enumerate(self.names):
                    adj[1][2 + i] = None
                    adj[2 + i] = {1: None}
                    label[2 + i] = n
            start = 0
        else:
            ids = {}

            def nid(c):
                if id(c) not in ids:
                    ids[id(c)] = len(ids)
                    adj[ids[id(c)]] = {}
                    label[ids[id(c)]] = c.name if c.is_terminal() else None
                return ids[id(c)]
            for c in tree.find_clades(order='preorder'):
                for ch in c.clades:
                    a, b = nid(c), nid(ch)
                    adj[a][b] = adj[b][a] = ch.branch_length
            root = nid(tree.root)
            leaves = [label[k] for k in adj if len(adj[k]) <= 1 and label[k] is not None]
            if attach_name in leaves:
                start = next(k for k in adj if label[k] == attach_name)
                if len(adj[start]) != 1:
                    raise ValueError('OutgroupSubtree: attachment leaf must have one neighbour')
            else:
                a = len(adj)
                adj[a] = {root: stem if stem is not None else tree.root.branch_length}
                adj[root][a] = adj[a][root]
                label[a] = attach_name
                start = a
            bad = set(l for l in leaves if l != attach_name) ^ names
            if bad:
                raise ValueError('OutgroupSubtree: tree leaves and sequences differ: %s' % sorted(bad))
            # suppress degree-2 internal nodes (e.g. a bifurcating input root): merge their edges
            for k in [k for k in adj if len(adj[k]) == 2 and label[k] is None]:
                (u, lu), (v, lv) = adj[k].items()
                ln = None if lu is None and lv is None else (lu or 0.0) + (lv or 0.0)
                del adj[u][k], adj[v][k], adj[k]
                adj[u][v] = adj[v][u] = ln
            if stem is not None:
                (nb,) = adj[start]
                adj[start][nb] = adj[nb][start] = stem
        # canonical numbering by DFS from the attachment, children ordered by their leaf sets
        leafset = {}

        def leaves_below(k, p):
            if label[k] is not None and k != start:
                s = (label[k],)
            else:
                s = tuple(sorted(x for w in adj[k] if w != p for x in leaves_below(w, k)))
            leafset[(k, p)] = s
            return s
        leaves_below(start, None)
        if len(adj[start]) != 1:
            raise ValueError('OutgroupSubtree: the attachment must have exactly one neighbour')
        # breadth-first from the attachment; indices assigned when enqueued
        parent, node_label, lengths, children = [-1], [label[start]], [], [[]]
        queue = [(start, None, 0)]
        while queue:
            k, p, me = queue.pop(0)
            for w in sorted((w for w in adj[k] if w != p), key=lambda w: leafset[(w, k)]):
                parent.append(me)
                children.append([])
                children[me].append(len(parent) - 1)
                node_label.append(label[w] if len(adj[w]) == 1 else None)
                ln = adj[k][w]
                lengths.append(float(ln) if ln is not None and ln > 0 else default_length)
                queue.append((w, k, len(parent) - 1))
        self.parent, self.node_label, self.children = parent, node_label, children
        self.lengths0 = np.clip(np.array(lengths), *self.bounds)
        self.lengths = self.lengths0.copy()
        self.edge_names = [STEM] + [self.node_label[k] or 'int:' + '|'.join(self._leafset(k))
                                    for k in range(2, len(self.parent))]
        self.leaf_nodes = {self.node_label[k]: k for k in range(1, len(self.parent)) if self.node_label[k]}
        self.postorder = sorted(range(1, len(self.parent)), key=lambda k: -self._depth(k))

    def _depth(self, k):
        d = 0
        while self.parent[k] >= 0:
            k, d = self.parent[k], d + 1
        return d

    def _leafset(self, k):
        if self.node_label[k] and not self.children[k]:
            return [self.node_label[k]]
        return sorted(x for c in self.children[k] for x in self._leafset(c))

    def newick(self, lengths=None):
        """Outgroup subtree as newick, rooted at the attachment leaf."""
        t = self.lengths if lengths is None else lengths

        def nw(k):
            s = self.node_label[k] if not self.children[k] else '(' + ','.join(nw(c) for c in self.children[k]) + ')'
            return '%s:%.10g' % (s, t[k - 1])
        return '(%s,%s:0);' % (nw(1), self.node_label[0])

    # -- patterns --------------------------------------------------------------
    def _build_patterns(self, sequences):
        seqs = {n: np.array(list(s.upper())) if isinstance(s, str) else np.asarray(s) for n, s in sequences.items()}
        L = {len(s) for s in seqs.values()}
        if len(L) != 1:
            raise ValueError('OutgroupSubtree: outgroup sequences differ in length')
        self.full_length = L.pop()
        leaves = [self.node_label[k] for k in sorted(self.leaf_nodes.values())]
        cols = np.stack([seqs[n] for n in leaves], axis=1)  # (L, n_leaves)
        self.patterns, self.pattern_of_position, self.pattern_counts = np.unique(
            cols, axis=0, return_inverse=True, return_counts=True)
        self.pattern_of_position = self.pattern_of_position.reshape(-1)
        self._leaf_order = leaves
        self.leaf_profiles = {n: seq2prof(self.patterns[:, j], self.gtr.profile_map) for j, n in enumerate(leaves)}
        self.neutral_patterns = np.all([np.all(p == 1.0, axis=1) for p in self.leaf_profiles.values()], axis=0)

    @property
    def n_patterns(self):
        return len(self.patterns)

    # -- likelihoods -------------------------------------------------------------
    def pattern_messages(self, lengths=None):
        """(n_patterns, K) messages m_o(a), absolute scale, for each distinct
        outgroup pattern; all-ambiguous patterns get exactly 1."""
        t = self.lengths if lengths is None else np.asarray(lengths, dtype=float)
        part, logs = {}, {}
        for k in self.postorder:
            if not self.children[k]:
                part[k], logs[k] = self.leaf_profiles[self.node_label[k]], np.zeros(self.n_patterns)
                continue
            p, lg = 1.0, 0.0
            for c in self.children[k]:
                p = p * (part[c] @ self.gtr.expQt(t[c - 1]))
                lg = lg + logs[c]
            mx = p.max(axis=1)
            part[k], logs[k] = p / mx[:, None], lg + np.log(mx)
        m = (part[1] @ self.gtr.expQt(t[0])) * np.exp(logs[1])[:, None]
        m[self.neutral_patterns] = 1.0
        return m

    def messages(self, lengths=None):
        """(full_length, K) root-state messages for the Stage 2 interface."""
        return self.pattern_messages(lengths)[self.pattern_of_position]

    def _groups(self, tt):
        """(ingroup pattern, outgroup pattern) groups of ``tt``'s current
        compression with multiplicities, and the root's ingroup partials."""
        if tt.data.full_length != self.full_length:
            raise ValueError('OutgroupSubtree: alignment length %d != outgroup length %d'
                             % (tt.data.full_length, self.full_length))
        tt.postorder_traversal_marginal()
        root = tt.tree.root
        cp = np.asarray(tt.data.full_to_compressed_sequence_map)
        g, n = np.unique(np.stack([cp, self.pattern_of_position]), axis=1, return_counts=True)
        lw = np.asarray(tt.gtr.Pi)[None, :] * root.marginal_subtree_LH[g[0]]
        return {'c': g[0], 'o': g[1], 'n': n, 'pi_l': lw, 'pref': root.marginal_subtree_LH_prefactor[g[0]]}

    @staticmethod
    def _value(groups, msg):
        return float(np.sum(groups['n'] * (groups['pref'] + np.log(np.sum(groups['pi_l'] * msg[groups['o']], axis=1)))))

    def combined_score(self, tt, lengths=None):
        """Combined log-likelihood (ingroup + outgroup) of the ingroup tree ``tt``
        at its current branch lengths, outgroup lengths ``lengths``. ``tt``'s own
        root-state messages, if any, are ignored (fresh postorder pass)."""
        return self._value(self._groups(tt), self.pattern_messages(lengths))

    def fit_lengths(self, tt, x0=None, starts=None):
        """Maximum-likelihood outgroup lengths at the fixed ingroup tree ``tt``.

        Bounded Brent on the log length for a single edge, else L-BFGS-B on log
        lengths; never returns a point worse than ``x0`` (default the current
        ``lengths``). ``starts``: extra initial points (lists of lengths).
        Returns (lengths, combined score)."""
        G = self._groups(tt)
        return self.maximise(lambda msg: self._value(G, msg), x0=x0, starts=starts)

    def maximise(self, value, x0=None, starts=None):
        """Outgroup lengths maximising ``value(pattern messages)`` (the optimiser of
        :meth:`fit_lengths`). Returns (lengths, value)."""
        from scipy.optimize import minimize, minimize_scalar
        lo, hi = np.log(self.bounds[0]), np.log(self.bounds[1])

        def f(x):
            return -value(self.pattern_messages(np.exp(x)))
        x0 = np.log(np.clip(self.lengths if x0 is None else np.asarray(x0, dtype=float), *self.bounds))
        best_x, best_f = x0, f(x0)
        inits = [x0] + [np.log(np.clip(np.asarray(s, dtype=float), *self.bounds)) for s in (starts or [])]
        for xi in inits:
            if len(xi) == 1:
                r = minimize_scalar(lambda v: f(np.array([v])), bounds=(lo, hi), method='bounded',
                                    options={'xatol': 1e-6})
                x, fx = np.array([r.x]), float(r.fun)
            else:
                r = minimize(f, xi, method='L-BFGS-B', bounds=[(lo, hi)] * len(xi),
                             options={'ftol': 1e-12, 'gtol': 1e-7, 'maxiter': 500})
                x, fx = r.x, float(r.fun)
            if fx < best_f:
                best_x, best_f = x, fx
        return np.exp(best_x), -best_f

    def initialise(self, tt):
        """Fit the lengths at ``tt`` from several stem starts and store them as
        the default ``lengths`` (used as the starting point everywhere else)."""
        starts = [np.concatenate([[s], self.lengths[1:]]) for s in (1e-3, 1e-2, 1e-1, 0.5)]
        self.lengths, val = self.fit_lengths(tt, starts=starts)
        self.lengths0 = self.lengths.copy()
        return val

    def copy(self):
        return copy.deepcopy(self)


# --------------------------------------------------------------------------
# per-candidate evaluator: alternating outgroup lengths <-> ingroup dated fit
# --------------------------------------------------------------------------
class OutgroupEvaluator:
    """Drop-in replacement for :func:`evaluate_root_candidate` (pass as
    ``rank_root_candidates(..., evaluator=)``): the outgroup subtree hangs off
    each candidate's root, with its lengths fitted per candidate.

    1. Fit the outgroup lengths at the starting ingroup tree (the baseline rooted
       at the candidate, or the warm-start state), starting from
       ``outgroup.lengths``.
    2. Set the messages m(t) and run the Stage 2 dated fit
       (:func:`evaluate_root_candidate`, bounded iterations, best state kept).
    3. Up to ``outer_iterations`` times: refit the lengths at the best fitted
       ingroup state; stop if the combined score gains less than ``tol``;
       otherwise continue the dated fit from that state under the new messages
       (a warm trajectory whose step 0 is the state at the new lengths).

    The returned :class:`FitRun` concatenates the trajectories; its best state
    (and score) is the best over all steps of all rounds, with the outgroup
    lengths it was scored at in ``run.extra['outgroup_lengths']``. With
    ``fit_lengths=False`` the lengths stay at ``outgroup.lengths`` (fixed
    messages: Stage 2b).

    ``attach_mixture_w`` = w > 0: the attachment mixture is part of the dated
    fit. The outgroup attaches at the candidate root with probability 1 - w and
    at the midpoint of one of its neighbouring edges with probability w, split
    uniformly (:class:`AttachmentMixture`; outgroup lengths fitted per
    attachment). Each round builds, at the current ingroup state, the
    responsibilities of the attachments and the surrogate root-state message
    (:meth:`AttachmentMixture.surrogate_messages`), runs the dated fit with it,
    and scores the fitted state by the exact mixture log-likelihood with the
    lengths refitted there; the candidate's score is the best such score. Rounds
    continue (up to ``outer_iterations`` after round 0) while refitting the
    lengths at the best state gains at least ``tol`` and the last round
    improved. A warm start state is itself scored by the exact mixture and
    kept if no fitted state beats it, so the score is never below the post-hoc
    mixture of that state. The trajectory ``scores`` are the surrogate's
    (close to the ingroup log-likelihood; ``trajectory_class``,
    ``tail_fluctuation`` and so the ``uncertainty`` read them as optimiser
    noise), while ``best_score`` is the exact mixture; ``run.extra`` holds the mixture
    terms, responsibilities and per-attachment lengths, and
    ``outgroup_lengths`` are those of the attachment at the root. The kept state
    carries the plain message of that attachment. w = 0 is exactly the
    unmixed evaluator.
    """

    def __init__(self, outgroup, outer_iterations=2, tol=1e-2, fit_lengths=True, attach_mixture_w=0.0):
        self.outgroup = outgroup
        self.outer_iterations, self.tol, self.fit = int(outer_iterations), float(tol), bool(fit_lengths)
        self.w = float(attach_mixture_w)
        if not 0.0 <= self.w <= 1.0:
            raise ValueError('attach_mixture_w must be in [0, 1]')
        if self.w > 0 and not self.fit:
            raise ValueError('attach_mixture_w > 0 needs fit_lengths=True')

    def __call__(self, baseline, cand, fit=None, keep_state=True, catch_errors=False, clock_rate=None,
                 initial_state=None):
        if self.w > 0:
            return self._call_mixture(baseline, cand, fit, keep_state, catch_errors, clock_rate, initial_state)
        og = self.outgroup
        t0 = time.time()
        cand = baseline.canonical(cand)
        runs, rounds = [], []
        try:
            if initial_state is not None:
                start = copy.deepcopy(initial_state)
                start.set_root_state_messages(None)
            else:
                start = baseline.make_treeanc(baseline.rooted_phylo(cand), messages=False)
            if self.fit:
                lengths, pre = og.fit_lengths(start)
            else:
                lengths, pre = og.lengths.copy(), og.combined_score(start)
            del start
            b = baseline.with_root_state_messages(og.messages(lengths))
            run = evaluate_root_candidate(b, cand, fit, keep_state=True, catch_errors=False, clock_rate=clock_rate,
                                          initial_state=initial_state)
            runs.append(run)
            rounds.append({'round': 0, 'lengths': lengths.tolist(), 'pre_fit_score': pre, 'best_score': run.best_score})
            best_run, best_len = run, lengths
            for k in range(self.outer_iterations if self.fit else 0):
                st = best_run.state
                new_len, val = og.fit_lengths(st, x0=best_len)
                if val - best_run.best_score < self.tol:
                    rounds.append({'round': k + 1, 'lengths': new_len.tolist(), 'pre_fit_score': val,
                                   'best_score': None, 'stopped': 'gain %.3g < tol' % (val - best_run.best_score)})
                    break
                b = baseline.with_root_state_messages(og.messages(new_len))
                run = evaluate_root_candidate(b, cand, fit, keep_state=True, catch_errors=False,
                                              clock_rate=best_run.clock_rate, initial_state=st)
                runs.append(run)
                rounds.append({'round': k + 1, 'lengths': new_len.tolist(), 'pre_fit_score': val,
                               'best_score': run.best_score})
                if run.best_score > best_run.best_score:
                    best_run, best_len = run, new_len
            out = _merge_runs(runs, best_run, keep_state)
            out.extra = {'outgroup_lengths': best_len.tolist(), 'outgroup_edges': list(og.edge_names),
                         'outer_rounds': rounds, 'outer_rounds_performed': len(runs) - 1,
                         'outgroup_newick': og.newick(best_len)}
            out.initial_fraction = None if initial_state is not None else getattr(cand, 'fraction', None)
        except FIT_ERRORS as exc:
            if not catch_errors:
                raise
            out = FitRun(candidate=cand, candidate_id=baseline.candidate_id(cand),
                         initial_fraction=None if initial_state is not None else getattr(cand, 'fraction', None),
                         initial_mode='warm' if initial_state is not None else (fit.initial_mode if fit else 'marginal'))
            out.error = '%s: %s' % (type(exc).__name__, exc)
            out.best_score = -np.inf
        out.runtime = time.time() - t0
        return out


    def _mixture_round(self, baseline, cand, fit, clock_rate, state, mix_state):
        """One dated fit under the surrogate message built at ``mix_state``
        (dict: mix, lengths, values); returns (run, its mixture record)."""
        og, w = self.outgroup, self.w
        msg = mix_state['mix'].surrogate_messages(w, mix_state['lengths'], mix_state['values'])
        b = baseline.with_root_state_messages(msg)
        run = evaluate_root_candidate(b, cand, fit, keep_state=True, catch_errors=False, clock_rate=clock_rate,
                                      initial_state=state)
        mix = AttachmentMixture(run.state, og)
        at_fit = mix.score(w, [mix.value(j, th) for j, th in enumerate(mix_state['lengths'])]) \
            if mix.n_terms == mix_state['mix'].n_terms else -np.inf
        lengths, values = mix.fit(term_starts=mix_state['lengths'] if mix.n_terms == mix_state['mix'].n_terms
                                  else None, x0=mix_state['lengths'][0])
        rec = {'mix': mix, 'lengths': lengths, 'values': values, 'score': mix.score(w, values),
               'score_at_fit_lengths': at_fit, 'surrogate_best': run.best_score}
        run.best_score = rec['score']
        return run, rec

    @staticmethod
    def _start_run(baseline, cand, start, clock_rate, st0):
        """FitRun holding the warm start state itself, scored by its exact mixture."""
        from .root_candidates import _root_summary, fitted_newick
        run = FitRun(candidate=cand, candidate_id=baseline.candidate_id(cand), initial_fraction=None,
                     initial_mode='warm')
        st = copy.deepcopy(start)
        st.set_root_state_messages(None)
        run.scores, run.best_index, run.best_score = [st0['score']], 0, st0['score']
        run.clock_rate = clock_rate
        run.clock_rates = [clock_rate]
        run.root = _root_summary(baseline, cand, st)
        run.fitted_newick = fitted_newick(st)
        run.state = st
        run.trajectory_class = classify_trajectory(run.scores)
        return run

    def _call_mixture(self, baseline, cand, fit, keep_state, catch_errors, clock_rate, initial_state):
        og, w = self.outgroup, self.w
        t0 = time.time()
        cand = baseline.canonical(cand)
        runs, rounds = [], []
        try:
            if initial_state is not None:
                start = copy.deepcopy(initial_state)
                start.set_root_state_messages(None)
            else:
                start = baseline.make_treeanc(baseline.rooted_phylo(cand), messages=False)
            mix0 = AttachmentMixture(start, og)
            lengths, values = mix0.fit(x0=og.lengths)
            st0 = {'mix': mix0, 'lengths': lengths, 'values': values, 'score': mix0.score(w, values)}
            start_run = None
            if initial_state is not None:
                # a dated start state is a candidate state too: the score is never below its exact mixture
                start_run = self._start_run(baseline, cand, start, clock_rate, st0)
            del start
            run, rec = self._mixture_round(baseline, cand, fit, clock_rate, initial_state, st0)
            runs.append(run)
            rounds.append({'round': 0, 'pre_fit_score': st0['score'], 'best_score': rec['score'],
                           'surrogate_best': rec['surrogate_best']})
            best_run, best = run, rec
            if start_run is not None and st0['score'] > rec['score']:
                runs.insert(0, start_run)
                best_run, best = start_run, dict(st0, score_at_fit_lengths=st0['score'])
            for k in range(self.outer_iterations):
                gain = best['score'] - best['score_at_fit_lengths']
                if gain < self.tol:
                    rounds.append({'round': k + 1, 'best_score': None, 'stopped': 'gain %.3g < tol' % gain})
                    break
                run, rec = self._mixture_round(baseline, cand, fit, best_run.clock_rate, best_run.state, best)
                runs.append(run)
                rounds.append({'round': k + 1, 'pre_fit_score': best['score'], 'best_score': rec['score'],
                               'surrogate_best': rec['surrogate_best']})
                if rec['score'] > best_run.best_score:
                    best_run, best = run, rec
                else:  # the next round would start from the same state with the same message
                    rounds[-1]['stopped'] = 'no gain'
                    break
            # the kept state carries the plain message of the attachment at the root
            best_run.state.set_root_state_messages(og.messages(best['lengths'][0]))
            out = _merge_runs(runs, best_run, keep_state)
            m = best['mix']
            out.extra = {'outgroup_lengths': best['lengths'][0].tolist(), 'outgroup_edges': list(og.edge_names),
                         'outer_rounds': rounds, 'outer_rounds_performed': len(runs) - 1,
                         'outgroup_newick': og.newick(best['lengths'][0]), 'attach_mixture_w': w,
                         'mixture_at_root': float(best['values'][0]),
                         'mixture_neighbours': [float(v) for v in best['values'][1:]],
                         'mixture_neighbour_edges': [nb['edge'] for nb in m.neighbours],
                         'mixture_responsibilities': m.responsibilities(w, best['values']).tolist(),
                         'mixture_lengths': [th.tolist() for th in best['lengths']]}
            out.initial_fraction = None if initial_state is not None else getattr(cand, 'fraction', None)
        except FIT_ERRORS as exc:
            if not catch_errors:
                raise
            out = FitRun(candidate=cand, candidate_id=baseline.candidate_id(cand),
                         initial_fraction=None if initial_state is not None else getattr(cand, 'fraction', None),
                         initial_mode='warm' if initial_state is not None else (fit.initial_mode if fit else 'marginal'))
            out.error = '%s: %s' % (type(exc).__name__, exc)
            out.best_score = -np.inf
        out.runtime = time.time() - t0
        return out


def _merge_runs(runs, best_run, keep_state):
    first = runs[0]
    out = FitRun(candidate=first.candidate, candidate_id=first.candidate_id, initial_fraction=first.initial_fraction,
                 initial_mode=first.initial_mode, rate_fitted=first.rate_fitted)
    offset = 0
    for r in runs:
        for name in ('scores', 'cached_LH', 'positional_LH', 'clock_rates', 'ndiff', 'root_state_changes'):
            getattr(out, name).extend(getattr(r, name))
        out.iterations_performed += r.iterations_performed
        out.rate_rounds_performed += r.rate_rounds_performed
        if out.treetime_stop_iteration is None and r.treetime_stop_iteration is not None:
            out.treetime_stop_iteration = r.treetime_stop_iteration
        out.warnings.extend(w for w in r.warnings if w not in out.warnings)
        if r is best_run:
            out.best_index = offset + r.best_index
        offset += len(r.scores)
    out.best_score, out.clock_rate, out.root = best_run.best_score, best_run.clock_rate, best_run.root
    out.fitted_newick = best_run.fitted_newick
    out.state = best_run.state if keep_state else None
    out.tail_fluctuation = runs[-1].tail_fluctuation
    out.trajectory_class = classify_trajectory(out.scores)
    return out


# --------------------------------------------------------------------------
# attachment scan and the Stage 1 / Stage 3 driver
# --------------------------------------------------------------------------
def attachment_scan(baseline, outgroup):
    """Every placement of the unrooted ``baseline`` rooted at the baseline
    lengths, with the outgroup subtree attached at that root and its lengths
    fitted by ML there. The outgroup's default lengths are first initialised at
    the baseline's reference rooting (several stem starts), and finally reset to
    the lengths fitted at the best placement.

    Returns a list of dicts ``{candidate, score, lengths}``, best score first.
    Costs one TreeAnc and one length fit per placement."""
    ta = baseline.make_treeanc(baseline.rooted_phylo(None), messages=False)
    outgroup.initialise(ta)
    out = []
    for i, c in enumerate(baseline.all_candidates()):
        ta = baseline.make_treeanc(baseline.rooted_phylo(c), messages=False)
        lengths, score = outgroup.fit_lengths(ta)
        out.append((-score, i, {'candidate': c, 'score': score, 'lengths': lengths}))
    out.sort(key=lambda r: r[:2])
    out = [r[2] for r in out]
    outgroup.lengths = out[0]['lengths'].copy()
    outgroup.lengths0 = outgroup.lengths.copy()
    return out


class Stage3Result:
    """Stage 1 and Stage 3 on one candidate set, plus separately scored extra
    candidates (e.g. an oracle true root) that never enter the selection."""

    def __init__(self, stage1, stage3, rows, extra=None, scan=None, config=None):
        self.stage1, self.stage3, self.rows = stage1, stage3, rows
        self.extra = extra or []  # rows of the extra candidates (same columns)
        self.scan, self.config = scan, config or {}

    def summary(self):
        return {'stage1_best': self.stage1.best.candidate_id, 'stage1_margin': self.stage1.margin,
                'stage3_best': self.stage3.best.candidate_id, 'stage3_best_root': self.stage3.best.root_id,
                'stage3_margin': self.stage3.margin,
                'stage3_resolved': self.stage3.best_is_resolved,
                'n_candidates': len(self.rows), 'n_extra': len(self.extra)}

    def mixture_ranking(self, w):
        """[(candidate_id, attachment-mixture score at weight w)], best first
        (needs ``mixture_ws`` containing w)."""
        w = float(w)
        sc = [(r['candidate_id'], r['mixture'][w]) for r in self.rows if r['mixture'] and w in r['mixture']]
        return sorted(sc, key=lambda t: -t[1])

    def to_tsv(self, path):
        rows = self.rows + self.extra
        cols = list(rows[0].keys()) if rows else []
        with open(path, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in rows:
                fh.write('\t'.join('' if r[c] is None else (repr(r[c]) if isinstance(r[c], float) else str(r[c]))
                                   for c in cols) + '\n')


def _warm_from(stage1):
    from .root_candidates import candidate_key
    warm = {}
    for r in stage1.results:
        b = r.best_run
        if b is not None and b.error is None and b.state is not None:
            warm[candidate_key(r.candidate)] = (b.state, b.clock_rate)
    return warm


def _mixture(baseline, outgroup, states, ws, lengths):
    """Per w, the best attachment-mixture score over the given fitted states."""
    best = {}
    for st in states:
        sc, _ = attachment_mixture_scores(baseline, st, outgroup, ws=ws, lengths=lengths)
        for w, v in sc.items():
            best[w] = max(best.get(w, -np.inf), v)
    return best


def _rows(stage1, stage3, outgroup, is_extra=False, baseline=None, mixture_ws=()):
    from .root_candidates import candidate_key, format_source_ranks, fresh_marginal_score
    by3 = {candidate_key(r.candidate): r for r in stage3.results}
    rows = []
    for r1 in stage1.results:
        r3 = by3.get(candidate_key(r1.candidate))
        row = {'candidate_id': r1.candidate_id, 'source': 'extra' if is_extra else r1.source,
               'source_ranks': format_source_ranks(r1.source_ranks), 'regression_rank': r1.regression_rank,
               'ingroup_1': r1.score, 'combined_3': None, 'ingroup_3': None, 'outgroup_3': None,
               'posthoc_3': None, 'outgroup_lengths': None, 'stem': None, 'outer_rounds': None, 'run_3': None,
               'root_date_1': r1.best_run.root.get('root_date') if r1.best_run else None, 'root_date_3': None,
               'runtime_1': sum(x.runtime for x in r1.runs), 'runtime_3': None, 'mixture': None}
        if r1.best_run is not None and r1.best_run.state is not None:
            # post hoc: Stage 1 state with the outgroup lengths fitted at it (no refit of the ingroup)
            row['posthoc_3'] = outgroup.fit_lengths(r1.best_run.state)[1]
        if r3 is not None and np.isfinite(r3.score):
            b = r3.best_run
            row.update(combined_3=r3.score, outgroup_lengths=b.extra.get('outgroup_lengths'),
                       stem=b.extra.get('outgroup_lengths', [None])[0], outer_rounds=b.extra.get('outer_rounds_performed'),
                       run_3=b.initial_mode, root_date_3=b.root.get('root_date'),
                       runtime_3=sum(x.runtime for x in r3.runs))
            if b.state is not None:
                st = copy.deepcopy(b.state)
                fresh_marginal_score(st)
                row.update(ingroup_3=float(st.tree.total_sequence_LH_ingroup),
                           outgroup_3=float(st.tree.total_root_message_LH))
        if mixture_ws:
            # post hoc at the Stage 1 and Stage 3 best states of this candidate; the better state counts per w
            states = [r.best_run.state for r in (r1, r3) if r is not None and r.best_run is not None
                      and r.best_run.error is None and r.best_run.state is not None]
            x0 = row['outgroup_lengths']
            row['mixture'] = _mixture(baseline, outgroup, states, mixture_ws, x0) if states else None
        rows.append(row)
    return rows


def rank_with_outgroup_subtree(baseline, outgroup, explicit=(), regression_k=5, attachment='scan',
                               attach_k=3, extra_candidates=(), stage1=None, warm_start=True, outer_iterations=2,
                               tol=1e-2, fit_lengths=True, mixture_ws=(), attach_mixture_w=0.0, **rank_kwargs):
    """Stage 1 and Stage 3 (outgroup subtree, lengths fitted per candidate) on
    one candidate set.

    Candidates as in :func:`~treetime.root_messages.rank_with_root_messages`:
    ``explicit``, the best ``attach_k`` placements of ``attachment``
    (``'scan'``: :func:`attachment_scan` of this outgroup subtree; or a list of
    placements, best first; label 'attach') and the regression sources
    (covariation top ``regression_k`` when attachment candidates are used,
    unless ``rank_kwargs`` names sources). Stage 1 (or the given ``stage1``,
    which must have been run on the same candidates with ``keep_states=True``)
    and Stage 3 see the same candidate list.

    Stage 3 ranks with :class:`OutgroupEvaluator` (
    ``outer_iterations``, ``tol``, ``fit_lengths``) through the unchanged
    :func:`rank_root_candidates`; with ``warm_start`` each candidate also gets a
    trajectory from its Stage 1 best state, so its Stage 3 score is at least the
    post hoc score of that state (outgroup lengths fitted at it).

    ``extra_candidates`` (e.g. the true root in a benchmark) are fitted with the
    same Stage 1 and Stage 3 procedure but outside the ranking: they appear in
    ``Stage3Result.extra``, never in ``stage3.results``, and cannot change the
    selected root. An extra candidate already on the shortlist is reported with
    its shortlist fits.

    ``attach_mixture_w``: attachment-mixture weight inside the Stage 3 dated
    fit (:class:`OutgroupEvaluator`); 0 (default) is the unmixed Stage 3. With
    w > 0 the Stage 3 scores and ``combined_3`` are mixture log-likelihoods,
    while ``ingroup_3`` + ``outgroup_3`` and ``posthoc_3`` are the attachment
    at the root only (compare ``combined_3`` with ``mixture[w]``, not with
    ``posthoc_3``).

    ``mixture_ws``: attachment-mixture weights w scored post hoc
    (:func:`attachment_mixture_scores`) at each candidate's Stage 1 and Stage 3
    best states, the better per w; in ``rows[i]['mixture']`` and
    :meth:`Stage3Result.mixture_ranking`. They do not change the Stage 3 answer.
    """
    from .root_candidates import (candidate_key, outgroup_candidate_sources, rank_root_candidates)
    keep = rank_kwargs.pop('keep_states', True)
    if 'ranked_sources' in rank_kwargs:
        raise ValueError('rank_with_outgroup_subtree: pass attachment candidates as attachment=')
    scan = None
    if isinstance(attachment, str):
        if attachment != 'scan':
            raise ValueError("attachment must be 'scan' or a list of placements")
        scan = attachment_scan(baseline, outgroup) if attach_k else None
        attachment = [s['candidate'] for s in scan] if scan else []
    attach = list(attachment)[:attach_k] if attach_k else []
    rank_kwargs['ranked_sources'] = {'attach': attach} if attach else None
    if attach and all(rank_kwargs.get(a) is None for a in ('candidate_sources', 'covariation', 'tip_slack')):
        rank_kwargs['candidate_sources'] = outgroup_candidate_sources(regression_k) if regression_k else ()
    if stage1 is None:
        stage1 = rank_root_candidates(baseline, explicit=explicit, regression_k=regression_k, keep_states=True,
                                      **rank_kwargs)
    elif any(r.best_run is not None and r.best_run.error is None and r.best_run.state is None for r in stage1.results):
        raise ValueError('rank_with_outgroup_subtree: the given stage1 result kept no states')
    ev = OutgroupEvaluator(outgroup, outer_iterations=outer_iterations, tol=tol, fit_lengths=fit_lengths,
                           attach_mixture_w=attach_mixture_w)
    s3 = rank_root_candidates(baseline, explicit=explicit, regression_k=regression_k, keep_states=True,
                              warm_starts=_warm_from(stage1) if warm_start else None, evaluator=ev, **rank_kwargs)
    k1 = {candidate_key(r.candidate) for r in stage1.results}
    k3 = {candidate_key(r.candidate) for r in s3.results}
    if k1 != k3:
        raise ValueError('rank_with_outgroup_subtree: Stage 1 and Stage 3 candidate sets differ '
                         '(%d only in Stage 1, %d only in Stage 3)' % (len(k1 - k3), len(k3 - k1)))
    mixture_ws = tuple(float(w) for w in mixture_ws)
    rows = _rows(stage1, s3, outgroup, baseline=baseline, mixture_ws=mixture_ws)
    # extra candidates: same fits, outside the ranking
    extra_rows = []
    listed = {candidate_key(baseline.canonical(c)) for c in extra_candidates} & k1
    new = [c for c in extra_candidates if candidate_key(baseline.canonical(c)) not in k1]
    for r1, row in zip(stage1.results, rows):  # rows follow stage1.results
        if candidate_key(r1.candidate) in listed:
            extra_rows.append(dict(row, source='extra(listed)'))
    if new:
        xkw = {k: v for k, v in rank_kwargs.items() if k not in ('ranked_sources', 'candidate_sources',
                                                                 'covariation', 'tip_slack', 'final_rate_refit',
                                                                 'final_rate_refit_window')}
        x1 = rank_root_candidates(baseline, explicit=new, regression_k=0, keep_states=True, candidate_sources=(),
                                  **xkw)
        x3 = rank_root_candidates(baseline, explicit=new, regression_k=0, keep_states=True, candidate_sources=(),
                                  warm_starts=_warm_from(x1) if warm_start else None, evaluator=ev, **xkw)
        extra_rows += _rows(x1, x3, outgroup, is_extra=True, baseline=baseline, mixture_ws=mixture_ws)
    if not keep:
        for res in (stage1, s3):
            for r in res.results:
                for x in r.runs:
                    x.state = None
            if res.final_rate_refit is not None:
                for c in [res.final_rate_refit] + (res.final_rate_refit.get('close') or []):
                    c['state'] = None
    config = {'outer_iterations': outer_iterations, 'tol': tol, 'fit_lengths': fit_lengths,
              'attach_k': attach_k, 'regression_k': regression_k, 'warm_start': warm_start, 'mixture_ws': mixture_ws,
              'attach_mixture_w': float(attach_mixture_w),
              'outgroup_newick': outgroup.newick(), 'n_outgroup_patterns': outgroup.n_patterns}
    return Stage3Result(stage1, s3, rows, extra=extra_rows, scan=scan, config=config)


# --------------------------------------------------------------------------
# outgroup topology from a joint (ingroup + outgroup) tree
# --------------------------------------------------------------------------
def outgroup_tree_from_joint(tree, outgroup_names, attach_name='ATTACH'):
    """Outgroup subtree induced by an unrooted joint tree of ingroup and outgroup
    leaves, for ``OutgroupSubtree(tree=...)``.

    The attachment is the ingroup end of the edge that best separates the
    ingroup from the outgroups (fewest leaves on the wrong side; a monophyletic
    ingroup has none). A leaf ``attach_name`` is added there with length 0, every
    ingroup leaf is pruned, and degree-2 nodes are merged, so the stem starts at
    the full length of the separating edge and the outgroup edges at their joint-
    tree lengths. Outgroups nested inside the ingroup stay where the induced
    subtree puts them. The rooting of ``tree`` is irrelevant.

    Returns a newick string (rooted at the attachment's neighbour; the attachment
    leaf makes the rooting irrelevant to :class:`OutgroupSubtree`)."""
    tree = _read_tree(tree)
    ogs = set(outgroup_names)
    adj, label = {}, {}
    ids = {}

    def nid(c):
        if id(c) not in ids:
            ids[id(c)] = len(ids)
            adj[ids[id(c)]] = {}
            label[ids[id(c)]] = c.name if c.is_terminal() else None
        return ids[id(c)]
    for c in tree.find_clades(order='preorder'):
        for ch in c.clades:
            a, b = nid(c), nid(ch)
            adj[a][b] = adj[b][a] = float(ch.branch_length or 0.0)
    nid(tree.root)
    leaves = {k: label[k] for k in adj if label[k] is not None and len(adj[k]) <= 1}
    missing = ogs - set(leaves.values())
    if missing:
        raise ValueError('outgroup_tree_from_joint: outgroups missing from the tree: %s' % sorted(missing))
    n_in = sum(1 for v in leaves.values() if v not in ogs)
    if n_in == 0:
        raise ValueError('outgroup_tree_from_joint: the tree has no ingroup leaves')

    # leaf counts below each directed edge (k -> its side away from p)
    memo = {}

    def below(k, p):
        if (k, p) not in memo:
            if k in leaves and p is not None:
                memo[(k, p)] = (0, 1) if leaves[k] in ogs else (1, 0)
            else:
                i = o = 0
                for w in adj[k]:
                    if w != p:
                        a, b = below(w, k)
                        i, o = i + a, o + b
                memo[(k, p)] = (i, o)
        return memo[(k, p)]
    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * len(adj) + 1000))
    best = None
    for u in adj:
        for x in adj[u]:
            i_x, o_x = below(x, u)            # the side of x
            i_u, o_u = n_in - i_x, len(ogs) - o_x  # the side of u (the ingroup side)
            cost = o_u + i_x                  # outgroups on the ingroup side + ingroup leaves on the outgroup side
            if o_x == 0:
                continue
            key = (cost, -i_u, adj[u][x], u, x)
            if best is None or key < best:
                best = key
    _, _, _, u, x = best
    a = max(adj) + 1
    adj[a], label[a] = {u: 0.0}, attach_name
    adj[u][a] = 0.0
    keep = ogs | {attach_name}
    # prune ingroup leaves and dangling internal nodes, then merge degree-2 nodes
    changed = True
    while changed:
        changed = False
        for k in list(adj):
            if len(adj[k]) <= 1 and label[k] not in keep:
                for w in adj[k]:
                    del adj[w][k]
                del adj[k]
                changed = True
    for k in [k for k in adj if len(adj[k]) == 2 and label[k] not in keep]:
        (p, lp), (q, lq) = adj[k].items()
        del adj[p][k], adj[q][k], adj[k]
        adj[p][q] = adj[q][p] = lp + lq
    (top,) = adj[a]

    def nw(k, p):
        kids = [w for w in sorted(adj[k], key=lambda w: str(label[w])) if w != p]
        s = label[k] if not kids else '(' + ','.join(nw(w, k) for w in kids) + ')'
        return '%s:%.10g' % (s, adj[k][p])
    kids = [w for w in adj[top]]
    return '(' + ','.join(nw(w, top) for w in kids) + ');'


def joint_tree_cmaple(sequences, iqtree='iqtree3', model='GTR', seed=1):
    """Quick joint tree (IQ-TREE's CMAPLE mode, ``--pathogen-force``) on
    ``sequences`` (name -> aligned sequence, ingroup and outgroups). Returns a
    Bio.Phylo tree. Needs an IQ-TREE >= 2.3 binary on PATH (``iqtree``)."""
    import os
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'aln.fasta'), 'w') as fh:
            for k, s in sequences.items():
                fh.write('>%s\n%s\n' % (k, s))
        cmd = [iqtree, '--pathogen-force', '-s', 'aln.fasta', '-m', model, '--prefix', 'j', '-T', '1',
               '-seed', str(seed), '-redo']
        p = subprocess.run(cmd, cwd=td, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError('joint_tree_cmaple: %s failed: %s' % (iqtree, p.stdout[-2000:]))
        return Phylo.read(os.path.join(td, 'j.treefile'), 'newick')


# --------------------------------------------------------------------------
# attachment mixture (post hoc): uncertainty about where the outgroup joins
# --------------------------------------------------------------------------
def _collapsed_side(tt, kids):
    """For a bifurcating root, the index (0/1) of the child the root has
    collapsed onto (its root edge within COLLAPSE_TOL expected mutations, the
    shorter one), else None. Such a state is that child's node-rooted tree,
    and gets that node's neighbours."""
    from .root_candidates import COLLAPSE_TOL
    if len(kids) != 2:
        return None
    n = tt.data.full_length
    ls = [float(c.branch_length) for c in kids]
    k = int(np.argmin(ls))
    return k if ls[k] * n <= COLLAPSE_TOL else None


def _unrooted_from_state(tt):
    """Adjacency {node index: {node index: length}} of a fitted ingroup state at
    the lengths its likelihood uses; a bifurcating root is suppressed. Returns
    (adj, names, neighbour edges of the root position, root edge or None).
    Neighbours: for a root inside edge (u, v), the other edges at u and at v;
    for a root at a node, its incident edges; a root collapsed onto u (within
    COLLAPSE_TOL expected mutations) counts as the node root at u."""
    nodes = list(tt.tree.find_clades(order='preorder'))
    idx = {id(n): i for i, n in enumerate(nodes)}
    adj = {i: {} for i in range(len(nodes))}
    for n in nodes:
        for c in n.clades:
            t = float(tt._branch_length_to_gtr(c))
            adj[idx[id(n)]][idx[id(c)]] = adj[idx[id(c)]][idx[id(n)]] = t
    names = {idx[id(n)]: n.name for n in nodes if n.is_terminal()}
    kids = list(adj[0])
    if len(kids) == 2:  # root inside edge (u, v): neighbours are the other edges at u and at v
        side = _collapsed_side(tt, tt.tree.root.clades)
        u, v = kids
        tu, tv = adj[0][u], adj[0][v]
        del adj[u][0], adj[v][0], adj[0]
        adj[u][v] = adj[v][u] = tu + tv
        if side is not None:  # collapsed onto one child: that node's incident edges
            x = kids[side]
            return adj, names, [(x, w) for w in adj[x]], (u, v, tu / (tu + tv) if tu + tv > 0 else 0.5)
        nb = [(u, w) for w in adj[u] if w != v] + [(v, w) for w in adj[v] if w != u]
        return adj, names, nb, (u, v, tu / (tu + tv) if tu + tv > 0 else 0.5)
    return adj, names, [(0, w) for w in kids], None


def _rooted_on_edge(adj, names, a, b, frac=0.5):
    from Bio.Phylo.BaseTree import Clade, Tree

    def build(v, p, bl):
        return Clade(name=names.get(v), branch_length=bl,
                     clades=[build(w, v, adj[v][w]) for w in sorted(adj[v]) if w != p])
    t = adj[a][b]
    root = Clade(branch_length=0.0, clades=[build(a, b, t * frac), build(b, a, t * (1 - frac))])
    return Tree(root=root, rooted=True)


def attachment_mixture_scores(baseline, tt, outgroup, ws=(0.0, 0.1, 0.3), lengths=None, fit=True):
    """Attachment-mixture log-likelihoods of a fitted ingroup state ``tt`` rooted
    at candidate R (post hoc: the ingroup state is not refitted).

        L_w = (1 - w) L(outgroup at R) + w * mean_e L(outgroup at the midpoint of e)

    over the edges e next to R: for R inside edge (u, v), every other edge at u
    or v; for R at a node, its incident edges. Each term is the joint likelihood
    of the ingroup at ``tt``'s branch lengths (rerooting a reversible tree leaves
    it unchanged) and of the outgroup subtree attached there, with the outgroup
    lengths fitted for that attachment (starting from ``lengths``, default
    ``outgroup.lengths``; ``fit=False`` keeps those lengths for every term).
    w = 0 is the post-hoc Stage 3 score of the state. The
    mixture is a proper likelihood over attachment points, and it bounds the
    penalty of a candidate whose own attachment the outgroup contradicts by
    log(w / n_e) plus its best neighbour's term.

    Returns ({w: log L_w}, {'at_root', 'neighbours', 'n_neighbours'})."""
    from scipy.special import logsumexp
    base = copy.deepcopy(tt)
    base.set_root_state_messages(None)
    x0 = outgroup.lengths if lengths is None else np.asarray(lengths, dtype=float)
    th_r, s_r = outgroup.fit_lengths(base, x0=x0) if fit else (x0, outgroup.combined_score(base, x0))
    adj, names, nb, _ = _unrooted_from_state(base)
    s_nb = []
    for a, b in nb:
        ta = baseline.make_treeanc(_rooted_on_edge(adj, names, a, b), messages=False)
        s_nb.append(outgroup.fit_lengths(ta, x0=th_r)[1] if fit else outgroup.combined_score(ta, th_r))
    out = {}
    for w in ws:
        w = float(w)
        if w == 0.0 or not s_nb:
            out[w] = s_r
        elif w == 1.0:
            out[w] = float(logsumexp([np.log(1.0 / len(s_nb)) + s for s in s_nb]))
        else:
            out[w] = float(logsumexp([np.log1p(-w) + s_r] + [np.log(w / len(s_nb)) + s for s in s_nb]))
    return out, {'at_root': s_r, 'neighbours': s_nb, 'n_neighbours': len(s_nb)}


class AttachmentMixture:
    """Attachment mixture of one fitted ingroup state, by message transport.

    For the state ``tt`` rooted at R, term 0 is the outgroup attached at R and
    term j >= 1 the outgroup attached at the midpoint of the j-th neighbouring
    edge (the edges of :func:`attachment_mixture_scores`: for R inside edge
    (u, v), the other edges at u and v; for R at a node, its incident edges).
    Each term is written as a root-state message at R,

        m~_j(a) = P(ingroup, outgroup | R = a, attachment j) / P(ingroup | R = a),

    computed from the state's subtree partials, so that
    sum_sites log sum_a pi_a l(a) m~_j(a) is that term's joint likelihood
    exactly at this state (no rerooted TreeAnc needed). For neighbour edge
    (p, c) with c below p, X its midpoint, t its length and l_c the partials of
    c: A(b) = sum_x P_bx(t/2) m(x) [P(t/2) l_c](x) and B(b) = [P(t) l_c](b) are
    the edge's contributions to p with and without the outgroup at X; then
    m~ = A / B if p = R, else m~ = [P(t_p) (l_p A / B)] / [P(t_p) l_p] with t_p
    the root edge above p.

    :meth:`fit` fits the outgroup lengths per term, :meth:`score` gives
    log[(1 - w) L_0 + w mean_j L_j], and :meth:`surrogate_messages` the
    root-state message that the dated fit uses under the mixture.
    """

    def __init__(self, tt, outgroup):
        base = copy.deepcopy(tt)
        base.set_root_state_messages(None)
        self.og = outgroup
        self.G = G = outgroup._groups(base)  # also runs the postorder pass
        cp = np.asarray(base.data.full_to_compressed_sequence_map)
        _, self.group_of_position = np.unique(np.stack([cp, outgroup.pattern_of_position]), axis=1,
                                              return_inverse=True)
        self.group_of_position = np.asarray(self.group_of_position).ravel()
        E = base.gtr.expQt
        root = base.tree.root
        kids = list(root.clades)
        side = _collapsed_side(base, kids)
        if side is not None:  # root collapsed onto one child x: x's incident edges (as _unrooted_from_state)
            x, y = kids[side], kids[1 - side]
            pairs = [(x, w) for w in x.clades] + [(root, y)]
        elif len(kids) == 2:  # root inside an edge: neighbours are the child edges of both root children
            pairs = [(u, w) for u in kids for w in u.clades]
        else:
            pairs = [(root, w) for w in kids]
        floor = ttconf.MIN_BRANCH_LENGTH * base.one_mutation  # each half-edge floored, as a rerooted TreeAnc does
        self.neighbours = []
        for p, c in pairs:
            t = float(base._branch_length_to_gtr(c))
            lc = c.marginal_subtree_LH[G['c']]
            if p is root and side is not None:
                # the edge x - y through the root: its midpoint X lies between the root and y, at h - t_x
                # from the root and h from y (h half of t_x + t_y)
                tx = float(base._branch_length_to_gtr(kids[side]))
                h = max((tx + t) / 2.0, floor)
                nb = {'edge': (kids[side].name, c.name), 'B': lc @ E(t), 'down': lc @ E(h), 'up': E(max(h - tx, 0.0))}
            else:
                h = max(t / 2.0, floor)
                nb = {'edge': (p.name, c.name), 'B': lc @ E(t), 'down': lc @ E(h), 'up': E(h)}
            if p is not root:
                tp = float(base._branch_length_to_gtr(p))
                lp = p.marginal_subtree_LH[G['c']]
                nb.update(lp=lp, Ep=E(tp), den=lp @ E(tp))
            self.neighbours.append(nb)
        self.n_terms = 1 + len(self.neighbours)

    def _transport(self, j, pattern_msg):
        m = pattern_msg[self.G['o']]
        if j == 0:
            return m
        nb = self.neighbours[j - 1]
        ratio = ((m * nb['down']) @ nb['up']) / nb['B']
        if 'lp' not in nb:
            return ratio
        return ((nb['lp'] * ratio) @ nb['Ep']) / nb['den']

    def site_messages(self, j, lengths):
        """(groups, K) transported message of term j at outgroup lengths ``lengths``."""
        return self._transport(j, self.og.pattern_messages(lengths))

    def _group_value(self, mt):
        G = self.G
        return float(np.sum(G['n'] * (G['pref'] + np.log(np.sum(G['pi_l'] * mt, axis=1)))))

    def value(self, j, lengths):
        """Joint log-likelihood of term j at outgroup lengths ``lengths``."""
        return self._group_value(self.site_messages(j, lengths))

    def fit(self, x0=None, term_starts=None):
        """Outgroup lengths fitted per term by ML. Term 0 starts from ``x0``
        (default the outgroup's lengths), each neighbour from term 0's fit;
        or ``term_starts``, one start per term (term 0's fit is then an extra
        start for the neighbours). Never worse than the start. Returns
        (lengths per term, values)."""
        per_term = term_starts is not None and len(term_starts) == self.n_terms
        lengths, values = [], []
        for j in range(self.n_terms):
            xj = term_starts[j] if per_term else (x0 if j == 0 else lengths[0])
            more = [lengths[0]] if per_term and j > 0 else None
            th, v = self.og.maximise(lambda msg, j=j: self._group_value(self._transport(j, msg)),
                                     x0=xj, starts=more)
            lengths.append(th)
            values.append(v)
        return lengths, values

    def log_weights(self, w):
        n = len(self.neighbours)
        if not n or w == 0.0:
            return np.array([0.0] + [-np.inf] * n)
        if w == 1.0:
            return np.array([-np.inf] + [np.log(1.0 / n)] * n)
        return np.array([np.log1p(-w)] + [np.log(w / n)] * n)

    def score(self, w, values):
        """log[(1 - w) L_0 + w mean_j L_j] from the term values."""
        from scipy.special import logsumexp
        return float(logsumexp(self.log_weights(float(w)) + np.asarray(values, dtype=float)))

    def responsibilities(self, w, values):
        lw = self.log_weights(float(w)) + np.asarray(values, dtype=float)
        return np.exp(lw - self.score(w, values))

    def surrogate_messages(self, w, lengths, values):
        """(full_length, K) root-state messages for the dated fit under the mixture.

        mu(a) = sum_j r_j m~_j(a) l_in / l_j per site, with r_j the term
        responsibilities at this state, l_j = sum_a pi_a l(a) m~_j(a) the
        term's site likelihood and l_in = sum_a pi_a l(a) the ingroup's. At this state the combined log-likelihood with
        mu equals the ingroup log-likelihood, and with the m~_j held fixed its
        derivative in an ingroup branch length is sum_j r_j dl_j / l_j per site,
        the derivative of the EM objective sum_j r_j log L_j, except that term
        j's part is approximate for the branches of neighbour j's own edge and
        the subtree below it and, when the edge hangs below a root child p, for
        the root edge above p and p's other subtrees: there the outgroup's pull
        is frozen at this state. So the dated fit with
        mu is a first-order M-step weighted by the responsibilities, exact for
        term 0 (the attachment at the root). The evaluator rebuilds mu at each
        new state and scores every state by the exact mixture. With all
        responsibility on term 0, mu is the plain Stage 3 message up to a
        per-site constant."""
        r = self.responsibilities(w, values)
        l_in = np.sum(self.G['pi_l'], axis=1)
        mu = 0.0
        for j in range(self.n_terms):
            if r[j] == 0.0:
                continue
            mt = self.site_messages(j, lengths[j])
            lj = np.sum(self.G['pi_l'] * mt, axis=1)
            mu = mu + r[j] * mt * (l_in / lj)[:, None]
        return np.asarray(mu)[self.group_of_position]
