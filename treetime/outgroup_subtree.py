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

Compression: the messages depend on an alignment column only through its
outgroup pattern, so they are computed once per distinct outgroup pattern and
indexed back by position. Positions with equal outgroup patterns get
byte-identical messages, so the Stage 2 pattern refinement groups ingroup-
constant columns by outgroup pattern with their multiplicities, as plan.md asks;
the length fit uses the same (ingroup pattern, outgroup pattern) groups.

"""
import copy

import numpy as np
from Bio import Phylo

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
        from scipy.optimize import minimize, minimize_scalar
        G = self._groups(tt)
        lo, hi = np.log(self.bounds[0]), np.log(self.bounds[1])

        def f(x):
            return -self._value(G, self.pattern_messages(np.exp(x)))
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
