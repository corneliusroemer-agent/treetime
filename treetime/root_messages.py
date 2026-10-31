"""External evidence on the root state (likelihood rooting, Stage 2).

A root-state message for alignment position i is the vector of likelihoods

    m_i(a) = P(D_ext,i | R_i = a),   a in gtr.alphabet,

of some external data (e.g. an outgroup) given the state of the root. Messages
are likelihoods, not posteriors: they multiply the root prior pi exactly once,

    L_i = sum_a pi_a l_i(a) m_i(a),

where l_i(a) is the ingroup likelihood of the subtree below the root. Neutral
messages (m = 1) leave every number unchanged.

Two uses:

* **post hoc (2a)** -- :func:`root_message_correction` scores a fitted ingroup
  tree with the messages, without refitting anything. It uses a grouped
  sidecar: positions are grouped by (compressed ingroup pattern, message), so
  the ingroup compression is untouched.
* **propagated (2b)** -- :meth:`TreeAnc.set_root_state_messages` attaches the
  messages to a TreeAnc/TreeTime object; the compressed patterns are refined so
  that each carries one message, and the marginal reconstruction (root
  posterior, internal profiles, branch profiles, hence date and rate fits)
  uses them.

Normalisation: a message's absolute scale enters the score (it is part of the
joint likelihood of ingroup and external data). For a fixed message set this is
the same constant for every candidate, so differences between candidates are
unaffected; it matters only when comparing scores across message sets, which
these functions never do.
"""
import copy

import numpy as np

from .root_candidates import fresh_marginal_score
from .seq_utils import normalize_profile, seq2prof


def normalise_root_messages(messages, full_length, n_states):
    """Validate messages and group identical ones.

    Returns ``keys`` (int per full position; 0 = neutral), ``table`` (one
    max-normalised message per key, row 0 = ones) and ``log_scale`` (log of each
    message's maximum, 0 for key 0).
    """
    if isinstance(messages, dict):
        items = [(int(p), np.asarray(v, dtype=float)) for p, v in messages.items()]
    else:
        arr = np.asarray(messages, dtype=float)
        if arr.shape != (full_length, n_states):
            raise ValueError('root-state messages: expected shape (%d, %d), got %s' % (full_length, n_states, arr.shape))
        items = list(enumerate(arr))
    keys = np.zeros(full_length, dtype=int)
    rows, scales, index = [np.ones(n_states)], [0.0], {}
    for p, v in items:
        if not 0 <= p < full_length:
            raise ValueError('root-state messages: position %d outside the alignment (length %d)' % (p, full_length))
        if v.shape != (n_states,):
            raise ValueError('root-state messages: position %d needs %d values (gtr alphabet), got %s' % (p, n_states, v.shape))
        if not np.all(np.isfinite(v)) or np.any(v < 0) or not v.max() > 0:
            raise ValueError('root-state messages: position %d must be finite, non-negative and not all zero' % p)
        if np.all(v == 1.0):
            continue
        k = v.tobytes()
        if k not in index:
            index[k] = len(rows)
            mx = v.max()
            rows.append(v / mx)
            scales.append(float(np.log(mx)))
        keys[p] = index[k]
    return keys, np.array(rows), np.array(scales)


def outgroup_leaf_messages(gtr, seq, d, positions=None):
    """Messages of a single outgroup leaf attached to the root by a branch of
    length ``d``: m_i(a) = sum_o p_i(o) P(o | a, d), where p_i is the profile of
    the outgroup character (``gtr.profile_map``). An ambiguous character whose
    profile is all ones gives an exactly neutral message. Returns a
    (len(seq), n_states) array; with ``positions``, only those rows are kept
    and the others set to neutral.

    This is the message a Felsenstein pass over the tree with the outgroup leaf
    sends to the root (TreeTime's ``expQt(t)[i, j]`` = P(i | j, t)).
    """
    prof = seq2prof(np.array(list(seq)) if isinstance(seq, str) else np.asarray(seq), gtr.profile_map)
    m = prof @ gtr.expQt(d)
    m[np.all(prof == 1.0, axis=1)] = 1.0
    if positions is not None:
        keep = np.zeros(len(m), dtype=bool)
        keep[np.asarray(positions, dtype=int)] = True
        m[~keep] = 1.0
    return m


def estimate_outgroup_distance(tt, seq, bounds=(1e-6, 2.0), xatol=1e-5):
    """Maximum-likelihood length of the branch joining a single outgroup leaf
    ``seq`` to the root of the fitted ingroup tree ``tt`` (no messages of its
    own), at ``tt``'s branch lengths: maximises the post-hoc combined score
    (:func:`root_message_correction` of :func:`outgroup_leaf_messages`) over d.
    Uses no information beyond the data and the fitted tree; the value depends
    on the rooting of ``tt``. Returns (d, combined score at d)."""
    from scipy.optimize import minimize_scalar
    fresh_marginal_score(tt)
    post = tt.tree.root.marginal_profile
    pat = tt.data.full_to_compressed_sequence_map
    prof = seq2prof(np.array(list(seq)) if isinstance(seq, str) else np.asarray(seq), tt.gtr.profile_map)
    known = ~np.all(prof == 1.0, axis=1)
    p, pr = post[pat[:len(prof)]][known], prof[known]

    def neg(logd):
        m = pr @ tt.gtr.expQt(float(np.exp(logd)))
        return -float(np.sum(np.log(np.sum(p * m, axis=1))))
    r = minimize_scalar(neg, bounds=tuple(np.log(bounds)), method='bounded', options={'xatol': xatol})
    d = float(np.exp(r.x))
    return d, root_message_correction(tt, outgroup_leaf_messages(tt.gtr, seq, d))['combined']


def root_message_correction(tt, messages):
    """Post-hoc (2a) score of a fitted ingroup tree with root-state messages.

    ``tt`` must carry no messages of its own. Recomputes the ingroup marginal
    likelihood at the current branch lengths (fresh postorder pass; no RNG use,
    no sequence reassignment) and adds, over positions grouped by (compressed
    pattern, message), count x log sum_a P_ingroup(R = a | pattern) m(a). The
    ingroup compression is not changed.

    Returns a dict with ``ingroup``, ``correction``, ``combined`` and
    ``n_groups`` (the number of sidecar groups).
    """
    if tt.root_state_message is not None:
        raise ValueError('root_message_correction: the tree already carries root-state messages')
    keys, table, log_scale = normalise_root_messages(messages, tt.data.full_length, len(tt.gtr.alphabet))
    ingroup = fresh_marginal_score(tt)
    post = tt.tree.root.marginal_profile  # pi * l, normalised per compressed pattern
    pos = np.nonzero(keys)[0]
    if len(pos) == 0:
        return {'ingroup': ingroup, 'correction': 0.0, 'combined': ingroup, 'n_groups': 0}
    pat = tt.data.full_to_compressed_sequence_map[pos]
    groups, counts = np.unique(np.stack([pat, keys[pos]]), axis=1, return_counts=True)
    p, k = groups
    corr = float(np.sum(counts * (np.log(np.sum(post[p] * table[k], axis=1)) + log_scale[k])))
    return {'ingroup': ingroup, 'correction': corr, 'combined': ingroup + corr, 'n_groups': int(len(counts))}


def root_state_posterior(tt, positions):
    """Root-state posterior (rows in gtr.alphabet order) at full-alignment
    ``positions``, from the root profile of the last likelihood computation."""
    cpos = tt.data.full_to_compressed_sequence_map[np.asarray(positions, dtype=int)]
    return np.array(tt.tree.root.marginal_profile[cpos])


def ingroup_root_posterior(tt, positions):
    """Root-state posterior at ``positions`` from the ingroup alone (pi * l),
    whether or not ``tt`` carries messages."""
    cpos = tt.data.full_to_compressed_sequence_map[np.asarray(positions, dtype=int)]
    return normalize_profile(tt.gtr.Pi[None, :] * tt.tree.root.marginal_subtree_LH[cpos])[0]


def score_tree_with_messages(baseline, tree, messages):
    """Combined marginal log-likelihood of the baseline alignment on a rooted
    Bio.Phylo ``tree`` at exactly its branch lengths, messages at its root
    (2b representation: refined patterns). Returns (combined, ingroup, correction)."""
    from .treeanc import TreeAnc
    ta = TreeAnc(tree=copy.deepcopy(tree), aln=baseline.aln, gtr=copy.deepcopy(baseline.gtr), verbose=0,
                 rng_seed=baseline.rng_seed)
    ta.set_root_state_messages(messages)
    s = fresh_marginal_score(ta)
    return s, float(ta.tree.total_sequence_LH_ingroup), float(ta.tree.total_root_message_LH)
