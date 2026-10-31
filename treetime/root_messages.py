"""External evidence on the root state: per-site root-state likelihood messages.

A message for alignment position i is the vector

    m_i(a) = P(D_ext,i | R_i = a),   a in the GTR alphabet,

a likelihood conditional on the root state (not a posterior, and not
normalised). Positions without a message are neutral (m_i = 1). The combined
per-site likelihood at the root is

    L_i = sum_a pi_a l_i(a) m_i(a),

with pi the ordinary root prior and l_i the ingroup message at the root, so the
prior enters exactly once.

Compression. TreeTime merges identical alignment columns into one pattern
(variable columns already get private patterns, so in practice only constant
columns are merged). Positions merged into one pattern may carry different
messages. Two representations keep compression:

* **grouped sidecar** (:meth:`RootStateMessages.groups`): for every compressed
  pattern, the distinct messages of its positions with their counts. Used for
  post-hoc scoring at a fixed fitted tree (Stage 2a), without touching the
  TreeAnc's data.
* **refined patterns** (:meth:`TreeAnc.set_root_state_messages`): positions with
  different messages get different compression tags, so each pattern carries
  exactly one message, and the ordinary marginal message passing propagates it
  (Stage 2b). The number of extra patterns is bounded by the number of distinct
  (column, message) pairs; the full alignment is never expanded.

Normalisation. Each message is stored as ``m_i = exp(c_i) * u_i`` with
``max_a u_i(a) = 1``; ``c_i`` is added back to the per-site log-likelihood, so
reported scores are absolute: log P(D_ingroup, D_ext | tree) under the
messages. Scaling a message by a constant changes every candidate's score by
the same amount and never changes a ranking.
"""
import numpy as np


class RootStateMessages:
    """Validated per-position root-state messages.

    Parameters
    ----------
    messages : dict or array
        ``{position: vector}`` with 0-based full-alignment positions, or an
        array of shape (L, n_states) where rows that are all NaN are neutral.
        Each vector has one non-negative finite value per state of the GTR
        alphabet, not all zero.
    n_states : int
        Alphabet size of the GTR model.
    length : int, optional
        Full alignment length, to validate positions.
    """

    def __init__(self, messages, n_states, length=None):
        if isinstance(messages, RootStateMessages):
            messages = messages.as_dict()
        if isinstance(messages, dict):
            items = sorted((int(p), np.asarray(v, dtype=float)) for p, v in messages.items())
        else:
            arr = np.asarray(messages, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != n_states:
                raise ValueError('RootStateMessages: array must have shape (L, %d)' % n_states)
            items = [(i, arr[i]) for i in range(arr.shape[0]) if not np.all(np.isnan(arr[i]))]
        self.n_states = n_states
        pos, vecs = [], []
        for p, v in items:
            if v.shape != (n_states,):
                raise ValueError('RootStateMessages: message at %d has shape %s, expected (%d,)' % (p, v.shape, n_states))
            if not np.all(np.isfinite(v)) or np.any(v < 0) or not np.any(v > 0):
                raise ValueError('RootStateMessages: message at %d must be finite, non-negative and not all zero' % p)
            if p < 0 or (length is not None and p >= length):
                raise ValueError('RootStateMessages: position %d outside the alignment' % p)
            if np.all(v == 1.0):
                continue  # neutral, exactly
            pos.append(p)
            vecs.append(v)
        self.positions = np.array(pos, dtype=int)
        self.vectors = np.array(vecs, dtype=float).reshape(len(pos), n_states)
        # distinct messages -> tag 1..k (0 = neutral)
        self.unique, inv = (np.unique(self.vectors, axis=0, return_inverse=True) if len(pos)
                            else (np.zeros((0, n_states)), np.zeros(0, dtype=int)))
        self.tag_of_position = dict(zip(self.positions.tolist(), (np.asarray(inv).ravel() + 1).tolist()))
        scale = self.unique.max(axis=1) if len(self.unique) else np.zeros(0)
        # tag -> normalised vector (max 1) and log scale; tag 0 neutral
        self.tag_vectors = np.vstack([np.ones((1, n_states)), self.unique / scale[:, None]])
        self.tag_log_scale = np.concatenate([[0.0], np.log(scale)])

    def __len__(self):
        return len(self.positions)

    @property
    def is_neutral(self):
        return len(self.positions) == 0

    def as_dict(self):
        return {int(p): v.copy() for p, v in zip(self.positions, self.vectors)}

    def tags(self, full_length):
        t = np.zeros(full_length, dtype=int)
        for p, k in self.tag_of_position.items():
            if p >= full_length:
                raise ValueError('RootStateMessages: position %d outside the alignment (length %d)' % (p, full_length))
            t[p] = k
        return t

    def groups(self, data):
        """Grouped sidecar for a compressed alignment ``data`` (SequenceData).

        Returns ``(pattern_idx, tag, count)`` arrays: for every compressed
        pattern, one entry per distinct message among its positions (tag 0 =
        neutral), with the number of positions carrying it. Counts per pattern
        sum to the pattern's multiplicity.
        """
        tags = self.tags(data.full_length)
        f2c = data.full_to_compressed_sequence_map
        L = len(f2c)
        key = np.asarray(f2c, dtype=np.int64) * (len(self.tag_vectors) + 1) + tags[:L]
        uniq, counts = np.unique(key, return_counts=True)
        pat, tag = np.divmod(uniq, len(self.tag_vectors) + 1)
        # columns beyond the mapped length (additional constant sites) are neutral
        mult = data.multiplicity()
        mapped = np.bincount(pat, weights=counts, minlength=len(mult))
        extra = mult - mapped
        if np.any(extra > 0.5):
            idx = np.where(extra > 0.5)[0]
            pat = np.concatenate([pat, idx])
            tag = np.concatenate([tag, np.zeros(len(idx), dtype=int)])
            counts = np.concatenate([counts, extra[idx]])
        return pat.astype(int), tag.astype(int), counts.astype(float)

    def compressed(self, data):
        """Per-pattern (normalised message, log scale) for a ``data`` whose
        compression was refined with :meth:`tags`; raises if a pattern still
        mixes messages."""
        pat, tag, _ = self.groups(data)
        if len(np.unique(pat)) != len(pat):
            raise ValueError('RootStateMessages: compressed patterns mix different messages; refine first')
        L = data.compressed_length
        t = np.zeros(L, dtype=int)
        t[pat] = tag
        return self.tag_vectors[t], self.tag_log_scale[t]


def outgroup_messages(gtr, sequence, distance, positions=None):
    """Messages from a single outgroup leaf attached at the root by a branch of
    length ``distance``: m_i(a) = sum_b P(b at the leaf | a at the root) p_i(b),
    with p_i the leaf's profile (one-hot, or summed over an ambiguity code), i.e.
    [e^{Qd}]_{a,o_i} for an unambiguous state o_i. ``sequence`` is a string or
    array over the full alignment; ``positions`` restricts the output to those
    sites (default all). Returns ``{position: vector}``.
    """
    from .seq_utils import seq2prof
    seq = np.array(list(sequence)) if isinstance(sequence, str) else np.asarray(sequence)
    pos = np.arange(len(seq)) if positions is None else np.asarray(positions, dtype=int)
    prof = seq2prof(seq[pos], gtr.profile_map)
    msg = gtr.propagate_profile(prof, distance, return_log=False)
    return {int(p): msg[k] for k, p in enumerate(pos)}
