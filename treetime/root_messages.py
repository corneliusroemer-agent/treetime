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
  uses them. ``UnrootedBaseline.with_root_state_messages`` does this for every
  candidate of the Stage 1 evaluator, and :func:`rank_with_root_messages` runs
  Stage 1, 2a and 2b on one candidate set and reports them side by side.

Normalisation: a message's absolute scale enters the score (it is part of the
joint likelihood of ingroup and external data). For a fixed message set this is
the same constant for every candidate, so differences between candidates are
unaffected; it matters only when comparing scores across message sets, which
these functions never do.
"""
import copy
from dataclasses import dataclass, field

import numpy as np

from .root_candidates import candidate_key, rank_root_candidates, restore_best_fitted_state, fresh_marginal_score
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
    ta = baseline.make_treeanc(copy.deepcopy(tree), messages=False)  # the baseline's data settings
    ta.set_root_state_messages(messages)
    s = fresh_marginal_score(ta)
    return s, float(ta.tree.total_sequence_LH_ingroup), float(ta.tree.total_root_message_LH)


# --------------------------------------------------------------------------
# Stage 1 / 2a / 2b side by side
# --------------------------------------------------------------------------
@dataclass
class Stage2Result:
    stage1: object         # RootSearchResult without messages
    stage2b: object        # RootSearchResult with messages propagated
    rows: list = field(default_factory=list)  # one dict per candidate (see rank_with_root_messages)

    def best(self, col):
        ok = [r for r in self.rows if r[col] is not None and np.isfinite(r[col])]
        return max(ok, key=lambda r: r[col])['candidate_id'] if ok else None

    def margin(self, col):
        sc = sorted((r[col] for r in self.rows if r[col] is not None and np.isfinite(r[col])), reverse=True)
        return sc[0] - sc[1] if len(sc) > 1 else None

    def summary(self):
        return {
            'stage1_best': self.stage1.best.candidate_id, 'stage1_margin': self.stage1.margin,
            'stage1_resolved': self.stage1.best_is_resolved,
            'stage2a_best': self.best('combined_2a'), 'stage2a_margin': self.margin('combined_2a'),
            'stage2b_best': self.stage2b.best.candidate_id, 'stage2b_margin': self.stage2b.margin,
            'stage2b_resolved': self.stage2b.best_is_resolved,
        }

    def to_tsv(self, path):
        cols = list(self.rows[0].keys()) if self.rows else []
        with open(path, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in self.rows:
                fh.write('\t'.join('' if r[c] is None else (repr(r[c]) if isinstance(r[c], float) else str(r[c]))
                                   for c in cols) + '\n')


def _root_edges(tt):
    return sorted((c.name, float(c.branch_length)) for c in tt.tree.root.clades)


def rank_with_root_messages(baseline, messages, explicit=(), regression_k=5, posterior_positions=(), stage1=None,
                            warm_start=True, **rank_kwargs):
    """Stage 1, 2a and 2b on one candidate set.

    1. Stage 1: :func:`rank_root_candidates` on the ingroup ``baseline`` (or the
       given ``stage1`` result, which must have been run with the same
       ``explicit``/``regression_k`` and ``keep_states``).
    2. 2a: every Stage 1 candidate's best fitted state is scored post hoc with
       the messages (:func:`root_message_correction`).
    3. 2b: :func:`rank_root_candidates` on
       ``baseline.with_root_state_messages(messages)`` with the same candidates
       and initialisations (the regression does not see the messages), so fits,
       rate fits and sensitivity refits all see the messages. With
       ``warm_start`` (default) each candidate also gets a trajectory that
       starts from its Stage 1 best state (step 0 of that trajectory is exactly
       the 2a state), so its 2b score is never below its 2a score.

    Per-candidate rows: ingroup-only, correction and combined at the Stage 1 fit
    (2a); the same three at the 2b state; refit minus post hoc; which 2b run won
    (``initial_mode`` 'warm' or the cold one) and the cold run's own score; root
    date and root-incident branch lengths before and after; clock rates;
    root-state posteriors at ``posterior_positions`` (ingroup-only and 2b).
    """
    if stage1 is None:
        stage1 = rank_root_candidates(baseline, explicit=explicit, regression_k=regression_k, **rank_kwargs)
    rows_in = messages.values() if isinstance(messages, dict) else messages
    neutral = all(np.all(np.asarray(v, dtype=float) == 1.0) for v in rows_in)
    warm = {}
    if warm_start and not neutral:
        for r in stage1.results:
            b = r.best_run
            if b is not None and b.error is None and b.state is not None:
                warm[candidate_key(r.candidate)] = (b.state, b.clock_rate)
    s2b = rank_root_candidates(baseline.with_root_state_messages(messages), explicit=explicit,
                               regression_k=regression_k, warm_starts=warm, **rank_kwargs)
    by2b = {candidate_key(r.candidate): r for r in s2b.results}
    rows = []
    pos = list(posterior_positions)
    for r1 in stage1.results:
        row = {'candidate_id': r1.candidate_id, 'source': r1.source, 'regression_rank': r1.regression_rank,
               'ingroup_1': r1.score}
        try:
            tt1 = copy.deepcopy(restore_best_fitted_state(r1))
            c = root_message_correction(tt1, messages)
            row.update(ingroup_2a=c['ingroup'], correction_2a=c['correction'], combined_2a=c['combined'],
                       root_date_1=float(tt1.tree.root.numdate), root_edges_1=_root_edges(tt1),
                       posterior_ingroup=root_state_posterior(tt1, pos).round(4).tolist() if pos else None)
        except ValueError:
            row.update(ingroup_2a=None, correction_2a=None, combined_2a=None, root_date_1=None, root_edges_1=None,
                       posterior_ingroup=None)
        r2 = by2b.get(candidate_key(r1.candidate))
        if r2 is not None and np.isfinite(r2.score) and r2.best_run.state is not None:
            tt2 = copy.deepcopy(restore_best_fitted_state(r2))
            comb = fresh_marginal_score(tt2)
            cold = [x.best_score for x in r2.runs if x.error is None and x.initial_mode != 'warm']
            row.update(combined_2b=comb, ingroup_2b=float(tt2.tree.total_sequence_LH_ingroup),
                       correction_2b=float(tt2.tree.total_root_message_LH),
                       refit_minus_posthoc=(comb - row['combined_2a']) if row['combined_2a'] is not None else None,
                       run_2b=r2.best_run.initial_mode, cold_2b=max(cold) if cold else None,
                       root_date_2b=float(tt2.tree.root.numdate), root_edges_2b=_root_edges(tt2),
                       clock_rate_1=r1.best_run.clock_rate, clock_rate_2b=r2.best_run.clock_rate,
                       posterior_2b=root_state_posterior(tt2, pos).round(4).tolist() if pos else None)
            if row['root_date_1'] is not None:
                row['root_date_shift'] = row['root_date_2b'] - row['root_date_1']
        else:
            row.update(combined_2b=None, ingroup_2b=None, correction_2b=None, refit_minus_posthoc=None,
                       run_2b=None, cold_2b=None, root_date_2b=None, root_edges_2b=None, clock_rate_1=None,
                       clock_rate_2b=None, posterior_2b=None)
        row.setdefault('root_date_shift', None)
        rows.append(row)
    return Stage2Result(stage1=stage1, stage2b=s2b, rows=rows)

