"""Tests for external root-state messages (treetime.root_messages, Stage 2).

Fixture binary20: simulated on a known tree, fitted tree estimated by IQ-TREE
CMAPLE and collapsed below 0.5/L (see test_root_candidates.py)."""
import copy
import io
import os
import sys

import numpy as np
import pytest
from Bio import Phylo, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Clade
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from test_root_candidates import load, make_baseline  # noqa: E402

from treetime.root_candidates import (  # noqa: E402
    FitConfig,
    evaluate_root_candidate,
    fitted_newick,
    fresh_marginal_score,
    rank_root_candidates,
    rank_with_root_messages,
    score_newick,
    score_with_root_messages,
)
from treetime.root_messages import RootStateMessages, outgroup_messages  # noqa: E402
from treetime.treeanc import TreeAnc  # noqa: E402

FIT1 = FitConfig(iterations=1)


@pytest.fixture(scope='module')
def fitted():
    b = make_baseline('binary20')
    aln = {r.id: str(r.seq) for r in SeqIO.parse(load('binary20')[1], 'fasta')}
    run = evaluate_root_candidate(b, b.all_candidates()[0], FIT1)
    return b, aln, run


def outgroup_seq(aln, n_mut=40, seed=1):
    rng = np.random.default_rng(seed)
    out = np.array(list(aln[sorted(aln)[0]]))
    idx = rng.choice(len(out), n_mut, replace=False)
    out[idx] = rng.choice(list('ACGT'), n_mut)
    return ''.join(out)


def whole_tree_score(b, aln, tt, out, d):
    """TreeAnc likelihood of the tree with the outgroup leaf attached at the root,
    at the branch lengths tt's likelihood uses."""
    t = Phylo.read(io.StringIO(fitted_newick(tt)), 'newick')
    bl = {n.name: tt._branch_length_to_gtr(n) for n in tt.tree.find_clades() if n.up is not None}
    for n in t.find_clades():
        if n is not t.root:
            n.branch_length = bl[n.name]
    t.root.clades.append(Clade(name='OUT', branch_length=d))
    recs = [SeqRecord(Seq(v), id=k) for k, v in aln.items()] + [SeqRecord(Seq(out), id='OUT')]
    ta = TreeAnc(tree=t, aln=MultipleSeqAlignment(recs), gtr=copy.deepcopy(b.gtr), verbose=0)
    return fresh_marginal_score(ta)


def test_neutral_messages_reproduce_stage1_exactly(fitted):
    b, aln, run = fitted
    tt = copy.deepcopy(run.state)
    s0 = fresh_marginal_score(tt)
    L0 = tt.data.compressed_length
    tt.set_root_state_messages({i: np.ones(tt.gtr.n_states) for i in range(tt.data.full_length)})
    assert tt.root_state_messages is None and tt.data.compressed_length == L0
    assert fresh_marginal_score(tt) == s0
    sc = score_with_root_messages(copy.deepcopy(run.state), {})
    assert sc['ingroup'] == s0 and sc['combined'] == s0 and sc['correction'] == 0.0


def test_neutral_messages_reproduce_stage1_ranking():
    b = make_baseline('binary20')
    kw = dict(regression_k=3, fit=FIT1, rate_rounds=0, sensitivity_window=None)
    r1 = rank_root_candidates(b, **kw)
    r2 = rank_root_candidates(b.with_root_state_messages({5: np.ones(5)}), **kw)
    assert [(r.candidate_id, r.score) for r in r1.results] == [(r.candidate_id, r.score) for r in r2.results]
    for x, y in zip(r1.results, r2.results):
        assert x.best_run.root == y.best_run.root


def test_combined_score_equals_whole_tree_felsenstein(fitted):
    """Fixed branch lengths: ingroup + single-outgroup message == likelihood of the
    whole tree with the outgroup leaf; both the refined (2b) and the sidecar (2a) path."""
    b, aln, run = fitted
    out, d = outgroup_seq(aln), 0.02
    ref = whole_tree_score(b, aln, run.state, out, d)
    m = outgroup_messages(b.gtr, out, d)
    tt = copy.deepcopy(run.state)
    tt.set_root_state_messages(m)
    assert fresh_marginal_score(tt) == pytest.approx(ref, abs=1e-8)
    sc = score_with_root_messages(copy.deepcopy(run.state), m)
    assert sc['combined'] == pytest.approx(ref, abs=1e-8)
    assert sc['correction'] == pytest.approx(ref - sc['ingroup'], abs=1e-8)


def test_compression_refined_not_expanded(fitted):
    b, aln, run = fitted
    tt = copy.deepcopy(run.state)
    L, Lc = tt.data.full_length, tt.data.compressed_length
    m = outgroup_messages(b.gtr, outgroup_seq(aln), 0.02)
    tt.set_root_state_messages(m)
    Lc2 = tt.data.compressed_length
    assert Lc < Lc2 < L
    assert tt.data.multiplicity().sum() == L
    # each refined pattern carries one message, positions map back consistently
    msgs = tt.root_state_messages
    for p, pos in tt.data.compressed_to_full_sequence_map.items():
        assert len({msgs.tag_of_position.get(int(i), 0) for i in pos}) == 1
    # sidecar groups on the unrefined data: counts sum to multiplicities
    pat, tag, cnt = RootStateMessages(m, 5).groups(run.state.data)
    assert np.allclose(np.bincount(pat, weights=cnt), run.state.data.multiplicity())
    # removing the messages restores the original compression
    tt.set_root_state_messages(None)
    assert tt.data.compressed_length == Lc


def test_message_scale_shifts_all_scores_equally(fitted):
    b, aln, run = fitted
    m = outgroup_messages(b.gtr, outgroup_seq(aln), 0.02)
    s1 = score_with_root_messages(copy.deepcopy(run.state), m)['combined']
    s2 = score_with_root_messages(copy.deepcopy(run.state), {k: 7.0 * v for k, v in m.items()})['combined']
    assert s2 - s1 == pytest.approx(len(m) * np.log(7.0), rel=1e-10)


def test_strong_message_shifts_root_posterior(fitted):
    b, aln, run = fitted
    tt = copy.deepcopy(run.state)
    fresh_marginal_score(tt)
    site = 3
    post0 = tt.tree.root.marginal_profile[tt.data.full_to_compressed_sequence_map[site]].copy()
    target = int(np.argsort(post0[:4])[-2])  # a state the ingroup disfavours
    assert post0[target] < 0.5
    vec = np.full(5, 1e-3 * post0[target])  # odds update far above the ingroup's preference
    vec[target] = 1.0
    sc = score_with_root_messages(copy.deepcopy(run.state), {site: vec}, sites=[site])
    assert sc['posterior'][site][target] > 0.99
    assert np.allclose(sc['posterior'][site], post0 * vec / np.sum(post0 * vec))
    tt.set_root_state_messages({site: vec})
    fresh_marginal_score(tt)
    post = tt.tree.root.marginal_profile[tt.data.full_to_compressed_sequence_map[site]]
    assert np.allclose(post, sc['posterior'][site])


def test_contradictory_message_moves_root_edges_and_date(fitted):
    """2b: a strong outgroup message at many sites changes the root-incident
    branch profiles and the fitted root date of the same candidate."""
    b, aln, run = fitted
    cand = run.candidate
    m = outgroup_messages(b.gtr, aln[sorted(aln)[0]], 0.002)
    run2 = evaluate_root_candidate(b.with_root_state_messages(m), cand, FIT1)
    tt1, tt2 = run.state, run2.state
    pp1 = [tt1.marginal_branch_profile(c)[0] for c in tt1.tree.root.clades]
    f2c1, f2c2 = tt1.data.full_to_compressed_sequence_map, tt2.data.full_to_compressed_sequence_map
    pp2 = [tt2.marginal_branch_profile(c)[0] for c in tt2.tree.root.clades]
    diff = max(np.abs(a[f2c1] - c[f2c2]).max() for a, c in zip(pp1, pp2))
    assert diff > 0.1
    assert abs(run2.root['root_date'] - run.root['root_date']) > 0.05
    assert run2.root['root_child_lengths'] != run.root['root_child_lengths']


def test_rank_with_root_messages_reports_all_stages():
    b = make_baseline('binary20')
    aln = {r.id: str(r.seq) for r in SeqIO.parse(load('binary20')[1], 'fasta')}
    m = outgroup_messages(b.gtr, aln[sorted(aln)[0]], 0.005)
    res = rank_with_root_messages(b, m, regression_k=3, fit=FIT1, rate_rounds=0, sensitivity_window=None)
    assert len(res.rows) == len(res.stage2b.results)
    for r in res.rows:
        assert r['combined_posthoc'] == pytest.approx(r['ingroup'] + r['correction'])
        assert r['logL_2b'] is not None and r['root_date_2b'] is not None
    assert res.best_2b == res.stage2b.best.candidate_id
    s = res.summary()
    assert set(s) >= {'best_stage1', 'best_2a', 'best_2b', 'regression_best'}
    res.table()
    res.to_json()


def test_joint_initialisation_with_messages():
    b = make_baseline('binary20')
    m = {i: np.array([1.0, 0.1, 0.1, 0.1, 0.1]) for i in range(0, 600, 7)}
    run = evaluate_root_candidate(b.with_root_state_messages(m), b.all_candidates()[0],
                                  FitConfig(iterations=1, initial_mode='joint'))
    assert np.isfinite(run.best_score)


def test_score_newick_with_messages(fitted):
    b, aln, run = fitted
    m = outgroup_messages(b.gtr, outgroup_seq(aln), 0.02)
    tt = copy.deepcopy(run.state)
    tt.set_root_state_messages(m)
    # newick from the fitted tree; score_newick uses branch_length, so compare at those lengths
    s = score_newick(b, fitted_newick(run.state), messages=m)
    assert np.isfinite(s)


def test_invalid_messages_rejected():
    with pytest.raises(ValueError):
        RootStateMessages({0: [1, 1, 1]}, 5)
    with pytest.raises(ValueError):
        RootStateMessages({0: [0, 0, 0, 0, 0]}, 5)
    with pytest.raises(ValueError):
        RootStateMessages({0: [1, -1, 1, 1, 1]}, 5)
    with pytest.raises(ValueError):
        RootStateMessages({700: np.ones(5) * 2}, 5, length=600)


def _bruteforce_loglik(gtr, tree, leaf_seqs, bl, out, d):
    """Plain numpy/scipy Felsenstein pruning on uncompressed columns, with an
    outgroup leaf attached at the root. Rate matrix is built here from W, Pi, mu;
    TreeTime's convention is Q[a, b] = mu * W[a, b] * Pi[a] (rate b -> a), diagonal
    set so columns sum to zero, so P(t)[a, b] = P(child a | parent b)."""
    from scipy.linalg import expm
    W, pi = np.asarray(gtr.W, float), np.asarray(gtr.Pi, float)
    Q = gtr.mu * W * pi[:, None]
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=0))
    alpha = list(gtr.alphabet)
    n = len(alpha)

    def P(t):
        return expm(Q * t)

    # one-time convention check against TreeTime's own transition matrix
    assert np.allclose(P(0.07), gtr.expQt(0.07), atol=1e-9)

    def profile(seq):
        prof = np.ones((len(seq), n))
        for i, ch in enumerate(seq.upper()):
            if ch in alpha:
                prof[i] = 0.0
                prof[i, alpha.index(ch)] = 1.0
        return prof  # unknown characters ('N', '?', ...) stay all ones

    L = len(next(iter(leaf_seqs.values())))
    prof = {k: profile(v) for k, v in leaf_seqs.items()}
    Ps = {}

    def prune(node):
        if node.is_terminal():
            return prof[node.name]
        lik = np.ones((L, n))
        for ch in node.clades:
            t = bl[ch.name]
            Pt = Ps.setdefault(t, P(t))
            lik = lik * (prune(ch) @ Pt.T)  # sum_b P[b, a] * l_child(b)
        return lik

    root = prune(tree.root)
    root = root * (profile(out) @ P(d).T)
    return float(np.log(root @ pi).sum())


def test_combined_score_equals_bruteforce_felsenstein(fitted):
    """Independent check: uncompressed plain pruning (scipy expm, one-hot leaves,
    root prior Pi) equals the root-message score, including an ambiguous outgroup character."""
    b, aln, run = fitted
    tt0 = run.state
    out = list(outgroup_seq(aln))
    out[3], out[250], out[251] = 'N', 'N', 'N'
    out = ''.join(out)
    d = 0.02
    t = Phylo.read(io.StringIO(fitted_newick(tt0)), 'newick')
    bl = {n.name: tt0._branch_length_to_gtr(n) for n in tt0.tree.find_clades() if n.up is not None}
    ref = _bruteforce_loglik(b.gtr, t, aln, bl, out, d)
    tt = copy.deepcopy(run.state)
    tt.set_root_state_messages(outgroup_messages(b.gtr, out, d))
    got = fresh_marginal_score(tt)
    print('bruteforce', ref, 'treetime', got, 'diff', got - ref)
    assert got == pytest.approx(ref, abs=1e-8)
