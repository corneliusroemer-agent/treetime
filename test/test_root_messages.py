"""Tests for external root-state messages (treetime.root_messages, Stage 2):
neutral messages reproduce Stage 1 exactly, compression is refined only where
messages differ, strong messages move the root posterior, contradictory
messages move root-incident branch lengths and the root date, and the Stage 1 /
2a / 2b driver. The brute-force Felsenstein check is in
test_root_messages_felsenstein.py."""
import copy

import numpy as np
import pytest

from test_root_candidates import load
from treetime.root_candidates import EdgeRoot, FitConfig, UnrootedBaseline, evaluate_root_candidate, fresh_marginal_score
from treetime.root_messages import (RootMessageBaseline, ingroup_root_posterior, outgroup_leaf_messages,
                                    rank_with_root_messages, root_message_correction, root_state_posterior)
from treetime.treeanc import TreeAnc

FIT = FitConfig(iterations=2)


@pytest.fixture(scope='module')
def b20():
    nwk, aln, dates, meta = load('binary20')
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    true = b.canonical(EdgeRoot(frozenset(meta['true_root_split']), 0.5))
    return b, true


@pytest.fixture(scope='module')
def fitted(b20):
    b, true = b20
    return evaluate_root_candidate(b, true, FIT)


def strong(L, K, pos, states, gtr, weak=1e-4):
    m = {}
    for p, s in zip(pos, states):
        v = np.full(K, weak)
        v[list(gtr.alphabet).index(s)] = 1.0
        m[int(p)] = v
    return m


def test_neutral_messages_reproduce_stage1_exactly(b20, fitted):
    b, true = b20
    L, K = fitted.state.data.full_length, len(fitted.state.gtr.alphabet)
    for msg in (np.ones((L, K)), {0: np.ones(K), 17: np.ones(K)}):
        run = evaluate_root_candidate(RootMessageBaseline(b, msg), true, FIT)
        assert run.scores == fitted.scores
        assert run.root == fitted.root
        assert np.array_equal(run.state.data.multiplicity(), fitted.state.data.multiplicity())
        assert run.state.tree.total_root_message_LH == 0.0
    c = root_message_correction(copy.deepcopy(fitted.state), np.ones((L, K)))
    assert c['correction'] == 0.0 and c['combined'] == c['ingroup'] == fitted.best_score


def test_compression_refined_only_where_messages_differ(b20, fitted):
    b, _ = b20
    data = fitted.state.data
    K = len(fitted.state.gtr.alphabet)
    mult = data.multiplicity()
    big = int(np.argmax(mult))  # a constant pattern shared by many positions
    pos = data.compressed_to_full_sequence_map[big][:3]
    msgs = {int(pos[0]): np.array([.9, .1, .1, .1, .1]), int(pos[1]): np.array([.9, .1, .1, .1, .1]),
            int(pos[2]): np.array([.1, .9, .1, .1, .1])}
    ta = TreeAnc(tree=copy.deepcopy(fitted.state.tree), aln=b.aln, gtr=copy.deepcopy(b.gtr), verbose=0)
    n0 = ta.data.compressed_length
    ta.set_root_state_messages(msgs)
    d = ta.data
    assert d.compressed_length == n0 + 2  # two new message groups split off one pattern
    assert d.multiplicity().sum() == d.full_length
    assert d.full_to_compressed_sequence_map[pos[0]] == d.full_to_compressed_sequence_map[pos[1]]
    assert d.full_to_compressed_sequence_map[pos[0]] != d.full_to_compressed_sequence_map[pos[2]]
    for p, full in d.compressed_to_full_sequence_map.items():  # maps consistent, alignment not expanded
        assert np.all(d.full_to_compressed_sequence_map[full] == p)
        for name, seq in d.compressed_alignment.items():
            assert all(d.aln[name][i] == seq[p] for i in full)
    # refinement alone does not change the ingroup likelihood
    ta.set_root_state_messages({k: 2.0 * np.ones(K) for k in pos})  # constant messages: posterior unchanged
    fresh_marginal_score(ta)
    assert ta.tree.total_sequence_LH_ingroup == pytest.approx(fitted.best_score, abs=1e-8)
    assert ta.tree.total_root_message_LH == pytest.approx(3 * np.log(2.0), abs=1e-12)
    ta.set_root_state_messages(None)
    assert ta.data.compressed_length == n0


def test_strong_message_shifts_root_posterior(b20, fitted):
    b, true = b20
    tt = fitted.state
    gtr = tt.gtr
    K = len(gtr.alphabet)
    tt0 = copy.deepcopy(tt)
    fresh_marginal_score(tt0)
    allpos = np.arange(tt.data.full_length)
    p_ing = root_state_posterior(tt0, allpos)
    # positions where the ingroup root state is confident; push them to a different state
    conf = allpos[p_ing.max(axis=1) > 0.99][:5]
    other = [gtr.alphabet[(np.argmax(p_ing[p]) + 1) % 4] for p in conf]
    msg = strong(tt.data.full_length, K, conf, other, gtr, weak=1e-8)
    run = evaluate_root_candidate(RootMessageBaseline(b, msg), true, FIT)
    post = root_state_posterior(run.state, conf)
    idx = [list(gtr.alphabet).index(s) for s in other]
    assert np.all(post[np.arange(len(conf)), idx] > 0.9)
    assert np.all(p_ing[conf, idx] < 0.01)
    # the ingroup-only posterior at the refitted 2b state is still reported separately
    assert np.all(ingroup_root_posterior(run.state, conf)[np.arange(len(conf)), idx] < 0.5)


def test_contradictory_message_moves_root_edges_and_date(b20, fitted):
    b, true = b20
    tt = fitted.state
    gtr = tt.gtr
    K = len(gtr.alphabet)
    kids = tt.tree.root.clades
    full = tt.data.full_to_compressed_sequence_map
    a, c = kids[0]._cseq[full], kids[1]._cseq[full]
    diff = np.nonzero(a != c)[0]
    assert len(diff) >= 1
    runs = {}
    for side, seq in (('a', a), ('c', c)):
        msg = strong(tt.data.full_length, K, diff, seq[diff], gtr, weak=1e-6)
        runs[side] = evaluate_root_candidate(RootMessageBaseline(b, msg), true, FIT)
    la = dict(zip(runs['a'].root['root_children'], runs['a'].root['root_child_lengths']))
    lc = dict(zip(runs['c'].root['root_children'], runs['c'].root['root_child_lengths']))
    # a root state matching child a puts the differences on c's edge, and vice versa
    assert la[kids[1].name] > lc[kids[1].name]
    assert lc[kids[0].name] > la[kids[0].name]
    dates = [fitted.root['root_date'], runs['a'].root['root_date'], runs['c'].root['root_date']]
    assert max(dates) - min(dates) > 0.1


def test_rank_with_root_messages_neutral_and_outgroup(b20):
    b, true = b20
    kw = dict(explicit=[true], regression_k=2, fit=FIT, sensitivity_window=None, rate_rounds=0)
    res = rank_with_root_messages(b, {0: np.ones(5)}, **kw)
    s1 = res.stage1
    for row in res.rows:
        assert row['combined_2a'] == row['ingroup_1'] == row['combined_2b']
        assert row['refit_minus_posthoc'] == 0.0
    # an outgroup-leaf message: correction is negative and 2a ranking uses combined scores
    tt = s1.best_treetime()
    full = tt.data.full_to_compressed_sequence_map
    o = tt.tree.root._cseq[full]
    msg = outgroup_leaf_messages(tt.gtr, o, 0.02)
    res2 = rank_with_root_messages(b, msg, stage1=s1, posterior_positions=[0, 1], **kw)
    for row in res2.rows:
        assert row['correction_2a'] < 0
        assert row['combined_2a'] == pytest.approx(row['ingroup_2a'] + row['correction_2a'])
        assert row['combined_2b'] >= row['combined_2a'] - 2.0  # refit should not be much worse than post hoc
        assert len(row['posterior_2b']) == 2
    assert set(res2.summary()) >= {'stage1_best', 'stage2a_best', 'stage2b_best'}
