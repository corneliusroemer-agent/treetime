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
from treetime.root_candidates import candidate_key, plain_candidate_sources
from treetime.root_messages import (attachment_candidates, estimate_outgroup_distance, ingroup_root_posterior,
                                    outgroup_leaf_messages, rank_with_root_messages, root_message_correction,
                                    root_state_posterior)
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
        run = evaluate_root_candidate(b.with_root_state_messages(msg), true, FIT)
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
    run = evaluate_root_candidate(b.with_root_state_messages(msg), true, FIT)
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
        runs[side] = evaluate_root_candidate(b.with_root_state_messages(msg), true, FIT)
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
        assert row['combined_2b'] >= row['combined_2a'] - 1e-8  # warm start: never below the 2a state
        assert len(row['posterior_2b']) == 2
    assert set(res2.summary()) >= {'stage1_best', 'stage2a_best', 'stage2b_best'}


def test_message_scale_enters_every_score_equally(b20, fitted):
    tt = fitted.state
    o = tt.tree.root._cseq[tt.data.full_to_compressed_sequence_map]
    m = outgroup_leaf_messages(tt.gtr, o, 0.02)
    s1 = root_message_correction(copy.deepcopy(tt), m)['combined']
    s7 = root_message_correction(copy.deepcopy(tt), 7.0 * m)['combined']
    assert s7 - s1 == pytest.approx(m.shape[0] * np.log(7.0), rel=1e-10)


def test_invalid_messages_rejected(fitted):
    tt = copy.deepcopy(fitted.state)
    for bad in ({0: [1, 1, 1]}, {0: np.zeros(5)}, {0: [1, -1, 1, 1, 1]}, {10 ** 6: 2 * np.ones(5)},
                {0: [1, np.nan, 1, 1, 1]}, np.ones((3, 5))):
        with pytest.raises(ValueError):
            tt.set_root_state_messages(bad)


def test_joint_initialisation_sees_messages(b20):
    b, true = b20
    m = {i: np.array([1.0, 1e-8, 1e-8, 1e-8, 1e-8]) for i in range(0, 300, 7)}
    run = evaluate_root_candidate(b.with_root_state_messages(m), true, FitConfig(iterations=1, initial_mode='joint'))
    assert np.isfinite(run.best_score)
    tt = copy.deepcopy(run.state)
    tt.infer_ancestral_sequences('ml', marginal=False)
    root = tt.tree.root._cseq[tt.data.full_to_compressed_sequence_map]
    assert np.mean(root[list(m)] == 'A') > 0.9  # joint root reconstruction follows the messages


def test_warm_start_continues_from_the_given_state(b20, fitted):
    """A warm run's step 0 is the given state scored with the messages (the 2a
    value), so its best score is never below it; without messages it is the
    state's own score."""
    b, true = b20
    tt = fitted.state
    o = tt.tree.root._cseq[tt.data.full_to_compressed_sequence_map]
    m = outgroup_leaf_messages(tt.gtr, o, 0.05)
    posthoc = root_message_correction(copy.deepcopy(tt), m)['combined']
    run = evaluate_root_candidate(b.with_root_state_messages(m), true, FIT, initial_state=tt,
                                  clock_rate=fitted.clock_rate)
    assert run.initial_mode == 'warm' and run.iterations_performed == FIT.iterations
    assert run.scores[0] == pytest.approx(posthoc, abs=1e-8)
    assert run.best_score >= posthoc - 1e-8
    assert tt.root_state_message is None  # the given state is not modified
    plain = evaluate_root_candidate(b, true, FIT, initial_state=tt, clock_rate=fitted.clock_rate)
    assert plain.scores[0] == pytest.approx(fitted.best_score, abs=1e-8)


def test_estimate_outgroup_distance(b20, fitted):
    """ML length of a single outgroup leaf's branch: recovers a simulated length
    roughly and is the maximum of the post-hoc combined score over d."""
    tt = fitted.state
    gtr = tt.gtr
    o = np.array(tt.tree.root._cseq[tt.data.full_to_compressed_sequence_map])
    rng = np.random.default_rng(3)
    d_true = 0.1
    P = gtr.expQt(d_true)  # P[i, j] = P(i | j, t)
    idx = {c: i for i, c in enumerate(gtr.alphabet)}
    og = ''.join(gtr.alphabet[rng.choice(len(gtr.alphabet), p=P[:, idx[c]])] for c in o)
    d, best = estimate_outgroup_distance(copy.deepcopy(tt), og)
    assert 0.05 < d < 0.2
    for f in (0.8, 1.25):
        assert root_message_correction(copy.deepcopy(tt), outgroup_leaf_messages(gtr, og, d * f))['combined'] < best


def _contradictory(tt, n=10, weak=1e-3):
    """Messages for a state other than the root's most probable one at n
    confidently reconstructed columns (other states ``weak``; far below 1e-3 the
    preorder pass's 1e-12 profile floor starts to bias root-edge profiles)."""
    fresh_marginal_score(tt)
    prof = tt.tree.root.marginal_profile[tt.data.full_to_compressed_sequence_map]
    cols = np.nonzero(prof.max(axis=1) > 0.99)[0][:n]
    m = {}
    for p in cols:
        v = np.full(prof.shape[1], weak)
        v[(int(np.argmax(prof[p])) + 1) % prof.shape[1]] = 1.0
        m[int(p)] = v
    return m


def test_root_edges_fitted_separately_with_messages(b20, fitted):
    """With messages at a bifurcating root, the two root edges are separate
    branches: optimize_tree must not merge them (the message would sit at one
    end). Each root edge's marginal optimum equals the 1-D optimum of the exact
    combined likelihood, and the fit does not depend on the order of the root's
    children."""
    from scipy.optimize import minimize_scalar
    b, true = b20
    m = _contradictory(copy.deepcopy(fitted.state))
    fits = []
    for reverse in (False, True):
        ta = TreeAnc(tree=b.rooted_phylo(true), aln=b.aln, gtr=copy.deepcopy(b.gtr), verbose=0, rng_seed=b.rng_seed)
        ta.set_root_state_messages(m)
        if reverse:
            ta.tree.root.clades.reverse()
        ta.optimize_tree(max_iter=2, method_anc='probabilistic', branch_length_mode='marginal')
        fits.append({c.name: c.branch_length for c in ta.tree.root.clades})
    assert fits[0].keys() == fits[1].keys()
    for k in fits[0]:
        assert fits[0][k] == pytest.approx(fits[1][k], rel=1e-9, abs=1e-12)
    # per-edge optimum against direct scoring, everything else fixed
    fresh_marginal_score(ta)
    ta.infer_ancestral_sequences(marginal=True)
    for c in ta.tree.root.clades:
        t_marg = ta.optimal_marginal_branch_length(c, tol=1e-10)
        tt2 = copy.deepcopy(ta)
        node = [x for x in tt2.tree.root.clades if x.name == c.name][0]

        def neg(t):
            node.branch_length = node.mutation_length = t
            return -fresh_marginal_score(tt2)
        t_brute = minimize_scalar(neg, bounds=(1e-6, 1.0), method='bounded', options={'xatol': 1e-9}).x
        assert t_marg == pytest.approx(t_brute, rel=2e-3, abs=2e-5)


def test_independent_scorers_keep_messages_and_data_settings():
    """score_newick and score_tree_with_messages use the baseline's data
    settings (here 20 added constant columns via seq_len) and score_newick the
    baseline's messages, including messages on the added columns."""
    import io
    from Bio import Phylo
    from treetime.root_candidates import fitted_newick, score_newick
    from treetime.root_messages import score_tree_with_messages
    nwk, aln, dates, meta = load('binary20')
    plain = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    L0 = plain.make_treeanc(plain.rooted_phylo(None)).data.full_length
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69', tt_kwargs=dict(seq_len=L0 + 20))
    true = b.canonical(EdgeRoot(frozenset(meta['true_root_split']), 0.5))
    run = evaluate_root_candidate(b, true, FitConfig(iterations=1))
    assert run.state.data.full_length == L0 + 20
    o = run.state.tree.root._cseq[run.state.data.full_to_compressed_sequence_map]
    m = outgroup_leaf_messages(run.state.gtr, o, 0.05)
    m[L0 + 3] = [0.9, 0.05, 0.02, 0.02, 0.01]  # an added constant column with its own message
    nw = fitted_newick(run.state)
    tree = Phylo.read(io.StringIO(nw), 'newick')
    expect = root_message_correction(b.make_treeanc(copy.deepcopy(tree)), m)
    assert b.make_treeanc(copy.deepcopy(tree)).data.full_length == L0 + 20
    assert score_newick(b, nw) == pytest.approx(expect['ingroup'], abs=1e-8)
    assert score_newick(b.with_root_state_messages(m), nw) == pytest.approx(expect['combined'], abs=1e-8)
    assert score_tree_with_messages(b, tree, m)[0] == pytest.approx(expect['combined'], abs=1e-8)
    assert score_newick(plain, nw) != pytest.approx(expect['ingroup'], abs=1e-3)  # the added columns count


def test_rank_with_root_messages_keep_states_false(b20, fitted):
    """keep_states=False still gives 2a/2b rows and warm starts; the states are
    dropped afterwards. A given Stage 1 result without states is rejected."""
    from treetime.root_candidates import rank_root_candidates
    b, true = b20
    tt = fitted.state
    m = outgroup_leaf_messages(tt.gtr, tt.tree.root._cseq[tt.data.full_to_compressed_sequence_map], 0.05)
    kw = dict(explicit=[true], regression_k=0, fit=FIT)
    res = rank_with_root_messages(b, m, keep_states=False, **kw)
    row = res.rows[0]
    assert row['combined_2a'] is not None and row['combined_2b'] >= row['combined_2a'] - 1e-8
    assert any(x.initial_mode == 'warm' for x in res.stage2b.results[0].runs)
    assert all(x.state is None for r in (res.stage1, res.stage2b) for c in r.results for x in c.runs)
    s1 = rank_root_candidates(b, keep_states=False, **kw)
    with pytest.raises(ValueError):
        rank_with_root_messages(b, m, stage1=s1, **kw)


def test_attachment_candidates_and_attach_k(b20, fitted):
    """The attachment scan ranks every placement for the outgroup with the smallest ML d;
    rank_with_root_messages adds its first attach_k placements, labelled 'attach' with their rank,
    to the regression shortlist (here plain top 2), deduplicated by bipartition."""
    b, true = b20
    tt = fitted.state
    full = tt.data.full_to_compressed_sequence_map
    near = ''.join(tt.tree.root._cseq[full])
    far = ''.join('A' if ch != 'A' else 'C' for ch in near[:40]) + near[40:]
    cands, og, d = attachment_candidates(b, {'far': far, 'near': near})
    assert og == 'near' and 0 < d < 0.05
    assert len(cands) == len(b.all_candidates()) and len({candidate_key(c) for c in cands}) == len(cands)
    msg = outgroup_leaf_messages(b.gtr, near, d)
    kw = dict(candidate_sources=plain_candidate_sources(2), fit=FitConfig(0), sensitivity_window=None, rate_rounds=0,
              warm_start=False, keep_states=False)
    for k in (0, 2, 3):
        res = rank_with_root_messages(b, msg, attachment=cands, attach_k=k, **kw)
        ranks = sorted(r.source_ranks['attach'] for r in res.stage1.results if 'attach' in r.source_ranks)
        assert ranks == list(range(1, k + 1))
        keys = {candidate_key(c) for c in cands[:k]} | {candidate_key(r.candidate) for r in res.stage1.results
                                                        if 'plain' in r.source_ranks}
        assert {candidate_key(r.candidate) for r in res.stage1.results} == keys
        assert len(res.rows) == len(keys)
        assert {candidate_key(r.candidate) for r in res.stage2b.results} == keys
        assert all(row['source_ranks'] for row in res.rows)
    # default regression part with attachment candidates: covariation top regression_k only
    res = rank_with_root_messages(b, msg, attachment=cands, regression_k=2, **{k: v for k, v in kw.items()
                                                                               if k != 'candidate_sources'})
    labels = {n for r in res.stage1.results for n in r.source_ranks}
    assert labels == {'cov', 'attach'} and [s['name'] for s in res.stage1.config['candidate_sources']] == ['cov']
    assert res.stage2b.config['candidate_sources'] == res.stage1.config['candidate_sources']
