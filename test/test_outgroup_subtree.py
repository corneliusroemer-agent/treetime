"""Stage 3: undated outgroup subtree as root-state evidence.

Reference calculations are independent of the code under test: a whole-tree Felsenstein pass with
scipy's expm (``independent_P`` from the Stage 2 test) over uncompressed columns, and a TreeAnc on the
explicit combined tree (ingroup + outgroup leaves), also rooted inside the outgroup subtree.
Outgroup sequences are drawn from the ingroup consensus, not from a generating tree (HANDOFF §6).
"""
import copy
import io
import os
import sys

import numpy as np
import pytest
from Bio import AlignIO, Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Clade, Tree
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

sys.path.insert(0, os.path.dirname(__file__))
from test_root_candidates import load  # noqa: E402
from test_root_messages_felsenstein import independent_P, nonuniform_gtr  # noqa: E402

from treetime.outgroup_subtree import (OutgroupEvaluator, OutgroupSubtree, attachment_scan,  # noqa: E402
                                       rank_with_outgroup_subtree)
from treetime.root_candidates import (FitConfig, UnrootedBaseline, evaluate_root_candidate,  # noqa: E402
                                      fresh_marginal_score, rank_root_candidates, candidate_key)
from treetime.root_messages import outgroup_leaf_messages, rank_with_root_messages  # noqa: E402
from treetime.treeanc import TreeAnc  # noqa: E402

TOL = 1e-8
LENGTHS3 = np.array([0.03, 0.012, 0.02, 0.008, 0.015])  # stem, (O1,O2) internal, O3, O1, O2


def _outgroups(seqs, L, seed=3):
    rng = np.random.default_rng(seed)
    cons = np.array([max('ACGT', key=[s[i] for s in seqs.values()].count) for i in range(L)])

    def mut(s, p):
        s = s.copy()
        for i in np.nonzero(rng.random(L) < p)[0]:
            s[i] = rng.choice([x for x in 'ACGT' if x != s[i]])
        return s
    anc = mut(cons, 0.02)
    o12 = mut(anc, 0.01)
    ogs = {'O1': mut(o12, 0.01), 'O2': mut(o12, 0.01), 'O3': mut(anc, 0.02)}
    variable = [i for i in range(L) if len({s_[i] for s_ in seqs.values()}) > 1]
    # missing and partially ambiguous outgroup characters, at variable columns and at constant ones
    ogs['O1'][variable[:3]] = 'N'
    ogs['O2'][variable[1:4]] = 'R'
    ogs['O3'][[7, 70, 150]] = 'N'
    ogs['O1'][[7, 70]] = 'N'
    ogs['O2'][[7]] = 'N'  # column 7: every outgroup missing -> neutral
    return {k: ''.join(v) for k, v in ogs.items()}


@pytest.fixture(scope='module', params=['binary20', 'poly20-gtr'])
def fx(request):
    name, _, model = request.param.partition('-')
    nwk, aln, dates, _ = load(name)
    b = UnrootedBaseline(nwk, aln, dates, gtr=nonuniform_gtr() if model else 'JC69')
    c = b.all_candidates()[3]
    tt = evaluate_root_candidate(b, c, FitConfig(iterations=2)).state
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    L = len(next(iter(seqs.values())))

    def clean(n_):  # the lengths TreeTime propagates with (see the Stage 2 test)
        cl = Clade(name=n_.name if n_.is_terminal() else None, clades=[clean(x) for x in n_.clades])
        if n_ is not tt.tree.root:
            cl.branch_length = tt._branch_length_to_gtr(n_)
        return cl
    tree = Tree(root=clean(tt.tree.root), rooted=True)
    return dict(b=b, tt=tt, seqs=seqs, L=L, ogs=_outgroups(seqs, L), tree=tree)


def og3(fx, newick='((O1,O2),O3);'):
    o = OutgroupSubtree(fx['b'].gtr, fx['ogs'], tree=newick)
    o.lengths = LENGTHS3.copy()
    return o


def combined_tree(fx, lengths=LENGTHS3):
    """Ingroup tree with the outgroup clade ((O1,O2),O3) attached at its root by the stem."""
    t = copy.deepcopy(fx['tree'])
    s, i12, l3, l1, l2 = lengths
    clade = Clade(branch_length=s, clades=[
        Clade(branch_length=i12, clades=[Clade(branch_length=l1, name='O1'), Clade(branch_length=l2, name='O2')]),
        Clade(branch_length=l3, name='O3')])
    t.root.clades.append(clade)
    return t


def brute(fx, tree):
    """Independent Felsenstein over uncompressed columns of the combined alignment."""
    gtr = fx['b'].gtr
    seqs = dict(fx['seqs'], **fx['ogs'])
    pm = {k: np.asarray(v, float) for k, v in gtr.profile_map.items()}
    P = independent_P(gtr, fx['tt'].one_mutation)
    cache = {}

    def Pt(t):
        if t not in cache:
            cache[t] = P(t)
        return cache[t]

    def up(node, i):
        if node.is_terminal():
            return pm[seqs[node.name][i]]
        v = np.ones(len(gtr.alphabet))
        for ch in node.clades:
            v = v * (Pt(ch.branch_length).T @ up(ch, i))
        return v
    return sum(np.log(np.sum(gtr.Pi * up(tree.root, i))) for i in range(fx['L']))


def treeanc_score(fx, tree, compress=True):
    seqs = dict(fx['seqs'], **fx['ogs'])
    recs = [SeqRecord(Seq(s), id=k, description='') for k, s in seqs.items()]
    ta = TreeAnc(tree=copy.deepcopy(tree), aln=MultipleSeqAlignment(recs), gtr=copy.deepcopy(fx['b'].gtr),
                 verbose=0, compress=compress)
    return fresh_marginal_score(ta)


# --- one outgroup = Stage 2 --------------------------------------------------------
def test_single_leaf_messages_identical_to_stage2(fx):
    o = OutgroupSubtree(fx['b'].gtr, {'O3': fx['ogs']['O3']})
    o.lengths = np.array([0.03])
    assert np.array_equal(o.messages(), outgroup_leaf_messages(fx['b'].gtr, fx['ogs']['O3'], 0.03))


def test_single_outgroup_reproduces_stage2b():
    nwk, aln, dates, _ = load('binary20')
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    L = len(next(iter(seqs.values())))
    og = _outgroups(seqs, L)['O3']
    d = 0.02
    cands = b.all_candidates()[2:5]
    fit = FitConfig(iterations=2)
    s2 = rank_with_root_messages(b, outgroup_leaf_messages(b.gtr, og, d), explicit=cands, regression_k=0, fit=fit)
    o = OutgroupSubtree(b.gtr, {'O': og})
    o.lengths = np.array([d])
    s3 = rank_with_outgroup_subtree(b, o, explicit=cands, regression_k=0, attachment=(), fit=fit,
                                    fit_lengths=False, stage1=s2.stage1)
    a = {r.candidate_id: r.score for r in s2.stage2b.results}
    c = {r.candidate_id: r.score for r in s3.stage3.results}
    assert a == c  # bit-identical: same messages, same evaluator
    assert s2.stage2b.best.candidate_id == s3.stage3.best.candidate_id


# --- multiple outgroups: full-tree Felsenstein --------------------------------------
def test_subtree_equals_full_tree_felsenstein(fx):
    o = og3(fx)
    s_msg = o.combined_score(copy.deepcopy(fx['tt']))
    ta = fx['b'].make_treeanc(copy.deepcopy(fx['tree']), messages=False)
    ta.set_root_state_messages(o.messages())
    s_2b = fresh_marginal_score(ta)  # Stage 2 representation: refined patterns
    t = combined_tree(fx)
    s_ta = treeanc_score(fx, t)
    s_bf = brute(fx, t)
    print('scores', s_msg, s_2b, s_ta, s_bf)
    assert abs(s_msg - s_bf) < TOL and abs(s_2b - s_bf) < TOL
    # TreeAnc on the explicit tree folds an N at an ingroup-constant column into the constant state
    # (tip-off s2a-ambiguous-at-constant-column); the next test compares it without such N
    if not _has_n_at_constant(fx):
        assert abs(s_ta - s_bf) < TOL


def _has_n_at_constant(fx):
    cols = [i for i in range(fx['L']) if len({s[i] for s in fx['seqs'].values()}) == 1]
    return any(s[i] in 'NR' for s in fx['ogs'].values() for i in cols)


def test_subtree_matches_treeanc_without_constant_column_ambiguity(fx):
    cols = [i for i in range(fx['L']) if len({s[i] for s in fx['seqs'].values()}) == 1]
    f2 = dict(fx)
    ogs = {}
    for k, s in fx['ogs'].items():
        s = np.array(list(s))
        s[[i for i in cols if s[i] in 'NR']] = 'A'
        ogs[k] = ''.join(s)
    f2['ogs'] = ogs
    o = og3(f2)
    s_msg = o.combined_score(copy.deepcopy(fx['tt']))
    s_ta = treeanc_score(f2, combined_tree(f2))
    assert abs(s_msg - s_ta) < TOL


# --- compression -------------------------------------------------------------------
def test_compressed_equals_uncompressed(fx):
    o = og3(fx)
    m = o.messages()
    scores = []
    for compress in (True, False):
        ta = TreeAnc(tree=copy.deepcopy(fx['tree']), aln=fx['b'].aln, gtr=copy.deepcopy(fx['b'].gtr), verbose=0,
                     compress=compress)
        ta.set_root_state_messages(m)
        scores.append(fresh_marginal_score(ta))
        if compress:
            # ingroup-constant columns are grouped by outgroup pattern, never expanded to one pattern per column
            assert ta.data.compressed_length < fx['L']
            n_in = TreeAnc(tree=copy.deepcopy(fx['tree']), aln=fx['b'].aln, gtr=copy.deepcopy(fx['b'].gtr),
                           verbose=0).data.compressed_length
            assert ta.data.compressed_length <= n_in * o.n_patterns
            assert ta.data.multiplicity().sum() == fx['L']
    assert abs(scores[0] - scores[1]) < TOL
    # identical outgroup columns give byte-identical messages
    p = o.pattern_of_position
    i, j = np.nonzero(p == np.argmax(np.bincount(p)))[0][:2]
    assert np.array_equal(m[i], m[j])


# --- invariances -------------------------------------------------------------------
def test_reroot_invariance(fx):
    base = og3(fx)
    # the same unrooted tree (outgroups + attachment leaf) written with three different rootings
    forms = ['(((O1:0.008,O2:0.015):0.012,O3:0.02):0.01,ATTACH:0.02);',
             '(O1:0.008,O2:0.015,(O3:0.02,ATTACH:0.03):0.012);',
             '(O3:0.01,((O1:0.008,O2:0.015):0.012,ATTACH:0.03):0.01);',
             '((O1:0.008,O2:0.015):0.012,O3:0.02,ATTACH:0.03);']
    outs = []
    for f in forms:
        o = OutgroupSubtree(fx['b'].gtr, fx['ogs'], tree=f)
        assert o.edge_names == base.edge_names
        assert np.allclose(o.lengths, LENGTHS3, atol=1e-12)
        o.lengths = LENGTHS3.copy()
        outs.append(o.combined_score(copy.deepcopy(fx['tt'])))
    assert max(outs) - min(outs) < TOL
    # pulley: the explicit combined tree rooted inside the outgroup subtree gives the same likelihood
    t2 = combined_tree(fx)
    t2.root_with_outgroup(next(c for c in t2.find_clades() if {x.name for x in c.get_terminals()} == {'O1', 'O2'}))
    assert {x.name for x in t2.root.clades[0].get_terminals()} in ({'O1', 'O2'}, set(fx['seqs']) | {'O3'})
    assert abs(brute(fx, t2) - outs[0]) < TOL


def test_large_stem_is_uninformative(fx):
    o = og3(fx)
    far = LENGTHS3.copy()
    far[0] = 1e4
    m = o.pattern_messages(far)
    assert np.allclose(m, m[:, :1], rtol=1e-9)  # constant over the root state
    o1 = OutgroupSubtree(fx['b'].gtr, {'O3': fx['ogs']['O3']})
    m1 = o1.messages([1e4])
    pi = np.asarray(fx['b'].gtr.Pi)
    known = [i for i, ch in enumerate(fx['ogs']['O3']) if ch in 'ACGT']
    idx = [list(fx['b'].gtr.alphabet).index(fx['ogs']['O3'][i]) for i in known]
    assert np.allclose(m1[known], pi[idx][:, None], rtol=1e-9)  # m_i(a) -> pi_{o_i}


def test_missing_states(fx):
    o = og3(fx)
    m = o.messages()
    assert np.all(m[7] == 1.0)  # all outgroups missing: exactly neutral
    # one leaf 'R' = indicator of {A, G}: equals the sum of the A and G versions
    j = next(i for i, ch in enumerate(fx['ogs']['O2']) if ch == 'R')
    vals = {}
    for ch in 'AG':
        ogs = dict(fx['ogs'])
        ogs['O2'] = ogs['O2'][:j] + ch + ogs['O2'][j + 1:]
        oo = OutgroupSubtree(fx['b'].gtr, ogs, tree='((O1,O2),O3);')
        vals[ch] = oo.messages(LENGTHS3)[j]
    assert np.allclose(m[j], vals['A'] + vals['G'], rtol=1e-12)


# --- lengths: fitted per candidate, outside the clock ---------------------------------
def test_length_fit_and_clock_untouched(fx):
    b = fx['b']
    o = og3(fx)
    tt = copy.deepcopy(fx['tt'])
    lengths, val = o.fit_lengths(tt)
    assert val >= o.combined_score(tt) - 1e-9
    ing = fresh_marginal_score(copy.deepcopy(fx['tt']))
    # the ingroup part does not depend on the outgroup lengths
    for ls in (LENGTHS3, lengths, LENGTHS3 * 3):
        ta = copy.deepcopy(fx['tt'])
        ta.set_root_state_messages(o.messages(ls))
        fresh_marginal_score(ta)
        assert abs(ta.tree.total_sequence_LH_ingroup - ing) < TOL
    ev = OutgroupEvaluator(o, outer_iterations=2, attach_mixture_w=0.0)
    c = b.all_candidates()[3]
    run = ev(b, c, FitConfig(iterations=2))
    st = run.state
    assert {x.name for x in st.tree.get_terminals()} == set(fx['seqs'])  # outgroups are not in the dated tree
    assert run.clock_rate == b.clock_rate and all(r == b.clock_rate for r in run.clock_rates)
    assert abs(fresh_marginal_score(copy.deepcopy(st)) - run.best_score) < 1e-9  # best state restored
    assert run.best_score == max(run.scores)
    # the retained state's messages are those of the reported lengths
    assert abs(o.combined_score(copy.deepcopy(st), run.extra['outgroup_lengths']) - run.best_score) < 1e-8
    assert len(run.extra['outer_rounds']) >= 1


def test_single_leaf_fit_matches_stage2_estimate():
    from treetime.root_messages import estimate_outgroup_distance
    nwk, aln, dates, _ = load('binary20')
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    og = _outgroups(seqs, len(next(iter(seqs.values()))))['O3']
    tt = evaluate_root_candidate(b, b.all_candidates()[3], FitConfig(iterations=1)).state
    d2, s2 = estimate_outgroup_distance(copy.deepcopy(tt), og)
    o = OutgroupSubtree(b.gtr, {'O': og})
    (d3,), s3 = o.fit_lengths(copy.deepcopy(tt))
    assert abs(np.log(d3 / d2)) < 1e-3 and abs(s3 - s2) < 1e-6


# --- driver: extra candidates stay out of the selection ---------------------------
def test_extra_candidates_do_not_change_selection():
    nwk, aln, dates, _ = load('binary20')
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    ogs = _outgroups(seqs, len(next(iter(seqs.values()))))
    fit = FitConfig(iterations=1)
    kw = dict(regression_k=2, attach_k=1, fit=fit, outer_iterations=1)
    r0 = rank_with_outgroup_subtree(b, OutgroupSubtree(b.gtr, ogs), **kw)
    assert r0.config['attach_mixture_w'] == 0.01  # the default reaches the driver config
    listed = {candidate_key(r.candidate) for r in r0.stage3.results}
    off = next(c for c in b.all_candidates() if candidate_key(b.canonical(c)) not in listed)
    on = r0.stage3.results[-1].candidate
    r1 = rank_with_outgroup_subtree(b, OutgroupSubtree(b.gtr, ogs), extra_candidates=[off, on], **kw)
    assert [(r.candidate_id, r.score) for r in r1.stage3.results] == [(r.candidate_id, r.score) for r in r0.stage3.results]
    assert len(r1.extra) == 2 and {x['source'] for x in r1.extra} == {'extra', 'extra(listed)'}
    new = next(x for x in r1.extra if x['source'] == 'extra')
    assert new['combined_3'] is not None and np.isfinite(new['combined_3'])
    assert b.candidate_id(b.canonical(off)) not in {r.candidate_id for r in r1.stage3.results}


def test_reroot_invariance_end_to_end():
    """Whole Stage 3 ranking (length fits, dated fits) with the outgroup tree written in two rootings."""
    nwk, aln, dates, _ = load('binary20')
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    ogs = _outgroups(seqs, len(next(iter(seqs.values()))))
    cands = b.all_candidates()[2:4]
    out = []
    for f in ('(((O1:0.008,O2:0.015):0.012,O3:0.02):0.01,ATTACH:0.02);',
              '(O3:0.01,((O1:0.008,O2:0.015):0.012,ATTACH:0.03):0.01);'):
        o = OutgroupSubtree(b.gtr, ogs, tree=f)
        r = rank_with_outgroup_subtree(b, o, explicit=cands, regression_k=0, attachment=(), fit=FitConfig(iterations=1),
                                       outer_iterations=1)
        out.append({x.candidate_id: (x.score, x.best_run.extra['outgroup_lengths']) for x in r.stage3.results})
    for k in out[0]:
        assert abs(out[0][k][0] - out[1][k][0]) < 1e-8
        assert np.allclose(out[0][k][1], out[1][k][1], rtol=1e-8)


# --- outgroup tree induced by a joint tree (CMAPLE helper) ---------------------------
def test_outgroup_tree_from_joint_tree(fx):
    from treetime.outgroup_subtree import outgroup_tree_from_joint
    joint = combined_tree(fx)
    names = ['O1', 'O2', 'O3']
    nwk = outgroup_tree_from_joint(joint, names)
    o = OutgroupSubtree(fx['b'].gtr, fx['ogs'], tree=nwk)
    base = og3(fx)
    assert o.edge_names == base.edge_names
    assert np.allclose(o.lengths, LENGTHS3, atol=1e-10)  # stem = separating edge, outgroup edges as in the joint tree
    # the joint tree's rooting is irrelevant
    j2 = copy.deepcopy(joint)
    j2.root_with_outgroup(next(c for c in j2.get_terminals() if c.name not in names))
    assert np.allclose(OutgroupSubtree(fx['b'].gtr, fx['ogs'], tree=outgroup_tree_from_joint(j2, names)).lengths,
                       LENGTHS3, atol=1e-10)
    # an outgroup nested in the ingroup: still a valid subtree on all outgroups
    j3 = copy.deepcopy(joint)
    o3 = next(c for c in j3.find_clades() if c.name == 'O3')
    j3.prune(o3)
    leaf = next(c for c in j3.get_terminals() if c.name not in names)
    leaf.split(n=2, branch_length=0.001)
    leaf.clades[0].name, leaf.clades[1].name, leaf.name = leaf.name, 'O3', None
    o = OutgroupSubtree(fx['b'].gtr, fx['ogs'], tree=outgroup_tree_from_joint(j3, names))
    assert sorted(o.leaf_nodes) == names


def test_joint_tree_cmaple_roundtrip():
    import shutil
    from treetime.outgroup_subtree import joint_tree_cmaple, outgroup_tree_from_joint
    if shutil.which('iqtree3') is None:
        pytest.skip('iqtree3 not on PATH')
    nwk, aln, dates, _ = load('binary20')
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    ogs = _outgroups(seqs, len(next(iter(seqs.values()))))
    t = joint_tree_cmaple(dict(seqs, **ogs), iqtree='iqtree3')
    o = OutgroupSubtree(b.gtr, ogs, tree=outgroup_tree_from_joint(t, list(ogs)))
    assert sorted(o.leaf_nodes) == sorted(ogs) and np.all(o.lengths > 0)


# --- attachment mixture -------------------------------------------------------------
def test_attachment_mixture(fx):
    from scipy.special import logsumexp
    from treetime.outgroup_subtree import _rooted_on_edge, _unrooted_from_state, attachment_mixture_scores
    o = og3(fx)
    tt = copy.deepcopy(fx['tt'])
    sc, det = attachment_mixture_scores(fx['b'], tt, o, ws=(0.0, 0.2, 1.0), lengths=LENGTHS3, fit=False)
    # w = 0 is the outgroup attached at the candidate root
    assert abs(sc[0.0] - o.combined_score(copy.deepcopy(fx['tt']), LENGTHS3)) < TOL
    adj, names, nb, root_edge = _unrooted_from_state(tt)
    assert det['n_neighbours'] == len(nb) >= 2
    # pulley: rerooting the state at its own root position reproduces the w = 0 term
    if root_edge is not None:
        u, v, f = root_edge
        ta = fx['b'].make_treeanc(_rooted_on_edge(adj, names, u, v, f), messages=False)
        assert abs(o.combined_score(ta, LENGTHS3) - sc[0.0]) < TOL
    # a neighbour term equals an independent Felsenstein on the full tree with the outgroup at that edge's midpoint
    a, b_ = nb[0]
    t = _rooted_on_edge(adj, names, a, b_)
    s, i12, l3, l1, l2 = LENGTHS3
    t.root.clades.append(Clade(branch_length=s, clades=[
        Clade(branch_length=i12, clades=[Clade(branch_length=l1, name='O1'), Clade(branch_length=l2, name='O2')]),
        Clade(branch_length=l3, name='O3')]))
    assert abs(brute(fx, t) - det['neighbours'][0]) < TOL
    # the mixture itself, and its bounds
    expect = logsumexp([np.log(0.8) + sc[0.0]] + [np.log(0.2 / len(nb)) + x for x in det['neighbours']])
    assert abs(sc[0.2] - expect) < 1e-10
    assert sc[1.0] <= max(det['neighbours']) + 1e-12
    assert sc[0.2] >= np.log(0.2 / len(nb)) + max(det['neighbours']) - 1e-12
    # with the lengths fitted every term can only improve
    sc_fit, det_fit = attachment_mixture_scores(fx['b'], tt, o, ws=(0.0,), lengths=LENGTHS3)
    assert sc_fit[0.0] >= sc[0.0] - 1e-9


def test_mixture_in_driver_leaves_answer_unchanged():
    nwk, aln, dates, _ = load('binary20')
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    ogs = _outgroups(seqs, len(next(iter(seqs.values()))))
    kw = dict(regression_k=2, attach_k=1, fit=FitConfig(iterations=1), outer_iterations=1)
    r0 = rank_with_outgroup_subtree(b, OutgroupSubtree(b.gtr, ogs), attach_mixture_w=0.0, **kw)
    r1 = rank_with_outgroup_subtree(b, OutgroupSubtree(b.gtr, ogs), mixture_ws=(0.0, 0.3), attach_mixture_w=0.0, **kw)
    assert [(r.candidate_id, r.score) for r in r1.stage3.results] == [(r.candidate_id, r.score) for r in r0.stage3.results]
    for row in r1.rows:
        assert row['mixture'][0.0] >= row['combined_3'] - 1e-9  # best state re-scored, lengths refitted
        assert np.isfinite(row['mixture'][0.3])
    rk = r1.mixture_ranking(0.3)
    assert len(rk) == len(r1.rows) and rk[0][1] >= rk[-1][1]


def test_stage3_final_rate_refit_carries_the_outgroup_subtree():
    """final_rate_refit=True through rank_with_outgroup_subtree: the refit runs through OutgroupEvaluator, so its
    state carries the outgroup-subtree messages at its own fitted outgroup lengths, and its logL is that combined
    score. The winner is the Stage 3 winner."""
    nwk, aln, dates, _ = load('binary20')
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69', clock_rate=2 * load('binary20')[3]['clock_rate'])
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    o = OutgroupSubtree(b.gtr, _outgroups(seqs, len(next(iter(seqs.values())))))
    r = rank_with_outgroup_subtree(b, o, regression_k=2, attach_k=1, fit=FitConfig(iterations=1), outer_iterations=1,
                                   final_rate_refit=True, attach_mixture_w=0.0)
    ref = r.stage3.final_rate_refit
    assert ref['candidate_id'] == r.stage3.best.candidate_id and ref['refit_logL'] >= ref['shared_logL']
    assert not ref['refit_is_ranking_state'] and ref['refit_clock_rate'] != b.clock_rate
    st, lengths = ref['state'], ref['refit_extra']['outgroup_lengths']
    assert st.root_state_message is not None
    assert {x.name for x in st.tree.get_terminals()} == set(seqs)  # outgroups are messages, not dated leaves
    assert abs(o.combined_score(copy.deepcopy(st), lengths) - ref['refit_logL']) < 1e-8
    assert r.stage1.final_rate_refit['state'].root_state_message is None  # Stage 1's refit has no outgroup


# --- attachment mixture inside the dated fit ---------------------------------------------------
def _with_outgroup_clade(t, lengths):
    s, i12, l3, l1, l2 = lengths
    t.root.clades.append(Clade(branch_length=s, clades=[
        Clade(branch_length=i12, clades=[Clade(branch_length=l1, name='O1'), Clade(branch_length=l2, name='O2')]),
        Clade(branch_length=l3, name='O3')]))
    return t


def _brute_mixture_terms(fx, tt, term_lengths):
    """Independent per-column Felsenstein of every attachment term: the outgroup clade at the state's root
    (pulley: the state rerooted at its own root position) and at the midpoint of each neighbouring edge."""
    from treetime.outgroup_subtree import _rooted_on_edge, _unrooted_from_state
    adj, names, nb, root_edge = _unrooted_from_state(copy.deepcopy(tt))
    if root_edge is not None:
        t0 = _rooted_on_edge(adj, names, root_edge[0], root_edge[1], root_edge[2])
    else:
        t0 = copy.deepcopy(fx['tree'])
    trees = [t0] + [_rooted_on_edge(adj, names, a, b_) for a, b_ in nb]
    return [brute(fx, _with_outgroup_clade(t, th)) for t, th in zip(trees, term_lengths)]


def test_attachment_mixture_transport_equals_full_tree(fx):
    """Each transported term equals the rerooted-TreeAnc term of attachment_mixture_scores and an independent
    full-tree Felsenstein; the surrogate message leaves the combined score equal to the ingroup score."""
    from treetime.outgroup_subtree import AttachmentMixture, attachment_mixture_scores
    o = og3(fx)
    tt = copy.deepcopy(fx['tt'])
    _, det = attachment_mixture_scores(fx['b'], tt, o, ws=(0.0,), lengths=LENGTHS3, fit=False)
    m = AttachmentMixture(tt, o)
    vals = [m.value(j, LENGTHS3) for j in range(m.n_terms)]
    assert m.n_terms == 1 + det['n_neighbours'] >= 3
    assert np.allclose(vals, [det['at_root']] + det['neighbours'], atol=1e-8, rtol=0)
    assert np.allclose(vals, _brute_mixture_terms(fx, tt, [LENGTHS3] * m.n_terms), atol=1e-7, rtol=0)
    # mixture formula and responsibilities
    from scipy.special import logsumexp
    n = m.n_terms - 1
    assert abs(m.score(0.3, vals) - logsumexp([np.log(0.7) + vals[0]] + [np.log(0.3 / n) + v for v in vals[1:]])) < 1e-10
    r = m.responsibilities(0.3, vals)
    assert abs(r.sum() - 1) < 1e-12 and m.score(0.0, vals) == vals[0]
    # surrogate: sum_a pi l(a) mu(a) = sum_a pi l(a) per site, so the combined score is the ingroup score
    mu = m.surrogate_messages(0.3, [LENGTHS3] * m.n_terms, vals)
    st = copy.deepcopy(tt)
    st.set_root_state_messages(None)
    ingroup = fresh_marginal_score(st)
    st.set_root_state_messages(mu)
    assert abs(fresh_marginal_score(st) - ingroup) < 1e-7
    # all responsibility on the root term: the plain message up to a per-site constant
    mu0 = m.surrogate_messages(0.0, [LENGTHS3] * m.n_terms, vals)
    plain = o.messages(LENGTHS3)
    ratio = mu0 / plain
    assert np.allclose(ratio, ratio[:, :1], rtol=1e-9, atol=0)
    # fitting the lengths per term never loses
    lens, fvals = m.fit(x0=LENGTHS3)
    assert all(f >= v - 1e-9 for f, v in zip(fvals, vals))
    assert np.allclose(fvals, [m.value(j, th) for j, th in enumerate(lens)], atol=1e-10, rtol=0)


@pytest.fixture(scope='module')
def mixture_runs():
    nwk, aln, dates, _ = load('binary20')
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    L = len(next(iter(seqs.values())))
    ogs = _outgroups(seqs, L)
    kw = dict(regression_k=3, attach_k=2, fit=FitConfig(iterations=2), outer_iterations=2)
    nw = '((O1,O2),O3);'
    plain = rank_with_outgroup_subtree(b, OutgroupSubtree(b.gtr, ogs, tree=nw), attach_mixture_w=0.0, **kw)
    zero = rank_with_outgroup_subtree(b, OutgroupSubtree(b.gtr, ogs, tree=nw), attach_mixture_w=0.0,
                                      stage1=plain.stage1, **kw)
    tiny = rank_with_outgroup_subtree(b, OutgroupSubtree(b.gtr, ogs, tree=nw), attach_mixture_w=1e-6,
                                      stage1=plain.stage1, **kw)
    mixed = rank_with_outgroup_subtree(b, OutgroupSubtree(b.gtr, ogs, tree=nw), attach_mixture_w=0.2,
                                       stage1=plain.stage1, **kw)
    fx_ = dict(b=b, seqs=seqs, L=L, ogs=ogs)
    return fx_, plain, zero, tiny, mixed


def test_attach_mixture_w0_is_stage3(mixture_runs):
    _, plain, zero, _, _ = mixture_runs
    key = lambda res: [(r.candidate_id, r.score, r.best_run.extra.get('outgroup_lengths')) for r in res.stage3.results]  # noqa: E731
    assert key(zero) == key(plain)
    assert all('attach_mixture_w' not in r.best_run.extra for r in zero.stage3.results)
    assert zero.config['attach_mixture_w'] == 0.0


def test_attach_mixture_small_w_matches_stage3_where_the_root_term_dominates(mixture_runs):
    """With w -> 0 the surrogate is the Stage 3 message up to per-site constants: candidates whose
    responsibility sits on the root attachment get the Stage 3 score back."""
    _, plain, _, tiny, _ = mixture_runs
    by0 = {r.candidate_id: r for r in plain.stage3.results}
    checked = 0
    for r in tiny.stage3.results:
        e = r.best_run.extra
        assert e['attach_mixture_w'] == 1e-6 and abs(sum(e['mixture_responsibilities']) - 1) < 1e-9
        if e['mixture_responsibilities'][0] > 1 - 1e-6:
            assert abs(r.score - by0[r.candidate_id].score) < 1e-3
            checked += 1
    assert checked >= 1


def test_attach_mixture_score_is_the_direct_mixture_of_its_state(mixture_runs):
    """The reported score of every candidate equals the mixture computed directly (independent Felsenstein
    over uncompressed columns, one full tree per attachment) at its kept state and reported lengths."""
    from scipy.special import logsumexp
    fx_, _, _, _, mixed = mixture_runs
    w = 0.2
    for r in mixed.stage3.results[:2]:
        st, e = r.best_run.state, r.best_run.extra
        assert st.root_state_message is not None  # carries the plain message of the root attachment
        fxs = dict(fx_, tt=st, tree=_clean_tree(st))
        terms = _brute_mixture_terms(fxs, st, [np.asarray(x) for x in e['mixture_lengths']])
        n = len(terms) - 1
        direct = logsumexp([np.log1p(-w) + terms[0]] + [np.log(w / n) + t for t in terms[1:]])
        assert abs(direct - r.score) < 1e-6
        assert np.allclose(terms, [e['mixture_at_root']] + e['mixture_neighbours'], atol=1e-6, rtol=0)
        assert e['outgroup_lengths'] == e['mixture_lengths'][0]
    assert mixed.config['attach_mixture_w'] == w


def _clean_tree(tt):
    def clean(n_):
        cl = Clade(name=n_.name if n_.is_terminal() else None, clades=[clean(x) for x in n_.clades])
        if n_ is not tt.tree.root:
            cl.branch_length = tt._branch_length_to_gtr(n_)
        return cl
    return Tree(root=clean(tt.tree.root), rooted=True)


def test_attach_mixture_hand_checkable_star():
    """Three ingroup leaves on a star rooted at the centre, one outgroup leaf, JC, two columns: the
    mixture of the attachment at the centre and at the midpoint of each leaf edge, written out by hand."""
    from scipy.linalg import expm
    from treetime import GTR
    from treetime.outgroup_subtree import AttachmentMixture
    recs = [SeqRecord(Seq(s), id=k, description='') for k, s in (('A', 'AC'), ('B', 'AA'), ('C', 'GA'))]
    gtr = GTR.standard('JC69', alphabet='nuc_nogap')
    t = Tree(root=Clade(clades=[Clade(name='A', branch_length=0.1), Clade(name='B', branch_length=0.2),
                                Clade(name='C', branch_length=0.3)]), rooted=True)
    ta = TreeAnc(tree=t, aln=MultipleSeqAlignment(recs), gtr=gtr, verbose=0)
    o = OutgroupSubtree(gtr, {'O': 'AG'})
    d = 0.15
    m = AttachmentMixture(ta, o)
    vals = [m.value(j, np.array([d])) for j in range(m.n_terms)]
    # by hand: P(t) = expm(Q t) with JC rate normalised to one substitution per unit time
    Q = (np.ones((4, 4)) - 4 * np.eye(4)) / 3.0
    P = lambda x: expm(Q * x)  # noqa: E731
    idx = {c: i for i, c in enumerate('ACGT')}
    leaves = {'A': ('AC', 0.1), 'B': ('AA', 0.2), 'C': ('GA', 0.3)}
    pi = np.full(4, 0.25)

    def col(i, at):
        oc = idx['AG'[i]]
        total = 0.0
        for a in range(4):  # centre state
            f = pi[a]
            for name, (s, bl) in leaves.items():
                x = idx[s[i]]
                if name == at:  # outgroup at the midpoint X of this leaf edge
                    f *= sum(P(bl / 2)[a, y] * P(d)[y, oc] * P(bl / 2)[y, x] for y in range(4))
                else:
                    f *= P(bl)[a, x]
            if at is None:
                f *= P(d)[a, oc]
            total += f
        return np.log(total)
    expect = [col(0, None) + col(1, None)] + [col(0, k) + col(1, k) for k in ('A', 'B', 'C')]
    assert m.n_terms == 4
    assert np.allclose(sorted(vals[1:]), sorted(expect[1:]), atol=1e-10, rtol=0)
    assert abs(vals[0] - expect[0]) < 1e-10
    w = 0.3
    assert abs(m.score(w, vals) - np.log(0.7 * np.exp(expect[0]) + 0.1 * sum(np.exp(expect[1:])))) < 1e-10


@pytest.mark.parametrize('case', ['edge', 'collapsed'])
def test_attachment_mixture_transport_edge_and_collapsed_roots(case):
    """Transport on edge-rooted states, including a fit that collapsed onto a node (poly20, edge L17-INT_0005):
    terms equal the rerooted-TreeAnc terms and an independent Felsenstein; a collapsed root has its node's
    neighbours, so it scores like the node-rooted fit of the same tree."""
    from treetime.outgroup_subtree import AttachmentMixture, attachment_mixture_scores
    nwk, aln, dates, _ = load('poly20')
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    L = len(next(iter(seqs.values())))
    cand = b.edge('L17', 0.5, 'INT_0005') if case == 'collapsed' else b.edge('INT_0004', 0.5)
    tt = evaluate_root_candidate(b, cand, FitConfig(iterations=2)).state
    kids = tt.tree.root.clades
    assert len(kids) == 2
    collapsed = min(c.branch_length for c in kids) * L <= 0.05
    assert collapsed == (case == 'collapsed')
    fxs = dict(b=b, tt=tt, seqs=seqs, L=L, ogs=_outgroups(seqs, L))
    fxs['tree'] = _clean_tree(tt)
    o = og3(fxs)
    _, det = attachment_mixture_scores(b, tt, o, ws=(0.0,), lengths=LENGTHS3, fit=False)
    m = AttachmentMixture(tt, o)
    vals = [m.value(j, LENGTHS3) for j in range(m.n_terms)]
    assert np.allclose(vals, [det['at_root']] + det['neighbours'], atol=1e-8, rtol=0)
    assert np.allclose(vals, _brute_mixture_terms(fxs, tt, [LENGTHS3] * m.n_terms), atol=1e-7, rtol=0)
    if collapsed:  # same neighbour set as the node-rooted fit of INT_0005
        tn = evaluate_root_candidate(b, b.node('INT_0005'), FitConfig(iterations=2)).state
        assert AttachmentMixture(tn, o).n_terms == m.n_terms


def test_surrogate_message_is_responsibility_weighted(fx):
    from scipy.special import logsumexp
    from treetime.outgroup_subtree import AttachmentMixture
    o = og3(fx)
    m = AttachmentMixture(copy.deepcopy(fx['tt']), o)
    lens = [LENGTHS3 * (1 + 0.3 * j) for j in range(m.n_terms)]
    vals = [m.value(j, th) for j, th in enumerate(lens)]
    w, n = 0.3, m.n_terms - 1
    lw = np.array([np.log1p(-w)] + [np.log(w / n)] * n) + np.array(vals)
    r = np.exp(lw - logsumexp(lw))
    assert np.allclose(m.responsibilities(w, vals), r, atol=1e-12)
    l_in = np.sum(m.G['pi_l'], axis=1)
    expect = sum(r[j] * m.site_messages(j, lens[j]) * (l_in / np.sum(m.G['pi_l'] * m.site_messages(j, lens[j]), axis=1))[:, None]
                 for j in range(m.n_terms))
    assert np.allclose(m.surrogate_messages(w, lens, vals), expect[m.group_of_position], rtol=1e-12, atol=0)


def test_attach_mixture_never_below_posthoc_start_and_keeps_plain_message(mixture_runs):
    """Each candidate's in-fit score is at least the post-hoc mixture of its Stage 1 state (the warm start), and
    the kept state carries the plain message of the root attachment (its fresh combined score is that term)."""
    from treetime.outgroup_subtree import AttachmentMixture
    fx_, plain, _, _, mixed = mixture_runs
    s1 = {r.candidate_id: r for r in mixed.stage1.results}
    og = OutgroupSubtree(fx_['b'].gtr, fx_['ogs'], tree='((O1,O2),O3);')
    for r in mixed.stage3.results:
        st1 = s1[r.candidate_id].best_run.state
        m = AttachmentMixture(st1, og)
        lens, vals = m.fit(term_starts=[np.asarray(x) for x in r.best_run.extra['mixture_lengths']])
        assert r.score >= m.score(0.2, vals) - 1e-3
        st = copy.deepcopy(r.best_run.state)
        assert abs(fresh_marginal_score(st) - r.best_run.extra['mixture_at_root']) < 1e-6


def test_attach_mixture_w_default_is_0p01_and_0_without_length_fit():
    from treetime.outgroup_subtree import DEFAULT_ATTACH_MIXTURE_W
    o = None  # the constructor only stores the outgroup
    assert DEFAULT_ATTACH_MIXTURE_W == 0.01
    assert OutgroupEvaluator(o).w == 0.01
    assert OutgroupEvaluator(o, fit_lengths=False).w == 0.0
    assert OutgroupEvaluator(o, attach_mixture_w=0.0).w == 0.0
    with pytest.raises(ValueError):
        OutgroupEvaluator(o, fit_lengths=False, attach_mixture_w=0.1)
