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
    ev = OutgroupEvaluator(o, outer_iterations=2)
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
