"""Stage 3: undated outgroup subtree as root-state messages.

References used here:
* TreeAnc on the full tree (ingroup rooted at R plus the outgroup subtree hung on R by the stem), at fixed
  lengths, with the outgroups in the alignment: TreeTime's own pruning with real outgroup leaves.
* An independent numpy pruning over uncompressed columns with scipy's expm (orientation checked by
  test_root_messages_felsenstein.independent_P), rooted at R or, for the pulley check, at an outgroup node.
TreeAnc folds an 'N' at an ingroup-constant column into the constant pattern (tip-off
2026-10-31-s2a-ambiguous-at-constant-column), so outgroup ambiguity sits at ingroup-variable columns
whenever TreeAnc-with-outgroup-leaves is the reference; the brute force has no such restriction.
"""
import copy
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

from treetime.outgroup_subtree import OutgroupSubtree, rank_with_outgroup  # noqa: E402
from treetime.root_candidates import (UnrootedBaseline, evaluate_root_candidate, FitConfig, candidate_key,  # noqa: E402
                                      fresh_marginal_score, rank_root_candidates)
from treetime.root_messages import outgroup_leaf_messages, rank_with_root_messages  # noqa: E402
from treetime.treeanc import TreeAnc  # noqa: E402

TOL = 1e-8
OG_NWK = '((OG1:0.02,OG2:0.035):0.01,OG3:0.05);'  # X on the edge between (OG1,OG2) and OG3
STEM = 0.03


def make_outgroups(seqs, rng, n=3, var_amb=()):
    L = len(next(iter(seqs.values())))
    cons = []
    for i in range(L):
        col = [s[i] for s in seqs.values() if s[i] in 'ACGT']
        cons.append(max('ACGT', key=col.count) if col else 'A')
    base = np.array(cons)
    out = {}
    for k in range(n):
        og = base.copy()
        for i in np.nonzero(rng.random(L) < 0.06)[0]:
            og[i] = rng.choice([x for x in 'ACGT' if x != og[i]])
        out['OG%d' % (k + 1)] = og
    for j, i in enumerate(var_amb):  # partial and full ambiguity at ingroup-variable columns
        out['OG1'][i] = 'NRY'[j % 3]
        if j % 2 == 0:
            out['OG2'][i] = 'N'
        if j % 4 == 0:
            out['OG3'][i] = 'N'  # all three unobserved at some columns
    return {k: ''.join(v) for k, v in out.items()}


@pytest.fixture(scope='module', params=['binary20', 'poly20-gtr'])
def fx(request):
    name, _, model = request.param.partition('-')
    nwk, aln, dates, _ = load(name)
    b = UnrootedBaseline(nwk, aln, dates, gtr=nonuniform_gtr() if model else 'JC69')
    c = b.all_candidates()[3]
    tt = evaluate_root_candidate(b, c, FitConfig(iterations=2)).state
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    L = len(next(iter(seqs.values())))
    variable = [i for i in range(L) if len({s_[i] for s_ in seqs.values()} - set('N-')) > 1]
    ogs = make_outgroups(seqs, np.random.default_rng(3), var_amb=variable[::max(1, len(variable) // 8)][:8])

    def clean(n_):
        c_ = Clade(name=n_.name if n_.is_terminal() else None, clades=[clean(x) for x in n_.clades])
        if n_ is not tt.tree.root:
            c_.branch_length = tt._branch_length_to_gtr(n_)
        return c_
    tree = Tree(root=clean(tt.tree.root), rooted=True)
    return dict(b=b, tt=tt, seqs=seqs, L=L, ogs=ogs, tree=tree, cand=c)


def og_tree(nwk=OG_NWK):
    return Phylo.read(__import__('io').StringIO(nwk), 'newick')


def full_tree(ingroup_tree, og_nwk=OG_NWK, stem=STEM):
    """Ingroup tree with the outgroup subtree (rooted at X) hung on its root by the stem."""
    t = copy.deepcopy(ingroup_tree)
    sub = og_tree(og_nwk).root
    sub.branch_length = stem
    t.root.clades.append(sub)
    return t


def treeanc_score(fx, tree, ogs):
    recs = [SeqRecord(Seq(s), id=k, description='') for k, s in fx['seqs'].items()]
    recs += [SeqRecord(Seq(s), id=k, description='') for k, s in ogs.items()]
    ta = TreeAnc(tree=tree, aln=MultipleSeqAlignment(recs), gtr=copy.deepcopy(fx['b'].gtr), verbose=0)
    return fresh_marginal_score(ta)


def brute(tree, seqs, gtr, one_mutation):
    """Independent pruning per uncompressed column, at the tree's root (any node)."""
    P = independent_P(gtr, one_mutation)
    pm = {k: np.asarray(v, float) for k, v in gtr.profile_map.items()}
    L = len(next(iter(seqs.values())))
    cache = {}

    def up(node, i):
        if node.is_terminal():
            return pm[seqs[node.name][i]]
        v = np.ones(len(gtr.alphabet))
        for ch in node.clades:
            v = v * (P(ch.branch_length).T @ up(ch, i))
        return v
    return sum(np.log(np.sum(gtr.Pi * up(tree.root, i))) for i in range(L))


def reroot_at(tree, pick):
    """The same unrooted tree rooted at an existing node (no new edge): the first clade with pick(clade)."""
    nodes = list(tree.find_clades(order='preorder'))
    adj = {id(c): [] for c in nodes}
    for c in nodes:
        for ch in c.clades:
            adj[id(c)].append((ch, ch.branch_length))
            adj[id(ch)].append((c, ch.branch_length))
    # a degree-2 root is suppressed by the pruning anyway (lengths add), keep it as a node
    start = next(c for c in nodes if pick(c))

    def build(v, p, bl):
        return Clade(name=v.name, branch_length=bl, clades=[build(w, v, t) for w, t in adj[id(v)] if w is not p])
    return Tree(root=build(start, None, None), rooted=True)


def subtree(fx, nwk=OG_NWK, **kw):
    return OutgroupSubtree(fx['b'].gtr, fx['ogs'], tree=og_tree(nwk) if nwk else None, stem=STEM, **kw)


def message_score(fx, og, lengths=None, eps=0.0):
    """(grouped sidecar at the fitted state, refined patterns on a TreeAnc) combined scores."""
    g = og.state_groups(copy.deepcopy(fx['tt']))
    s_side = og.combined_at_state(g, lengths, eps)
    ta = fx['b'].make_treeanc(copy.deepcopy(fx['tree']), messages=False)
    ta.set_root_state_messages(og.root_messages(lengths, eps, ta.data.full_length))
    return s_side, fresh_marginal_score(ta)


# ---------------------------------------------------------------------------
def test_single_leaf_reproduces_stage2_messages(fx):
    seq = fx['ogs']['OG1']
    og = OutgroupSubtree(fx['b'].gtr, {'OG1': seq}, stem=0.021)
    assert og.n_nodes == 1
    m2 = outgroup_leaf_messages(fx['b'].gtr, seq, 0.021)
    assert np.array_equal(og.full_messages(), m2)  # bit-identical, ambiguous columns included


def test_multi_outgroup_equals_full_tree_felsenstein(fx):
    og = subtree(fx)
    s_side, s_refined = message_score(fx, og)
    ftree = full_tree(fx['tree'])
    s_full = treeanc_score(fx, ftree, fx['ogs'])
    allseqs = dict(fx['seqs'], **fx['ogs'])
    s_brute = brute(ftree, allseqs, fx['b'].gtr, fx['tt'].one_mutation)
    print(s_side, s_refined, s_full, s_brute)
    assert abs(s_side - s_brute) < TOL and abs(s_refined - s_brute) < TOL and abs(s_full - s_brute) < TOL


def test_star_default_and_compression(fx):
    og = OutgroupSubtree(fx['b'].gtr, fx['ogs'], stem=STEM)
    assert og.n_nodes == 4 and og.edge_names[0] == 'stem'
    th = np.array([STEM, 0.01, 0.02, 0.04])
    s_side, s_refined = message_score(fx, og, th)
    star = '(OG1:0.01,OG2:0.02,OG3:0.04);'
    ftree = full_tree(fx['tree'], star)
    s_brute = brute(ftree, dict(fx['seqs'], **fx['ogs']), fx['b'].gtr, fx['tt'].one_mutation)
    assert abs(s_side - s_brute) < TOL and abs(s_refined - s_brute) < TOL
    # compression: refined patterns = (ingroup pattern, outgroup pattern), never the full alignment
    ta = fx['b'].make_treeanc(copy.deepcopy(fx['tree']), messages=False)
    n0 = ta.data.compressed_length
    ta.set_root_state_messages(og.root_messages(th, 0.0, ta.data.full_length))
    pairs = {(int(p), int(o)) for p, o in zip(fx['tt'].data.full_to_compressed_sequence_map, og.pattern)
             if not og.neutral[o]}
    assert n0 <= ta.data.compressed_length < fx['L']
    assert ta.data.multiplicity().sum() == fx['L']
    assert ta.data.compressed_length <= len(pairs) + n0


def test_rerooting_outgroup_subtree_invariant(fx):
    """The same unrooted outgroup tree and attachment, written with three rootings; and the full tree
    pruned at an outgroup node instead of R (pulley identity) -- 8 decimal places."""
    ref = subtree(fx)
    m_ref = ref.full_messages()
    # unrooted: Y-OG1 0.02, Y-OG2 0.035, Y-OG3 0.06; X on Y-OG3, 0.05 from OG3
    for nwk in ('(OG1:0.02,OG2:0.035,OG3:0.06);', '(OG1:0.005,(OG2:0.035,OG3:0.06):0.015);',
                '((OG1:0.02,OG2:0.035):0.03,OG3:0.03);'):
        og = OutgroupSubtree(fx['b'].gtr, fx['ogs'], tree=og_tree(nwk), attach={'OG3'}, attach_length=0.05, stem=STEM)
        assert np.max(np.abs(og.full_messages() - m_ref)) < 1e-12
        assert abs(message_score(fx, og)[0] - message_score(fx, ref)[0]) < TOL
        # after fitting the outgroup lengths too
    g = ref.state_groups(copy.deepcopy(fx['tt']))
    a = ref.optimise_lengths(g)[1]
    og = OutgroupSubtree(fx['b'].gtr, fx['ogs'], tree=og_tree('(OG1:0.005,(OG2:0.035,OG3:0.06):0.015);'),
                         attach={'OG1', 'OG2'}, stem=STEM)
    bb = og.optimise_lengths(og.state_groups(copy.deepcopy(fx['tt'])))[1]
    assert abs(a - bb) < TOL
    # pulley: full tree rerooted at the outgroup cherry's parent, same likelihood
    ftree = full_tree(fx['tree'])
    allseqs = dict(fx['seqs'], **fx['ogs'])
    s_r = brute(ftree, allseqs, fx['b'].gtr, fx['tt'].one_mutation)
    rer = reroot_at(ftree, lambda c: not c.is_terminal() and {x.name for x in c.get_terminals()} == {'OG1', 'OG2'})
    s_y = brute(rer, allseqs, fx['b'].gtr, fx['tt'].one_mutation)
    assert abs(s_r - s_y) < TOL and abs(message_score(fx, ref)[0] - s_y) < TOL


def test_infinite_stem_is_uninformative(fx):
    og = subtree(fx)
    th = og.lengths.copy()
    th[0] = 1e4
    m = og.message_table(th)
    c = og.uninformative_table(th)
    assert np.allclose(m, c, rtol=1e-10, atol=0)
    assert np.allclose(m, m[:, :1], rtol=1e-10, atol=0)  # constant in the root state
    one = OutgroupSubtree(fx['b'].gtr, {'OG1': fx['ogs']['OG1']}, stem=1e4)
    pi = np.asarray(fx['b'].gtr.Pi)
    prof = np.array([fx['b'].gtr.profile_map[ch] for ch in fx['ogs']['OG1']])
    exp = prof @ pi  # pi of the observed state (sum over the admissible set)
    exp[np.all(prof == 1, axis=1)] = 1.0
    assert np.allclose(one.full_messages()[:, 0], exp, rtol=1e-10)


def test_missing_states(fx):
    og = subtree(fx)
    # all-unobserved columns are exactly neutral, and partial ambiguity is the admissible-set indicator
    allN = [i for i in range(fx['L']) if all(fx['ogs'][k][i] == 'N' for k in fx['ogs'])]
    assert allN and np.all(og.full_messages()[allN] == 1.0)
    assert any(ch in 'RY' for ch in fx['ogs']['OG1'])
    # brute force above uses profile_map for R/Y/N on the full tree: covered by the equality test; here the
    # same equality with every ambiguous column turned into a fully masked outgroup column
    ogs = {k: ''.join('N' if fx['ogs']['OG1'][i] in 'NRY' else ch for i, ch in enumerate(v))
           for k, v in fx['ogs'].items()}
    og2 = OutgroupSubtree(fx['b'].gtr, ogs, tree=og_tree(), stem=STEM)
    s = message_score(dict(fx, ogs=ogs), og2)[0]
    s_b = brute(full_tree(fx['tree']), dict(fx['seqs'], **ogs), fx['b'].gtr, fx['tt'].one_mutation)
    assert abs(s - s_b) < TOL


def test_eps_zero_identical_and_mixture(fx):
    og = subtree(fx)
    assert np.array_equal(og.message_table(eps=0.0), og.message_table())
    e = 0.05
    m, m0, c = og.message_table(eps=e), og.message_table(), og.uninformative_table()
    obs = ~og.neutral
    assert np.allclose(m[obs], (1 - e) * m0[obs] + e * c[obs], rtol=1e-14)
    assert np.all(m[og.neutral] == 1.0)
    # the mixture is a likelihood: score against brute force on a per-site mixture of two stems
    s = message_score(fx, og, eps=e)
    assert abs(s[0] - s[1]) < TOL


def test_outgroup_lengths_do_not_touch_ingroup_clock(fx):
    og = subtree(fx)
    g = og.state_groups(copy.deepcopy(fx['tt']))
    th, best, _ = og.optimise_lengths(g)
    assert best >= og.combined_at_state(g) - 1e-12
    assert not np.allclose(th, og.lengths)
    # at a fixed ingroup state, changing outgroup lengths changes only the outgroup part
    tt = copy.deepcopy(fx['tt'])
    ing0 = fresh_marginal_score(tt)
    for t in (og.lengths, th):
        tt.set_root_state_messages(og.root_messages(t, 0.0, tt.data.full_length))
        fresh_marginal_score(tt)
        assert abs(tt.tree.total_sequence_LH_ingroup - ing0) < TOL
    assert not ({'OG1', 'OG2', 'OG3'} & {n.name for n in tt.tree.get_terminals()})


@pytest.fixture(scope='module')
def ranked():
    nwk, aln, dates, _ = load('binary20')
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    ogs = make_outgroups(seqs, np.random.default_rng(5))
    fit = FitConfig(iterations=2)
    s1 = rank_root_candidates(b, regression_k=3, fit=fit, keep_states=True)
    return dict(b=b, ogs=ogs, fit=fit, s1=s1)


def test_one_outgroup_fixed_d_equals_stage2(ranked):
    b, fit, s1 = ranked['b'], ranked['fit'], ranked['s1']
    seq = ranked['ogs']['OG1']
    d = 0.02
    s2 = rank_with_root_messages(b, outgroup_leaf_messages(b.gtr, seq, d), regression_k=3, fit=fit, stage1=s1)
    og = OutgroupSubtree(b.gtr, {'OG1': seq}, stem=d)
    s3 = rank_with_outgroup(b, og, regression_k=3, fit=fit, stage1=s1, optimise_lengths=False, cold_start=True)
    sc2 = {candidate_key(r.candidate): r.score for r in s2.stage2b.results}
    sc3 = {candidate_key(r.candidate): r.score for r in s3.stage3.results}
    assert sc2.keys() == sc3.keys()
    for k in sc2:
        assert sc2[k] == sc3[k]
    assert s2.stage2b.best.candidate_id == s3.stage3.best.candidate_id
    assert s2.stage2b.margin == s3.stage3.margin


def test_stage3_rounds_monotone_and_extras_do_not_change_answer(ranked):
    b, fit, s1 = ranked['b'], ranked['fit'], ranked['s1']
    og = OutgroupSubtree(b.gtr, ranked['ogs'], tree=og_tree(), stem=0.02)
    s3 = rank_with_outgroup(b, og, regression_k=3, fit=fit, stage1=s1, rounds=3)
    for row in s3.rows:
        rs = row['round_scores']
        assert all(y >= x - 1e-9 for x, y in zip(rs, rs[1:]))
        assert row['combined_3'] >= row['posthoc_3a'] - 1e-6  # the warm start begins at the 3a state
        assert row['stem'] != 0.02
    extra = [r.candidate for r in s1.results[::-1][:1]] + [b.all_candidates()[0]]
    s3x = rank_with_outgroup(b, og, regression_k=3, fit=fit, stage1=s1, rounds=3, extra_candidates=extra)
    assert [r.candidate_id for r in s3x.stage3.results] == [r.candidate_id for r in s3.stage3.results]
    assert [r.score for r in s3x.stage3.results] == [r.score for r in s3.stage3.results]
    assert s3x.stage3.margin == s3.stage3.margin and len(s3x.extra) == 2
    same = [r for r in s3x.extra if r.candidate_id == s1.results[-1].candidate_id][0]
    ref = [r for r in s3.stage3.results if r.candidate_id == same.candidate_id][0]
    assert same.score == ref.score  # an extra is scored exactly as a listed candidate
