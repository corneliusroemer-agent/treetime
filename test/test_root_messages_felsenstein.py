"""Root-state messages against an independent brute-force Felsenstein calculation.

What was worked out:
* Scores must use the lengths TreeTime itself propagates with, ``tt._branch_length_to_gtr(node)``
  (mutation_length when ``use_mutation_length`` is set, else branch_length, floored at
  MIN_BRANCH_LENGTH * one_mutation). The fixture's fitted TreeTime has use_mutation_length False but
  mutation_length != branch_length, so the tree handed to 2b/3/4 gets those lengths copied in.
* The brute force works on uncompressed columns, so it also checks that pattern refinement (2b) and
  the (pattern, message) sidecar (2a) lose nothing; 'N' and '-' have all-ones profiles (neutral messages).
* Method 3 (a real outgroup leaf in a TreeAnc) scores an 'N' at an ingroup-constant column as the constant
  state (pre-existing "constant up to ambiguous" compression), so the outgroup's N sit at variable columns.
* P(i | j, t) = gtr.expQt(t)[i, j]; a child message is expQt(t).T @ profile, root prior is gtr.Pi.
"""
import copy
import sys
import os

import numpy as np
import pytest
from Bio import AlignIO, Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.BaseTree import Clade

sys.path.insert(0, os.path.dirname(__file__))
from test_root_candidates import load  # noqa: E402

from treetime import config as ttconf  # noqa: E402
from treetime.root_candidates import UnrootedBaseline, evaluate_root_candidate, FitConfig  # noqa: E402
from treetime.root_messages import outgroup_leaf_messages, root_message_correction, score_tree_with_messages  # noqa: E402
from treetime.treeanc import TreeAnc  # noqa: E402

D = 0.03
TOL = 1e-8


@pytest.fixture(scope='module', params=['binary20', 'poly20'])
def fx(request):
    nwk, aln, dates, _ = load(request.param)
    b = UnrootedBaseline(nwk, aln, dates, gtr='JC69')
    c = b.all_candidates()[3]
    tt = evaluate_root_candidate(b, c, FitConfig(iterations=2)).state
    seqs = {r.id: str(r.seq).upper() for r in AlignIO.read(aln, 'fasta')}
    L = len(next(iter(seqs.values())))
    cons = []
    for i in range(L):
        col = [s[i] for s in seqs.values() if s[i] in 'ACGT']
        cons.append(max('ACGT', key=col.count) if col else 'A')
    rng = np.random.default_rng(7)
    og = np.array(cons)
    for i in np.nonzero(rng.random(L) < 0.1)[0]:
        og[i] = rng.choice([x for x in 'ACGT' if x != og[i]])
    # 'N' only at ingroup-variable columns: TreeAnc's compression (sequence_data.py, "one state plus
    # ambiguous") folds an N at an otherwise constant column into the constant pattern, i.e. scores it as
    # that state, which differs from the all-ones profile that the messages and the brute force use.
    variable = [i for i in range(L) if len({s_[i] for s_ in seqs.values()} - set('N-')) > 1]
    og[[variable[0], variable[len(variable) // 2], variable[-1]]] = 'N'
    og[[5, 120]] = '-'  # constant columns too: '-' keeps its own pattern
    og = ''.join(og)
    # clean tree (no cached TreeTime attributes) with the lengths TreeTime propagates with
    def clean(n_):
        c = Clade(name=n_.name if n_.is_terminal() else None, clades=[clean(x) for x in n_.clades])
        if n_ is not tt.tree.root:
            c.branch_length = tt._branch_length_to_gtr(n_)
        return c
    tree = Phylo.BaseTree.Tree(root=clean(tt.tree.root), rooted=True)
    lens = None
    return dict(b=b, tt=tt, seqs=seqs, L=L, og=og, tree=tree, lens=lens)


def brute_force(tree, seqs, gtr, one_mutation, og=None, d=D):
    """Felsenstein pruning per uncompressed column; optional outgroup leaf on the root."""
    L = len(next(iter(seqs.values())))
    alpha = list(gtr.alphabet)
    pm = {k: np.asarray(v, float) for k, v in gtr.profile_map.items()}

    def P(t):
        return gtr.expQt(max(ttconf.MIN_BRANCH_LENGTH * one_mutation, t))

    def up(node, i):
        if node.is_terminal():
            return pm[seqs[node.name][i]]
        v = np.ones(len(alpha))
        for ch in node.clades:
            v = v * (P(ch.branch_length).T @ up(ch, i))  # sum_x P(x|a,t) l_ch(x)
        return v

    total = 0.0
    for i in range(L):
        v = np.ones(len(alpha))
        for ch in tree.root.clades:
            v = v * (P(ch.branch_length).T @ up(ch, i))
        if og is not None:
            v = v * (P(d).T @ pm[og[i]])
        total += np.log(np.sum(gtr.Pi * v))
    return total


def all_four(fx, m):
    tt, b = fx['tt'], fx['b']
    s1 = root_message_correction(copy.deepcopy(tt), m)['combined']
    s2 = score_tree_with_messages(b, fx['tree'], m)[0]
    # 3: real outgroup leaf in a TreeAnc
    tree3 = copy.deepcopy(fx['tree'])
    tree3.root.clades.append(Clade(branch_length=D, name='OUTGROUP'))
    recs = [SeqRecord(Seq(s), id=k, description='') for k, s in fx['seqs'].items()]
    recs.append(SeqRecord(Seq(fx['og']), id='OUTGROUP', description=''))
    ta = TreeAnc(tree=tree3, aln=MultipleSeqAlignment(recs), gtr=copy.deepcopy(b.gtr), verbose=0)
    ta.postorder_traversal_marginal()
    ta.total_LH_and_root_sequence()
    s3 = float(ta.tree.total_sequence_LH)
    s4 = brute_force(fx['tree'], fx['seqs'], b.gtr, tt.one_mutation, og=fx['og'])
    return s1, s2, s3, s4


def test_outgroup_four_ways(fx):
    m = outgroup_leaf_messages(fx['b'].gtr, fx['og'], D)
    s = all_four(fx, m)
    print('scores', s, 'max abs diff', max(s) - min(s))
    for x in s[1:]:
        assert abs(x - s[0]) < TOL


def test_messages_are_expQt_columns(fx):
    gtr = fx['b'].gtr
    m = outgroup_leaf_messages(gtr, fx['og'], D)
    i = fx['og'].index('A') if 'A' in fx['og'] else 0
    prof = gtr.profile_map[fx['og'][i]]
    expect = np.array([sum(prof[x] * gtr.expQt(D)[x, a] for x in range(len(prof))) for a in range(len(prof))])
    assert np.allclose(m[i], expect, atol=1e-14)


def test_neutral_messages_equal_ingroup(fx):
    tt, b = fx['tt'], fx['b']
    ones = np.ones((fx['L'], len(b.gtr.alphabet)))
    r = root_message_correction(copy.deepcopy(tt), ones)
    assert r['combined'] == r['ingroup'] and r['correction'] == 0.0
    s2, ing2, corr2 = score_tree_with_messages(b, fx['tree'], ones)
    assert s2 == ing2 and corr2 == 0.0
    assert abs(s2 - brute_force(fx['tree'], fx['seqs'], b.gtr, tt.one_mutation)) < TOL
    assert abs(r['ingroup'] - s2) < TOL


def test_subset_positions_with_constant_columns(fx):
    tt, b, L = fx['tt'], fx['b'], fx['L']
    cmap = tt.data.full_to_compressed_sequence_map
    counts = np.bincount(cmap)
    # constant-column patterns: the largest pattern groups many identical columns
    big = int(np.argmax(counts))
    cols = np.nonzero(cmap == big)[0]
    assert len(cols) >= 4
    rng = np.random.default_rng(11)
    other = rng.choice(np.nonzero(cmap != big)[0], 6, replace=False)
    n = len(b.gtr.alphabet)
    msgs = {}
    for p in list(cols[:6]) + list(other):
        v = rng.random(n) + 0.05  # distinct, not max-normalised: scale enters the score
        msgs[int(p)] = v
    full = np.ones((L, n))
    for p, v in msgs.items():
        full[p] = v
    # brute force with explicit per-column messages: reuse the outgroup path via a message-aware variant
    s1 = root_message_correction(copy.deepcopy(tt), msgs)
    s2 = score_tree_with_messages(b, fx['tree'], msgs)[0]
    assert s1['n_groups'] >= 5
    ref = brute_force_messages(fx['tree'], fx['seqs'], b.gtr, tt.one_mutation, full)
    print('subset diffs', s1['combined'] - ref, s2 - ref)
    assert abs(s1['combined'] - ref) < TOL and abs(s2 - ref) < TOL
    # array form of the same messages agrees with the dict form
    assert abs(root_message_correction(copy.deepcopy(tt), full)['combined'] - s1['combined']) < TOL


def brute_force_messages(tree, seqs, gtr, one_mutation, msgs):
    L = msgs.shape[0]
    alpha = len(gtr.alphabet)
    pm = {k: np.asarray(v, float) for k, v in gtr.profile_map.items()}
    P = lambda t: gtr.expQt(max(ttconf.MIN_BRANCH_LENGTH * one_mutation, t))  # noqa: E731

    def up(node, i):
        if node.is_terminal():
            return pm[seqs[node.name][i]]
        v = np.ones(alpha)
        for ch in node.clades:
            v = v * (P(ch.branch_length).T @ up(ch, i))
        return v

    total = 0.0
    for i in range(L):
        v = msgs[i].copy()
        for ch in tree.root.clades:
            v = v * (P(ch.branch_length).T @ up(ch, i))
        total += np.log(np.sum(gtr.Pi * v))
    return total
