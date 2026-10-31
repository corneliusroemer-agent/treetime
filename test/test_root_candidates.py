"""Tests for treetime.root_candidates (likelihood root selection, stage 1).

Fixtures in test/root_candidates_data/ are simulated on a known dated tree;
the fitted tree is estimated from the alignment with IQ-TREE (CMAPLE mode) and
near-zero internal edges are collapsed (see make_fixtures.py and meta.json).
"""

import copy
import csv
import json
import os

import numpy as np
import pytest
from Bio import Phylo

from treetime import TreeAnc, TreeTime
from treetime.root_candidates import (
    EdgeRoot,
    NodeRoot,
    RootCandidateEvaluator,
    apply_candidate_root,
    candidate_key,
    canonical_split,
    dedupe_candidates,
    fresh_marginal_score,
    leaf_names,
    node_key,
    restore_best_fitted_state,
    unrooted_degrees,
    unrooted_splits,
    _leafsets,
)

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'root_candidates_data')


def load(fx):
    d = os.path.join(DATA, fx)
    meta = json.load(open(os.path.join(d, 'meta.json')))
    dates = {r[0]: float(r[1]) for r in csv.reader(open(os.path.join(d, 'dates.tsv')), delimiter='\t') if r[0] != 'name'}
    return os.path.join(d, 'fitted_tree.nwk'), os.path.join(d, 'aln.fasta'), dates, meta


def make_ev(fx, iterations=2, **kw):
    nwk, aln, dates, meta = load(fx)
    return RootCandidateEvaluator(nwk, aln, dates, gtr='JC69', clock_rate=meta['clock_rate'], iterations=iterations, **kw)


@pytest.fixture(scope='module')
def ev():
    return make_ev('binary20')


@pytest.fixture(scope='module')
def top(ev):
    """Regression-best candidate on binary20 (an edge with non-zero length)."""
    c = ev.regression_candidates(1)[0][0]
    assert isinstance(c, EdgeRoot)
    return c


def high_degree_node(tree):
    return max(tree.get_nonterminals(), key=lambda n: len(n.clades) + (n is not tree.root))


def dates_by_split(tt):
    sets = _leafsets(tt.tree)
    return {sets[id(n)]: n.numdate for n in tt.tree.find_clades()}


# --- regression candidates ---------------------------------------------------


def test_regression_ranking_reuses_find_best_root(ev):
    tt = ev.new_treetime(copy.deepcopy(ev.baseline))
    for n in tt.tree.find_clades():
        n.branch_length = n.mutation_length
    treg = tt.setup_TreeRegression(covariation=False)
    best = treg.find_best_root(slope=ev.clock_rate)
    ranked = treg.rank_roots(slope=ev.clock_rate)
    assert ranked[0]['node'] is best['node'] and ranked[0]['split'] == best['split']
    assert [r['edge_chisq'] for r in ranked] == sorted(r['edge_chisq'] for r in ranked)


def test_top_regression_candidate_is_treetime_least_squares_root(ev, top):
    tt = ev.new_treetime(copy.deepcopy(ev.baseline))
    tt._set_branch_length_mode('marginal')
    from treetime.root_candidates import treetime_seq_kwargs

    tt.optimize_tree(infer_gtr=False, max_iter=1, **treetime_seq_kwargs())
    tt.reroot('least-squares', clock_rate=ev.clock_rate)
    all_leaves = leaf_names(tt.tree)
    kids = tt.tree.root.clades
    assert len(kids) == 2
    assert canonical_split(frozenset(n.name for n in kids[0].get_terminals()), all_leaves) == top.split


# --- 1. explicit root reproduces a fixed-root TreeTime fit -------------------


@pytest.mark.parametrize('max_iter', [0, 2])
def test_explicit_root_reproduces_fixed_root_treetime_run(ev, top, max_iter):
    fit = ev.fit(top)
    tt = TreeTime(tree=apply_candidate_root(ev.baseline, top), aln=ev.aln, dates=ev.dates, gtr='JC69',
                  verbose=0, rng_seed=ev.rng_seed)
    tt.run(root=None, max_iter=max_iter, infer_gtr=False, fixed_clock_rate=ev.clock_rate, branch_length_mode='marginal',
           resolve_polytomies=False, prune_short=False, Tc=None, time_marginal='never')
    performed = len(tt.trace_run) - 1  # run() may stop early on ndiff == 0
    assert fresh_marginal_score(tt) == pytest.approx(fit.scores[performed], abs=1e-9)
    assert tt.tree.positional_LH == pytest.approx(fit.positional_LH[performed], abs=1e-9)


# --- 2 + 8. order independence, exhaustive reproducibility -------------------


def test_exhaustive_reproducible_and_order_independent():
    ev = make_ev('poly20', iterations=1, keep_states=False)
    fwd = ev.exhaustive()
    cands = ev.all_candidates()
    assert len(fwd.results) == len(cands)
    rev = ev.finish([ev.evaluate(c, 'exhaustive') for c in reversed(cands)], close_delta=None)
    a = {r.candidate.id: r.score for r in fwd.results}
    b = {r.candidate.id: r.score for r in rev.results}
    assert a == b  # exact equality
    assert [r.candidate.id for r in fwd.results] == [r.candidate.id for r in rev.results]
    ok = [s for s in a.values() if np.isfinite(s)]
    assert len(ok) == len(a)


# --- 3 + 4. fresh score at saved branch lengths, reproducible best state -----


def test_fresh_score_matches_saved_branch_lengths_and_state_reproducible(ev, top):
    r = ev.evaluate(top)
    state = restore_best_fitted_state(r)
    run = r.best_run
    assert run.best_score == max(run.scores)
    assert fresh_marginal_score(copy.deepcopy(state)) == pytest.approx(r.score, abs=1e-9)

    # independent evaluation from the saved branch lengths only
    tree = copy.deepcopy(state.tree)
    plain = Phylo.BaseTree.Tree(root=_strip(tree.root))
    ta = TreeAnc(tree=plain, aln=ev.aln, gtr='JC69', verbose=0)
    assert fresh_marginal_score(ta) == pytest.approx(r.score, abs=1e-6)

    # re-fit from scratch: same best iteration, score and dates
    r2 = ev.evaluate(top)
    assert r2.best_run.best_index == run.best_index and r2.score == r.score
    d1, d2 = dates_by_split(state), dates_by_split(restore_best_fitted_state(r2))
    assert d1 == d2


def _strip(clade):
    new = Phylo.BaseTree.Clade(branch_length=float(clade.branch_length or 0.0), name=clade.name)
    new.clades = [_strip(c) for c in clade.clades]
    return new


# --- 5. node and edge semantics ------------------------------------------------


def test_node_root_is_exactly_at_node(ev):
    node = high_degree_node(ev.baseline)
    deg = len(node.clades) + (node is not ev.baseline.root)
    assert deg >= 4
    c = NodeRoot.at(ev.baseline, node.name)
    t = apply_candidate_root(ev.baseline, c)
    assert len(t.root.clades) == deg
    assert node_key(t.root, t) == c.key


def test_edge_root_fraction_and_endpoint_snapping(ev):
    tree = ev.baseline
    clade = [n for n in tree.get_nonterminals() if n is not tree.root and n.branch_length > 0][0]
    parent = [n for n in tree.find_clades() if clade in n.clades][0]
    w = clade.branch_length
    c = EdgeRoot.above(tree, clade, 0.3)
    t = apply_candidate_root(tree, c)
    below = frozenset(x.name for x in clade.get_terminals())
    kids = {frozenset(x.name for x in k.get_terminals()): k.branch_length for k in t.root.clades}
    assert len(kids) == 2 and kids[below] == pytest.approx(0.3 * w)
    assert sum(kids.values()) == pytest.approx(w)
    # endpoints are node roots
    assert EdgeRoot.above(tree, clade, 0.0) == NodeRoot.at(tree, clade)
    assert EdgeRoot.above(tree, clade, 1.0) == NodeRoot.at(tree, parent)
    # a leaf is not a root: its node root is the leaf-end of its edge
    leaf = tree.get_terminals()[0]
    lr = NodeRoot.at(tree, leaf.name)
    assert isinstance(lr, EdgeRoot) and candidate_key(lr) == candidate_key(EdgeRoot.above(tree, leaf, 0.5))
    # same edge at different initial fractions is one candidate
    assert len(dedupe_candidates([c, EdgeRoot.above(tree, clade, 0.7), NodeRoot.at(tree, clade)])) == 2


def test_candidate_identity_is_stable_under_rerooting(ev):
    keys = {candidate_key(c) for c in ev.all_candidates()}
    node = high_degree_node(ev.baseline)
    other = RootCandidateEvaluator(apply_candidate_root(ev.baseline, NodeRoot.at(ev.baseline, node)), ev.aln,
                                   ev.dates, clock_rate=ev.clock_rate)
    assert {candidate_key(c) for c in other.all_candidates()} == keys


# --- 6. fixed topology, high-degree nodes preserved --------------------------


def test_topology_and_polytomy_preserved(ev, top):
    splits, degrees = unrooted_splits(ev.baseline), unrooted_degrees(ev.baseline)
    node = high_degree_node(ev.baseline)
    for c in (top, NodeRoot.at(ev.baseline, node)):
        r = ev.evaluate(c)
        t = restore_best_fitted_state(r).tree
        assert unrooted_splits(t) == splits
        assert unrooted_degrees(t) == degrees


def test_zero_length_root_edge_is_fitted(ev):
    leaf = [n for n in ev.baseline.get_terminals() if n.branch_length == 0]
    if not leaf:
        pytest.skip('no zero-length leaf edge in fixture')
    r = ev.evaluate(EdgeRoot.above(ev.baseline, leaf[0], 0.5))
    assert r.best_run.error is None and np.isfinite(r.score)
    assert any('zero-length' in w for w in r.best_run.warnings)


# --- 7 + 9. known root, close-candidate diagnostics --------------------------


def test_known_root_and_close_candidate_diagnostics(ev):
    _, _, _, meta = load('binary20')
    res = ev.run(n_regression=4, close_delta=3.0)
    true_split = canonical_split(meta['true_root_split'], leaf_names(ev.baseline))
    assert isinstance(res.best.candidate, EdgeRoot) and res.best.candidate.split == true_split
    close = [r for r in res.results if res.best.score - r.score <= 3.0]
    assert len(close) >= 2
    for r in close:
        assert len(r.runs) > 1 and r.optimiser_spread is not None
    far = [r for r in res.results if res.best.score - r.score > 3.0]
    assert all(len(r.runs) == 1 for r in far)
    assert res.verdict.startswith(('resolved', 'unresolved'))
    rows = res.rows()
    assert rows[0]['delta_logL'] == 0 and all(r['delta_logL'] <= 0 for r in rows)
    assert res.to_tsv().count('\n') == len(rows) + 1
    json.loads(res.to_json())


def test_explicit_and_regression_candidates_are_merged(ev, top):
    res = ev.run(candidates=[EdgeRoot(top.split, 0.2)], n_regression=1, close_delta=None)
    assert len(res.results) == 1 and res.best.source == 'explicit+regression'
    assert res.verdict == 'single candidate'


# --- TreeTime's own example data ---------------------------------------------


def test_example_data():
    d = os.path.join(HERE, 'treetime_examples', 'data', 'h3n2_na')
    if not os.path.isdir(d):
        pytest.skip('treetime_examples not checked out')
    from treetime.utils import parse_dates

    dates = parse_dates(os.path.join(d, 'h3n2_na_20.metadata.csv'))
    ev = RootCandidateEvaluator(os.path.join(d, 'h3n2_na_20.nwk'), os.path.join(d, 'h3n2_na_20.fasta'), dates,
                                gtr='JC69', iterations=1, keep_states=False)
    assert ev.clock_rate > 0
    res = ev.run(n_regression=3, close_delta=None)
    assert len(res.results) == 3 and all(np.isfinite(r.score) for r in res.results)
