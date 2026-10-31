"""Tests for likelihood-based root selection (treetime.root_candidates) and the
regression ranking it draws candidates from (TreeRegression.rank_roots).

Fixtures in test/root_candidates_data/ were simulated on a known dated tree
(JC69, SeqGen). The tree TreeTime sees, ``fitted_tree.nwk``, was estimated from
the alignment with IQ-TREE 3.1.2 in CMAPLE mode (``--pathogen-force``) and every
internal edge shorter than 0.5/L (and every exact zero) was fully collapsed; see
make_fixtures.py and each meta.json. ``true_tree.nwk`` is only the reference
answer and is never fitted.
"""
import csv
import json
import os

import pytest
from Bio import Phylo

from treetime import TreeTime

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'root_candidates_data')


def load(fx):
    d = os.path.join(DATA, fx)
    with open(os.path.join(d, 'meta.json')) as fh:
        meta = json.load(fh)
    with open(os.path.join(d, 'dates.tsv')) as fh:
        dates = {r[0]: float(r[1]) for r in csv.reader(fh, delimiter='\t') if r[0] != 'name'}
    return os.path.join(d, 'fitted_tree.nwk'), os.path.join(d, 'aln.fasta'), dates, meta


# --- regression ranking --------------------------------------------------------


@pytest.mark.parametrize('covariation', [False, True])
@pytest.mark.parametrize('fixed_slope', [False, True])
def test_rank_roots_first_entry_is_find_best_root(covariation, fixed_slope):
    nwk, aln, dates, meta = load('binary20')
    tt = TreeTime(tree=nwk, aln=aln, dates=dates, gtr='JC69', verbose=0)
    treg = tt.setup_TreeRegression(covariation=covariation)
    slope = meta['clock_rate'] if fixed_slope else None
    best = treg.find_best_root(slope=slope)
    ranked = treg.rank_roots(slope=slope)
    assert ranked[0]['node'] is best['node'] and ranked[0]['split'] == best['split']
    assert ranked[0]['chisq'] == best['chisq']
    assert [r['edge_chisq'] for r in ranked] == sorted(r['edge_chisq'] for r in ranked)
    assert all(r['slope'] >= 0 for r in ranked)
    assert len({id(r['node']) for r in ranked}) == len(ranked)
