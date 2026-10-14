import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'examples'))


def study():
    return importlib.import_module('recursive_preview_design')


def test_exact_identities_defaults_and_shards():
    s = study()
    cases = s.specs()
    assert len(cases) == len({c['id'] for c in cases}) == 480
    assert {c['n'] for c in cases} == {80, 200}
    assert (cases[160]['scenario'], cases[240]['n']) == ('chain', 200)
    assert cases[240]['analysis_seed'] == 1_700_240_000
    assert len(s.shards('noise')) == 80 and len(s.shards('signal')) == 160
    assert s.shards('signal')[0] == [160, 161]
    c = s.settings(cases[0])
    assert (c.analysis.root, c.analysis.max_depth, c.learner.library) == ('Y', 3, ('elastic_net',))
    assert (c.performance_cv.folds, c.performance_cv.repeats, c.tuning_cv.folds) == (5, 2, 3)
    assert (c.stability.n_resamples, c.thresholds.stability, c.thresholds.unique_relative) == (100, .70, .05)


def test_generator_replays_exact_approved_formulas():
    s = study()
    for i in (0, 160, 320):
        spec = s.specs()[i]
        frame = s.generate(spec)
        innovations = np.random.default_rng(spec['generator_seed']).normal(size=(5, spec['n']))
        x1, e2, ey, z1, z2 = innovations
        x2 = e2 if i == 0 else (.6*x1+.8*e2 if i == 160 else .4*x1+np.sqrt(.84)*e2)
        y = ey if i == 0 else (.8*x2+.6*ey if i == 160 else .6*x1+.6*x2+np.sqrt(.672)*ey)
        np.testing.assert_array_equal(frame.to_numpy(), np.column_stack((y, x1, x2, z1, z2)))
        assert list(frame) == ['Y', 'X1', 'X2', 'Z1', 'Z2']


def test_upstream_misses_and_joint_group_are_not_correct_unique_edges():
    s = study()
    spec = s.specs()[160]
    score = s.structure(spec, [('X2', 'Y')], [], {'Y': 0, 'X2': 1})
    assert score['missing'] == [['X1', 'X2']]
    assert score['depth_matches']['X1'] is False and not score['exact']
    score = s.structure(s.specs()[320], [], [('Y', ('X1', 'X2'))], {'Y':0, 'X1':1, 'X2':1})
    assert score['incorrect'] and not score['exact'] and len(score['missing']) == 2


def test_cp_integer_boundaries_unknowns_and_stage_gate():
    s = study()
    rows = [dict(status='complete', structure=dict(incorrect=i < 2, exact=i < 72)) for i in range(80)]
    cell = s.summarize_cell(rows, signal=True)
    assert cell['qualified'] and cell['incorrect']['upper'] < .10
    rows[2]['structure']['incorrect'] = True
    assert not s.summarize_cell(rows, signal=True)['qualified']
    rows[2]['structure']['incorrect'] = False
    rows[-1] = dict(status='missing')
    cell = s.summarize_cell(rows, signal=True)
    assert not cell['qualified'] and cell['unknown'] == 1
    assert cell['incorrect']['upper'] is None
    assert cell['incorrect']['possible_counts'] == [2, 3]
    with pytest.raises(ValueError):
        s.shards('all')
