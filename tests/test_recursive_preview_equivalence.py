import sys
from pathlib import Path
import importlib
import math

sys.path.insert(0, str(Path(__file__).parents[1]/'examples'))


def comparator():
    return importlib.import_module('recursive_preview_equivalence').compare_payloads


def test_roundoff_does_not_require_bit_identical_risk_values():
    value = .40676605290832346
    report = comparator()({'raw':math.nextafter(value, math.inf)}, {'raw':value})
    assert report['equivalent'] and report['changed_float_fields'] == 1
    assert report['max_absolute_difference'] < 1e-15


def test_material_numeric_or_discrete_decision_change_is_rejected():
    compare = comparator()
    assert not compare({'raw':.4001},{'raw':.4})['equivalent']
    assert not compare({'qualifies':True},{'qualifies':False})['equivalent']
    assert not compare({'selected':70},{'selected':69})['equivalent']
    assert not compare({'members':['a','b']},{'members':['a','c']})['equivalent']


def test_parameters_data_frequencies_and_types_remain_exact():
    compare = comparator()
    for field in ('fitted_parameters','observed','selection_frequency','evaluation_frequency'):
        assert not compare({field:math.nextafter(.5,math.inf)}, {field:.5})['equivalent']
    assert not compare({'raw':1}, {'raw':1.})['equivalent']


def test_missing_fields_and_nonfinite_values_cannot_pass():
    compare = comparator()
    assert not compare({'raw':.5}, {})['equivalent']
    assert not compare({'raw':float('nan')},{'raw':.5})['equivalent']
