"""Synthetic numerical checks only: no model fixtures or model execution."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts/expert_deletion_temporal_prediction_v1'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('temporal_analysis',SCRIPTS/'analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

@pytest.mark.parametrize('rule',['current','history_mean','history_max'])
def test_future_leakage(rule):
    current=np.array([.1,-.2,.03]);previous=np.array([.2,.1,-.4]);ids=['a','b','c']
    predictions=[]
    for future in [np.array([1.,2.,3.]),np.array([-100.,999.,-8.])]:
        s=a.score(current,previous,rule)
        predictions.append((s.copy(),a.select(s,ids)))
        # Future is consumed only after prediction, by the outcome expression.
        low,high,_=predictions[-1][1]
        outcome=future[high]-future[low]
        assert np.isfinite(outcome)
    np.testing.assert_array_equal(predictions[0][0],predictions[1][0])
    assert predictions[0][1]==predictions[1][1]

def test_signed_actual_count_weighting():
    n=np.array([[-1.,2.],[9.,-3.]])
    d=np.array([1.,9.]);weights=np.array([[1.,1.],[2.,0.]])
    mean,boot=a.summarize(n,d,np.arange(2),weights)
    np.testing.assert_allclose(mean,[.8,-.1])
    np.testing.assert_allclose(boot,[[.8,-.1],[-1.,2.]])
    assert mean[0]!=np.mean(n[:,0]/d)

def test_zero_denominators():
    mean,boot=a.summarize(np.zeros((2,3)),np.zeros(2),np.arange(2),np.ones((4,2)))
    assert np.isnan(mean).all() and np.isnan(boot).all()
    assert a.ci(boot[:,0])==dict(ci95_low=None,ci95_high=None,undefined_bootstrap_fraction=1.)
    assert a.select(mean,['a','b','c'])==(None,None,True)

def test_ties_singletons_and_insufficient_rank():
    assert a.rho([1,1,1],[1,2,3]) is None
    assert a.rho([1,2],[2,1]) is None
    assert a.select([1],['a'])==(None,None,True)
    assert a.select([1,1,1],['b','a','c'])==(1,2,True)
    assert a.ties([1,1,2])==1
    assert a.rho([1,1,2],[1,1,2])==pytest.approx(1)

def test_identical_selections_exact_zero():
    future=np.array([[1e100,-1e100],[np.nan,np.nan]])
    np.testing.assert_array_equal(a.policy_difference(future,0,0),[0.,0.])
    np.testing.assert_array_equal(a.policy_difference(np.array([2.,5.]),0,1),-3.)

def test_overlap_and_empty_selection():
    sets=[{1,2,3},{2,3,4},{5}]
    assert a.coordinate_union(sets,[0,1])=={1,2,3,4}
    assert len(a.coordinate_union(sets,[0,1]))==4
    assert a.coordinate_union(sets,[])==set()

def test_history_shared_batch_resamples():
    n=np.array([[1.,2.,3.],[4.,5.,6.]])
    w=np.array([[2.,0.],[0.,2.]])
    _,current=a.summarize(n,np.ones(2),np.arange(2),w)
    _,previous=a.summarize(-n,np.ones(2),np.arange(2),w)
    np.testing.assert_array_equal(a.score(current,previous,'history_mean'),np.zeros((2,3)))

def test_all_bootstrap_candidates_paired():
    draws=np.array([[1.,4.,2.],[3.,1.,2.],[np.nan,np.nan,np.nan]])
    low,high,tied=a.boot_select(draws,['a','b','c'])
    np.testing.assert_array_equal(low,[0,1,-1])
    np.testing.assert_array_equal(high,[1,0,-1])
    np.testing.assert_allclose(a.picked(draws,low),[1.,1.,np.nan],equal_nan=True)
