"""Independent protocol tests with tiny deterministic fixtures, no training."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/ramp_imp_correspondence_v1/run.py'
spec=importlib.util.spec_from_file_location('ramp_pipeline',SCRIPT)
pipeline=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_registration_rejects_changed_calibration_or_inputs(tmp_path):
    p=tmp_path/'registration.json'
    pipeline.register(p,dict(seed=1,hash='a'))
    pipeline.register(p,dict(seed=1,hash='a'))
    with pytest.raises(RuntimeError,match='Registration mismatch'):
        pipeline.register(p,dict(seed=2,hash='a'))
    with pytest.raises(RuntimeError):
        pipeline.register(p,dict(seed=1,hash='b'))


def test_correlations_ties_and_constant_exposure():
    assert pipeline.correlation([1,1,1],[1,2,3])!=pipeline.correlation([1,1,1],[1,2,3])
    assert pipeline.correlation([1,2,3,4],[1,1,2,2])==pytest.approx(np.sqrt(.8))
    assert pipeline.correlation([1,np.nan,3,4],[4,3,2,1])==pytest.approx(-1)


def test_run_outcome_never_replicated_across_neurons():
    n=20
    df=pd.DataFrame(dict(layer=np.repeat([0,1],10),expert=np.tile(np.repeat([0,1],5),2),neuron=np.tile(range(5),4),
                         surviving_weights=np.arange(1,n+1),retention=np.linspace(.1,.5,n),up_retention=np.linspace(.1,.5,n),down_retention=np.linspace(.1,.5,n)))
    for i,f in enumerate(pipeline.FEATURES):
        df[f]=np.linspace(.1,1,n) if f!='utilization' else 5.
    groups,deciles,summary=pipeline.analyze_pair(df,dict(seed=7,recovery_gap=.04))
    assert isinstance(summary,dict)
    assert summary['recovery_gap']==.04
    assert summary['ramp_weighted_retention']==pytest.approx(np.average(df.retention,weights=df.ramp))
    assert summary['utilization_within_expert_valid_groups']==0
    assert summary['ramp_within_expert_valid_groups']==4
    assert len(groups)==len(pipeline.FEATURES)*6
    assert deciles.neurons.sum()==n*len(pipeline.FEATURES)


def test_calibration_disjoint_balanced_and_reproducible(tmp_path,monkeypatch):
    train=tmp_path/'mixed.txt';val=tmp_path/'val.txt';tiny=tmp_path/'tiny.txt';wiki=tmp_path/'wiki.txt'
    tiny.write_text('abcdefghABCDEFGHijklmnopIJKLMNOPqrstuvwxQRSTUVWX')
    wiki.write_text('1234567890!@#$%^8765432109)(*&^%01234567!@#$%^&*')
    train.write_text(tiny.read_text()+wiki.read_text())
    val.write_text('abcdefgh') # An exact training block must be excluded.
    meta=tmp_path/'data/multidomain_balanced';meta.mkdir(parents=True)
    import json
    (meta/'metadata.json').write_text(json.dumps(dict(train=dict(tiny_path=str(tiny),wiki_path=str(wiki)))))
    source=tmp_path/'src/moe_lth';source.mkdir(parents=True)
    (source/'data.py').write_text('fixture')
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    dc=dict(tokenizer='byte',seq_len=8,train_path=str(train),validation_path=str(val))
    out=tmp_path/'out';out.mkdir()
    b,l,m=pipeline.calibrate(dict(data=dc),out,4,123)
    bb,ll,mm=pipeline.calibrate(dict(data=dc),out,4,123)
    assert m==mm and l==ll
    assert m['domain_counts']==dict(tinystories=2,wikitext=2)
    assert all(bytes(x.tolist()) not in val.read_bytes() for batch in b for x in batch)
    assert all(np.array_equal(a.numpy(),b.numpy()) for a,b in zip(b,bb))
