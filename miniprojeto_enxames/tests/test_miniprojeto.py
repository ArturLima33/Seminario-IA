"""Testes integrados sem depender de rede; a base sintética NUNCA vira resultado científico."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from src.data import load_data, split_data
from src.objective import SearchSpace, Objective, final_test, BudgetExhausted
from src.optimizers import METHODS, run_optimizer
from src.runner import effective_budget, specs


@pytest.fixture
def sample():
    X,y = make_classification(n_samples=160,n_features=8,n_informative=5,
                              n_redundant=1,n_classes=3,n_clusters_per_class=1,random_state=4)
    return split_data(X,y)


def test_data_smoke_and_real_guard():
    X,y, meta = load_data(smoke=True)
    assert X.shape == (240,12)
    assert meta['source'].startswith('SYNTHETIC')
    with pytest.raises(FileNotFoundError):
        load_data('/path/notfound.tsv.gz')


def test_specs_limit_tasks():
    cfg = {'extension': {'methods':['pso','tpe'],'spaces':['wide'],'scales':['linear','log'],'budgets':[20,50],'seeds':[0,1]}}
    tasks = list(specs(cfg, 'extension', max_tasks=3))
    assert len(tasks) == 3


def test_space():
    s = SearchSpace((0.1,10),(0.01,1),'log')
    assert np.allclose(s.decode([0,0]),[0.1,0.01])
    assert np.allclose(s.decode([1,1]),[10,1])
    with pytest.raises(ValueError): s.decode([np.nan,1])


def test_budget_and_no_test_leak(sample):
    Xtr,Xte,ytr,yte = sample
    obj = Objective(Xtr,ytr,SearchSpace((0.1,10),(0.01,1)),budget=2)
    a = obj([0.5,0.5]); b = obj([0.5,0.5])
    assert a == b
    assert obj.n_evals == 2
    with pytest.raises(BudgetExhausted): obj([0.5,0.5])
    met, cm = final_test(Xtr,ytr,Xte,yte,1.0,0.1)
    assert met['accuracy'] > 0 and cm.sum() == len(yte)


def test_cv(sample):
    Xtr,_,ytr,_ = sample
    obj = Objective(Xtr,ytr,SearchSpace((0.1,10),(0.01,1)),3,
                    validation={'type':'cv','cv_folds':3})
    assert 0 <= -obj([0.4,0.4]) <= 1


@pytest.mark.parametrize('method', METHODS)
def test_every_optimizer(method, sample):
    Xtr,_,ytr,_ = sample
    with open(Path(__file__).resolve().parents[1] / 'config/experiments.json') as f: cfg=json.load(f)
    task = {'method':method,'budget':10}
    budget, params_sha = effective_budget(task)
    obj = Objective(Xtr,ytr,SearchSpace((0.1,10),(0.01,1)), budget)
    params = {**cfg['params'][method],**params_sha}
    res = run_optimizer(method,obj,seed=0,params=params)
    assert res['n_evals'] == budget
    assert 0 <= res['best_val_accuracy'] <= 1
    assert 0.1 <= res['best_C'] <= 10 and 0.01 <= res['best_gamma'] <= 1
    assert obj.evaluations and not pd.DataFrame(obj.evaluations).empty


def test_sha_full_fidelity(sample):
    Xtr,_,ytr,_ = sample
    obj=Objective(Xtr,ytr,SearchSpace((0.1,10),(0.01,1)), 30)
    res=run_optimizer('sha',obj,seed=1,params={'k':4})
    assert res['n_evals'] == 30
    assert sum(e['resource_fraction'] == 1 for e in obj.evaluations) == 2
    assert res['equivalent_full_evals'] < 30
