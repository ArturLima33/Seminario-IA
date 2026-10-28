"""Regressões específicas da extensão orientada a classes minoritárias."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from src.data import split_data
from src.objective import Objective, SearchSpace, final_test
from src.runner import specs, make_imbalance_analysis

CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'experiments.json'


def sample():
    X, y = make_classification(n_samples=180, n_features=9, n_informative=5,
                               n_redundant=1, n_classes=3, weights=[.68,.24,.08],
                               n_clusters_per_class=1, random_state=21)
    return split_data(X,y)


def test_factorial_has_all_four_variants_and_two_methods():
    cfg = json.loads(CONFIG.read_text())
    tasks=list(specs(cfg,'imbalance',budget_override=[10],seeds_override=[0]))
    assert len(tasks)==8
    assert {(t['fitness_metric'],t['class_weight']) for t in tasks} == {
        ('accuracy','none'), ('f1_macro','none'), ('accuracy','balanced'), ('f1_macro','balanced')}


def test_metric_and_class_weight_propagated_with_no_test_access():
    Xtr,Xte,ytr,yte = sample()
    space = SearchSpace((.1,10),(.01,1),'log')
    objs = [Objective(Xtr,ytr,space,1,seed=3,fitness_metric=metric,
                      class_weight=weight) for metric,weight in
            [('accuracy','none'),('f1_macro','balanced')]]
    vals = [o([.52,.52]) for o in objs]
    for o,v in zip(objs,vals):
        ev=o.evaluations[0]
        assert np.isclose(-v, ev['val_score'])
        assert np.isclose(ev['val_score'], ev['val_accuracy'] if o.fitness_metric == 'accuracy'
                          else ev['val_f1_macro'])
        assert o.n_evals==1
        assert o.svm_class_weight == (None if o.class_weight=='none' else 'balanced')
    # Só o teste final conhece yte, e mantém métricas comparáveis.
    metrics, cm, report=final_test(Xtr,ytr,Xte,yte,1.,.2,
                                 class_weight='balanced',return_per_class=True)
    assert cm.sum()==len(yte)
    assert {'accuracy','f1_macro','min_class_recall','balanced_accuracy'} <= metrics.keys()
    assert len(report)==3 and (report['recall']>=0).all()


def test_no_label_leakage_in_scaler_selection():
    Xtr, Xte, ytr, yte = sample()
    o=Objective(Xtr,ytr,SearchSpace((.1,10),(.01,1)),1,
                fitness_metric='f1_macro',class_weight='balanced')
    o([.2,.2])
    assert set(o.evaluations[0]) >= {'best_val_f1_macro','best_val_accuracy','class_weight'}
    assert Xte.shape[0] > 0


def test_validation_of_extension_arguments():
    Xtr,_,ytr,_=sample()
    for kw in ({'fitness_metric':'invalid'},{'class_weight':'invalid'}):
        with pytest.raises(ValueError):
            Objective(Xtr,ytr,SearchSpace((.1,10),(.01,1)),1,**kw)


def test_paired_delta_uses_same_seed(tmp_path):
    for folder in ['figures','statistics']:(tmp_path/folder).mkdir()
    rows=[]
    for seed in (0,1):
        for fit,weight,delta in [('accuracy','none',0),('f1_macro','none',.04),
                                 ('accuracy','balanced',.02),('f1_macro','balanced',.05)]:
            rows.append({'method':'pso','space':'wide','scale':'log','budget':10,
                         'fitness_metric':fit,'class_weight':weight,'seed':seed,
                         'test_accuracy':.8+delta/2, 'test_f1_macro':.7+delta,
                         'test_balanced_accuracy':.6+delta,
                         'test_min_class_recall':.5+delta,'optimization_s':1.})
    make_imbalance_analysis(pd.DataFrame(rows),tmp_path)
    delta=pd.read_csv(tmp_path/'statistics'/'ablation_paired_deltas.csv')
    x=delta[(delta.variant=='f1_macro+balanced') & (delta.metric=='test_f1_macro')]
    assert len(x)==1 and np.isclose(x.delta_mean.iloc[0],.05)
    assert int(x.seeds_paired.iloc[0])==2
