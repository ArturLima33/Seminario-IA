"""Testes regressivos das falhas identificadas na auditoria v2."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.objective import Objective, SearchSpace
from src.optimizers import _pso
from src.runner import make_imbalance_analysis


class ConstantObjective:
    """Oráculo determinístico sem SVM para testar o fluxo do otimizador."""
    def __init__(self, budget):
        self.budget = budget
        self.evaluations = []
        self.iteration = 0

    @property
    def n_evals(self):
        return len(self.evaluations)

    def __call__(self, candidate):
        assert np.asarray(candidate).shape == (2,)
        self.evaluations.append({'iteration': self.iteration})
        return -0.5  # platô: força a estagnação


def test_adaptive_inertia_starts_at_w_start_and_ends_at_w_end():
    obj = ConstantObjective(60)
    params = {'population': 10, 'w_start': 0.9, 'w_end': 0.4,
              'cognitive': 1.5, 'social': 1.5, 'max_velocity': 0.25,
              'stagnation_limit': 2, 'restart_sigma': 0.15}
    _pso(obj, np.random.default_rng(0), params, adaptive=True)
    assert obj.n_evals == 60
    updates = pd.DataFrame(obj.evaluations).query("pso_phase == 'update'")
    assert len(updates) == 50
    by_generation = updates.groupby('iteration').pso_inertia.first()
    assert np.allclose(by_generation.to_numpy(), [0.9, 0.775, 0.65, 0.525, 0.4])


def test_adaptive_restart_is_reachable_and_logged():
    obj = ConstantObjective(60)
    params = {'population': 10, 'w_start': 0.9, 'w_end': 0.4,
              'cognitive': 1.5, 'social': 1.5, 'max_velocity': 0.25,
              'stagnation_limit': 2, 'restart_sigma': 0.15}
    _pso(obj, np.random.default_rng(3), params, adaptive=True)
    restarts = [e for e in obj.evaluations if e.get('pso_restart')]
    assert restarts and restarts[0]['iteration'] == 3
    assert all(e['pso_phase'] == 'update' for e in restarts)


def test_sha_prefixes_maintain_original_class_proportions():
    X, y = make_classification(n_samples=800, n_features=10, n_informative=7,
                              n_redundant=0, n_classes=3, weights=[.7,.2,.1],
                              n_clusters_per_class=1, random_state=12)
    obj = Objective(X, y, SearchSpace((0.1, 10), (.01, 1)), 1, seed=2)
    train, _ = obj.folds[0]
    order = obj.orders[0]
    full_labels, counts = np.unique(y[train], return_counts=True)
    proportions = counts / counts.sum()
    for n in (60, 90, 150, 300):
        observed = np.array([(y[order[:n]] == cl).sum() for cl in full_labels])
        # Quantis intercalados por proporção: arredondamento por no máximo dois.
        assert np.all(np.abs(observed - n * proportions) <= 2.0)
        assert len(np.unique(y[order[:n]])) == 3
    # Os subconjuntos são aninhados (SHA amplia recursos, não muda dados arbitrariamente).
    assert set(order[:60]).issubset(set(order[:150]))


def test_ablation_interaction_and_adaptive_comparison(tmp_path):
    (tmp_path / 'statistics').mkdir()
    (tmp_path / 'figures').mkdir()
    rows = []
    arms = [('accuracy','none',0.70),('f1_macro','none',0.74),
            ('accuracy','balanced',0.72),('f1_macro','balanced',0.78)]
    for method, offset in [('pso',0.0),('adaptive_pso',0.01)]:
        for seed in [0, 1, 2]:
            for metric,weight,score in arms:
                rows.append({'method':method, 'space':'wide','scale':'log','budget':100,
                             'fitness_metric':metric, 'class_weight':weight, 'seed':seed,
                             'test_accuracy':score+offset, 'test_f1_macro':score+offset,
                             'test_balanced_accuracy':score+offset,
                             'test_min_class_recall':score+offset,
                             'optimization_s':2.0})
    make_imbalance_analysis(pd.DataFrame(rows), tmp_path)
    interaction = pd.read_csv(tmp_path / 'statistics' / 'ablation_interactions.csv')
    ps = interaction.query("method == 'pso' and metric == 'test_f1_macro'")
    assert len(ps) == 1
    assert np.isclose(ps.interaction_mean.iloc[0], 0.78-0.74-0.72+0.70)
    comparison = pd.read_csv(tmp_path / 'statistics' / 'adaptive_vs_pso_paired.csv')
    diff = comparison.query("fitness_metric == 'accuracy' and class_weight == 'none' and metric == 'test_f1_macro'")
    assert len(diff) == 1
    assert np.isclose(diff.adaptive_minus_pso_mean.iloc[0], .01)
    assert int(diff.seeds_paired.iloc[0]) == 3
