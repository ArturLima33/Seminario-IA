"""Função fitness comum. A classe Objective NÃO recebe o conjunto de teste."""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class SearchSpace:
    C: tuple[float, float]
    gamma: tuple[float, float]
    scale: str = 'linear'

    def __post_init__(self):
        for bounds in (self.C, self.gamma):
            if len(bounds) != 2 or not (0 < bounds[0] < bounds[1]):
                raise ValueError(f'Intervalo inválido: {bounds}')
        if self.scale not in ('linear', 'log'):
            raise ValueError('Escala deve ser linear ou log')

    def decode(self, position):
        v = np.asarray(position, dtype=float)
        if v.shape != (2,) or not np.all(np.isfinite(v)):
            raise ValueError('Candidato inválido, posição deve ter dois valores finitos')
        v = np.clip(v, 0, 1)
        out = []
        for x, (lo, hi) in zip(v, (self.C, self.gamma)):
            if self.scale == 'linear':
                out.append(lo + x * (hi - lo))
            else:
                out.append(math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo))))
        return tuple(float(x) for x in out)


class BudgetExhausted(RuntimeError):
    pass


def scaler_for(name):
    if name == 'standard': return StandardScaler()
    if name == 'minmax': return MinMaxScaler()
    if name == 'none': return None
    raise ValueError(f'Pré-processamento inválido: {name}')


class Objective:
    def __init__(self, X, y, space, budget, *, seed=42, validation=None, scaler='standard'):
        self.space, self.budget = space, int(budget)
        self.X, self.y = np.asarray(X), np.asarray(y)
        self.scaler_name = scaler
        self.validation = validation or {'type':'holdout', 'fraction':0.2}
        self.evaluations = []
        self.best_score = -np.inf
        self.best_position = None
        self.start = time.perf_counter()
        self.iteration = 0
        indices = np.arange(len(self.y))
        if self.validation['type'] == 'holdout':
            a, b = train_test_split(indices, test_size=self.validation.get('fraction', 0.2),
                                    stratify=self.y, random_state=seed)
            self.folds = [(a, b)]
        elif self.validation['type'] == 'cv':
            k = int(self.validation.get('cv_folds', 3))
            self.folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                              .split(self.X, self.y))
        else:
            raise ValueError('validation.type deve ser holdout ou cv')
        # Ordem de treino fixada por fold: mesma amostra reduzida para cada candidato SHA.
        self.orders = []
        for i, (tr, _) in enumerate(self.folds):
            rng = np.random.default_rng(seed + i)
            # A amostragem deve ser estratificada para manter todas as classes.
            buckets = [list(rng.permutation(tr[self.y[tr] == label])) for label in np.unique(self.y)]
            merged = []
            while any(buckets):
                for b in buckets:
                    if b: merged.append(b.pop())
            self.orders.append(np.asarray(merged, dtype=int))

    @property
    def n_evals(self):
        return len(self.evaluations)

    def __call__(self, position, resource_fraction=1.0):
        if self.n_evals >= self.budget:
            raise BudgetExhausted()
        if not 0 < resource_fraction <= 1:
            raise ValueError('resource_fraction fora de (0,1]')
        C, gamma = self.space.decode(position)
        start = time.perf_counter()
        try:
            scores = []
            for (tr, va), order in zip(self.folds, self.orders):
                if resource_fraction < 1:
                    n = min(len(tr), max(len(np.unique(self.y)) * 2,
                                          int(np.ceil(len(tr) * resource_fraction))))
                    # Ordenação cíclica estratificada; não reutiliza subset de outra divisão.
                    tr_use = order[:n]
                else:
                    tr_use = tr
                xfit, xval = self.X[tr_use], self.X[va]
                s = scaler_for(self.scaler_name)
                if s is not None:
                    xfit = s.fit_transform(xfit)
                    xval = s.transform(xval)
                model = SVC(C=C, gamma=gamma, kernel='rbf', decision_function_shape='ovo',
                            cache_size=256)
                model.fit(xfit, self.y[tr_use])
                scores.append(accuracy_score(self.y[va], model.predict(xval)))
            score = float(np.mean(scores))
            valid, error = True, ''
        except Exception as ex:
            score, valid, error = -1.0, False, repr(ex)
        if valid and resource_fraction == 1 and score > self.best_score:
            self.best_score = score
            self.best_position = np.asarray(position, float).copy()
        self.evaluations.append({'eval': self.n_evals + 1, 'iteration': self.iteration,
                                 'C': C, 'gamma': gamma, 'val_accuracy': score,
                                 'best_val_accuracy': self.best_score if self.best_position is not None else np.nan,
                                 'resource_fraction': resource_fraction, 'valid': valid,
                                 'error': error, 'elapsed_s': time.perf_counter() - start})
        return -score if valid else 1.0


def final_test(X_train, y_train, X_test, y_test, C, gamma, scaler='standard'):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    s = scaler_for(scaler)
    xtr, xte = X_train, X_test
    if s is not None:
        xtr, xte = s.fit_transform(xtr), s.transform(xte)
    model = SVC(C=C, gamma=gamma, kernel='rbf', decision_function_shape='ovo', cache_size=256)
    model.fit(xtr, y_train)
    yp = model.predict(xte)
    metrics = {'accuracy': accuracy_score(y_test, yp)}
    for avg in ('macro','weighted','micro'):
        metrics.update({f'precision_{avg}': precision_score(y_test, yp, average=avg, zero_division=0),
                        f'recall_{avg}': recall_score(y_test, yp, average=avg, zero_division=0),
                        f'f1_{avg}': f1_score(y_test, yp, average=avg, zero_division=0)})
    return {k: float(v) for k,v in metrics.items()}, confusion_matrix(y_test, yp, labels=np.unique(y_train))
