"""Oito otimizadores aproximados do artigo + proposta original e baselines.
Todos usam a MESMA Objective e contam cada chamada como uma avaliação.
"""
from __future__ import annotations
import math
import os
import sys
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

try:
    from .objective import BudgetExhausted
except ImportError:  # execução direta do arquivo
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.objective import BudgetExhausted

PAPER = ['pso', 'sa', 'ga', 'cs', 'bo', 'sha', 'tpe', 'mcmc']
METHODS = PAPER + ['adaptive_pso', 'random_search', 'grid_search']


def _levy(rng, dim, beta=1.5):
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, dim)
    v = rng.normal(0, 1, dim)
    return u / np.maximum(np.abs(v), 1e-9) ** (1 / beta)


def _pso(obj, rng, p, *, adaptive=False):
    """PSO no cubo [0,1]^2, com orçamento contado por avaliação SVM.

    A inércia adaptativa depende da GERAÇÃO de movimento, não das avaliações
    já gastas na inicialização. Reinicia a pior partícula (posição, velocidade e
    pBest) após ``stagnation_limit`` gerações completas sem novo gBest.
    Reinícios são registrados no histórico; não criam avaliações gratuitas.
    """
    n = min(obj.budget, int(p['population']))
    if n < 1:
        raise ValueError('PSO requer população positiva e orçamento >= 1')
    if adaptive:
        if not (0 <= p['w_end'] <= p['w_start']):
            raise ValueError('Requer 0 <= w_end <= w_start')
        if int(p['stagnation_limit']) < 1:
            raise ValueError('stagnation_limit deve ser >= 1')
    x = rng.random((n, 2))
    velocity = rng.uniform(-0.1, 0.1, (n, 2))
    individual_best = x.copy()
    individual_fitness = []
    obj.iteration = 0
    for z in x:
        individual_fitness.append(obj(z))
        obj.evaluations[-1].update(pso_phase='initialization', pso_inertia=None,
                                   pso_restart=False, pso_stalled_generations=0)
    individual_fitness = np.asarray(individual_fitness)
    global_idx = int(np.argmin(individual_fitness))
    global_best = x[global_idx].copy()
    global_fitness = float(individual_fitness[global_idx])

    stalled = 0
    moves = obj.budget - n
    total_generations = math.ceil(moves / n) if moves else 0
    for generation in range(total_generations):
        obj.iteration = generation + 1
        # Primeira geração usa w_start; a última usa w_end quando existem >=2.
        progress = generation / max(1, total_generations - 1)
        inertia = (float(p['w_start'] * (1 - progress) + p['w_end'] * progress)
                   if adaptive else float(p['inertia']))
        restarted_idx = None
        if adaptive and stalled >= int(p['stagnation_limit']):
            # Reiniciar a pior partícula permite explorar região não visitada.
            restarted_idx = int(np.argmax(individual_fitness))
            x[restarted_idx] = rng.random(2)
            velocity[restarted_idx] = rng.uniform(-0.1, 0.1, 2)
            # Descartar seu pBest antigo; gBest é preservado.
            individual_best[restarted_idx] = x[restarted_idx].copy()
            individual_fitness[restarted_idx] = np.inf
            stalled = 0

        previous_global = global_fitness
        for i in range(n):
            if obj.n_evals >= obj.budget:
                break
            restarted = i == restarted_idx
            if not restarted:
                r1, r2 = rng.random(2), rng.random(2)
                velocity[i] = (inertia * velocity[i] +
                               p['cognitive'] * r1 * (individual_best[i] - x[i]) +
                               p['social'] * r2 * (global_best - x[i]))
                velocity[i] = np.clip(velocity[i], -p['max_velocity'], p['max_velocity'])
                x[i] = np.clip(x[i] + velocity[i], 0, 1)
            score = obj(x[i])
            obj.evaluations[-1].update(pso_phase='update', pso_inertia=inertia,
                                       pso_restart=bool(restarted),
                                       pso_stalled_generations=stalled)
            if score < individual_fitness[i]:
                individual_best[i], individual_fitness[i] = x[i].copy(), score
            if score < global_fitness:
                global_best, global_fitness = x[i].copy(), score
        if global_fitness < previous_global - 1e-12:
            stalled = 0
        else:
            stalled += 1


def _sa(obj, rng, p):
    current = rng.random(2)
    cur = obj(current)
    for i in range(1, obj.budget):
        obj.iteration = i
        fraction = i / max(1, obj.budget - 1)
        temp = p['initial_temp'] * (p['final_temp'] / p['initial_temp']) ** fraction
        proposed = np.clip(current + rng.normal(0, p['sigma'], 2), 0, 1)
        fit = obj(proposed)
        if fit < cur or rng.random() < math.exp(min(0, (cur - fit) / temp)):
            current, cur = proposed, fit


def _ga(obj, rng, p):
    n = min(int(p['population']), obj.budget)
    pop = rng.random((n, 2))
    fit = np.array([obj(z) for z in pop])
    iteration = 0
    while obj.n_evals < obj.budget:
        iteration += 1
        obj.iteration = iteration
        order = np.argsort(fit)
        elites = max(1, min(n, round(n * p['elite_fraction'])))
        winners = pop[order[:elites]].copy()
        win_fit = fit[order[:elites]].copy()
        new, new_fit = list(winners), list(win_fit)
        while len(new) < n and obj.n_evals < obj.budget:
            a, b = rng.integers(0, elites, 2)
            blend = rng.random()
            child = blend * winners[a] + (1 - blend) * winners[b]
            mutate = rng.random(2) < p['mutation_rate']
            child[mutate] = rng.random(int(mutate.sum()))
            new.append(child)
            new_fit.append(obj(child))
        pop, fit = np.asarray(new), np.asarray(new_fit)
        if obj.n_evals >= obj.budget: break


def _cs(obj, rng, p):
    n = min(obj.budget, int(p['nests']))
    nests = rng.random((n, 2))
    values = np.array([obj(z) for z in nests])
    iteration = 0
    while obj.n_evals < obj.budget:
        iteration += 1
        obj.iteration = iteration
        best = nests[np.argmin(values)].copy()
        for i in range(n):
            if obj.n_evals >= obj.budget: break
            cand = np.clip(nests[i] + p['levy_alpha'] * _levy(rng, 2) * (nests[i] - best), 0, 1)
            fit = obj(cand)
            if fit < values[i]:
                nests[i], values[i] = cand, fit
        if obj.n_evals >= obj.budget: break
        worst_n = max(1, round(n * p['abandonment']))
        worst = np.argsort(values)[-worst_n:]
        for i in worst:
            if obj.n_evals >= obj.budget: break
            nests[i] = rng.random(2)
            values[i] = obj(nests[i])


def _bo(obj, rng, p):
    import warnings
    X, y = [], []
    for _ in range(min(obj.budget, int(p['startup']))):
        z = rng.random(2)
        X.append(z)
        y.append(obj(z))
    iteration = 0
    while obj.n_evals < obj.budget:
        iteration += 1
        obj.iteration = iteration
        kernel = ConstantKernel(1, (0.01, 100)) * Matern(length_scale=[0.25, 0.25], nu=2.5) + WhiteKernel(1e-5)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0,
                                      random_state=int(rng.integers(2**30)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gp.fit(np.asarray(X), np.asarray(y))
        candidate = np.vstack([rng.random((p['candidates'], 2)),
                               np.clip(np.asarray(X)[np.argmin(y)] + rng.normal(0, 0.06, (50, 2)), 0, 1)])
        means, deviations = gp.predict(candidate, return_std=True)
        imp = min(y) - means - p['xi']
        dev = np.maximum(deviations, 1e-12)
        z = imp / dev
        expected_improvement = imp * norm.cdf(z) + dev * norm.pdf(z)
        x = candidate[np.argmax(expected_improvement)]
        X.append(x)
        y.append(obj(x))


def _log_density(candidates, points, bandwidth=0.10):
    if not len(points): return np.zeros(len(candidates))
    sigma = np.maximum(0.05, np.std(points, axis=0) * max(1, len(points)) ** -0.2)
    sigma = np.maximum(sigma, bandwidth / 2)
    z = (candidates[:, None, :] - points[None, :, :]) / sigma
    component = (-0.5 * z**2 - np.log(sigma) - 0.5 * math.log(2 * math.pi)).sum(axis=2)
    # prior uniforme no cubo unitário
    return logsumexp(np.column_stack([component, np.zeros(len(candidates))]), axis=1) - math.log(len(points) + 1)


def _tpe(obj, rng, p):
    X, y = [], []
    for _ in range(min(obj.budget, int(p['startup']))):
        z = rng.random(2)
        X.append(z)
        y.append(obj(z))
    iteration = 0
    while obj.n_evals < obj.budget:
        iteration += 1
        obj.iteration = iteration
        order = np.argsort(y)
        ng = max(1, int(math.ceil(len(X) * p['good_fraction'])))
        good, bad = np.asarray(X)[order[:ng]], np.asarray(X)[order[ng:]]
        candidates = np.vstack([rng.random((p['candidates'] // 4, 2)),
                                np.clip(good[rng.integers(len(good), size=p['candidates'])] +
                                        rng.normal(0, 0.1, (p['candidates'], 2)), 0, 1)])
        ratio = _log_density(candidates, good) - _log_density(candidates, bad)
        z = candidates[np.argmax(ratio)]
        X.append(z)
        y.append(obj(z))


def _mcmc(obj, rng, p):
    # MH com alvo de Boltzmann de acurácia; deixa essa hipótese explícita no relatório.
    cur = rng.random(2)
    fitness = obj(cur)
    for i in range(1, obj.budget):
        obj.iteration = i
        proposal = cur + rng.normal(0, p['sigma'], 2)
        # Reflexão robusta para valores fora [0, 1] (simetria do random walk).
        proposal = 1 - np.abs(1 - np.mod(proposal, 2))
        next_fitness = obj(proposal)
        if math.log(max(rng.random(), 1e-200)) < min(0, (fitness - next_fitness) / p['temperature']):
            cur, fitness = proposal, next_fitness


def _sha(obj, rng, p):
    # k=4/5/6 implica 30/62/126 avaliações; fidelidade vai até 100%.
    k = int(p['k'])
    candidates = rng.random((2**k, 2))
    for stage in range(k):
        obj.iteration = stage
        fraction = 2 ** (stage - (k-1))
        scores = np.array([obj(z, resource_fraction=fraction) for z in candidates])
        if stage != k - 1:
            candidates = candidates[np.argsort(scores)[:len(candidates)//2]]


def _random(obj, rng, p):
    for i in range(obj.budget):
        obj.iteration = i
        obj(rng.random(2))


def _grid(obj, rng, p):
    # Usa EXATAMENTE budget candidatos (grade retangular ou grade + truncamento)
    n = int(math.ceil(math.sqrt(obj.budget)))
    axis = np.linspace(0, 1, n)
    coords = np.array([(a, b) for a in axis for b in axis])
    # Distribui as avaliações por toda a grade, em vez de excluir a última linha.
    chosen = np.linspace(0, len(coords) - 1, num=obj.budget, dtype=int)
    for i, coord in enumerate(coords[chosen]):
        obj.iteration = i
        obj(coord)


DISPATCH = {'pso': _pso, 'adaptive_pso': lambda o,r,p: _pso(o,r,p,adaptive=True),
            'sa': _sa, 'ga': _ga, 'cs': _cs, 'bo': _bo, 'sha': _sha,
            'tpe': _tpe, 'mcmc': _mcmc, 'random_search': _random, 'grid_search': _grid}


def run_optimizer(method, objective, seed, params):
    if method not in DISPATCH:
        raise ValueError(f'Método desconhecido: {method}')
    rng = np.random.default_rng(seed)
    started = __import__('time').perf_counter()
    try:
        DISPATCH[method](objective, rng, params)
    except BudgetExhausted:
        # Esse caso pode ocorrer com SHA quando o orçamento foi configurado incorretamente.
        raise RuntimeError('Orçamento excedido: verifique k/budget para SHA')
    elapsed = __import__('time').perf_counter() - started
    if objective.best_position is None:
        raise RuntimeError('Nenhuma avaliação de fidelidade completa válida')
    return {'best_C': objective.space.decode(objective.best_position)[0],
            'best_gamma': objective.space.decode(objective.best_position)[1],
            'best_val_accuracy': objective.best_val_accuracy,
            'best_val_f1_macro': objective.best_val_f1_macro,
            'best_val_score': objective.best_score,
            'optimization_s': elapsed,
            'n_evals': objective.n_evals,
            'equivalent_full_evals': sum(e['resource_fraction'] for e in objective.evaluations),
            'failed_evals': sum(not e['valid'] for e in objective.evaluations)}
