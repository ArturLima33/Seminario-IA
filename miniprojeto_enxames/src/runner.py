"""Execução reproduzível: seleção somente na validação, teste somente no final."""
from __future__ import annotations
import json
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import friedmanchisquare, wilcoxon
from .data import load_data, describe, split_data
from .objective import Objective, SearchSpace, final_test
from .optimizers import run_optimizer


def specs(cfg, mode, method_override=None, space_override=None, scale_override=None,
          budget_override=None, seeds_override=None, fitness_override=None,
          class_weight_override=None, max_tasks=None):
    plan = cfg[mode]
    methods = method_override or plan['methods']
    spaces = space_override or plan['spaces']
    scales = scale_override or plan['scales']
    budgets = budget_override or plan['budgets']
    seeds = seeds_override or plan['seeds']
    metrics = [fitness_override] if fitness_override else plan.get('objective_metrics', ['accuracy'])
    weights = [class_weight_override] if class_weight_override else plan.get('class_weights', ['none'])
    count = 0
    for sp in spaces:
        for sc in scales:
            for budget in budgets:
                for method in methods:
                    for seed in seeds:
                        for metric in metrics:
                            for weight in weights:
                                if max_tasks is not None and count >= max_tasks:
                                    return
                                count += 1
                                yield {'mode': mode, 'space': sp, 'scale': sc, 'budget': int(budget),
                                       'method': method, 'seed': int(seed),
                                       'fitness_metric': metric, 'class_weight': weight}


def effective_budget(task):
    if task['method'] != 'sha': return task['budget'], {}
    # artigo: k=4/5/6 quando identifica iterações 20/50/100; valor registrado.
    nominal = task['budget']
    k = {20: 4, 50: 5, 100: 6}.get(nominal)
    if k is None:
        k = max(1, int(np.floor(np.log2(nominal + 2))) - 1)
    return 2 ** (k + 1) - 2, {'k': k}


def _save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def run_experiment(cfg, mode, *, output, data_path=None, smoke=False, quick=False,
                   validation_override=None, method_override=None, space_override=None,
                   scale_override=None, budget_override=None, seeds_override=None,
                   fitness_override=None, class_weight_override=None, dry_run=False, max_tasks=None):
    tasks = list(specs(cfg, mode, method_override, space_override, scale_override,
                       budget_override, seeds_override, fitness_override, class_weight_override,
                       max_tasks=max_tasks))
    for task in tasks:
        task['actual_evals'], task['sha_params'] = effective_budget(task)
    if mode == 'reproduction' and any(t['fitness_metric'] != 'accuracy' or t['class_weight'] != 'none' for t in tasks):
        raise ValueError('Reproduction exige fitness accuracy e class_weight none')
    if dry_run:
        print(f'Plano: {len(tasks)} execuções; {sum(t["actual_evals"] for t in tasks)} avaliações da SVM')
        print(pd.DataFrame(tasks).groupby(['method', 'space', 'scale', 'budget', 'fitness_metric', 'class_weight'], as_index=False)
              .agg(runs=('seed', 'count'), evaluations=('actual_evals', 'sum')).to_string(index=False))
        return None
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for d in ['evaluations', 'confusion', 'per_class', 'statistics', 'figures']:
        (out / d).mkdir(exist_ok=True)
    X, y, meta = load_data(data_path, smoke=smoke)
    report = describe(X, y, meta)
    val = dict(cfg['validation'])
    if validation_override:
        val['type'] = validation_override
    if quick and not smoke:
        # Preserva o teste real de 20%; quick NÃO é resultado final para o artigo.
        print('ATENÇÃO: --quick usa subamostra da base real, não interpretar como resultado científico completo')
    cfg_record = {'parameters': cfg, 'mode': mode, 'smoke': smoke, 'quick': quick,
                  'validation_effective': val, 'dataset': report, 'tasks': tasks,
                  'environment': {'python': sys.version, 'scikit_learn': sklearn.__version__,
                                  'platform': platform.platform()}, 'generated_at': datetime.now().isoformat()}
    _save_json(out / 'experiment_manifest.json', cfg_record)
    _save_json(out / 'dataset_report.json', report)
    rows = []
    for index, task in enumerate(tasks, start=1):
        ident = '__'.join(str(task[k]) for k in ['mode', 'space', 'scale', 'budget', 'method', 'seed', 'fitness_metric', 'class_weight'])
        print(f'[{index}/{len(tasks)}] {ident}', flush=True)
        row = {**task, 'run_id': ident, 'smoke': smoke, 'quick': quick, 'status': 'failed', 'error': ''}
        started = time.perf_counter()
        try:
            Xtr, Xte, ytr, yte = split_data(X, y, cfg['dataset']['test_fraction'], cfg['dataset']['seed'])
            if quick and not smoke:
                from sklearn.model_selection import train_test_split
                Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=0.25,
                                                 stratify=ytr, random_state=cfg['dataset']['seed'])
            bounds = cfg['search_spaces'][task['space']]
            space = SearchSpace(tuple(bounds['C']), tuple(bounds['gamma']), task['scale'])
            objective = Objective(Xtr, ytr, space, task['actual_evals'],
                                  seed=cfg['dataset']['seed'], validation=val, scaler=cfg['scaler'],
                                  fitness_metric=task['fitness_metric'], class_weight=task['class_weight'])
            params = dict(cfg['params'].get(task['method'], {}))
            params.update(task['sha_params'])
            result = run_optimizer(task['method'], objective, task['seed'], params)
            metrics, cm, per_class = final_test(Xtr, ytr, Xte, yte, result['best_C'], result['best_gamma'],
                                     scaler=cfg['scaler'], class_weight=task['class_weight'],
                                     return_per_class=True)
            # O teste é mensurado e publicado, nunca usado para selecionar candidatos.
            pd.DataFrame(objective.evaluations).to_csv(out / 'evaluations' / (ident + '.csv'), index=False)
            np.savetxt(out / 'confusion' / (ident + '.csv'), cm, fmt='%d', delimiter=',')
            per_class.to_csv(out / 'per_class' / (ident + '.csv'))
            row.update(result)
            row.update({f'test_{k}': v for k, v in metrics.items()})
            row.update({'train_rows': len(ytr), 'test_rows': len(yte), 'status': 'ok',
                        'elapsed_total_s': time.perf_counter() - started})
            print(f"  validação={row['best_val_accuracy']:.4f}; teste={row['test_accuracy']:.4f}; "
                  f"avaliações={row['n_evals']}; tempo={row['elapsed_total_s']:.2f}s", flush=True)
        except Exception:
            row['error'] = traceback.format_exc()
            print('  FALHA: ' + row['error'].splitlines()[-1], flush=True)
        rows.append(row)
        # checkpoint por execução: resultados parciais não se perdem em interrupções
        pd.DataFrame(rows).to_csv(out / 'runs.csv', index=False)
    runs = pd.DataFrame(rows)
    valid = runs[runs['status'] == 'ok'].copy()
    if not valid.empty:
        columns = ['test_accuracy','test_f1_macro','test_balanced_accuracy','test_min_class_recall',
                   'test_f1_weighted', 'test_precision_weighted',
                   'test_recall_weighted','best_val_accuracy','best_val_f1_macro',
                   'best_val_score','optimization_s','elapsed_total_s',
                   'n_evals','equivalent_full_evals']
        group = ['mode','space','scale','budget','method','fitness_metric','class_weight']
        aggregation = valid.groupby(group)[columns].agg(['mean','std','count'])
        aggregation.columns = [f'{c}_{stat}' for c, stat in aggregation.columns]
        aggregation.reset_index().to_csv(out / 'aggregated.csv', index=False)
        if mode == 'imbalance':
            make_imbalance_analysis(valid, out)
        else:
            make_statistics(valid, out / 'statistics')
            make_figures(valid, out)
    return out


def make_statistics(runs, directory):
    """Friedman/Wilcoxon exploratórios sobre blocos completos; NÃO prova generalização entre bases."""
    directory = Path(directory)
    notes = [
        'Estatística exploratória; blocos (escala, espaço, orçamento) da mesma base não são independentes.',
        'Use apenas grupos de algoritmos presentes em TODOS os blocos comparados.',
        'Não confundir repetição das sementes no mesmo teste com replicações independentes.',
        'Não concluir superioridade apenas pelo p-valor; verificar efeito, variância e orçamento.',
        'Testes Wilcoxon exploratórios SEM correção por múltiplas comparações; não inferência confirmatória.'
    ]
    if runs['method'].nunique() < 3:
        notes.append('Friedman não aplicado: exige pelo menos 3 métodos.')
    for metric in ['test_accuracy', 'test_f1_macro', 'elapsed_total_s']:
        tab = (runs.groupby(['scale','space','budget','method'])[metric].mean()
               .unstack('method').dropna())
        tab.to_csv(directory / f'blocks_{metric}.csv')
        if tab.shape[1] >= 3 and tab.shape[0] >= 2:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                statistic, p = friedmanchisquare(*[tab[c] for c in tab.columns])
            (directory / f'friedman_{metric}.txt').write_text(
                f'chi2={statistic}; p={p}; blocos={len(tab)}; algoritmos={len(tab.columns)}\n',
                encoding='utf-8')
            rank = tab.rank(axis=1, ascending=metric != 'elapsed_total_s').mean().sort_values(ascending=False)
            rank.to_csv(directory / f'ranks_{metric}.csv')
            # pares exploratórios
            pairs = []
            import itertools
            for a,b in itertools.combinations(tab.columns,2):
                diff = tab[a].values - tab[b].values
                if np.allclose(diff,0): pval = 1.0
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        try: pval = wilcoxon(diff).pvalue
                        except ValueError: pval = np.nan
                pairs.append({'algorithm_A':a, 'algorithm_B':b, 'p_unadjusted':pval})
            pd.DataFrame(pairs).to_csv(directory / f'wilcoxon_{metric}.csv',index=False)
        else:
            notes.append(f'{metric}: não há pelo menos 2 blocos completos e 3 algoritmos.')
    (directory / 'LEIA_ANTES_DA_ESTATISTICA.txt').write_text('\n'.join(notes),encoding='utf-8')


def make_figures(runs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    means = (runs.groupby(['method','scale'])['test_accuracy'].agg(['mean','std'])
             .reset_index().sort_values('mean', ascending=False))
    plt.figure(figsize=(11,5))
    labels = [f'{row.method} ({row.scale})' for row in means.itertuples()]
    plt.bar(labels, means['mean'])
    plt.xticks(rotation=35, ha='right')
    plt.ylabel('Acurácia no conjunto de teste (média das execuções)')
    plt.title('Comparação exploratória; conferir espaço e orçamento em aggregated.csv')
    plt.tight_layout()
    plt.savefig(out / 'figures' / 'overview_test_accuracy.png', dpi=160)
    plt.close()
    # Comparação de configurações só dentro do MESMO espaço, escala e orçamento.
    for (space, scale, budget), grp in runs.groupby(['space','scale','budget']):
        plt.figure(figsize=(9,4))
        agg = grp.groupby('method')['test_accuracy'].agg(['mean','std'])
        plt.bar(agg.index, agg['mean'], yerr=agg['std'].fillna(0), capsize=3)
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Acurácia no teste')
        plt.title(f'{space} | {scale} | orçamento nominal {budget}')
        plt.tight_layout()
        plt.savefig(out / 'figures' / f'accuracy_{space}_{scale}_{budget}.png',dpi=150)
        plt.close()
        plt.figure(figsize=(9,4))
        for method, subset in grp.groupby('method'):
            histories = [pd.read_csv(out / 'evaluations' / (rid + '.csv')) for rid in subset['run_id']]
            if not histories: continue
            curve = pd.concat([h.set_index('eval')['best_val_accuracy'] for h in histories],axis=1)
            yy = curve.ffill().mean(axis=1).dropna()
            if len(yy): plt.plot(yy.index, yy.values, label=method)
        plt.xlabel('Avaliações acumuladas')
        plt.ylabel('Melhor acurácia de validação')
        plt.title(f'Convergência | {space} | {scale} | {budget}')
        plt.legend(loc='lower right',fontsize=8)
        plt.tight_layout()
        plt.savefig(out / 'figures' / f'convergence_{space}_{scale}_{budget}.png',dpi=150)
        plt.close()


# Experimento NOVO: desenho fatorial 2x2 (fitness x pesos), analisado separadamente
# para PSO e PSO adaptativo, sem misturar configurações no resumo.
def make_imbalance_analysis(runs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    cols = ['test_accuracy', 'test_f1_macro', 'test_balanced_accuracy',
            'test_min_class_recall', 'optimization_s']
    keys = ['method', 'space', 'scale', 'budget', 'fitness_metric', 'class_weight']
    summary = runs.groupby(keys)[cols].agg(['mean','std','count'])
    summary.columns = [f'{col}_{stat}' for col,stat in summary.columns]
    summary.reset_index().to_csv(out / 'statistics' / 'ablation_summary.csv', index=False)
    records = []
    # A mesma semente identifica repetições pareadas; testes de hipótese não
    # são inferidos sobre uma única divisão de dados.
    for group_key, subset in runs.groupby(['method','space','scale','budget']):
        wide = subset.pivot_table(index='seed', columns=['fitness_metric','class_weight'],
                                  values=cols, aggfunc='mean')
        baseline = ('accuracy','none')
        for target in [('f1_macro','none'),('accuracy','balanced'),('f1_macro','balanced')]:
            for metric in cols:
                if (metric, *baseline) not in wide.columns or (metric, *target) not in wide.columns:
                    continue
                differences = (wide[(metric,*target)] - wide[(metric,*baseline)]).dropna()
                if len(differences):
                    records.append(dict(zip(['method','space','scale','budget'],group_key)) |
                        {'variant':f'{target[0]}+{target[1]}', 'metric':metric,
                         'seeds_paired':len(differences),
                         'delta_mean':float(differences.mean()),
                         'delta_std':float(differences.std(ddof=1)) if len(differences)>1 else None})
    pd.DataFrame(records).to_csv(out / 'statistics' / 'ablation_paired_deltas.csv', index=False)
    # Interação fatorial em pares completos por semente: I = D - B - C + A.
    # A comparação é EXPLORATÓRIA porque o mesmo teste externo é reutilizado.
    interaction_rows = []
    for group_key, subset in runs.groupby(['method','space','scale','budget']):
        for metric in cols:
            # Não agregar seeds duplicadas silenciosamente.
            pivot = subset.pivot_table(index='seed',
                                       columns=['fitness_metric','class_weight'],
                                       values=metric, aggfunc='mean')
            arms = [('accuracy','none'),('f1_macro','none'),
                    ('accuracy','balanced'),('f1_macro','balanced')]
            if any(a not in pivot.columns for a in arms):
                continue
            four = pivot.loc[:, arms].dropna()
            if four.empty:
                continue
            a, b, c, d = (four[arm] for arm in arms)
            effect = d - b - c + a
            interaction_rows.append(dict(zip(['method','space','scale','budget'],group_key)) |
                {'metric':metric, 'seeds_paired':len(effect),
                 'interaction_mean':float(effect.mean()),
                 'interaction_std':float(effect.std(ddof=1)) if len(effect)>1 else np.nan})
    pd.DataFrame(interaction_rows, columns=['method','space','scale','budget',
        'metric','seeds_paired','interaction_mean','interaction_std']).to_csv(
        out / 'statistics' / 'ablation_interactions.csv', index=False)

    # Diferenças pareadas do otimizador adaptativo vs. PSO tradicional
    # SOMENTE com braço, budget, escala, espaço e semente correspondentes.
    optimizer_rows = []
    for group_key, subset in runs.groupby(['space','scale','budget','fitness_metric','class_weight']):
        for metric in cols:
            wide_methods = subset.pivot_table(index='seed', columns='method',
                                             values=metric, aggfunc='mean')
            if not {'pso','adaptive_pso'}.issubset(wide_methods.columns):
                continue
            differences = (wide_methods['adaptive_pso'] - wide_methods['pso']).dropna()
            if differences.empty:
                continue
            optimizer_rows.append(dict(zip(['space','scale','budget','fitness_metric','class_weight'],group_key)) |
                {'metric':metric, 'seeds_paired':len(differences),
                 'adaptive_minus_pso_mean':float(differences.mean()),
                 'adaptive_minus_pso_std':float(differences.std(ddof=1)) if len(differences)>1 else np.nan})
    pd.DataFrame(optimizer_rows, columns=['space','scale','budget','fitness_metric',
        'class_weight','metric','seeds_paired','adaptive_minus_pso_mean',
        'adaptive_minus_pso_std']).to_csv(
        out / 'statistics' / 'adaptive_vs_pso_paired.csv', index=False)

    labels = {('accuracy','none'):'A - acurácia / sem pesos',
              ('f1_macro','none'):'B - F1-macro / sem pesos',
              ('accuracy','balanced'):'C - acurácia / balanceado',
              ('f1_macro','balanced'):'D - F1-macro / balanceado'}
    for (method,space,scale,budget), subset in runs.groupby(['method','space','scale','budget']):
        data = subset.groupby(['fitness_metric','class_weight'])[cols].mean()
        variants = [v for v in labels if v in data.index]
        if not variants: continue
        for metric, ylabel in [('test_accuracy','Acurácia no teste'),
                              ('test_f1_macro','F1-macro no teste'),
                              ('test_min_class_recall','Menor recall entre as classes')]:
            fig,ax=plt.subplots(figsize=(10,4.6))
            ax.bar(range(len(variants)),[data.loc[v,metric] for v in variants])
            ax.set_xticks(range(len(variants)),[labels[v] for v in variants],rotation=12,ha='right')
            ax.set_ylim(0,1.01)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{method} | {space} | {scale} | {budget} avaliações')
            fig.tight_layout()
            fig.savefig(out/'figures'/f'ablation_{method}_{space}_{scale}_{budget}_{metric}.png', dpi=150)
            plt.close(fig)
    (out/'statistics'/'INTERPRETACAO.txt').write_text(
        'Os quatro cenários são: A=accuracy+none, B=f1_macro+none, '
        'C=accuracy+balanced e D=f1_macro+balanced.\n'
        'Diferenças pareadas = variante - controle A, mesma semente, método, espaço, escala e orçamento.\n'
        'Interação fatorial = D - B - C + A, com comparação dentro do mesmo método e semente.\n'
        'adaptive_vs_pso_paired.csv confronta métodos no mesmo braço, semente e orçamento.\n'
        'O teste não é usado na escolha de C/gamma; comparações de teste são uma análise '
        'exploratória de configurações pré-definidas. Uma única base/divisão não prova superioridade geral.\n'
        'F1-macro pondera igualmente as seis classes; class_weight=balanced altera '
        'a penalização da SVM inversamente à frequência no fold de treino.\n'
        'SMOTE e Tomek links não foram aplicados.\n',encoding='utf-8')
