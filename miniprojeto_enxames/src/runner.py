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
          budget_override=None, seeds_override=None):
    plan = cfg[mode]
    methods = method_override or plan['methods']
    spaces = space_override or plan['spaces']
    scales = scale_override or plan['scales']
    budgets = budget_override or plan['budgets']
    seeds = seeds_override or plan['seeds']

    for sp in spaces:
        for sc in scales:
            for budget in budgets:
                for method in methods:
                    for seed in seeds:
                        yield {
                            'mode': mode,
                            'space': sp,
                            'scale': sc,
                            'budget': int(budget),
                            'method': method,
                            'seed': int(seed)
                        }


def effective_budget(task):
    if task['method'] != 'sha':
        return task['budget'], {}

    nominal = task['budget']
    k = {20: 4, 50: 5, 100: 6}.get(nominal)

    if k is None:
        k = max(1, int(np.floor(np.log2(nominal + 2))) - 1)

    return 2 ** (k + 1) - 2, {'k': k}


def _save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )


def run_experiment(cfg, mode, *, output, data_path=None, smoke=False, quick=False,
                   validation_override=None, method_override=None, space_override=None,
                   scale_override=None, budget_override=None, seeds_override=None,
                   dry_run=False):
    tasks = list(specs(
        cfg,
        mode,
        method_override,
        space_override,
        scale_override,
        budget_override,
        seeds_override
    ))

    for task in tasks:
        task['actual_evals'], task['sha_params'] = effective_budget(task)

    if dry_run:
        print(
            f'Plano: {len(tasks)} execuções; '
            f'{sum(t["actual_evals"] for t in tasks)} avaliações da SVM'
        )
        print(
            pd.DataFrame(tasks)
            .groupby(['method', 'space', 'scale', 'budget'], as_index=False)
            .agg(
                runs=('seed', 'count'),
                evaluations=('actual_evals', 'sum')
            )
            .to_string(index=False)
        )
        return None

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    for directory in ['evaluations', 'confusion', 'statistics', 'figures']:
        (out / directory).mkdir(parents=True, exist_ok=True)

    print(f'Resultados serão salvos em: {out.resolve()}', flush=True)

    X, y, meta = load_data(data_path, smoke=smoke)
    report = describe(X, y, meta)

    val = dict(cfg['validation'])
    if validation_override:
        val['type'] = validation_override

    if quick and not smoke:
        print(
            'ATENÇÃO: --quick usa subamostra da base real, '
            'não interpretar como resultado científico completo'
        )

    cfg_record = {
        'parameters': cfg,
        'mode': mode,
        'smoke': smoke,
        'quick': quick,
        'validation_effective': val,
        'dataset': report,
        'tasks': tasks,
        'environment': {
            'python': sys.version,
            'scikit_learn': sklearn.__version__,
            'platform': platform.platform()
        },
        'generated_at': datetime.now().isoformat()
    }

    _save_json(out / 'experiment_manifest.json', cfg_record)
    _save_json(out / 'dataset_report.json', report)

    rows = []

    for index, task in enumerate(tasks, start=1):
        ident = '__'.join(
            str(task[k])
            for k in ['mode', 'space', 'scale', 'budget', 'method', 'seed']
        )

        print(f'[{index}/{len(tasks)}] {ident}', flush=True)

        row = {
            **task,
            'run_id': ident,
            'smoke': smoke,
            'quick': quick,
            'status': 'failed',
            'error': ''
        }

        started = time.perf_counter()

        try:
            Xtr, Xte, ytr, yte = split_data(
                X,
                y,
                cfg['dataset']['test_fraction'],
                cfg['dataset']['seed']
            )

            if quick and not smoke:
                from sklearn.model_selection import train_test_split

                Xtr, _, ytr, _ = train_test_split(
                    Xtr,
                    ytr,
                    train_size=0.25,
                    stratify=ytr,
                    random_state=cfg['dataset']['seed']
                )

            bounds = cfg['search_spaces'][task['space']]
            space = SearchSpace(
                tuple(bounds['C']),
                tuple(bounds['gamma']),
                task['scale']
            )

            objective = Objective(
                Xtr,
                ytr,
                space,
                task['actual_evals'],
                seed=cfg['dataset']['seed'],
                validation=val,
                scaler=cfg['scaler']
            )

            params = dict(cfg['params'].get(task['method'], {}))
            params.update(task['sha_params'])

            result = run_optimizer(
                task['method'],
                objective,
                task['seed'],
                params
            )

            metrics, cm = final_test(
                Xtr,
                ytr,
                Xte,
                yte,
                result['best_C'],
                result['best_gamma'],
                scaler=cfg['scaler']
            )

            pd.DataFrame(objective.evaluations).to_csv(
                out / 'evaluations' / f'{ident}.csv',
                index=False
            )

            np.savetxt(
                out / 'confusion' / f'{ident}.csv',
                cm,
                fmt='%d',
                delimiter=','
            )

            row.update(result)
            row.update({f'test_{k}': v for k, v in metrics.items()})
            row.update({
                'train_rows': len(ytr),
                'test_rows': len(yte),
                'status': 'ok',
                'elapsed_total_s': time.perf_counter() - started
            })

            print(
                f"  validação={row['best_val_accuracy']:.4f}; "
                f"teste={row['test_accuracy']:.4f}; "
                f"avaliações={row['n_evals']}; "
                f"tempo={row['elapsed_total_s']:.2f}s",
                flush=True
            )

        except Exception:
            row['error'] = traceback.format_exc()
            print(
                '  FALHA: ' + row['error'].splitlines()[-1],
                flush=True
            )

        rows.append(row)

        pd.DataFrame(rows).to_csv(
            out / 'runs.csv',
            index=False
        )

    runs = pd.DataFrame(rows)
    valid = runs[runs['status'] == 'ok'].copy()

    if not valid.empty:
        columns = [
            'test_accuracy',
            'test_f1_macro',
            'test_f1_weighted',
            'test_precision_weighted',
            'test_recall_weighted',
            'best_val_accuracy',
            'optimization_s',
            'elapsed_total_s',
            'n_evals',
            'equivalent_full_evals'
        ]

        group = ['mode', 'space', 'scale', 'budget', 'method']
        aggregation = valid.groupby(group)[columns].agg(['mean', 'std', 'count'])
        aggregation.columns = [
            f'{column}_{stat}'
            for column, stat in aggregation.columns
        ]
        aggregation.reset_index().to_csv(out / 'aggregated.csv', index=False)

        make_statistics(valid, out / 'statistics')
        make_figures(valid, out)

    return out


def make_statistics(runs, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

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
        tab = (
            runs.groupby(['scale', 'space', 'budget', 'method'])[metric]
            .mean()
            .unstack('method')
            .dropna()
        )

        tab.to_csv(directory / f'blocks_{metric}.csv')

        if tab.shape[1] >= 3 and tab.shape[0] >= 2:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                statistic, p = friedmanchisquare(
                    *[tab[column] for column in tab.columns]
                )

            (directory / f'friedman_{metric}.txt').write_text(
                f'chi2={statistic}; p={p}; blocos={len(tab)}; '
                f'algoritmos={len(tab.columns)}\n',
                encoding='utf-8'
            )

            rank = tab.rank(
                axis=1,
                ascending=metric != 'elapsed_total_s'
            ).mean().sort_values(ascending=False)

            rank.to_csv(directory / f'ranks_{metric}.csv')

            pairs = []
            import itertools

            for a, b in itertools.combinations(tab.columns, 2):
                diff = tab[a].values - tab[b].values

                if np.allclose(diff, 0):
                    pval = 1.0
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        try:
                            pval = wilcoxon(diff).pvalue
                        except ValueError:
                            pval = np.nan

                pairs.append({
                    'algorithm_A': a,
                    'algorithm_B': b,
                    'p_unadjusted': pval
                })

            pd.DataFrame(pairs).to_csv(
                directory / f'wilcoxon_{metric}.csv',
                index=False
            )
        else:
            notes.append(
                f'{metric}: não há pelo menos 2 blocos completos e 3 algoritmos.'
            )

    (directory / 'LEIA_ANTES_DA_ESTATISTICA.txt').write_text(
        '\n'.join(notes),
        encoding='utf-8'
    )


def make_figures(runs, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out)
    figures_dir = out / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    means = (
        runs.groupby(['method', 'scale'])['test_accuracy']
        .agg(['mean', 'std'])
        .reset_index()
        .sort_values('mean', ascending=False)
    )

    plt.figure(figsize=(11, 5))
    labels = [
        f'{row.method} ({row.scale})'
        for row in means.itertuples()
    ]
    plt.bar(labels, means['mean'])
    plt.xticks(rotation=35, ha='right')
    plt.ylabel('Acurácia no conjunto de teste (média das execuções)')
    plt.title('Comparação exploratória; conferir espaço e orçamento em aggregated.csv')
    plt.tight_layout()
    plt.savefig(figures_dir / 'overview_test_accuracy.png', dpi=160)
    plt.close()

    for (space, scale, budget), grp in runs.groupby(['space', 'scale', 'budget']):
        plt.figure(figsize=(9, 4))
        agg = grp.groupby('method')['test_accuracy'].agg(['mean', 'std'])
        plt.bar(agg.index, agg['mean'], yerr=agg['std'].fillna(0), capsize=3)
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Acurácia no teste')
        plt.title(f'{space} | {scale} | orçamento nominal {budget}')
        plt.tight_layout()
        plt.savefig(
            figures_dir / f'accuracy_{space}_{scale}_{budget}.png',
            dpi=150
        )
        plt.close()

        plt.figure(figsize=(9, 4))

        for method, subset in grp.groupby('method'):
            histories = [
                pd.read_csv(out / 'evaluations' / f'{run_id}.csv')
                for run_id in subset['run_id']
            ]

            if not histories:
                continue

            curve = pd.concat(
                [
                    history.set_index('eval')['best_val_accuracy']
                    for history in histories
                ],
                axis=1
            )

            yy = curve.ffill().mean(axis=1).dropna()

            if len(yy):
                plt.plot(yy.index, yy.values, label=method)

        plt.xlabel('Avaliações acumuladas')
        plt.ylabel('Melhor acurácia de validação')
        plt.title(f'Convergência | {space} | {scale} | {budget}')
        plt.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        plt.savefig(
            figures_dir / f'convergence_{space}_{scale}_{budget}.png',
            dpi=150
        )
        plt.close()