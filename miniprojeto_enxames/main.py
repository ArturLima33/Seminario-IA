from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from src.optimizers import METHODS
from src.runner import run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description='Miniprojeto SVM Satimage: métodos de otimização de C e gamma')
    parser.add_argument('--mode', choices=['reproduction', 'extension'], default='extension')
    parser.add_argument('--config', default='config/experiments.json')
    parser.add_argument('--data-path', default=None, help='Caminho do Satimage TSV/TSV.GZ real')
    parser.add_argument('--out', default=None)
    parser.add_argument('--smoke', action='store_true', help='BASE SINTÉTICA de 240 exemplos: SOMENTE TESTE FUNCIONAL')
    parser.add_argument('--quick', action='store_true', help='Subamostra de treino real: NÃO é resultado completo')
    parser.add_argument('--dry-run', action='store_true', help='Planeja experimentos sem carregar a base')
    parser.add_argument('--validation', choices=['holdout', 'cv'], default=None)
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--spaces', nargs='+', choices=['narrow', 'wide'])
    parser.add_argument('--scales', nargs='+', choices=['linear', 'log'])
    parser.add_argument('--budgets', nargs='+', type=int)
    parser.add_argument('--seeds', nargs='+', type=int)

    args = parser.parse_args(argv)

    if args.budgets and any(n < 2 for n in args.budgets):
        parser.error('--budgets exige valores >= 2')

    if args.smoke and args.quick:
        parser.error('Escolha --smoke OU --quick, não ambos')

    with open(args.config, encoding='utf-8') as f:
        config = json.load(f)

    if args.mode == 'reproduction' and args.methods and any(
        m not in config['reproduction']['methods'] for m in args.methods
    ):
        parser.error('Modo reproduction não permite métodos extras; use --mode extension')

    if args.out:
        output = Path(args.out)
    else:
        results_dir = Path('results')
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = 'smoke_' if args.smoke else 'quick_' if args.quick else ''
        output = results_dir / f'{args.mode}_{suffix}{timestamp}'

        counter = 1
        while output.exists():
            output = results_dir / f'{args.mode}_{suffix}{timestamp}_{counter}'
            counter += 1

    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    run_experiment(
        config,
        args.mode,
        output=output,
        data_path=args.data_path,
        smoke=args.smoke,
        quick=args.quick,
        validation_override=args.validation,
        method_override=args.methods,
        space_override=args.spaces,
        scale_override=args.scales,
        budget_override=args.budgets,
        seeds_override=args.seeds,
        dry_run=args.dry_run
    )

    if not args.dry_run:
        print('\nArquivos de saída:', output.resolve())


if __name__ == '__main__':
    main()