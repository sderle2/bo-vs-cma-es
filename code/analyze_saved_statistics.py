"""Reproduce report statistics from saved runs without writing report files."""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    summaries = ROOT / 'results' / 'summaries'
    paired_final = {}
    for function in ('sphere', 'ellipsoid'):
        differences = {'bo': [], 'cma': []}
        for trial in range(10):
            stem = f'{function}_d10_t{trial}'
            path = ROOT / 'results/lengthscale_sensitivity/full' / f'{stem}_bo_bound4.json'
            bound4 = json.loads(path.read_text())['trajectory'][-1]
            for method in differences:
                path = ROOT / 'results/budget_extension_1000' / f'{stem}_{method}.csv'
                baseline = pd.read_csv(path).iloc[-1]
                if (bound4['evaluation'] != 1000 or baseline.evaluation != 1000
                        or bound4['seed'] != baseline.seed
                        or bound4['trial'] != trial or baseline.trial != trial):
                    raise ValueError(f'Mismatched final comparison for {stem}: {method}')
                differences[method].append(bound4['best'] - baseline.best)
        for method, values in differences.items():
            q25, median, q75 = pd.Series(values).quantile([.25, .5, .75])
            paired_final[function, method] = f'{median:.3g} [{q25:.3g}, {q75:.3g}]'
    extension_dir = ROOT / 'results' / 'budget_extension_1000'
    final_kernels = [
        json.loads(path.read_text())['final_kernel']
        for path in sorted(extension_dir.glob('*_bo.json'))
    ]
    at_upper_bound = sum(
        bool(re.search(r'length_scale=2(?:\.0+)?(?=,|\))', kernel))
        for kernel in final_kernels
    )

    finals = pd.read_csv(summaries / 'final_runs.csv')
    timing = finals.groupby('method').elapsed.agg(['median', 'min', 'max'])
    follow = pd.read_csv(summaries / 'followup_pairs.csv')
    wins = (follow.bo < follow.bo_fixed).groupby(follow.function).sum()
    audit = json.loads((summaries / 'objective_audit.json').read_text())
    extension = json.loads((summaries / 'budget_extension_validation.json').read_text())
    total_warnings = sum(item['fit_warnings_recorded'] for item in audit.values())
    total_warnings += extension['fit_warnings_recorded']
    print('Final paired differences: bound-4 BO minus comparator, median [Q1, Q3]')
    for (function, method), stats in paired_final.items():
        print(function, 'bound-2 BO' if method == 'bo' else 'CMA-ES', stats)
    print(f'Final bound-2 BO length-scales at upper bound: {at_upper_bound}/{len(final_kernels)}')
    print('Learned-kernel wins:', ', '.join(f'{f} {n}' for f, n in wins.items()))
    print('GP-fit warnings, extension / main + follow-up + extension:',
          extension['fit_warnings_recorded'], '/', total_warnings)
    print('Main-run elapsed seconds:')
    print(timing.to_string(float_format=lambda value: f'{value:.4f}'))


if __name__ == '__main__':
    main()
