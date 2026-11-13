"""Validate and analyze the exploratory 1,000-evaluation budget extension."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark import make_objective


ROOT = Path(__file__).resolve().parents[1]
LABEL = 'budget_extension_1000'
CHECKPOINTS = [200, 500, 1000]
TIE_SCALE = 1e-8
LOG_EPS = 1e-12


def load_run(folder, function, trial, method):
    path = folder / f'{function}_d10_t{trial}_{method}.csv'
    frame = pd.read_csv(path)
    return path, frame


def winner(bo, cma):
    tolerance = TIE_SCALE * max(1.0, abs(bo), abs(cma))
    if abs(bo - cma) <= tolerance:
        return 'tie'
    return 'bo' if bo < cma else 'cma'


def validate_extension():
    folder = ROOT / 'results' / LABEL
    main = ROOT / 'results' / 'main'
    manifest = json.loads((folder / 'manifest.json').read_text())
    config_path = ROOT / 'configs' / f'{LABEL}.json'
    source_path = ROOT / 'code' / 'benchmark.py'
    assert manifest['config_sha256'] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert manifest['source_sha256'] == hashlib.sha256(source_path.read_bytes()).hexdigest()

    max_prefix_x = 0.0
    max_prefix_value = 0.0
    max_prefix_best = 0.0
    max_objective_error = 0.0
    evaluations = 0
    warning_count = 0
    seconds = 0.0
    frames = []

    for function in manifest['config']['functions']:
        for trial in manifest['config']['trials']:
            initial = []
            for method in manifest['config']['methods']:
                path, frame = load_run(folder, function, trial, method)
                assert len(frame) == 1000, path
                np.testing.assert_array_equal(frame.evaluation, np.arange(1, 1001))
                assert np.isfinite(frame[['value', 'best', 'elapsed']]).all().all(), path
                np.testing.assert_allclose(frame.best, np.minimum.accumulate(frame.value), rtol=0, atol=1e-12)
                assert np.all(np.diff(frame.best) <= 1e-12), path
                xs = np.array([json.loads(x) for x in frame.x])
                assert xs.shape == (1000, 10), path
                assert np.all((xs >= 0) & (xs <= 1)), path
                initial.append(xs[:10])

                main_path = main / f'{function}_d10_t{trial}_{method}.csv'
                old = pd.read_csv(main_path)
                old_x = np.array([json.loads(x) for x in old.x])
                max_prefix_x = max(max_prefix_x, float(np.max(np.abs(xs[:200] - old_x))))
                max_prefix_value = max(max_prefix_value, float(np.max(np.abs(frame.value.iloc[:200].to_numpy() - old.value.to_numpy()))))
                max_prefix_best = max(max_prefix_best, float(np.max(np.abs(frame.best.iloc[:200].to_numpy() - old.best.to_numpy()))))

                objective, _ = make_objective(function, 10, trial)
                recalculated = np.array([objective(x) for x in xs])
                max_objective_error = max(max_objective_error, float(np.max(np.abs(recalculated - frame.value.to_numpy()))))
                np.testing.assert_allclose(recalculated, frame.value, rtol=1e-12, atol=1e-10)

                metadata = json.loads(path.with_suffix('.json').read_text())
                warning_count += metadata['warnings']
                seconds += metadata['seconds']
                evaluations += len(frame)
                frames.append(frame)
            np.testing.assert_array_equal(initial[0], initial[1])

    prefix_ok = max(max_prefix_x, max_prefix_value, max_prefix_best) <= 1e-12
    assert prefix_ok
    return pd.concat(frames, ignore_index=True), {
        'runs': len(frames),
        'evaluations_recomputed': evaluations,
        'source_and_config_hash_match': True,
        'first_200_prefix_match': prefix_ok,
        'max_prefix_x_absolute_difference': max_prefix_x,
        'max_prefix_value_absolute_difference': max_prefix_value,
        'max_prefix_best_absolute_difference': max_prefix_best,
        'max_objective_recomputation_error': max_objective_error,
        'fit_warnings_recorded': warning_count,
        'extension_run_seconds_sum': seconds,
    }


def summarize_extension(data, destination):
    points = data[data.evaluation.isin(CHECKPOINTS)]
    summary = points.groupby(['function', 'evaluation', 'method']).agg(
        median_gap=('best', 'median'),
        q25=('best', lambda x: x.quantile(.25)),
        q75=('best', lambda x: x.quantile(.75)),
    ).reset_index()
    summary.to_csv(destination / 'budget_extension_checkpoints.csv', index=False)

    rows = []
    for function in ['sphere', 'ellipsoid']:
        for trial in range(10):
            subset = points[(points.function == function) & (points.trial == trial)]
            for evaluation in CHECKPOINTS:
                selected = subset[subset.evaluation == evaluation].set_index('method').best
                bo = float(selected['bo'])
                cma = float(selected['cma'])
                rows.append({
                    'function': function,
                    'trial': trial,
                    'evaluation': evaluation,
                    'bo_gap': bo,
                    'cma_gap': cma,
                    'raw_difference_cma_minus_bo': cma - bo,
                    'log10_ratio_cma_over_bo': np.log10((cma + LOG_EPS) / (bo + LOG_EPS)),
                    'winner': winner(bo, cma),
                })
    paired = pd.DataFrame(rows)
    paired.to_csv(destination / 'budget_extension_pairs.csv', index=False)

    counts = paired.groupby(['function', 'evaluation', 'winner']).size().unstack(fill_value=0).reset_index()
    for column in ['bo', 'cma', 'tie']:
        if column not in counts:
            counts[column] = 0
    counts = counts[['function', 'evaluation', 'bo', 'cma', 'tie']]
    counts.to_csv(destination / 'budget_extension_wins.csv', index=False)

    closure = []
    for function, group in paired.groupby('function'):
        wide = group.pivot(index='trial', columns='evaluation', values='log10_ratio_cma_over_bo')
        change = wide[1000] - wide[200]
        closure.append({
            'function': function,
            'median_log10_ratio_at_200': float(wide[200].median()),
            'median_log10_ratio_at_500': float(wide[500].median()),
            'median_log10_ratio_at_1000': float(wide[1000].median()),
            'median_change_1000_minus_200': float(change.median()),
            'trials_cma_closed_relative_gap': int((change < -1e-12).sum()),
            'trials_bo_extended_relative_advantage': int((change > 1e-12).sum()),
            'trials_unchanged': int((np.abs(change) <= 1e-12).sum()),
        })
    pd.DataFrame(closure).to_csv(destination / 'budget_extension_closure.csv', index=False)
    return summary, paired, counts, closure


def plot_extension(data):
    colors = {'bo': '#0072B2', 'cma': '#D55E00'}
    labels = {'bo': 'BO', 'cma': 'CMA-ES'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for axis, function in zip(axes, ['sphere', 'ellipsoid']):
        for method in ['bo', 'cma']:
            subset = data[(data.function == function) & (data.method == method)]
            pivot = subset.pivot(index='evaluation', columns='trial', values='best').loc[10:]
            quantiles = np.quantile(pivot, [.25, .5, .75], axis=1)
            axis.plot(pivot.index, np.maximum(quantiles[1], 1e-12), color=colors[method], label=labels[method])
            axis.fill_between(pivot.index, np.maximum(quantiles[0], 1e-12), np.maximum(quantiles[2], 1e-12), color=colors[method], alpha=.16)
        for checkpoint in CHECKPOINTS:
            axis.axvline(checkpoint, color='#999999', linewidth=.6, alpha=.35)
        axis.set(title=f'{function.title()} | d = 10', xlabel='Total objective evaluations', ylabel='Best gap', yscale='log')
        axis.grid(alpha=.18)
    axes[0].legend()
    fig.tight_layout()
    path = ROOT / 'report' / 'figures' / 'budget_extension.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_extension_crossovers(data, destination):
    report = {}
    for function in ['sphere', 'ellipsoid']:
        subset = data[data.function == function]
        medians = subset.groupby(['evaluation', 'method']).best.median().unstack()
        difference = medians.cma - medians.bo
        after_200 = difference.loc[201:]
        first = int(after_200[after_200 <= 0].index[0])
        sustained = None
        for evaluation in after_200.index:
            if (after_200.loc[evaluation:] <= 0).all():
                sustained = int(evaluation)
                break
        report[function] = {
            'first_evaluation_median_cma_no_larger_than_bo': first,
            'sustained_crossover_evaluation': sustained,
            'bo_median_at_sustained_crossover': float(medians.loc[sustained, 'bo']),
            'cma_median_at_sustained_crossover': float(medians.loc[sustained, 'cma']),
        }
    (destination / 'budget_extension_crossovers.json').write_text(json.dumps(report, indent=2))
    return report


def analyze_followup_timing(destination):
    folder = ROOT / 'results' / 'followup'
    rows = []
    for path in sorted(folder.glob('*.csv')):
        frame = pd.read_csv(path)
        last = frame.iloc[-1]
        metadata = json.loads(path.with_suffix('.json').read_text())
        rows.append({
            'function': last.function,
            'trial': int(last.trial),
            'method': last.method,
            'seconds': metadata['seconds'],
            'fit_seconds': float(last.fit_seconds),
            'acquisition_seconds': float(last.acquisition_seconds),
            'objective_seconds': float(last.objective_seconds),
            'other_seconds': metadata['seconds'] - float(last.fit_seconds) - float(last.acquisition_seconds) - float(last.objective_seconds),
        })
    timing = pd.DataFrame(rows)
    summary = timing.groupby(['function', 'method']).agg(
        median_seconds=('seconds', 'median'),
        median_fit_seconds=('fit_seconds', 'median'),
        median_acquisition_seconds=('acquisition_seconds', 'median'),
        median_objective_seconds=('objective_seconds', 'median'),
        median_other_seconds=('other_seconds', 'median'),
    ).reset_index()
    summary.to_csv(destination / 'followup_timing_components.csv', index=False)
    return summary


def analyze_ellipsoid_crossover(destination):
    folder = ROOT / 'results' / 'main'
    frames = []
    for trial in range(10):
        for method in ['bo', 'cma']:
            frame = pd.read_csv(folder / f'ellipsoid_d2_t{trial}_{method}.csv')
            frames.append(frame[['evaluation', 'best', 'trial', 'method']])
    data = pd.concat(frames)
    medians = data.groupby(['evaluation', 'method']).best.median().unstack()
    difference = medians.cma - medians.bo
    after_initial = difference.loc[11:]
    sustained = None
    for evaluation in after_initial.index:
        if (after_initial.loc[evaluation:] <= 0).all():
            sustained = int(evaluation)
            break
    report = {
        'definition': 'earliest evaluation after initialization from which median CMA gap stays no larger than median BO gap through evaluation 200',
        'sustained_crossover_evaluation': sustained,
        'bo_median_at_50': float(medians.loc[50, 'bo']),
        'cma_median_at_50': float(medians.loc[50, 'cma']),
        'bo_median_at_100': float(medians.loc[100, 'bo']),
        'cma_median_at_100': float(medians.loc[100, 'cma']),
        'bo_median_at_200': float(medians.loc[200, 'bo']),
        'cma_median_at_200': float(medians.loc[200, 'cma']),
    }
    (destination / 'ellipsoid_d2_crossover.json').write_text(json.dumps(report, indent=2))
    return report


def select_failure_case(paired, data, destination):
    final = paired[paired.evaluation == 1000].copy()
    final['worse_method'] = np.where(final.bo_gap > final.cma_gap, 'bo', 'cma')
    final['worse_gap'] = np.maximum(final.bo_gap, final.cma_gap)
    final['better_gap'] = np.minimum(final.bo_gap, final.cma_gap)
    final['failure_ratio'] = (final.worse_gap + LOG_EPS) / (final.better_gap + LOG_EPS)
    chosen = final.sort_values(['failure_ratio', 'function', 'trial'], ascending=[False, True, True]).iloc[0]
    function = chosen.function
    trial = int(chosen.trial)
    trace = {}
    for method in ['bo', 'cma']:
        frame = data[(data.function == function) & (data.trial == trial) & (data.method == method)].set_index('evaluation')
        trace[method] = {str(evaluation): float(frame.loc[evaluation, 'best']) for evaluation in [10, 50, 100, 200, 500, 1000]}
    report = {
        'selection_rule': 'largest final worse-method/better-method gap ratio across the 20 matched function-trial comparisons; ties broken by function then trial',
        'status': 'illustrative, not representative',
        'function': function,
        'dimension': 10,
        'trial': trial,
        'worse_method': chosen.worse_method,
        'failure_ratio': float(chosen.failure_ratio),
        'traces': trace,
    }
    (destination / 'illustrative_failure_case.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    destination = ROOT / 'results' / 'summaries'
    data, validation = validate_extension()
    summary, paired, counts, closure = summarize_extension(data, destination)
    plot_extension(data)
    crossovers = analyze_extension_crossovers(data, destination)
    timing = analyze_followup_timing(destination)
    crossover = analyze_ellipsoid_crossover(destination)
    failure = select_failure_case(paired, data, destination)
    (destination / 'budget_extension_validation.json').write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))
    print('\nCHECKPOINTS\n', summary.to_string(index=False))
    print('\nWINS\n', counts.to_string(index=False))
    print('\nCLOSURE\n', pd.DataFrame(closure).to_string(index=False))
    print('\nEXTENSION CROSSOVERS\n', json.dumps(crossovers, indent=2))
    print('\nFOLLOWUP TIMING\n', timing.to_string(index=False))
    print('\nELLIPSOID CROSSOVER\n', json.dumps(crossover, indent=2))
    print('\nILLUSTRATIVE FAILURE\n', json.dumps(failure, indent=2))


if __name__ == '__main__':
    main()
