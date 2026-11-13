"""Compare the saved BO upper bound of 2 with a new upper bound of 4."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys

import benchmark as bm

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/budget_extension_1000.json'
BASE = ROOT / 'results/budget_extension_1000'
OUTPUT = ROOT / 'results/lengthscale_sensitivity'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_atomic(path, content):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(content)
    tmp.replace(path)


@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare_runner():
    # Reuse the validated loop verbatim apart from the bound and telemetry.
    source = inspect.getsource(bm.run_one)
    substitutions = {
        'Matern(.25, (.02, 2.), nu=2.5)': 'Matern(.25, (.02, 4.), nu=2.5)',
        '    warning_count = 0': '    warning_count = 0\n    fitted = []',
        '            kernel = gp.kernel_': (
            '            kernel = gp.kernel_\n'
            '            if learn:\n'
            '                fitted.append(dict(observations=len(y), '
            'length_scale=float(kernel.k2.length_scale)))'
        ),
        '    return frame,meta': '    meta["fitted_length_scales"] = fitted\n    return frame,meta',
    }
    for original, replacement in substitutions.items():
        if source.count(original) != 1:
            raise RuntimeError(f'Benchmark loop changed: {original}')
        source = source.replace(original, replacement)
    namespace = vars(bm).copy()
    exec(compile(source, '<lengthscale sensitivity runner>', 'exec'), namespace)
    return namespace['run_one']


def baseline(config):
    manifest = json.loads((BASE / 'manifest.json').read_text())
    if manifest['config_sha256'] != digest(CONFIG) or manifest['source_sha256'] != digest(bm.__file__):
        raise RuntimeError('Saved baseline source/config hash mismatch')
    traces = {}
    missing = []
    for function in config['functions']:
        for trial in config['trials']:
            for method in ('bo', 'cma'):
                path = BASE / f'{function}_d10_t{trial}_{method}.csv'
                if not path.is_file():
                    missing.append(str(path))
                    continue
                frame = bm.pd.read_csv(path)
                if list(frame.evaluation) != list(range(1, config['budget'] + 1)):
                    raise ValueError(f'Incomplete or mismatched budget: {path}')
                traces[function, trial, method] = frame
    if missing:
        raise FileNotFoundError('Missing equal-budget comparisons:\n' + '\n'.join(missing))
    return manifest, traces


def check_run(record, expected, original, budget):
    if record['identity'] != expected:
        raise ValueError('Run metadata mismatch')
    frame = bm.pd.DataFrame(record['trajectory'])
    if list(frame.evaluation) != list(range(1, budget + 1)):
        raise ValueError('Run budget mismatch')
    values = frame.value.to_numpy()
    if not bm.np.isfinite(values).all() or not bm.np.array_equal(
        frame.best.to_numpy(), bm.np.minimum.accumulate(values)
    ):
        raise ValueError('Invalid best-so-far trace')
    n = expected['n_initial']
    if list(frame.x.iloc[:n]) != list(original.x.iloc[:n]):
        raise ValueError('Initial inputs differ from saved BO')
    bm.np.testing.assert_allclose(values[:n], original.value.iloc[:n], rtol=1e-12, atol=1e-12)
    fits = record['metadata']['fitted_length_scales']
    if [fit['observations'] for fit in fits] != list(range(n, budget, expected['refit_every'])):
        raise ValueError('Missing length-scale refits')
    if not all(.02 <= fit['length_scale'] <= 4.000000000001 for fit in fits):
        raise ValueError('Invalid fitted length-scale')
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'run', 'analyze'))
    mode = parser.parse_args().mode
    cfg = json.loads(CONFIG.read_text())
    manifest, originals = baseline(cfg)
    versions = dict(python=sys.version, numpy=bm.np.__version__, scipy=bm.scipy.__version__,
                    sklearn=bm.sklearn.__version__, cma=bm.cma.__version__)
    if mode != 'analyze':
        for package in ('numpy', 'scipy', 'sklearn', 'cma'):
            if versions[package] != manifest[package]:
                raise RuntimeError(f'{package} version differs from saved baseline')
    directory = OUTPUT / ('smoke' if mode == 'smoke' else 'full')
    budget = cfg['n_initial'] + 2 if mode == 'smoke' else cfg['budget']
    trials = cfg['trials'][:1] if mode == 'smoke' else cfg['trials']
    runner = prepare_runner() if mode != 'analyze' else None
    rows = []
    with locked(directory), bm.threadpool_limits(limits=1):
        for function in cfg['functions']:
            for trial in trials:
                path = directory / f'{function}_d10_t{trial}_bo_bound4.json'
                identity = dict(function=function, dimension=10, trial=trial, budget=budget,
                                bound=4, seed=cfg['seed_base'] + trial*100 + 10,
                                n_initial=cfg['n_initial'], refit_every=cfg['refit_every'],
                                source_sha256=digest(bm.__file__), script_sha256=digest(__file__),
                                config_sha256=digest(CONFIG))
                original = originals[function, trial, 'bo']
                if path.exists():
                    record = json.loads(path.read_text())
                    frame = check_run(record, identity, original, budget)
                    if mode != 'analyze':
                        print(f'SKIP {path.name}', flush=True)
                elif mode == 'analyze':
                    raise SystemExit(f'Full experiment incomplete: missing {path}')
                else:
                    print(f'START {path.name} budget={budget}', flush=True)
                    frame, metadata = runner(function, 10, trial, 'bo', dict(cfg, budget=budget))
                    record = dict(identity=identity, metadata=metadata, trajectory=frame.to_dict('records'),
                                  versions=versions, platform=platform.platform(), threads=1)
                    check_run(record, identity, original, budget)
                    write_atomic(path, json.dumps(record, allow_nan=False))
                    print(f'DONE {path.name} seconds={metadata["seconds"]:.2f}', flush=True)
                if mode == 'analyze':
                    for t in (200, 500, 1000):
                        rows.append(dict(function=function, evaluation=t,
                                         bo_bound2=original.best.iloc[t-1],
                                         bo_bound4=frame.best.iloc[t-1],
                                         cma_es=originals[function, trial, 'cma'].best.iloc[t-1]))
        if mode == 'analyze':
            pairs = bm.pd.DataFrame(rows)
            summary = pairs.groupby(['function', 'evaluation']).median()
            grouped = pairs.assign(bound4_wins=pairs.bo_bound4 < pairs.bo_bound2,
                                   bound4_beats_cma=pairs.bo_bound4 < pairs.cma_es)
            for column in ('bound4_wins', 'bound4_beats_cma'):
                summary[column] = grouped.groupby(['function', 'evaluation'])[column].sum()
            write_atomic(directory / 'comparison.csv', summary.to_csv())
            print(summary.to_string())


if __name__ == '__main__':
    main()
