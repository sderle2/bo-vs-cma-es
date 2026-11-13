"""Reproducible bounded black-box benchmark; objectives are custom, not BBOB."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse, hashlib, json, platform, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.linalg import solve_triangular
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from threadpoolctl import threadpool_limits
import cma

ROOT = Path(__file__).resolve().parents[1]

def make_objective(name, d, instance):
    rng = np.random.default_rng(10000 + instance * 100 + d)
    center = rng.uniform(.25, .75, d)
    rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
    def objective(x):
        x = np.asarray(x)
        if x.shape != (d,) or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
            raise ValueError('Invalid objective input')
        z = 10 * (x - center)
        if name == 'sphere':
            return float(z @ z)
        if name == 'ellipsoid':
            z = rotation @ z
            return float(np.sum(np.logspace(0, 3, d) * z**2))
        if name == 'rastrigin':
            return float(np.sum(z**2 + 10 * (1 - np.cos(2*np.pi*z))))
        raise ValueError(name)
    return objective, center

def expected_improvement(mu, sd, best):
    delta = best - mu
    z = np.divide(delta, sd, out=np.zeros_like(delta), where=sd > 1e-14)
    ei = delta * ndtr(z) + sd * np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    return np.where(sd > 1e-14, np.maximum(ei, 0), np.maximum(delta, 0))

def predict(gp, points):
    # Same latent posterior as sklearn, using explicit contractions to avoid
    # host BLAS matmul floating-point flag warnings observed in the pilot.
    cross = gp.kernel_(points, gp.X_train_)
    mean = np.einsum('ij,j->i', cross, gp.alpha_) * gp._y_train_std + gp._y_train_mean
    v = solve_triangular(gp.L_, cross.T, lower=True, check_finite=False)
    variance = np.maximum(gp.kernel_.diag(points)-np.einsum('ij,ij->j', v, v), 0)
    sd = np.sqrt(variance) * gp._y_train_std
    if not np.isfinite(mean).all() or not np.isfinite(sd).all():
        raise FloatingPointError('Nonfinite GP posterior')
    return mean, sd

def propose(gp, X, best, rng, starts, pool_size):
    d = X.shape[1]
    pool = rng.uniform(0, 1, (pool_size, d))
    mu, sd = predict(gp, pool)
    ei = expected_improvement(mu, sd, best)
    candidates = list(pool)
    # Optimize log(EI + small floor): monotone transformation improves scaling.
    def loss(x):
        m, s = predict(gp, x[None, :])
        return -float(np.log(expected_improvement(m, s, best)[0] + 1e-30))
    for i in np.argsort(ei)[-starts:]:
        result = minimize(loss, pool[i], method='L-BFGS-B', bounds=[(0., 1.)]*d,
                          options={'maxiter': 35, 'ftol': 1e-8})
        # Valid evaluated endpoints are candidates even if iteration limit was reached.
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            candidates.append(np.clip(result.x, 0, 1))
    candidates = np.array(candidates)
    keep = np.min(np.linalg.norm(candidates[:, None, :] - X[None, :, :], axis=2), axis=1) > 1e-7
    candidates = candidates[keep]
    if not len(candidates):
        raise RuntimeError('No distinct acquisition candidate')
    m, s = predict(gp, candidates)
    return candidates[np.argmax(expected_improvement(m, s, best))]

def run_one(name, d, trial, method, cfg):
    obj, _ = make_objective(name, d, trial)
    seed = cfg['seed_base'] + trial*100 + d
    rng = np.random.default_rng(seed)
    n_init = cfg['n_initial']
    initial = rng.uniform(0, 1, (n_init, d))
    X, y, rows = [], [], []
    start = time.perf_counter()
    obj_time = fit_time = acq_time = 0.
    warning_count = 0
    def evaluate(x):
        nonlocal obj_time
        before = time.perf_counter()
        value = obj(x)
        obj_time += time.perf_counter()-before
        X.append(np.array(x)); y.append(value)
        rows.append(dict(evaluation=len(y),value=value,best=min(y),elapsed=time.perf_counter()-start,
                         objective_seconds=obj_time,fit_seconds=fit_time,acquisition_seconds=acq_time,
                         x=json.dumps(np.asarray(x).tolist())))
        return value
    for x in initial:
        evaluate(x)
    # All methods pay for and retain identical initial points. CMA starts at their best.
    es = None
    if method == 'cma':
        es = cma.CMAEvolutionStrategy(X[int(np.argmin(y))], .25,
            {'bounds':[0,1], 'seed':seed+1, 'verbose':-9, 'verb_log':0})
    kernel = ConstantKernel(1., (.01, 100.))*Matern(.25, (.02, 2.), nu=2.5)
    while len(y) < cfg['budget']:
        if method == 'random':
            evaluate(rng.uniform(0, 1, d))
        elif method == 'cma':
            batch = es.ask()
            remaining = cfg['budget']-len(y)
            chosen = batch[:remaining]
            vals = [evaluate(x) for x in chosen]
            if len(chosen) == len(batch):
                es.tell(batch, vals)
            # Incomplete last generation is evaluated but not used for an update.
        else:
            learn = method == 'bo' and ((len(y)-n_init) % cfg['refit_every'] == 0)
            before = time.perf_counter()
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-8,
                optimizer='fmin_l_bfgs_b' if learn else None,
                n_restarts_optimizer=0, random_state=seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                gp.fit(np.asarray(X), np.asarray(y))
            warning_count += len(caught)
            kernel = gp.kernel_
            fit_time += time.perf_counter()-before
            before = time.perf_counter()
            point = propose(gp,np.asarray(X),min(y),rng,cfg['acquisition_starts'],cfg['candidate_pool'])
            acq_time += time.perf_counter()-before
            evaluate(point)
    frame = pd.DataFrame(rows)
    for k,v in dict(function=name,dimension=d,trial=trial,method=method,seed=seed).items():
        frame[k]=v
    meta=dict(seconds=time.perf_counter()-start,warnings=warning_count,
              final_kernel=str(kernel) if method.startswith('bo') else None,
              cma_stop={str(k):str(v) for k,v in es.stop().items()} if es else None)
    return frame,meta

def main():
    parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args()
    config_path=Path(args.config);cfg=json.loads(config_path.read_text())
    out=ROOT/'results'/cfg['label'];out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(config_path.read_bytes()).hexdigest()
    source_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous['config_sha256'] != digest or previous['source_sha256'] != source_digest:
            raise RuntimeError('Refusing resume with changed source/config')
    else:
        manifest=dict(config=cfg,config_sha256=digest,source_sha256=source_digest,
            created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            python=sys.version,platform=platform.platform(),numpy=np.__version__,
            scipy=scipy.__version__,sklearn=sklearn.__version__,cma=cma.__version__,
            threads=1)
        manifest_path.write_text(json.dumps(manifest,indent=2))
    with threadpool_limits(limits=1):
        for name in cfg['functions']:
            for d in cfg['dimensions']:
                for trial in cfg['trials']:
                    for method in cfg['methods']:
                        stem=f'{name}_d{d}_t{trial}_{method}';path=out/(stem+'.csv')
                        if path.exists() and (out/(stem+'.json')).exists(): continue
                        frame,meta=run_one(name,d,trial,method,cfg)
                        temp=out/(stem+'.tmp');frame.to_csv(temp,index=False);temp.replace(path)
                        (out/(stem+'.json')).write_text(json.dumps(meta,indent=2))
                        print(stem, 'gap=',round(frame.best.iloc[-1],6),'seconds=',round(meta['seconds'],2),flush=True)

if __name__=='__main__': main()
