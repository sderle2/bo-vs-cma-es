# Bayesian Optimization versus CMA-ES under Limited Evaluation Budgets

DSA4212 research project by Esther Lee. The [report](report/final%20DSA4212%20report.pdf) examines how evaluation budget, dimension and objective geometry affect Gaussian-process Bayesian optimization (BO) and CMA-ES, and whether a GP length-scale bound changes their relative performance. Uniform random search provides a baseline.

## Methods and experimental design

The main study compares the three methods on custom shifted Sphere, rotated Ellipsoid (condition number 1,000) and shifted Rastrigin objectives in 2, 5 and 10 dimensions. Each setting uses ten matched trials and 200 objective evaluations per method. The domain is `[0,1]^d` and each known minimum is zero. All methods spend ten evaluations on the same uniform initial points. BO conditions on all ten observations, and CMA-ES starts at the best initial point with step size 0.25. Each objective call counts toward the budget, including calls in a partial CMA-ES generation. Optimizers receive only evaluated inputs and objective values.

BO uses an isotropic Matérn 5/2 GP and Expected Improvement (EI). It refits hyperparameters every five proposals and updates the posterior at each proposal. EI search combines a 128-point candidate pool with three bounded L-BFGS-B starts. CMA-ES uses pycma with box constraints and the default population size. See [Methods](report/report.md#2-methods) and the configurations for model settings and evaluation accounting.

The follow-ups compare learned and fixed GP kernels, extend the matched 10D Sphere and Ellipsoid runs to 1,000 evaluations, and repeat BO with length-scale upper bound 4 in place of 2. The budget and bound comparisons reuse instances selected after the main study. Performance is measured by best-so-far optimality gap and sequential wall-clock time, with gaps summarized within each objective.

## Findings

- At 200 evaluations, bound-2 BO has the smallest median gap among the compared methods on all three 10D objectives.
- At 1,000 evaluations, CMA-ES leads bound-2 BO by median gap on both extended objectives. With bound 4, BO leads CMA-ES on Ellipsoid by median gap and wins 6/10 matched trials. CMA-ES wins all ten Sphere trials against either BO setting.
- On Ellipsoid, the median paired gap difference (bound-4 BO minus CMA-ES) is −2.53, with quartiles [−14.6, 2.73]. Rankings depend on the budget and GP setting in these instances. BO has higher wall-clock overhead on these inexpensive objectives.

## Repository structure

| Path | Contents |
| --- | --- |
| `code/`, `configs/`, `tests/` | Optimizers, experiment settings, analysis, report builders and tests |
| `results/main/`, `results/followup/` | Main comparisons and learned-versus-fixed kernel runs |
| `results/budget_pilot/`, `results/budget_extension_1000/` | Timing pilot and matched budget extension |
| `results/lengthscale_sensitivity/full/` | Bound-4 trajectories, fitted length-scales, runtime metadata and `comparison.csv` |
| `results/summaries/` | Analysis tables and validation results |
| `report/`, `research_log.md` | Report source and artifacts, figures, bibliography and experimental decisions |

## Reproducibility

From the repository root on macOS or Linux, use Python 3.10 and the pinned dependencies in [requirements.txt](requirements.txt):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

**Analyze the saved runs.** The objectives are generated from seeded configurations and require no external dataset. These commands regenerate analysis outputs from the supplied trajectories:

```bash
python code/analyze.py
python code/audit_results.py
python code/analyze_budget_extension.py
python code/lengthscale_sensitivity.py analyze
python code/analyze_saved_statistics.py
```

**Run the experiments.** Use a separate working copy for fresh runs. The benchmark resumes compatible runs when their outputs are present. Move the corresponding output folders aside in that copy to start fresh.

```bash
python code/benchmark.py configs/pilot_verified.json
python code/benchmark.py configs/main.json
python code/benchmark.py configs/followup.json
python code/benchmark.py configs/budget_pilot.json
python code/benchmark.py configs/budget_extension_1000.json
python code/lengthscale_sensitivity.py run
```

The sensitivity runner uses the saved budget-extension trajectories and checks source and configuration hashes and core dependency versions. It uses a POSIX file lock. Runs are sequential with one BLAS thread. Manifests record the platform, package versions and hashes. Floating-point libraries and local optimization can affect exact trajectories across machines. The extension audit checks that the first 200 evaluations match the main runs.

**Report files.** The [authoritative Word report](report/final%20DSA4212%20report.docx), its [PDF export](report/final%20DSA4212%20report.pdf), and the [Markdown source](report/report.md) are in `report/` with the bibliography and figures. The analysis commands reproduce summaries and figures from saved runs. The saved-statistics script prints the paired gap differences and diagnostic counts.

## Provenance

The BO loop and EI implementation follow Alexandre Thiéry’s DSA4212 GP and BO teaching notebooks, cited in the report. GP fitting uses scikit-learn, acquisition optimization uses SciPy, and CMA-ES uses pycma. The benchmark adds matched multidimensional objectives, evaluation accounting and validation. AI assistance was used for code, experiment execution, analysis and report drafting.
