# Research log

## 2026-11-12 — initial implementation

- Main design: three methods, three custom function families, dimensions 2/5/10, maximum 200 evaluations, ten paired runs.
- All three methods receive the same 10 initial evaluations. CMA uses best point as initial mean; no full covariance estimate is inferred from initial points.
- Pilot uses separate instance 90; main uses 0-9. Pilot timing supports keeping the original 270-run plan.
- Initial pilot produced numerous host BLAS prediction warnings. Replaced posterior matrix-vector contraction with explicit einsum and triangular solve. A numerical equivalence test against sklearn passed. Verified pilot is the relevant timing evidence; initial pilot is excluded from report.
- Four checks passed: known objective optimum, EI against Monte Carlo, exact budget/common initialization, posterior equivalence.
- GP uses an isotropic length-scale, not ARD. This limits claims about learning rotated geometry. Ellipsoid is rotated; Sphere is rotation invariant; Rastrigin remains axis-aligned after translation.
- Main config saved before main results. Fixed initial kernel values were not tuned against main outcomes. Kernel optimization only learns from observations within each run.
- No parallel experiments used for main timings. Pilot/tests briefly overlapped in the first diagnostic pass, which is excluded.

## Exploratory follow-up selection

After the Sphere and Ellipsoid main runs and initial Rastrigin runs, selected ten-dimensional Ellipsoid and Rastrigin as contrasting difficult landscapes. This selection is exploratory and occurred before all Rastrigin main runs had completed. New paired instances/seeds 100-109 are used, with the same 200-evaluation cap and initialization. This examines sensitivity to kernel fitting, not a causal decomposition of dimension effects.

A representative Sphere GP refit warning was an upper length-scale bound warning (bound=2); posterior values remained finite. Per-run fit-warning counts are saved. No silent exclusion of warning-producing runs is used.

## 2026-11-12 — exploratory budget-extension protocol (recorded before new runs)

**Question.** Is CMA-ES's disadvantage at 200 evaluations partly a short-budget effect?

This analysis was selected after inspecting the completed main results, including BO's advantage at d=10 and 200 evaluations. It is therefore exploratory. It extends the same Sphere and Ellipsoid instances and trial IDs rather than providing independent confirmation. No statistical-significance claim will be made.

**Design.** Compare the unchanged `bo` and `cma` implementations on Sphere and Ellipsoid at d=10, trials 0-9, with seed base 20261112, ten shared initial evaluations, BO refitting every five proposals, three acquisition starts and a 128-point candidate pool. Both methods and both functions must receive the same final evaluation cap. Runs are fresh deterministic reruns from their original seeds under a new label; they are not continuations because optimizer state is not restored. Full histories and metadata will be saved. The first 200 inputs and values must match the saved main trajectories within numerical tolerance before extended outcomes are interpreted.

**Runtime decision, made without target-trial outcomes.** First run both methods on separate Sphere d=10 trial 91 to 500 evaluations. Use timing only. Estimate a 1,000-evaluation BO runtime by fitting a power relationship to cumulative elapsed time at 200, 300, 400 and 500 evaluations, then add a 25% safety margin for 20 BO runs; CMA-ES cost is included directly but expected to be small. Use a 1,000 cap only if the projected full design plus pilot fits within 30 minutes. Otherwise choose a common cap of 500 before examining comparative outcomes. Retain all ten target trials and both functions. The protocol required equal budgets and all ten trials at either cap.

**Competing predictions.**

- Short-budget prediction: with more evaluations, CMA-ES reduces its disadvantage—its matched gap relative to BO shrinks from evaluation 200, and it wins more matched trials at later checkpoints, potentially overtaking BO on one or both functions.
- Persistent-advantage prediction: BO remains ahead at the extended cap, with no consistent reduction in the matched relative advantage across trials.
- A mixed result is possible and must be reported: Sphere and rotated Ellipsoid may show different budget sensitivity.

**Analysis definitions.** Summarize evaluations 200 and the selected later checkpoints using within-function medians and IQRs. For each matched trial define the raw difference `D = gap_CMA - gap_BO` (positive favors BO) and the log ratio `L = log10((gap_CMA + 1e-12)/(gap_BO + 1e-12))`. CMA “closes the gap” only descriptively when `L` declines from 200 to the cap; raw differences will also be reported. A tie is declared when `abs(gap_CMA-gap_BO) <= 1e-8 * max(1, abs(gap_CMA), abs(gap_BO))`. Checkpoints are repeated observations from the same trajectories, not independent samples. Results will be compared within each function; raw gaps will not be pooled across functions. This experiment tests budget sensitivity. It cannot prove that covariance adaptation causes any observed change.

**Cap decision from timing only.** The separate trial-91 Sphere pilot took 26.49 seconds for BO and 0.027 seconds for CMA-ES at 500 evaluations. BO cumulative elapsed times at evaluations 200/300/400/500 were 7.89/13.43/20.07/26.48 seconds. A log-log power fit projected 67.0 seconds per 1,000-evaluation BO run. Twenty BO runs with the prespecified 25% margin project to 27.9 minutes; adding the approximately 0.44-minute pilot and CMA cost remains below 30 minutes. Therefore the common target cap is fixed at **1,000 evaluations** before inspecting target-trial outcomes. Planned checkpoints are 200, 500 and 1,000 evaluations.

**Runtime deviation recorded before outcome analysis.** The completed extension plus pilot took 32.22 minutes by summed per-run metadata, exceeding the 30-minute operational budget by 2.22 minutes. The Sphere-only pilot underestimated function-dependent BO cost: median 1,000-evaluation BO time was 51.94 seconds on Sphere but 139.39 seconds on Ellipsoid. The cap rule should have piloted both functions or reserved a larger margin. The completed 1,000-evaluation histories are retained and the overrun is disclosed; the design was not changed after outcomes. Both methods still received equal budgets in every comparison.

### Budget-extension validation and results

- All 40 target runs reached 1,000 evaluations. All 40,000 objective values were recomputed from saved inputs; maximum CSV round-trip discrepancy was `1.4551915228366852e-11`. Source and configuration hashes match.
- The first 200 inputs, values and best-so-far traces of all 40 longer reruns match their original main trajectories exactly. These are fresh reruns, not restored optimizer continuations.
- Sphere median gaps for BO/CMA were 0.000361/4.938 at 200, 0.000125/0.0469 at 500 and 0.000125/0.00000888 at 1,000. BO/CMA/tie wins were 10/0/0, 10/0/0 and 0/10/0.
- Ellipsoid median gaps for BO/CMA were 115.7/576.4 at 200, 36.8/101.3 at 500 and 13.2/4.93 at 1,000. BO/CMA/tie wins were 10/0/0, 9/1/0 and 4/6/0.
- Median CMA becomes and remains no worse at evaluation 875 on Sphere and 803 on Ellipsoid. The matched median log10(CMA/BO) ratio falls from 4.10 to -1.15 on Sphere and from 0.73 to -0.55 on Ellipsoid between 200 and 1,000.
- Supporting outcome: CMA closes and reverses the median gap on both functions by 1,000. Contradictory/qualifying outcomes: BO still wins every Sphere and nine Ellipsoid trials at 500; BO remains better in four Ellipsoid trials at 1,000; its relative advantage increases in two Ellipsoid trials.
- Under the predeclared failure rule (largest final worse/better ratio across 20 matched function-trial pairs), Ellipsoid trial 9 is illustrative: BO/CMA gaps change from 66.1/534.6 at 200 to 21.7/0.158 at 1,000. This selected case is not representative.

### Deeper analysis of saved experiments

- The original d=2 Ellipsoid median ordering reverses only at the end: BO/CMA are 0.0236/5.69 at 50 and 0.000949/0.417 at 100; CMA becomes sustainably no worse at evaluation 197 and ends 0.000237/0.000259 (CMA/BO). This is an early/late crossover, not a stable broad advantage.
- Fixed-kernel BO saves fitting time but spends more time in acquisition. Learned/fixed median fit and acquisition seconds are 0.224/0.127 and 6.815/8.160 on Ellipsoid, and 0.268/0.135 and 6.829/7.552 on Rastrigin. Different trajectories or local-optimizer work are plausible explanations, but optimizer iteration counts were not logged.
- Observed evidence: rankings reverse late with a longer evaluation horizon. Plausible explanation: CMA may need more ranked generations while BO exploits early data quickly and can plateau. Unresolved: the experiment does not isolate covariance adaptation, kernel limitations, acquisition-search behavior or warning-bound effects.
