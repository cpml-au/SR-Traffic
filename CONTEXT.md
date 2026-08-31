# SR-Traffic fundamental-diagram improvement — project context

> **Living automodel document.** Update this after every phase and meta-iteration.

## Goal

Find an interpretable multiplicative symbolic correction for each calibrated basic
fundamental diagram used by the repository's I80/prediction benchmark:
Greenshields, Weidmann, triangular, IDM, and Del Castillo. Preserve the first-order
LWR/Godunov formulation and improve the reported SR data score while retaining the
physical velocity constraint `dV/drho <= 0`. Record every evaluated expression and
its coefficients/metrics in durable, machine-readable artifacts.

## Problem definition

- **Inputs:** dimensionless traffic density `rho(x,t)` on the repository's uniform
  one-dimensional I80 mesh; the correction may use local algebraic and DEC spatial
  operators available in `src/sr_traffic/sr/sr_traffic.yaml` and
  `src/sr_traffic/sr/primitives.py`.
- **Outputs:** simulated density, velocity, and flow fields from the LWR continuity
  equation. Density and velocity determine the search score; flow is reported only
  as a diagnostic.
- **Model form:** `q(rho) = q_baseline(rho; theta_fixed) * g[rho; c]`, equivalently
  `V(rho) = V_baseline(rho; theta_fixed) * g[rho; c]`.
- **Fixed components:** I80 preprocessing, the five stored I80/prediction baseline
  coefficient vectors in `src/sr_traffic/fd/results.py`, the Godunov solver, and
  observed initial/boundary densities.
- **Physical constraints:** the corrected velocity must be finite and
  non-increasing over normalized density `rho in [0,1]`. Candidate expressions
  should remain compact and interpretable.

## Data

- **Files:** `src/sr_traffic/data/I80/NGSIM_I80_4pm_{Density,Velocity,Flow}_Data.txt`.
  These contain the 4:00–4:15 PM I80 fields used by `preprocess_data("I80")`.
- **Key variables:** density is the model state/input; measured density and velocity
  are the scored targets; flow is a diagnostic.
- **Cleaning/preprocessing:** `src/sr_traffic/data/data.py::preprocess_data` removes
  the first/last raw spatial rows, nondimensionalizes density by its observed
  maximum and velocity by 100 ft/s, interpolates fields onto DEC cochains, and
  constructs refined-time boundary data.
- **Selection score:** repository-native penalized calibration fitness on the first
  60% of time: `0.5*(E_rho + E_v) + 0.01*tree_nodes`, where each `E` is 100 times
  relative squared error. This complexity-penalized in-sample score avoids using
  the external test interval to choose structures. Unpenalized `E_data` and tree
  size are also logged separately.
- **Splits:** chronological prediction split already implemented by
  `build_dataset`: first 60% (`X_train_val`) for candidate fitting/ranking and final
  40% (`X_test`) for the external check. The nested 36%/24% train/validation arrays
  remain available for diagnostics but are not the primary ranker.
- **External check (Phase 4):** final 40% of the I80 time interval. It must not be
  queried while generating or selecting candidate structures.

## Objective / metrics

- **Coefficient-fitting objective:** minimize unpenalized training
  `E_data = 0.5*(E_rho + E_v)` subject to velocity feasibility, matching
  `src/sr_traffic/sr/sr_traffic.py`.
- **Structure selection:** minimize `E_data + 0.01*tree_nodes` on the first 60%.
- **Evaluation metrics:** training and final-test `E_data`; density and velocity
  rRMSE; flow rRMSE as a diagnostic; number of free constants; expression/tree
  complexity; feasibility/convergence status.
- **Runtime and memory:** wall-clock seconds and peak resident memory are recorded
  for the baseline and every attempt.
- **Reference result:** the current smoke-test artifact for the triangular seed has
  training `E_data=7.6192` (fitness 7.7692 with 15 nodes) and test
  `E_data=8.0742627959`; the paper-result table reports rRMSEs for all five
  baselines and their common published SR structure.

## Parameter optimization routine

The existing SR implementation compiles typed DEAP expressions, then tunes their
scalar constants with PyGMO's simple evolutionary algorithm (`sea`, 10 generations,
population 10, bounds `[-10,10]`) and rejects candidates whose corrected velocity
increases with density. The improvement harness may use deterministic seeds,
multiple starts, and a stronger library optimizer while calling the same JAX/DEC
simulation objective; all optimizer settings and convergence outcomes must be
logged. Baseline coefficients remain the stored paper calibrations so improvements
are attributable to `g`.

## Mock model / baseline

- **Files:** `automodel/model.py` defines parameterized corrections;
  `automodel/evaluate.py` loads I80, builds DEC operators, runs the LWR solver,
  fits constants, checks velocity feasibility, and emits structured JSON.
- **Form:** the end-to-end mock model is the one-node correction `g[rho]=c0`;
  `g=1` provides the unmodified-FD reference. The paper's prediction expression is
  also encoded without changing its structure.
- **Parameters:** `c0`, a dimensionless scalar multiplier of baseline flux and
  velocity. The sub-agent fit found `c0=0.9766975596` for triangular.
- **Baseline metrics:** triangular identity training `E_data=8.0593198750`,
  penalized fitness `8.0693198750`, density rRMSE `0.2698447421`, velocity rRMSE
  `0.2972712779`, flow rRMSE `0.2518347597`. The independently optimized constant
  model reached `E_data=8.0267517009`, fitness `8.0367517009`, density rRMSE
  `0.2698847370`, velocity rRMSE `0.2961372363`, and flow rRMSE `0.2610436691`.
- **Runtime:** 2.4 s for identity and 2.8 s for the small constant optimization;
  peak process RSS was about 675 MB (both include Python/JAX startup and compile).
- **Sanity/environment:** all metrics and predictions were finite and non-degenerate.
  The independent sub-agent initially found `/usr/bin/python` 3.14 without project
  packages; the intended interpreter is
  `/home/alucantonio/.local/share/mamba/envs/sr_traffic/bin/python` (Python 3.12,
  JAX/dctkit/PyGMO/Flex installed). JAX runs on CPU. Shared read/write/execute access
  was verified via `automodel/phase2_subagent.json`.

## Search plan

- **M × S × I:** 2 meta-iterations × 3 parallel sub-agents × 2 sequential
  structural attempts. Each structure is independently coefficient-fitted and
  scored on all five baselines; test data remain inaccessible until Phase 4.
- **Directory layout:** `meta_m/agent_s/attempt_i/{plan.md,model.py,results.json,evaluation.md}`.
  Agents write only inside their assigned `agent_s` directory.
- **Diversification:** meta 1 explores (1) physically constrained variants of the
  published upwind-gradient correction, (2) alternative DEC direction/window
  geometry, and (3) compact local/nonlocal hybrids. Meta 2 will refine distinct
  winners or failure modes found in meta 1.
- **Compute budget:** a simple fit/evaluation costs roughly 2.5–3 s and 675 MB RSS;
  the paper DEC expression costs about 3 s and 820 MB for one supplied-parameter
  evaluation. Three parallel workers are expected to remain near 2.5 GB aggregate
  RSS; optimizer runs use deterministic seeds and modest populations.

## Hypotheses for meta_1

- The published `1 + gradient-exp convolution` structure can improve when given an
  explicit amplitude/positive exponential envelope while retaining feasible
  negative-gradient parameter regions.
- Downwind/right-flat and three-point look-ahead variants may better align the DEC
  nonlocality with traffic propagation than the published one-point upwind term.
- A compact local density modulation combined with a gradient term may repair
  baseline-specific residuals, especially for IDM and Del Castillo, without a large
  tree penalty.

## Meta_1 outcome (2026-08-31)

All six attempt directories contain plan/model/results/evaluation artifacts and
explicitly record that the test split was not accessed. The root recheck in
`meta_1/recheck/` reproduced the winning common structure on every baseline.

| Source | Correction family | Mean training fitness | Eligibility |
|---|---|---:|---|
| baseline | identity (`g=1`) | 7.500854 | all feasible |
| agent_1/attempt_1 | additive amplitude + upwind conv-1 | 7.325706 raw | rejected: 4/5 infeasible; survivor out of bounds |
| agent_1/attempt_2 | exponential upwind conv-1 envelope | 7.435213 | all feasible/in bounds |
| agent_2/attempt_1 | downwind conv-1 | 25.933160 constraint-aware | rejected: infeasible/out-of-bounds rows |
| agent_2/attempt_2 | downwind conv-3 | **6.940513** | all feasible/in bounds; root-reproduced |
| agent_3/attempt_1 | local + signed-gradient exponential | 7.246717 | feasible, but two legacy-polish rows out of bounds |
| agent_3/attempt_2 | local + squared-gradient exponential | 7.198418 | all feasible/in bounds |

The winner is
`g = 1 + (delta flat_downwind exp(c1*rho)) *_3 exp(c2*rho)` (15 nodes).
Root-rechecked per-baseline fitnesses are Greenshields 7.004365, Weidmann
6.364392, triangular 7.648460, IDM 7.220317, and Del Castillo 6.465031. The
corresponding identity fitnesses are 8.220503, 6.929012, 8.069320, 7.402856,
and 6.882578, so the same structure improves all five under the selection score.

**Key findings:**

- Three-point downwind look-ahead is decisively better than one-point geometry;
  its mean fitness improves over identity by 0.560341 and over the triangular
  README smoke fitness 7.7692 by 0.120740 at equal tree size.
- Positive exponential envelopes reliably avoid feasibility plateaus, but the
  upwind conv-1 version often selects zero amplitude and pays unnecessary nodes.
- The compact squared-gradient hybrid is useful for IDM/Del Castillo, but does
  not match conv-3 on Weidmann, triangular, or Del Castillo.
- Hard feasibility creates flat objective-100 regions. Bounded polish and a
  feasible identity default are mandatory; old unbounded-polish attempts remain
  logged but are ineligible.
- The original IDM Newton inversion and density/spacing feasibility mapping were
  inconsistent. `diagrams.IDM_v` plus fixed bisection now gives finite,
  non-increasing equilibrium speed on `rho in [0,1]`.

**Hypotheses for meta_2:**

- An explicit amplitude can decouple conv-3 response magnitude from the inner and
  kernel exponential slopes, several of which approached their bounds.
- Replacing the inner exponential by a linear density gradient may retain the
  useful three-point downwind geometry with fewer nodes and better conditioning.
- A positive downwind conv-3 envelope or a local/conv-3 hybrid may combine the
  winner's Weidmann/triangular gains with the hybrid's IDM behavior.

## Meta_2 outcome (2026-08-31)

All six attempts are complete, train-only, finite, feasible, and in bounds. The
full machine-readable history is `automodel/leaderboard.csv`; unique structures
are listed in `automodel/expressions.md`, and
`automodel/training_fitness_progress.png` visualizes cumulative best fitness.

| Source | Model family | Mean training fitness | Main result |
|---|---|---:|---|
| agent_1/attempt_1 | amplitude + nonlinear downwind conv-3 | 7.295421 | tiny Greenshields/IDM gains only |
| agent_1/attempt_2 | linear downwind conv-3 | 7.277023 | simplification loses nonlinear gain |
| agent_2/attempt_1 | linear downwind conv-3 (independent seed) | 7.351449 | confirms simplification loss |
| agent_2/attempt_2 | linear upwind conv-3 | 7.527641 | direction reversal is worse |
| agent_3/attempt_1 | positive nonlinear downwind conv-3 envelope | 7.242966 | best IDM specialization |
| agent_3/attempt_2 | local exponential + nonlinear downwind conv-3 | 7.192254 | best Greenshields specialization |

**Key findings:** the nonlinear inner exponential and right/downwind three-point
geometry are both essential. Extra amplitude is poorly identified and added
complexity does not help globally. Baseline-specific positive/local envelopes do
improve IDM and Greenshields enough to overcome their node penalties. Root
rechecks in `meta_2/recheck/` exactly reproduced both specialized scores.

## Final model pointer (end of Phase 3)

- **Best model file:** `automodel/final_model.py` maps each baseline to its selected
  correction, parameters, complexity, and source attempt.
- **Training selection fitness:** Greenshields 6.419024; Weidmann 6.364392;
  triangular 7.648460; IDM 7.142832; Del Castillo 6.465031. All are below their
  identity values (8.220503, 6.929012, 8.069320, 7.402856, 6.882578).
- **Selected structures:** local exponential + downwind conv-3 for Greenshields;
  positive exponential downwind conv-3 for IDM; additive downwind conv-3 for
  Weidmann, triangular, and Del Castillo.
- **Why chosen:** lowest eligible complexity-penalized training score per baseline
  after two structural meta-iterations, with independent exact rechecks and hard
  velocity-feasibility enforcement.
- **Stopping rationale:** the goal of a feasible improvement for all five basic
  diagrams is met; meta 2 establishes that simpler/upwind alternatives lose the
  core gain. Proceed to the untouched final 40% I80 prediction interval.

## Phase 4 — Finalize (2026-08-31)

**Decision: accept the frozen Phase-3 selections.** The final 40% of the I80
prediction interval was evaluated once after expressions and parameters were
fixed. Every selected correction is feasible and improves the corresponding
uncorrected diagram on test `E_data`. The selected model registry is
`automodel/final_model.py`; run the reproducible check with
`python -m automodel.finalize`. Full metrics are in
`automodel/final_results.{json,md}`.

| Baseline | Train `E_data` | Train fitness | Identity test `E_data` | Selected test `E_data` | Test delta |
|---|---:|---:|---:|---:|---:|
| Greenshields | 6.239024 | 6.419024 | 9.230426 | 6.121604 | -3.108822 |
| Weidmann | 6.214392 | 6.364392 | 7.651073 | 6.706620 | -0.944453 |
| Triangular | 7.498460 | 7.648460 | 7.918859 | 6.816831 | -1.102028 |
| IDM | 6.982832 | 7.142832 | 6.957441 | 6.828271 | -0.129170 |
| Del Castillo | 6.315031 | 6.465031 | 7.196903 | 6.822161 | -0.374742 |

The mean unpenalized training error is 6.649948 and mean selected test error is
6.659097, a small +0.009149 validation-to-test difference. Mean identity test
error is 7.790941, so the frozen corrections improve it by 1.131843 (14.53%).
The triangular model's test score is 6.816831, improving the README SR reference
8.074263 by 1.257432 (15.57%). The complete final check took 14.62 s on CPU and
reached 1589.5 MB peak resident memory. Per-model runtime and density, velocity,
and flow rRMSEs are stored in `automodel/final_results.json`.

**Acceptance criteria:** 5/5 test improvements over the uncorrected diagrams;
all five velocity-feasibility checks pass; all fields and metrics are finite;
and the triangular test score improves the existing README SR result. The final
check therefore agrees with the training selection rather than showing a material
generalization gap.

**Limitations:** this is one chronological holdout from one road/dataset; the
expressions have not been selected or independently checked on US101 or the
reconstruction task. Greenberg and Underwood are not included because the repo
has neither calibration configs nor I80/prediction reference coefficients for
them; the benchmark's five calibrated basic diagrams are the declared scope.
Flow error was diagnostic rather than optimized and can increase even when the
scored density/velocity error improves. Some fitted values reach search bounds
(Greenshields and IDM), but widening or retuning after viewing this test split
would require a fresh external check. The corrections are DEC spatial operators,
not pointwise equilibrium curves, and require the right/downwind flat and the
same mesh/boundary conventions.

### Paper-style reporting handoff

`src/sr_traffic/fd/automodel_results.py` simulates all five calibrated baselines,
their five published SR corrections, and their five frozen Automodel variants
through the same full-horizon Godunov path used by `fd/results.py`. The 15-row
LaTeX/Markdown table ranks all three families jointly using the paper's four
rRMSE metrics. Paired fundamental-diagram, spatiotemporal, and
predicted-versus-actual plot suites remain focused on baseline/Automodel pairs.
The authoritative installed registry is
`src/sr_traffic/fd/automodel_registry.py`; the top-level search registry imports
its expressions and coefficients to avoid drift. Default outputs are isolated
in `results/I80/prediction/automodel/`.
