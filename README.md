# SR-Traffic

<p align="center">
<img src="readme_figure.png" alt="My figure" width="100%">
</p>

This repository contains the code used to produce the results of the paper [SR-Traffic: Discovering Macroscopic Traffic Flow Models with Symbolic Regression](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_105.pdf)

## Installation

The dependencies are collected in `environment.yaml` and can be installed, after cloning the repository, using [`mamba`]("https://github.com/mamba-org/mamba"):

```bash
mamba env create -f environment.yaml
```

Once the environment is installed and activated, install the library using

```bash
pip install -e .
```

## Usage

To reproduce the paper's figures and error tables using the precomputed model
parameters, run

```bash
python src/sr_traffic/fd/results.py --road_name {road_name} --task {task_name}
```

where `{road_name}` is either `US101` or `I80`, and `{task_name}` is either
`prediction` or `reconstruction`. This script does not perform parameter
calibration or symbolic-regression search: it reruns the traffic simulations
with fixed parameters, evaluates the models, and writes the figures under
`results/<road_name>/<task_name>/`. Each run also writes its error table in
LaTeX and Markdown formats as `error_table.tex` and `error_table.md` in the same
directory.

To re-calibrate a given fundamental diagram, run

```bash
python src/sr_traffic/fd/calibration.py --config src/sr_traffic/fd/configs/{fnd_name}.yaml
```

where `{fnd_name}` is either `greenshields`, `triangular`, `weidmann`,
`del_castillo`, or `idm`.

This command calibrates only the predefined baseline diagrams; SR-discovered
models are searched for and fitted separately by the SR-Traffic pipeline below.

Each file in `src/sr_traffic/fd/configs/` defines one calibration problem:

- `road_name` selects the dataset (`I80` or `US101`).
- `task` selects the data split. `prediction` calibrates on earlier time points
  and tests on later ones, while `reconstruction` calibrates on a subset of
  spatial locations and tests on the held-out locations.
- `flux` is the case-sensitive name of the fundamental-diagram function in
  `src/sr_traffic/fd/diagrams.py`.
- `bounds` contains two lists: the lower bounds followed by the upper bounds.
  Their entries follow the order of the selected flux function's parameters
  after density, as listed below.
- `opt.num_ind` is the population size used by the PyGMO simple evolutionary
  algorithm, and `opt.num_gen` is the number of generations. The supplied
  values (`1000` and `100`) are full calibration settings and can be reduced
  for a quick trial.

| Config | Parameter order in `bounds` | Meaning |
| --- | --- | --- |
| `greenshields` | `v_max`, `rho_max` | Free-flow speed and jam density |
| `triangular` | `V_0`, `l_eff`, `T` | Free-flow speed, effective vehicle length, and time headway |
| `weidmann` | `v_max`, `rho_max`, `lambda_w` | Free-flow speed, jam density, and shape parameter |
| `del_castillo` | `C_jam`, `V_max`, `rho_max`, `theta` | Congested-wave speed scale, free-flow speed, jam density, and shape parameter |
| `idm` | `s0`, `T`, `delta`, `v0` | Minimum gap, time headway, acceleration exponent, and desired speed |

Calibration minimizes the average of the normalized squared density and
velocity errors on the training split. The data and parameters are
nondimensionalized by the preprocessing code, so the bounds are expressed in
the model's normalized units.

Finally, to perform a new model search with SR-Traffic, run

```bash
python -m sr_traffic.sr.sr_traffic
```

You can change the parameters of the algorithm by modifying `sr_traffic.yaml`.
The provided defaults use a seeded evolution with one individual and one
generation. This is intended as a quick smoke test of the implementation, not
as a search for a new model. Run artifacts, including `run.log`, are written to
`results/<road_name>/<task_name>/sr/` rather than the repository root.

### Selecting the baseline fundamental diagram for SR

SR-Traffic searches for a multiplicative correction to a predefined baseline
fundamental diagram. If the symbolic-regression individual is denoted by
`g_SR`, the flux used in the traffic simulation is

```text
q_SR(rho) = q_baseline(rho; opt_coeffs) * g_SR(rho).
```

The baseline is selected by the `gp.ansatz` entry in
`src/sr_traffic/sr/sr_traffic.yaml`. The default is

```yaml
ansatz:
  flux: triangular_flux
  v: triangular_v
  opt_coeffs: [0.37013956, 1.48964708, 6.59672108]
```

Thus, the default search modifies the triangular fundamental diagram. The
three coefficients are, in order, `V_0`, `l_eff`, and `T`. The baseline
coefficients are kept fixed during the SR run; the search changes `g_SR` and
fits the constants occurring in that expression.

To select another baseline, change all three fields in `ansatz`: `flux` is the
case-sensitive flux-function name, `v` is its matching velocity function, and
`opt_coeffs` follows the parameter order below.

| Baseline | `flux` | `v` | Order of `opt_coeffs` |
| --- | --- | --- | --- |
| Greenshields | `Greenshields_flux` | `Greenshields_v` | `v_max`, `rho_max` |
| Triangular | `triangular_flux` | `triangular_v` | `V_0`, `l_eff`, `T` |
| Weidmann | `Weidmann_flux` | `Weidmann_v` | `v_max`, `rho_max`, `lambda_w` |
| Del Castillo | `del_castillo_flux` | `del_castillo_v` | `C_jam`, `V_max`, `rho_max`, `theta` |
| IDM | `IDM_flux` | `IDM_v` | `s0`, `T`, `delta`, `v0` |

For example, an IDM baseline can be selected with

```yaml
ansatz:
  flux: IDM_flux
  v: IDM_v
  opt_coeffs: [s0, T, delta, v0]
```

where the placeholders must be replaced by numeric coefficients calibrated for
the selected `road_name` and `task`.

Notice that the `opt_coeffs` are obtained by calibrating the chosen baseline before running
symbolic regression. Run `src/sr_traffic/fd/calibration.py` with the appropriate
fundamental-diagram config, as described above. The calibration minimizes the
training density and velocity error and prints `pop.champion_x`; copy that
printed vector into `gp.ansatz.opt_coeffs`. This copy is currently manual: the
calibration script does not update `sr_traffic.yaml` automatically. Make sure
the calibration config's `road_name` and `task` match the corresponding SR
settings.

The coefficient vectors used for the paper's combinations of road, task, and
baseline are also stored as the `opt_*` variables in
`src/sr_traffic/fd/results.py`. The default triangular vector in
`sr_traffic.yaml`, `[0.37013956, 1.48964708, 6.59672108]`, is the stored
triangular calibration for I80/prediction. The commented IDM example currently
in that YAML, `[0.13046561, 0.74381154, 0.05752636, 0.54561196]`, comes from
US101/reconstruction, so it should not be used unchanged with the default
I80/prediction settings. The stored I80/prediction IDM vector is
`[0.43936351, 0.93094344, 0.16251414, 0.61353022]`.

### SR model score

The data error used to evaluate an SR model is the average of the relative
squared density and velocity errors:

```text
E_rho  = 100 * sum((rho_computed - rho_data)^2) / sum(rho_data^2)
E_v    = 100 * sum((v_computed - v_data)^2) / sum(v_data^2)
E_data = 0.5 * (E_rho + E_v).
```

Lower values are better. The factors of `100` express the two relative squared
errors as percentages. This quantity is a relative squared error, not a mean
squared error: the squared residuals are normalized by the squared norm of the
observations rather than by the number of observations. Although the simulated
flow is also computed, it is not included in `E_data`.

During the symbolic-regression search, the expression-tree length penalty from
`gp.penalty.reg_param` is added to the data error:

```text
E_fitness = E_data + reg_param * number_of_tree_nodes.
```

The default `reg_param` is `0.01`. The search minimizes `E_fitness`, thereby
trading off agreement with the observed density and velocity against symbolic
expression complexity. A candidate receives a data error of `100` if it fails
the velocity feasibility check, which requires velocity to be non-increasing
with density. Invalid expressions and expressions rejected by the tree checks
are penalized similarly.

The final test error printed after the search is `E_data` alone; it does not
include the expression-length penalty.

## Results obtained with Automodel

### How the search was run

Automodel was run on the I80 prediction problem while keeping each calibrated
baseline FD fixed and searching only for a multiplicative DEC correction. The
search used two meta-iterations, three structural workers per iteration, and
two sequential attempts per worker, for 12 candidate expression families in
total. Coefficients were fitted with differential evolution followed by bounded
Nelder–Mead refinement. Expressions were ranked on the first 60% of the time
interval using `E_data + 0.01 * number_of_tree_nodes` and were rejected if the
corrected velocity was not finite and non-increasing with density. The final
40% was evaluated once after the expressions and coefficients had been frozen.

### Selected corrections and scores

An Automodel search over the repository's DEC primitives found a three-cell
right/downwind correction that improves all five calibrated basic diagrams on
the held-out I80 prediction interval. The common multiplicative term is

```text
g[rho] = 1 + (delta flat_downwind exp(c1*rho)) *_3 exp(c2*rho).
```

Greenshields uses a local-exponential variant and IDM uses a positive
exponential envelope. Expressions and fitted coefficients are registered in
`automodel/final_model.py`; reusable implementations live in
`src/sr_traffic/fd/diagrams.py`.

| Baseline | Selected correction parameters | Identity test `E_data` | Corrected test `E_data` |
| --- | --- | ---: | ---: |
| Greenshields | `(0.5, -5.957330756, -10)` | 9.230426 | **6.121604** |
| Weidmann | `(0.2017540235, 8.151159515)` | 7.651073 | **6.706620** |
| Triangular | `(0.1327074798, 8.704037661)` | 7.918859 | **6.816831** |
| IDM | `(0.6140045959, -10, -0.4413702102)` | 6.957441 | **6.828271** |
| Del Castillo | `(0.1300955079, 9.519560936)` | 7.196903 | **6.822161** |

The mean test error decreases from 7.790941 to 6.659097. In particular, the
triangular correction improves the previous SR test score from 8.074263 to
6.816831. The model structures were selected and coefficient-tuned on the first
60% of I80; the final 40% was inspected only after freezing the selections.

> [!IMPORTANT]
> **Interpreting the ranking.** The current paper-compatible rank is based on
> global, uncentered density and velocity rRMSE. A smooth prediction can score
> well by matching average field magnitudes while reproducing congestion waves
> poorly, so the lowest average rank should not by itself be interpreted as the
> best qualitative traffic dynamics. Future evaluations should supplement the
> paper metrics with interior-only rRMSE, density/velocity correlation, spatial-
> and temporal-gradient rRMSE, flow rRMSE, and a structural image metric such as
> SSIM. The generated Automodel table ranks each corrected model against the
> same five calibrated baselines. The separate corrected-model table below
> makes the comparison with the published SR models explicit without mixing SR
> rows into the Automodel output table.

> [!WARNING]
> **Automodel-Greenshields is not physically admissible over the full normalized
> I80 density range.** Its stored `feasible: true` flag only certifies that the
> corrected velocity is finite and non-increasing on uniform-density states. The
> calibrated Greenshields parameter is `rho_max = 0.559951`, while normalized I80
> density extends to 1, so both velocity and flux become negative for
> `rho > rho_max`. The final simulation contains 4 negative interior
> velocity/flow samples out of 13,500. The model is therefore more accurately
> described as *monotonicity-feasible*, not fully admissible. A future search
> should additionally enforce nonnegative velocity and flux, a jam density that
> covers the modeled domain, positivity on nonuniform states, and optionally flux
> concavity; refitting under those constraints would require a fresh external
> validation check.

### Comparison with the published SR models

The following table reports the published SR corrections and frozen Automodel
corrections together using the four paper metrics. Values are uncentered rRMSE;
lower is better, and parenthesized ranks are computed only across these ten
corrected models. This table is kept separate from the generated Automodel error
table, whose comparison set is the five calibrated baselines and five Automodel
variants.

| Model | $E^{\mathrm{tr}}_\rho$ | $E^{\mathrm{tr}}_v$ | $E^{\mathrm{ts}}_\rho$ | $E^{\mathrm{ts}}_v$ | Avg Rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| SR-Greenshields | 0.296 (10) | 0.234 (4) | 0.287 (10) | 0.285 (10) | 8.50 |
| Automodel-Greenshields | 0.252 (7) | 0.233 (3) | 0.239 (3) | **0.244 (1)** | 3.50 |
| SR-IDM | **0.244 (1)** | 0.257 (7) | 0.247 (6) | 0.259 (4) | 4.50 |
| Automodel-IDM | 0.253 (8) | 0.258 (8) | 0.254 (9) | 0.256 (2) | 6.75 |
| SR-Weidmann | 0.249 (3) | **0.226 (1)** | 0.250 (8) | 0.261 (5) | 4.25 |
| Automodel-Weidmann | 0.246 (2) | 0.238 (6) | 0.238 (2) | 0.266 (6) | **4.00** |
| SR-Triangular | 0.251 (6) | 0.272 (9) | 0.248 (7) | 0.258 (3) | 6.25 |
| Automodel-Triangular | 0.253 (9) | 0.277 (10) | 0.241 (5) | 0.267 (7) | 7.75 |
| SR-Del Castillo | 0.251 (5) | 0.230 (2) | 0.240 (4) | 0.268 (8) | 4.75 |
| Automodel-Del Castillo | 0.250 (4) | 0.238 (5) | **0.235 (1)** | 0.273 (9) | 4.75 |

### Stored artifacts and reproduction

The search and reporting artifacts are organized as follows:

- `meta_1/` and `meta_2/` contain every attempted structure, fitted parameters,
  training metrics, and per-attempt evaluation notes.
- `automodel/leaderboard.csv`, `automodel/expressions.md`, and
  `automodel/training_fitness_progress.png` summarize the full search history.
- `automodel/final_model.py` contains the frozen selections, while
  `automodel/final_results.json` and `automodel/final_results.md` contain the
  one-time held-out evaluation.
- Generated paper-style plots and comparison tables are written to
  `results/I80/prediction/automodel/`.

To reproduce the frozen external check and regenerate its JSON/Markdown report,
run

```bash
python -m automodel.finalize
```

To generate paper-style plots and a paired comparison table containing every
baseline FD and its frozen Automodel-corrected variant, run

```bash
python src/sr_traffic/fd/automodel_results.py --road_name I80 --task prediction
```

The command writes ten plots, a 10-row `error_table.tex`/`error_table.md`, and
the raw `metrics.json` values under `results/I80/prediction/automodel/`. The
table jointly ranks the five baselines and five Automodel variants using the
same four metrics as `fd/results.py`: training/test density and velocity rRMSE
plus average rank. Published SR results are intentionally omitted from that
table and reported separately above. Use `--tables-only` to skip plot rendering.
The search covers the five diagrams with calibration configs and stored I80
coefficients; Greenberg and Underwood are not included in this benchmark.

## Citing

```
@article{manti2025,
  title={{SR}-{T}raffic: {D}iscovering {M}acroscopic {T}raffic {F}low {M}odels with {S}ymbolic {R}egression},
  author={Manti, S. and Mohammadian, S. and Treiber, M. and Lucantonio, A.},
  journal={Neural Information Processing Systems, ML4PS Workshop},
  year={2025}
}
```

## Acknowledgements

This work is supported by the European Union (European Research Council (ERC), ALPS, 101039481). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the ERC Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
