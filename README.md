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

## Citing

```
@article{mantisr,
  title={{SR}-{T}raffic: {D}iscovering {M}acroscopic {T}raffic {F}low {M}odels with {S}ymbolic {R}egression},
  author={Manti, S. and Mohammadian, S. and Treiber, M. and Lucantonio, A.},
  journal={Neural Information Processing Systems, ML4PS Workshop},
  year={2025}
}
```
