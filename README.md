# SR-Traffic

This repository contains the code used to produce the results of the paper [SR-Traffic: Discovering Macroscopic Traffic Flow Models with Symbolic Regression](link)

## Installation

The dependencies are collected in `environment.yaml` and can be installed, after cloning the repository, using [`mamba`]("https://github.com/mamba-org/mamba"):

```bash
$ mamba env create -f environment.yaml
```

Once the environment is installed and activated, install the library using

```bash
$ pip install -e .
```

## Usage

To reproduce the results of the paper just run

```bash
$ python src/sr_traffic/fund_diagrams/fund_diagrams_results.py
```

Make sure to update the task name in `fund_diagrams_results.py` so it matches the specific task you want to reproduce.

To re-calibrate a given fundamental diagram, run

```bash
$ python src/sr_traffic/fund_diagrams/fund_diagrams_calibration.py
```

By modifying just a few lines in `fund_diagrams_calibration.py`, you can easily switch the fundamental diagram, select a different task, and adjust the optimizer type or its parameters.

Finally, to perform a run of SR-Traffic, run

```bash
$ python src/sr_traffic/learning/stgp_traffic.py
```

You can change the parameters of the algorithm modifying `stgp_traffic.yaml`.

## Citing
