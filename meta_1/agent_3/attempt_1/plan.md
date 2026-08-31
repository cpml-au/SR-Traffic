# Attempt 1 plan: positive local/right-gradient hybrid

## Hypothesis

IDM and Del Castillo have relatively rigid local velocity-density shapes.  A
small exponential tilt in local density can repair that shape, while a signed
right-flat DEC gradient term can react to spatially developing congestion.  By
placing both contributions in one exponential, the multiplicative correction
stays positive for all finite inputs.

## Candidate

```text
g[rho] = exp(c0*rho
             + c1*(delta flat_right_P rho *_1 exp(c2*rho)))
```

This uses only `MF`, `AddC`, `Exp`, `del`, `flat_lin_rightP0`, and `conv_1`, all
represented by `src/sr_traffic/sr/sr_traffic.yaml` and
`src/sr_traffic/sr/primitives.py`.  Its GP-style tree has 15 nodes and three
free dimensionless constants.  Bounds are deliberately moderate to reduce
overflow and optimizer time: `c0 in [-1.5, 0.5]`, `c1 in [-4, 4]`, and
`c2 in [-4, 4]`.

## Protocol

- Dataset/problem: I80 prediction.
- Selection/evaluation split: first 60% training interval only; never query
  the external `test` split.
- Baselines: every key in `automodel.evaluate.BASELINES`, fit independently.
- Optimizer: `I80PredictionProblem.fit`, differential evolution plus its
  built-in Nelder-Mead polish, `seed=31031`, `maxiter=8`, `popsize=5`.
- Record fitted constants, unpenalized `E_data`,
  `E_data + 0.01*tree_nodes`, rho/velocity/flow rRMSE, convergence message,
  physical feasibility, wall time, and peak-process-RSS estimate.

Attempt 2 will be chosen only after inspecting this attempt's per-baseline
training results, especially whether the local tilt or signed gradient is
driven to a bound and whether IDM/Del Castillo improve.
