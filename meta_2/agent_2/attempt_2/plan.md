# Meta 2 attempt 2 — linear upwind gradient with a three-point kernel

## Structural response

The attempt-1 downwind linearization was feasible but worse than the nonlinear
meta-1 parent for every FD.  Its near-zero amplitude/kernel fits for three
baselines suggest that the direct gradient may have the wrong endpoint
orientation.  Mirror only the flat direction:

`g[rho] = 1 + a * conv_3(delta(flat_left(rho)), exp(b*rho))`.

This is a genuine DEC geometry comparison, not a coefficient restart.  It
keeps the same three-point look-ahead, two-parameter conditioning, and 14-node
complexity as attempt 1, isolating left/upwind versus right/downwind sampling.

## Fit/evaluation protocol

- Parameters `(a,b)`, exact-identity default `(0,0)`, bounds
  `[-10,10] x [-10,10]`, 14 nodes.
- Verify default feasibility independently for all five fixed baselines.
- I80/prediction training only; never evaluate the external test split.
- Fixed seed 2202; differential evolution `maxiter=2`, `popsize=3`; bounded
  Nelder-Mead `polish_maxiter=10`.
- Record objective, complexity-penalized fitness, rho/v/flow rRMSE, runtime,
  peak RSS, convergence, physical feasibility, and bound/acceptance status.
