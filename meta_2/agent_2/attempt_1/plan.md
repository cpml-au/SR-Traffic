# Meta 2 attempt 1 — linear downwind gradient with a three-point kernel

## Structural hypothesis

The meta-1 winner used

`1 + conv_3(delta(flat_right(exp(c1*rho))), exp(c2*rho))`.

Its best coefficients split into sign regimes, and several inner slopes were
small.  Replace the nonlinear inner exponential by the linear DEC density
gradient and expose its scale as a separate amplitude:

`g[rho] = 1 + a * conv_3(delta(flat_right(rho)), exp(b*rho))`.

This preserves the winning right/downwind three-point geometry, decouples
gradient amplitude from density-dependent amplification, and reduces the
nominal tree size from 15 to 14 nodes.

## Fit/evaluation protocol

- `CorrectionSpec` parameters: `(a,b)`, defaults `(0,0)`, bounds
  `[-10,10] x [-10,10]`, 14 nodes.
- The zero-amplitude default is exactly `g=1` and must pass feasibility for all
  five baselines before fitting.
- I80/prediction training interval only; no test evaluation.
- Fixed seed 2201; differential evolution `maxiter=2`, `popsize=3`; bounded
  Nelder-Mead `polish_maxiter=10` through `I80PredictionProblem.fit`.
- Log unpenalized objective, 14-node penalized fitness, all three rRMSEs,
  runtime/RSS, convergence, physical feasibility, and bound status.
