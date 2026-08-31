# Attempt 2 plan: positive local/squared-gradient hybrid

## Structural response to attempt 1

Attempt 1 reached mean training `E_data=7.096717` and mean penalized fitness
`7.246717`.  It was highly effective for Greenshields, but on Weidmann,
triangular, IDM, and Del Castillo the signed-gradient amplitude was small
(`abs(c1) <= 0.062`).  The kernel exponent was consequently weakly identified;
the older unbounded Nelder-Mead polish even moved the Weidmann and Del Castillo
kernel parameters just outside their declared bounds.  This attempt removes
that redundant kernel and changes the spatial response from signed gradient to
gradient magnitude squared.

## Candidate

```text
g[rho] = exp(c0*rho + c1*square(delta flat_right_P rho))
```

The squared response is insensitive to front direction and can alter the flux
at both compressive and dispersive sharp fronts.  The outer exponential keeps
the multiplier positive.  The structure uses only `MF`, `AddC`, `Exp`,
`Square`, `del`, and `flat_lin_rightP0`, all enabled SR/DEC primitives.  It has
11 GP-style nodes and only two constants.  The identity point `(0, 0)` is used
as a guaranteed finite, velocity-feasible optimizer seed.  Bounds are
`c0 in [-0.5, 0.5]`, `c1 in [-4, 4]`.

## Protocol and reduced compute budget

- Dataset/problem: I80 prediction; evaluate the first 60% training interval
  only and never query `split='test'`.
- Independently fit every `automodel.evaluate.BASELINES` key.
- Use the shared `I80PredictionProblem.fit` in a fresh interpreter after the
  IDM/bounded-polish fixes.
- Fixed seed `31032`, differential evolution `maxiter=3`, `popsize=4`, and
  bounded Nelder-Mead `polish_maxiter=20`.

This is intentionally lower than attempt 1's `maxiter=8`, `popsize=5`: measured
full-horizon nonlocal fits were much slower than the smoke estimate, and the
phase lead set the reduced global attempt-2 budget.  All constants, metrics,
fitness, convergence/feasibility, runtime, and peak RSS estimate will still be
recorded for every baseline.
