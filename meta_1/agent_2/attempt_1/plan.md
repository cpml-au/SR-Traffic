# Attempt 1 plan — downwind one-point geometry

## Structural hypothesis

Mirror the published upwind/left flat while leaving every other part of its
ansatz unchanged:

`g[rho] = 1 + conv_1(delta(flat_right(exp(c1*rho))), exp(c2*rho))`.

The right endpoint interpolation samples the density on the downwind side of
each oriented primal edge.  This tests whether the learned nonlocal correction
is sensitive to spatial direction rather than merely to the presence of a
discrete derivative.  Keeping the one-point convolution makes the comparison
to the published 15-node term direct and interpretable.

## Fit/evaluation protocol

- Dataset/task: I80 prediction.
- Baselines: Greenshields, Weidmann, triangular, IDM, Del Castillo.
- Baseline FD coefficients: fixed values in `automodel.evaluate.BASELINES`.
- Parameter bounds: `c1,c2 in [-10,10]`.
- Independent differential-evolution fits, then the harness's Nelder-Mead
  polish, for every baseline.
- Fixed seed: 1201; `maxiter=8`; `popsize=5`.
- Selection score: training `E_data + 0.01 * 15`.
- Only the first 60% training interval is evaluated.  The test split is not
  accessed.
