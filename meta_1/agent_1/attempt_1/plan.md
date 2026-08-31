# Attempt 1 plan — amplitude-scaled published correction

## Hypothesis

The published prediction correction fixes the nonlocal term's outer amplitude at
one. That unnecessarily couples the spatial-response magnitude to the two
exponential slopes. Introduce an explicit dimensionless amplitude `a` while
preserving the published upwind, one-point convolution geometry:

`g[rho] = 1 + a * ((delta flat_left exp(b*rho)) *_1 exp(c*rho))`.

The bounds `a in [0, 2.5]`, `b in [0, 6]`, and `c in [-8, 0]` preserve the
published slope-sign pattern and include the identity limit (`a=0` or `b=0`).
The common harness rejects any fitted parameters for which corrected velocity is
non-finite or increases on its density grid.

## Protocol

- Fit a fresh parameter vector for every key in `automodel.evaluate.BASELINES`.
- Use `I80PredictionProblem.fit(seed=1101, maxiter=8, popsize=5)`.
- Evaluate only `problem.evaluate(..., split="train")`; never inspect test data.
- Record the unpenalized objective and data error, `E_data + 0.01*tree_nodes`,
  all three rRMSE values, optimizer status, feasibility, elapsed time, and peak
  process RSS estimate.
- Complexity is 17 nodes: the published 15-node expression plus scalar amplitude
  multiplication and its parameter leaf.
