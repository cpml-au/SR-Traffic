# Attempt 2 plan — positive upwind-gradient envelope

## Evidence from attempt 1

Attempt 1 found one physically feasible fit out of five. The other four fits sat
on the objective's value-100 infeasibility plateau, all at a small positive inner
slope. The only feasible fit also left the declared `b <= 6` range during the
unconstrained polish step. Thus merely adding amplitude to the published additive
form did not provide a robust feasible search region.

## Structural hypothesis

Use a non-positive inner slope and put the amplitude-scaled nonlocal term inside
a positive exponential envelope:

`g[rho] = exp(a * ((delta flat_left exp(b*rho)) *_1 exp(c*rho)))`.

This is genuinely different from attempt 1's `1 + a*z`: `g` is strictly positive
for every finite `z`, while `a=0` or `b=0` recovers identity. Bounds are
`a in [0, 2.5]`, `b in [-8, 0]`, and `c in [-8, 0]`. The negative inner-slope
half-space is selected specifically to make the velocity-feasible region
reachable under the repository's density-grid check.

## Revised protocol

- Fit each key in `automodel.evaluate.BASELINES` independently with fixed seed
  1102 through `I80PredictionProblem.fit`.
- Use the revised global budget `maxiter=3`, `popsize=4` because attempt 1 took
  284 seconds. In the local process only, cap the method's unconstrained
  Nelder-Mead polish at 20 iterations/evaluations; shared source remains untouched.
- Evaluate only `problem.evaluate(..., split="train")`; never inspect test data.
- Log params, objective, E_data, penalized fitness, all rRMSE values, runtime,
  peak process RSS estimate, convergence, finiteness, bound adherence, and
  velocity feasibility.
- Explicitly discard any fit with objective 100, non-finite values, failed
  feasibility, or parameters outside the declared bounds. Raw E_data from a
  discarded fit is retained only as a diagnostic.
- Complexity is 16 nodes: the published convolution subtree plus amplitude
  multiplication and an outer cochain exponential, without the additive-one node.
