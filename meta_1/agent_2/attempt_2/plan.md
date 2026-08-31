# Attempt 2 plan — downwind three-point geometry

## Response to attempt 1

Attempt 1's right-flat/one-point term passed the physical check for four FDs
and gave low training errors for Greenshields and Del Castillo, but Weidmann was
uniformly rejected and several fitted gradient amplitudes collapsed near zero.
This attempt preserves the promising direction while widening the convolution
to the repository's other enabled nonlocal primitive:

`g[rho] = 1 + conv_3(delta(flat_right(exp(c1*rho))), exp(c2*rho))`.

The three-point window tests whether a compact downstream neighborhood smooths
the sharp one-point response.  The symbolic tree remains the same 15 nodes;
only the geometry encoded by the convolution primitive changes.

## Fit/evaluation protocol

- Dataset/task: I80 prediction; all five fixed baseline diagrams.
- Bounds: `c1,c2 in [-10,10]`, enforced during both optimizer stages.
- Feasible optimizer seed/default: `(c1,c2)=(0,0)`; fixed random seed: 1202.
- Revised compute budget requested after attempt 1 timing:
  differential evolution `maxiter=3`, `popsize=4`, followed by bounded
  bounded Nelder-Mead capped at 20 iterations through
  `I80PredictionProblem.fit(..., polish_maxiter=20)`.
- Only training is evaluated.  No test-split access or `--include-test`.
- Fitness: constraint-aware `E_data + 0.01 * 15` (an infeasible candidate has
  objective 100 before the tree penalty).
