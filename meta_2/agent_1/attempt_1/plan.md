# Attempt 1 plan — amplitude-decoupled downwind conv-3

## Parent lineage

The meta-1 winner is the 15-node additive downwind/right-flat correction

`1 + (delta flat_right exp(c1*rho)) *_3 exp(c2*rho)`.

Its five fits were eligible and achieved mean training fitness 6.9405128680.
However, IDM reached `c1=c2=-10`, Del Castillo reached `c2=9.5196`, and
triangular reached `c2=8.7040`, showing pressure near the original slope bounds.

## Structural hypothesis

Introduce an independent amplitude:

`g[rho] = 1 + a * ((delta flat_right exp(c1*rho)) *_3 exp(c2*rho))`.

This separates response magnitude from exponential shape. The identity is an
exact, physically feasible default at `(a,c1,c2)=(0,0,0)`. Use expanded slope
bounds `c1,c2 in [-15,15]` and a compact signed amplitude range `a in [-2,2]`.
The model has 17 tree nodes, two more than the parent.

## Evaluation protocol

- Independently fit every key in `automodel.evaluate.BASELINES` using seed 2101.
- Call `I80PredictionProblem.fit(maxiter=2, popsize=3, polish_maxiter=10)`.
- Evaluate `problem.evaluate(..., split="train")` only; never access test.
- Record parameters, constrained objective, E_data, penalized fitness, all rRMSE
  values, runtime, peak process RSS estimate, optimizer status, finiteness,
  physical feasibility, bound adherence, and eligibility.
- A row is eligible only when finite, feasible, in bounds, and below the
  objective-100 constraint plateau.
- Compare with the parent conv-3 fitness only for eligible new rows.
