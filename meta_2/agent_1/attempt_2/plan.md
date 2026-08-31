# Attempt 2 plan — amplitude-scaled linear-gradient conv-3

## Evidence-driven structural change

Attempt 1 improved only two parent rows and exposed confounding between amplitude
`a` and inner slope `c1`: three fits set `c1` close to zero while amplitude was
otherwise arbitrary or boundary-limited. Near zero,
`delta flat_right exp(c1*rho)` is approximately `c1 * delta flat_right rho`, so
only the product `a*c1` is identifiable.

Remove the inner exponential and its slope entirely:

`g[rho] = 1 + a * ((delta flat_right rho) *_3 exp(c*rho))`.

The amplitude now directly controls gradient response. This retains the winning
downwind three-point geometry, reduces the model from three coefficients to two,
and reduces relative complexity from 17 to 14 nodes. The exact identity default
is `(a,c)=(0,0)`. Bounds `a in [-5,5]` and `c in [-15,15]` cover the effective
parent gradient scales and the kernel-slope bound pressure seen in attempt 1.

## Evaluation protocol

- Fit every baseline independently with fixed seed 2102 using
  `I80PredictionProblem.fit(maxiter=2, popsize=3, polish_maxiter=10)`.
- Evaluate the training selection split only; never access test.
- Log the complete metric, resource, optimizer, feasibility, bound, and
  eligibility fields used in attempt 1.
- Compare against meta-1's conv-3 parent per baseline, and count/report a gain
  only when the new row is eligible and has lower penalized fitness.
