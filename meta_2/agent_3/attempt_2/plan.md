# Attempt 2 plan: local exponential plus downwind window-3 response

## Structural response to attempt 1

The strictly positive envelope was eligible on all baselines and improved the
meta-1 common winner on Greenshields by 0.065462 fitness points and IDM by
0.077485.  It did not retain the parent's key Weidmann and triangular gains:
their fitness worsened by 0.706861 and 0.568542, respectively.  Both fitted
inner slopes collapsed close to zero, indicating that the product of an outer
amplitude and inner exponential slope is poorly identified under the short
optimizer budget.

This attempt removes the outer amplitude product and changes the parent's
constant branch into a fitted local exponential:

```text
g[rho] = exp(c0*rho)
         + ((delta flat_right_P exp(c1*rho)) *_3 exp(c2*rho))
```

At `c0=0` this contains the 15-node meta-1 winner exactly.  The local branch is
intended to repair the IDM velocity-density residual without sacrificing the
right/downwind three-point response responsible for Weidmann and triangular.

## Parsimony, primitives, bounds, and seed

Replacing the parent's constant terminal with `exp(MF(rho,c0))` adds three
nodes, giving 18 nodes and three fitted constants.  This is smaller than a
four-parameter outer-envelope/local/conv-3 combination and avoids a separate
amplitude.  It uses only `AddC`, `MF`, `Exp`, `del`, `flat_lin_rightP0`, and
`conv_3`.

- `c0 in [-0.5, 0.5]`
- `c1 in [-10, 10]`
- `c2 in [-10, 10]`
- default `(c0,c1,c2)=(0,0,8)` returns identity exactly because the discrete
  gradient of `exp(0*rho)` vanishes.

The feasible default places the kernel slope near the meta-1 Weidmann and
triangular winners (`8.15` and `8.70`) so the short bounded polish can discover
their small positive `c1`; differential evolution still explores both slope
regimes for IDM.

## Evaluation protocol

- I80 prediction, first 60% training interval only; never access `test`.
- Independently fit all five `BASELINES`.
- Shared bounded `I80PredictionProblem.fit`, seed `2302`, differential
  evolution `maxiter=2`, `popsize=3`, Nelder-Mead `polish_maxiter=10`.
- Record all parameters, `E_data`, 18-node fitness, rho/velocity/flow rRMSE,
  runtime/RSS, convergence, finite/feasible/in-bounds/eligible status, and
  differences from both the meta-1 common winner and meta-2 attempt 1.
