# Attempt 1 plan: exponential downwind window-3 envelope

## Parents and hypothesis

The common meta-1 winner is the 15-node right/downwind window-3 response

```text
1 + (delta flat_right_P exp(b*rho)) *_3 exp(c*rho).
```

It obtained mean training fitness 6.940513 and particularly strong Weidmann
(6.364392), triangular (7.648460), and Del Castillo (6.465031) scores.  Its
additive correction is not structurally positive, however.  This attempt wraps
the same response in an exponential and adds one fitted amplitude:

```text
g[rho] = exp(a * ((delta flat_right_P exp(b*rho)) *_3 exp(c*rho)))
```

The result is strictly positive for every finite response.  It uses only `MF`,
`Exp`, `del`, `flat_lin_rightP0`, and `conv_3`, which are enabled in the SR
configuration/primitives.  Following the corresponding meta-1 positive
envelope's counting convention, it has 16 tree nodes and three constants.

## Bounds and feasible seed

- `a in [0, 2.5]`
- `b in [-10, 10]`
- `c in [-10, 10]`
- default `(a,b,c)=(0,0,0)`, which returns identity exactly.

The nonnegative amplitude removes a redundant overall sign because `b` already
controls the gradient response direction.  The runner explicitly checks the
identity default for every baseline before optimization.

## Evaluation protocol

- I80 prediction, first 60% training interval only; never access `test`.
- Fit each of the five `BASELINES` independently.
- `I80PredictionProblem.fit`, fixed seed `2301`, differential evolution
  `maxiter=2`, `popsize=3`, bounded Nelder-Mead `polish_maxiter=10`.
- Record all parameters, `E_data`, 16-node fitness, rho/velocity/flow rRMSE,
  runtime, peak RSS estimate, convergence, finiteness, physical feasibility,
  declared-bound membership, eligibility, and per-baseline difference from the
  meta-1 common winner.

Attempt 2 will be selected only after inspecting whether this envelope retains
the parent winner's Weidmann/triangular advantage and whether IDM remains the
weak baseline.
