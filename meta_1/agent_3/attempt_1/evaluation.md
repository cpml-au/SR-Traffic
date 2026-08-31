# Attempt 1 evaluation

## Outcome

The positive local/right-gradient hybrid was finite and satisfied the corrected
velocity non-increasing check for all five independently fitted baselines.  Its
mean unpenalized training error was **7.096717**, and its mean penalized training
fitness was **7.246717** (15 nodes, penalty 0.15).

Only the first 60% I80/prediction training interval was evaluated.  The test
split was never queried and is `null` in `results.json`.

| Baseline | Fitted `(c0, c1, c2)` | E_data | Fitness | rho rRMSE | v rRMSE | flow rRMSE | Runtime (s) | Peak RSS estimate (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Greenshields | `(0.140408, -0.723989, -2.941217)` | 6.612569 | 6.762569 | 0.261262 | 0.252969 | 0.255777 | 90.87 | 886.43 |
| Weidmann | `(0.023186, 0.024760, -4.242087)` | 6.864676 | 7.014676 | 0.270714 | 0.252997 | 0.255720 | 108.68 | 1006.35 |
| Triangular | `(-0.035416, 0.061826, 1.455773)` | 8.041865 | 8.191865 | 0.270185 | 0.296374 | 0.256215 | 79.40 | 1046.80 |
| IDM | `(-0.090078, 0.034214, 0.147285)` | 7.206705 | 7.356705 | 0.258729 | 0.277837 | 0.204424 | 218.38 | 1119.04 |
| Del Castillo | `(-0.026277, -0.026916, 4.932898)` | 6.757771 | 6.907771 | 0.261045 | 0.258864 | 0.261387 | 94.60 | 1119.04 |

Total runtime was 591.93 s and process peak RSS was approximately 1119.04 MB.
All metrics were finite.  Every physical feasibility check passed.  None of the
optimizers reported numerical convergence: each reported that its iteration
limit was exceeded, so these are budget-limited best points rather than proven
optima.

## Interpretation and optimizer caveat

Greenshields used a substantial signed spatial term and obtained the lowest
fitness in this attempt.  In contrast, Weidmann, triangular, IDM, and Del
Castillo fitted small gradient amplitudes (`abs(c1) <= 0.062`).  This makes the
kernel exponent weakly identified and motivated removing it in attempt 2.

This process began before the shared harness added bounded polishing.  The
differential-evolution phase respected the model bounds, but its subsequent
50-iteration Nelder-Mead polish was unbounded.  Consequently Weidmann's
`c2=-4.242087` and Del Castillo's `c2=4.932898` lie just outside the declared
`[-4,4]` structural-search bounds.  Both final expressions are nevertheless
finite and passed the explicit velocity check.  Attempt 2 uses the updated
bounded polish and reports the cap explicitly.
