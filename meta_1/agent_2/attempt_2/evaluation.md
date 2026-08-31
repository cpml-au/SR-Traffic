# Attempt 2 evaluation — downwind three-point geometry

The three-point downwind correction was independently fitted to every fixed
baseline on the I80/prediction training interval.  Before fitting, the `(0,0)`
default was explicitly checked with `I80PredictionProblem.is_feasible` and
passed for Greenshields, Weidmann, triangular, IDM, and Del Castillo.

The revised compute budget used seed 1202, differential evolution with
`maxiter=3`, `popsize=4`, and the shared bounded Nelder-Mead polish with
`polish_maxiter=20`.  No test split was evaluated.

| Baseline | Parameters `(c1,c2)` | `E_data` | Fitness | rho rRMSE | v rRMSE | flow rRMSE | Feasible / bounded | Converged | Runtime (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| Greenshields | `(-0.368374, -7.950539)` | 6.854365 | 7.004365 | 0.255942 | 0.267546 | 0.279513 | yes | no (budget) | 42.87 |
| Weidmann | `(0.201754, 8.151160)` | 6.214392 | 6.364392 | 0.251027 | 0.247534 | 0.261739 | yes | no (budget) | 38.73 |
| Triangular | `(0.132707, 8.704038)` | 7.498460 | 7.648460 | 0.258740 | 0.288137 | 0.254618 | yes | no (budget) | 41.63 |
| IDM | `(-10.000000, -10.000000)` | 7.070317 | 7.220317 | 0.260274 | 0.271410 | 0.205329 | yes | yes | 70.85 |
| Del Castillo | `(0.130096, 9.519561)` | 6.315031 | 6.465031 | 0.255592 | 0.246928 | 0.258720 | yes | no (budget) | 45.73 |

Mean training `E_data` is **6.790513** and mean 15-node penalized fitness is
**6.940513**.  All metrics are finite, every corrected velocity passes
`dV/drho <= 0`, and every parameter lies in `[-10,10]`.  Total wall time was
239.82 s and peak process RSS was about 1191 MB.  Four optimizers exhausted
the deliberately short iteration budget; IDM reported convergence at the
lower bounds, so its constants should be viewed as boundary-limited.

## Comparison and interpretation

Relative to attempt 1, window 3 changes training `E_data` by +0.024484 for
Greenshields, -10.704182 for the previously rejected Weidmann simulation,
-0.325021 for triangular, -0.322538 for IDM, and -0.554554 for Del Castillo.
Most importantly, it turns all five candidates into accepted bounded models.

For the triangular benchmark highlighted in project context, the published
15-node term has training `E_data=7.6192` and fitness 7.7692.  This attempt has
the same node count and improves those values to 7.498460 and 7.648460,
respectively (an absolute improvement of about 0.12074).  This is a training-
only comparison; external test performance remains intentionally unknown.

The fitted signs reveal two regimes: Greenshields and IDM prefer negative
inner slopes, while Weidmann, triangular, and Del Castillo prefer small
positive `c1` paired with a large positive kernel slope.  A future iteration
could refine these regimes separately or test a centered left/right blend,
but the current three-point structure is already compact, feasible, and
substantially more robust than the one-point downwind mirror.
