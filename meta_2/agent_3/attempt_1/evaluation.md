# Attempt 1 evaluation: exponential downwind window-3 envelope

## Outcome

The strictly positive window-3 envelope completed independent fits for all five
baselines.  Every identity default and fitted expression passed the physical
velocity check; every metric was finite, every coefficient remained within its
declared bounds, and all candidates were eligible.  No external test data were
accessed.

Mean training `E_data` was **7.082966** and mean 16-node penalized fitness was
**7.242966**.  This is 0.302453 worse than the 15-node meta-1 common winner's
mean fitness of 6.940513.

| Baseline | Fitted `(a,b,c)` | E_data | Fitness | Meta-1 winner | Change vs winner | rho rRMSE | v rRMSE | flow rRMSE | Runtime (s) | Peak RSS MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Greenshields | `(0.082179, -8.556288, -8.627367)` | 6.778903 | 6.938903 | 7.004365 | **-0.065462** | 0.255063 | 0.265558 | 0.279299 | 45.10 | 1314.20 |
| Weidmann | `(0.633524, 0.008917, 2.559748)` | 6.911253 | 7.071253 | 6.364392 | +0.706861 | 0.269544 | 0.256069 | 0.259545 | 33.43 | 1360.78 |
| Triangular | `(0.852672, -0.000770, 9.228379)` | 8.057001 | 8.217001 | 7.648460 | +0.568542 | 0.269842 | 0.297195 | 0.251815 | 38.53 | 1360.78 |
| IDM | `(0.614005, -10.000000, -0.441370)` | 6.982832 | 7.142832 | 7.220317 | **-0.077485** | 0.258299 | 0.270070 | 0.205213 | 64.08 | 1473.06 |
| Del Castillo | `(0.320806, -0.244327, -7.092519)` | 6.684842 | 6.844842 | 6.465031 | +0.379811 | 0.258891 | 0.258210 | 0.257132 | 32.34 | 1519.85 |

Total wall time was 213.48 s and peak process RSS was approximately 1519.85
MB.  All optimizers exhausted the deliberately short iteration budget, so none
reported convergence; their feasible best points remain eligible for ranking.

## Interpretation

The exponential envelope achieved its physical goal and improved both
Greenshields and IDM relative to the common winner.  IDM's inner slope reached
its `-10` bound, consistent with the common winner's negative-slope IDM regime.
For Weidmann and triangular, however, the fitted inner slopes collapsed almost
to zero.  Under this budget the optimizer could not identify both an outer
amplitude and the inner-gradient slope, and most of the parent's conv-3 benefit
was lost.  That failure motivated attempt 2's removal of the outer amplitude.

The one extra tree node over the parent costs only 0.01 fitness, so the observed
mean regression is dominated by data fit rather than the parsimony penalty.
