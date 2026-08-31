# Attempt 2 evaluation: local exponential/downwind window-3 hybrid

## Outcome

The 18-node local/conv-3 hybrid completed all five independent train-only fits.
Every identity default and fitted expression was finite and physically
feasible, all parameters remained bounded, and all five candidates were
eligible.  The test split was never queried.

Mean training `E_data` was **7.012254** and mean penalized fitness was
**7.192254**.  This improves meta-2 attempt 1 by 0.050712, but remains 0.251741
worse than the 15-node meta-1 common winner.

| Baseline | Fitted `(c0,c1,c2)` | E_data | Fitness | Meta-1 winner | Delta vs winner | Delta vs attempt 1 | rho rRMSE | v rRMSE | flow rRMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Greenshields | `(0.500000, -5.957331, -10.000000)` | 6.239024 | **6.419024** | 7.004365 | **-0.585341** | **-0.519879** | 0.256879 | 0.242474 | 0.206912 |
| Weidmann | `(-0.000250, 0.003417, 6.533333)` | 6.909358 | 7.089358 | 6.364392 | +0.724966 | +0.018105 | 0.269353 | 0.256195 | 0.259613 |
| Triangular | `(-0.006278, 0.006028, 7.155556)` | 8.032270 | 8.212270 | 7.648460 | +0.563811 | -0.004731 | 0.269198 | 0.296948 | 0.252665 |
| IDM | `(-0.002858, -7.739894, -10.000000)` | 7.088134 | 7.268134 | 7.220317 | +0.047817 | +0.125302 | 0.260312 | 0.272030 | 0.205400 |
| Del Castillo | `(-0.011000, 0.006583, 10.000000)` | 6.792483 | 6.972483 | 6.465031 | +0.507452 | +0.127641 | 0.264792 | 0.256389 | 0.259234 |

| Baseline | Runtime (s) | Peak RSS estimate (MB) | Converged | Finite / feasible / in-bounds / eligible |
| --- | ---: | ---: | --- | --- |
| Greenshields | 33.65 | 1206.59 | no (budget) | yes / yes / yes / yes |
| Weidmann | 31.65 | 1528.91 | no (budget) | yes / yes / yes / yes |
| Triangular | 33.41 | 1528.91 | no (budget) | yes / yes / yes / yes |
| IDM | 65.90 | 1528.91 | no (budget) | yes / yes / yes / yes |
| Del Castillo | 38.77 | 1528.91 | no (budget) | yes / yes / yes / yes |

Total runtime was 203.38 s and peak process RSS was approximately 1528.91 MB.

## Interpretation and parsimony

The response partially succeeded: removing the amplitude degeneracy improved
mean fitness and slightly improved triangular relative to attempt 1.  It also
found the strongest Greenshields result among the compared candidates.  It did
not retain the meta-1 winner's Weidmann/triangular values or improve IDM; the
small fitted positive slopes for Weidmann and triangular show that ten local
polish iterations were still insufficient to move from the exact-identity
seed toward the parent's `c1 approximately 0.13--0.20` regime.

The hybrid adds three nodes to the common winner, increasing the complexity
penalty by 0.03.  Its mean `E_data` is already 0.221741 worse, so removing the
penalty would not change the structural conclusion: keep the meta-1 conv-3
winner globally, use attempt 1 as an IDM-specialized positive candidate, and
retain attempt 2 only as a promising Greenshields specialization.  Additional
optimization, rather than extra expression complexity, is the most direct way
to test the intended exact-parent subspace.
