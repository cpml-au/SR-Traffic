# Attempt 1 evaluation — amplitude-decoupled conv-3

## Structure and protocol

The meta-1 winner was extended to

`g[rho] = 1 + a * ((delta flat_right exp(c1*rho)) *_3 exp(c2*rho))`.

The identity default `(0,0,0)` passed feasibility for all five baselines. Each
baseline was fitted independently on training data only with seed 2101,
`maxiter=2`, `popsize=3`, and `polish_maxiter=10`. No test split was accessed.

| Baseline | Parameters `[a,c1,c2]` | E_data | Fitness | Parent fitness | Eligible improvement | rho rRMSE | v rRMSE | flow rRMSE | Runtime (s) | Peak RSS (MB) | Eligible | Converged |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Greenshields | `[0.386188, -0.978746, -14.403340]` | 6.830335 | 7.000335 | 7.004365 | +0.004029 | 0.256123 | 0.266473 | 0.279662 | 34.96 | 1165.43 | yes | no |
| Weidmann | `[0.753822, 0.006583, 0.052415]` | 6.914483 | 7.084483 | 6.364392 | -0.720091 | 0.269636 | 0.256097 | 0.259521 | 32.15 | 1260.80 | yes | no |
| Triangular | `[1.396112, 0.000587, 0.446106]` | 8.057658 | 8.227658 | 7.648460 | -0.579198 | 0.269843 | 0.297217 | 0.251831 | 36.14 | 1334.23 | yes | no |
| IDM | `[0.646635, -15.000000, -9.046129]` | 7.038808 | 7.208808 | 7.220317 | +0.011510 | 0.259448 | 0.271041 | 0.205166 | 80.91 | 1363.04 | yes | no |
| Del Castillo | `[2.000000, -0.014333, 1.257949]` | 6.785823 | 6.955823 | 6.465031 | -0.490792 | 0.264208 | 0.256731 | 0.257082 | 36.18 | 1485.67 | yes | no |

Mean E_data is **7.1254213720** and mean 17-node fitness is
**7.2954213720**, which is 0.3549085040 worse than the parent's mean
6.9405128680. Every row is finite, physically feasible, in bounds, and below
the objective-100 plateau, but none formally converged under the short budget.
Total runtime was 220.33 seconds; peak process RSS was about 1485.67 MB.

Only the eligible Greenshields and IDM rows improve the parent. The remaining
three do not and are not reported as improvements.

## Motivation for attempt 2

The amplitude is poorly identified separately from the inner exponential slope.
Weidmann and triangular make `c1` almost zero, and Del Castillo combines an
amplitude at its upper bound with `c1` near zero; all three effectively suppress
the response. Greenshields pushes the expanded kernel slope near -15, while IDM
pushes the expanded inner slope to -15, so simply widening the three-parameter
box has not cured conditioning.

Attempt 2 removes `exp(c1*rho)` and applies a single amplitude directly to the
linear downwind density gradient. This eliminates the small-slope product
`a*c1`, reduces the fit to two coefficients, preserves conv-3 geometry, and keeps
an exact feasible identity at zero amplitude.
