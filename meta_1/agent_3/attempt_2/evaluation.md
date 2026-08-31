# Attempt 2 evaluation

## Outcome

The positive local/squared-right-gradient hybrid was finite and velocity
non-increasing for every baseline.  Its mean unpenalized training error was
**7.088418**, and its mean penalized training fitness was **7.198418** (11
nodes, penalty 0.11).  It therefore beat attempt 1's mean fitness by
**0.048300** despite the substantially smaller optimization budget.

Only the first 60% I80/prediction training interval was evaluated.  The test
split was never queried and is `null` in `results.json`.

| Baseline | Fitted `(c0, c1)` | E_data | Fitness | rho rRMSE | v rRMSE | flow rRMSE | Runtime (s) | Peak RSS estimate (MB) | Converged |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Greenshields | `(0.002263, 0.000325)` | 6.747839 | 6.857839 | 0.254460 | 0.264965 | 0.278572 | 17.65 | 792.47 | no |
| Weidmann | `(-0.000014, 0.000007)` | 6.917947 | 7.027947 | 0.269352 | 0.256531 | 0.259484 | 12.04 | 857.44 | yes |
| Triangular | `(-0.000067, 0.000028)` | 8.046829 | 8.156829 | 0.268751 | 0.297841 | 0.251869 | 11.38 | 946.50 | yes |
| IDM | `(-0.000741, 0.001493)` | 7.070265 | 7.180265 | 0.257017 | 0.274495 | 0.207033 | 55.87 | 1033.93 | no |
| Del Castillo | `(0.000647, 0.000062)` | 6.659208 | 6.769208 | 0.258386 | 0.257722 | 0.256922 | 20.55 | 1091.22 | no |

Total runtime was 117.50 s and process peak RSS was approximately 1091.22 MB.
All metrics were finite and every physical feasibility check passed.  Weidmann
and triangular reported successful convergence; the other three reached the
iteration limit but returned feasible best points.

## Structural comparison with attempt 1

| Baseline | Attempt 1 fitness | Attempt 2 fitness | Change (attempt 2 - attempt 1) |
| --- | ---: | ---: | ---: |
| Greenshields | 6.762569 | 6.857839 | +0.095270 |
| Weidmann | 7.014676 | 7.027947 | +0.013271 |
| Triangular | 8.191865 | 8.156829 | -0.035037 |
| IDM | 7.356705 | 7.180265 | **-0.176440** |
| Del Castillo | 6.907771 | 6.769208 | **-0.138562** |
| **Mean** | **7.246717** | **7.198418** | **-0.048300** |

The structural response achieved its main goal: IDM and Del Castillo both
improved materially, and removing four complexity-penalty nodes also made the
slightly worse triangular data error a better penalized model.  Attempt 1
remains the better Greenshields correction, while attempt 2 is the stronger
cross-baseline hybrid by mean fitness and is particularly preferable for IDM
and Del Castillo.

The very small coefficients should not be interpreted as a negligible spatial
effect without accounting for the DEC operator's mesh scaling: the
codifferential of the interpolated edge cochain is not normalized to a unit
finite difference.  All attempt-2 parameters stayed inside their declared
bounds under the updated bounded Nelder-Mead polish.
