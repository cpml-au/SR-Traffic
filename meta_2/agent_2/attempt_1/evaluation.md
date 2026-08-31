# Meta 2 attempt 1 evaluation — linear downwind gradient

The 14-node correction

`1 + a * ((delta flat_right rho) *_3 exp(b*rho))`

was fitted independently for every baseline with seed 2201,
`maxiter=2`, `popsize=3`, and bounded `polish_maxiter=10`.  The exact-identity
default `(0,0)` passed a direct physical-feasibility check on all five
baselines before fitting.  Only the I80/prediction training interval was used.

| Baseline | Parameters `(a,b)` | `E_data` | Fitness | rho rRMSE | v rRMSE | flow rRMSE | Status | Runtime (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Greenshields | `(-1.132424, -8.176665)` | 7.165236 | 7.305236 | 0.271266 | 0.264044 | 0.278927 | accepted | 30.49 |
| Weidmann | `(0.027496, -0.003880)` | 6.901064 | 7.041064 | 0.269961 | 0.255230 | 0.259607 | accepted | 23.94 |
| Triangular | `(0.000781, -0.000086)` | 8.057377 | 8.197377 | 0.269839 | 0.297211 | 0.251830 | accepted | 26.57 |
| IDM | `(-1.329073, -10.000000)` | 7.197283 | 7.337283 | 0.259803 | 0.276493 | 0.205864 | accepted, at bound | 71.10 |
| Del Castillo | `(-0.047232, 0.008178)` | 6.736285 | 6.876285 | 0.262108 | 0.256953 | 0.256997 | accepted | 37.67 |

Mean `E_data` is **7.211449** and mean penalized fitness is **7.351449**.
Every record is finite, physically feasible, and within its declared bounds.
All five optimizers stopped at the intentionally short iteration budget; IDM's
kernel slope also reached the lower bound.  Total wall time was 189.77 s and
peak process RSS was about 1399 MB.

## Comparison with the meta-1 winner

Relative to the accepted 15-node downwind exponential-gradient winner, the
14-node linear form changes fitness by +0.300871 (Greenshields), +0.676672
(Weidmann), +0.548917 (triangular), +0.116966 (IDM), and +0.411254
(Del Castillo).  Mean fitness is 0.410936 worse.  The one-node saving therefore
does not compensate for removing density-dependent amplification inside the
gradient.

Three fitted kernel slopes (Weidmann, triangular, Del Castillo) collapse near
zero and their amplitudes also approach zero, while Greenshields and IDM retain
strong negative kernel slopes.  Rather than merely retuning the same geometry,
attempt 2 switches the flat to the left/upwind endpoint.  This tests whether
the direct linear gradient needs the opposite sampling direction while
preserving the compact three-point look-ahead and identical parameterization.
