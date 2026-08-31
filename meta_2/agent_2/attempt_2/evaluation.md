# Meta 2 attempt 2 evaluation — linear upwind gradient

The 14-node upwind counterpart

`1 + a * ((delta flat_left rho) *_3 exp(b*rho))`

was fitted independently on all five I80/prediction training problems with
seed 2202, `maxiter=2`, `popsize=3`, and bounded
`polish_maxiter=10`.  Its `(0,0)` exact-identity default passed direct
feasibility checks for every baseline.  The test interval was not accessed.

| Baseline | Parameters `(a,b)` | `E_data` | Fitness | rho rRMSE | v rRMSE | flow rRMSE | Status | Runtime (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Greenshields | `(-0.652461, -6.717213)` | 7.716523 | 7.856523 | 0.271792 | 0.283654 | 0.290066 | accepted | 21.26 |
| Weidmann | `(0.000000, -4.454622)` | 6.919012 | 7.059012 | 0.269608 | 0.256304 | 0.259503 | accepted | 19.21 |
| Triangular | `(-0.000625, -4.844401)` | 8.057848 | 8.197848 | 0.269811 | 0.297252 | 0.251844 | accepted | 21.52 |
| IDM | `(-0.020477, -5.371647)` | 7.383228 | 7.523228 | 0.257089 | 0.285604 | 0.202661 | accepted | 49.48 |
| Del Castillo | `(-0.027246, -7.911304)` | 6.861593 | 7.001593 | 0.266514 | 0.257298 | 0.257584 | accepted | 34.20 |

Mean training `E_data` is **7.387641** and mean 14-node penalized fitness is
**7.527641**.  All fields and metrics are finite; all corrected velocities are
non-increasing; all parameters are within bounds, with none exactly at a
bound.  Each optimizer stopped at the deliberately short iteration limit.
Total runtime was 145.68 s and peak process RSS was about 1385 MB.

## Structural comparison

Changing only the flat from downwind/right to upwind/left worsens fitness by
0.551288 (Greenshields), 0.017948 (Weidmann), 0.000471 (triangular), 0.185945
(IDM), and 0.125308 (Del Castillo), or 0.176192 on average.  Four fitted
amplitudes are essentially zero, indicating that the model retreats toward the
identity rather than exploiting the upwind linear gradient.

Against the nonlinear 15-node meta-1 winner, the upwind linear candidate is
worse by 0.852159, 0.694620, 0.549388, 0.302911, and 0.536562 across the same
baseline order; mean fitness is 0.587128 worse.  Thus neither direction of the
linear-gradient simplification recovers the parent's accuracy.  The useful
structure is specifically the downwind three-point geometry combined with
`delta(flat_right(exp(c1*rho)))`, whose exponential supplies a
density-dependent gradient amplitude that the compact linear alternatives
lose.
