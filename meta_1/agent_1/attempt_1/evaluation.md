# Attempt 1 evaluation

## Structure and protocol

The fitted structure was

`g[rho] = 1 + a * ((delta flat_left exp(b*rho)) *_1 exp(c*rho))`

with three free constants, 17 tree nodes, seed 1101, `maxiter=8`, and
`popsize=5`. Every baseline was independently fitted and evaluated on the
first-60%-of-time training selection split only. No test split was called.

## Results

| Baseline | Parameters `[a,b,c]` | E_data | Fitness | rho rRMSE | v rRMSE | flow rRMSE | Runtime (s) | Peak RSS (MB) | Feasible | Optimizer converged |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Greenshields | `[0.277341, 6.202214, -3.883445]` | 6.558994 | 6.728994 | 0.277714 | 0.232497 | 0.278969 | 70.59 | 988.12 | yes | no |
| Weidmann | `[0.851464, 0.038621, -4.321196]` | 6.914345 | 7.084345 | 0.269499 | 0.256237 | 0.259530 | 30.86 | 1074.64 | no | yes |
| Triangular | `[0.851464, 0.038621, -4.321196]` | 8.053578 | 8.223578 | 0.269734 | 0.297179 | 0.251807 | 28.29 | 1155.12 | no | yes |
| IDM | `[0.851464, 0.038621, -4.321196]` | 7.384924 | 7.554924 | 0.257098 | 0.285656 | 0.202420 | 113.01 | 1193.11 | no | yes |
| Del Castillo | `[0.851464, 0.038621, -4.321196]` | 6.866689 | 7.036689 | 0.267278 | 0.256702 | 0.257319 | 41.49 | 1213.52 | no | yes |

The arithmetic mean of the five logged training fitness values is
**7.3257058580**. All reported metrics are finite.

## Assessment and next-step motivation

Only Greenshields escaped the physical-feasibility penalty. Its optimizer did
not converge within the polish iteration budget and returned `b=6.2022`, just
outside the declared upper bound of 6, so it is useful evidence but not a fully
admissible bounded optimum. For Weidmann, triangular, IDM, and Del Castillo, the
fitting objective is exactly 100 and the explicit velocity check fails. Their
unguarded post-fit E_data values are diagnostics only and are **discarded as
candidates**, regardless of the optimizer's success message on the flat penalty.

The common failed vector has a small positive inner slope. This indicates that
the attempt-1 sign restriction made the feasible basin effectively unreachable
for most fundamental diagrams. Attempt 2 therefore makes two structural changes:
it restricts the inner slope to be non-positive, and it replaces the additive
outer form with a strictly positive exponential envelope. The identity remains
reachable at zero amplitude or zero inner slope.
