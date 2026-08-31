# Attempt 2 evaluation — linear-gradient conv-3

## Structure and protocol

Attempt 1's confounded `a*c1` behavior motivated the reduced correction

`g[rho] = 1 + a * ((delta flat_right rho) *_3 exp(c*rho))`.

This preserves downwind conv-3 geometry while replacing the nonlinear inner
exponential with a directly scaled density gradient. It has two parameters and
14 tree nodes. The `(0,0)` identity default passed feasibility for all baselines.

Every baseline was fitted independently on training data only with seed 2102,
`maxiter=2`, `popsize=3`, and `polish_maxiter=10`. No test split was accessed.

| Baseline | Parameters `[a,c]` | E_data | Fitness | Parent fitness | rho rRMSE | v rRMSE | flow rRMSE | Runtime (s) | Peak RSS (MB) | Eligible | Converged |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Greenshields | `[-0.353115, -11.791381]` | 6.845748 | 6.985748 | 7.004365 | 0.256188 | 0.266988 | 0.279693 | 28.67 | 1155.98 | yes | no |
| Weidmann | `[0.014117, 13.412877]` | 6.721889 | 6.861889 | 6.364392 | 0.260509 | 0.258017 | 0.260940 | 21.05 | 1214.15 | yes | no |
| Triangular | `[0.020688, 3.080239]` | 8.032793 | 8.172793 | 7.648460 | 0.269759 | 0.296455 | 0.251840 | 18.64 | 1251.32 | yes | no |
| IDM | `[0.004469, 11.829631]` | 7.343070 | 7.483070 | 7.220317 | 0.256435 | 0.284785 | 0.202491 | 43.54 | 1283.86 | yes | no |
| Del Castillo | `[-0.033609, -15.000000]` | 6.741615 | 6.881615 | 6.465031 | 0.262713 | 0.256543 | 0.257169 | 31.19 | 1305.14 | yes | no |

Mean E_data is **7.1370232270** and mean penalized fitness is
**7.2770232270**. All rows are finite, feasible, in bounds, and below the
objective-100 plateau. None formally converged within the deliberately short
budget. Total runtime was 143.09 seconds and peak process RSS was approximately
1305.14 MB.

## Eligible improvements over the parent

Only Greenshields improves the meta-1 conv-3 parent: fitness falls from
7.0043645229 to **6.9857483280**, an eligible improvement of **0.0186161948**.
No improvement is claimed for the other four baselines.

The reduced form lowers mean fitness by 0.0183981450 relative to attempt 1 and
cuts runtime by about 77 seconds, but remains 0.3365103589 worse than the parent
mean. Its near-zero amplitudes for Weidmann, triangular, and IDM show that those
baselines reject the linear-gradient approximation under this budget; the
parent's nonlinear inner exponential remains important. Del Castillo also pushes
the kernel slope to -15 without improving the parent.

Across this agent's two attempts, the only eligible parent improvements are:

- Greenshields: attempt 2, +0.0186161948 (attempt 1 also improved by +0.0040290498).
- IDM: attempt 1, +0.0115098043.

The parent remains preferable for Weidmann, triangular, and Del Castillo.
