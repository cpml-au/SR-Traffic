# Attempt 1 evaluation — downwind one-point geometry

The exact downwind mirror was optimized independently for all five fixed FDs
on the I80/prediction training interval only.  The optimizer used seed 1201,
differential evolution with `maxiter=8`, `popsize=5`, followed by the
`I80PredictionProblem.fit` Nelder-Mead polish.  No test metric was requested.

| Baseline | Parameters `(c1,c2)` | `E_data` | Fitness | rho rRMSE | v rRMSE | flow rRMSE | Feasible | In bounds | Runtime (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| Greenshields | `(-0.377669, -38.177258)` | 6.829880 | 6.979880 | 0.255474 | 0.267078 | 0.279734 | yes | **no** | 51.76 |
| Weidmann | `(-3.220670, -2.645521)` | 16.918574 (diagnostic) | **100.150000** | 0.434844 | 0.386370 | 0.265254 | **no** | yes | 24.25 |
| Triangular | `(0.036406, 11.835864)` | 7.823481 | 7.973481 | 0.267396 | 0.291494 | 0.250532 | yes | **no** | 51.99 |
| IDM | `(0.000000, 3.826329)` | 7.392856 | 7.542856 | 0.257206 | 0.285836 | 0.202463 | yes | yes | 202.57 |
| Del Castillo | `(-0.000655, 2.774733)` | 6.869585 | 7.019585 | 0.267310 | 0.256782 | 0.257316 | yes | yes | 67.48 |

Mean raw simulator `E_data` is 9.166875.  Applying the physical rejection
threshold to Weidmann gives a mean constraint-aware objective of 25.783160 and
a mean 15-node penalized fitness of **25.933160**.  Peak process RSS was about
1011 MB; total wall time was 398.05 s.

Four candidates had finite fields and passed `dV/drho <= 0`; the Weidmann
candidate was rejected by that check even though its diagnostic simulation was
finite.  Differential evolution stopped at its mandated iteration limit for
the four feasible records.  Weidmann reported global convergence only because
its explored population was uniformly rejected at objective 100.

The harness version used when this attempt ran did not enforce
`CorrectionSpec` bounds during Nelder-Mead, so the Greenshields and triangular
polished `c2` values escaped `[-10,10]`.  They are retained verbatim for
auditability but are flagged as out-of-bounds rather than accepted bounded
fits.  The shared harness was subsequently updated before attempt 2.

## Structural response for attempt 2

The strong Greenshields and Del Castillo errors support retaining the downwind
direction, whereas Weidmann's rejection and the near-zero gradient amplitudes
for triangular, IDM, and Del Castillo suggest sensitivity to a sharp one-point
operator.  Attempt 2 therefore keeps the right flat but changes the available
DEC convolution primitive from window 1 to window 3, testing whether a short
nonlocal average stabilizes the correction without adding expression nodes.
