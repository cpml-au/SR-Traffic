# Attempt 2 evaluation

## Structure and protocol

Guided by attempt 1's infeasible positive-slope plateau, this attempt used

`g[rho] = exp(a * ((delta flat_left exp(b*rho)) *_1 exp(c*rho)))`

with `b <= 0`. The outer exponential makes the correction strictly positive and
is a structural change from attempt 1's additive `1 + a*z`. The structure has
three free constants and 16 tree nodes. Its default vector was explicitly checked
and found feasible for all five baselines before fitting.

Each baseline was independently fitted with seed 1102 through
`I80PredictionProblem.fit(maxiter=3, popsize=4, polish_maxiter=20)`. The bounded
polish and lower global budget reflect attempt 1's measured 284-second runtime.
Only `split="train"` was evaluated; no test data were called.

## Results

| Baseline | Parameters `[a,b,c]` | E_data | Fitness | rho rRMSE | v rRMSE | flow rRMSE | Runtime (s) | Peak RSS (MB) | Eligible | Converged |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Greenshields | `[0.999591, -2.594177, -3.451367]` | 7.626750 | 7.786750 | 0.266061 | 0.285913 | 0.280743 | 27.54 | 1019.60 | yes | no |
| Weidmann | `[0, -0.340988, -2.574546]` | 6.919012 | 7.079012 | 0.269608 | 0.256304 | 0.259503 | 35.56 | 1104.43 | yes | no |
| Triangular | `[1.332074, -2.581647, -5.355257]` | 7.601896 | 7.761896 | 0.264066 | 0.286893 | 0.252585 | 17.89 | 1180.59 | yes | no |
| IDM | `[0, -0.394030, -2.732172]` | 7.392856 | 7.552856 | 0.257206 | 0.285836 | 0.202463 | 142.30 | 1180.59 | yes | no |
| Del Castillo | `[0.874399, -6.905258, -7.799147]` | 6.835554 | 6.995554 | 0.261280 | 0.261618 | 0.257945 | 38.49 | 1269.87 | yes | no |

The mean training E_data is **7.2752134362** and the mean penalized training
fitness is **7.4352134362**. All five fits are finite, physically feasible, within
bounds, and below the objective-100 penalty plateau, so the eligible-only mean is
the same. Total runtime was 261.79 seconds and peak process RSS was approximately
1269.87 MB.

All optimizers report `Maximum number of iterations has been exceeded.` This is
expected under the deliberately reduced search/polish budget: each fit ran to
completion, but none is claimed as a converged optimum. Additional fitting could
change the point estimates.

## Comparison with attempt 1

Attempt 1's five raw logged fitness values average **7.3257058580**, versus
**7.4352134362** here. That numerical mean is not an admissible model comparison:
four attempt-1 fits have objective 100 and fail physical feasibility, and the
remaining Greenshields fit left its declared parameter bounds. Attempt 2 is the
surviving structure because it yields eligible fits for every baseline.

The amplitude collapsed to zero for Weidmann and IDM, making their fitted
correction the identity while still paying the 16-node complexity penalty. The
nontrivial feasible envelope was retained for Greenshields, triangular, and Del
Castillo. This suggests any refinement should either reduce complexity for the
identity-preferring baselines or give the nonlocal amplitude more optimization
budget before judging its benefit.
