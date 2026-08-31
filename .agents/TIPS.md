# automodel — optional tips

These are practices that can sharpen an `automodel` search but are **deliberately not baked into the core recipes**, because their relevance and best implementation are use-case-dependent. Treat them as a menu: raise them with the agent during a run, or fold the useful ones into your own fork of the skill.

---

## Attach uncertainty to metrics

The meta-agent selects the "best" model from point estimates, but a validation-metric gap of, say, 0.102 vs. 0.105 may be pure optimizer/sampling noise. Before committing to one structure over another:

- Report an error bar on the validation metric — via bootstrap resampling of the validation set, or the spread across cross-validation folds.
- Or run a paired test on per-example residuals between the candidate and the best-so-far, so a change is kept only when the improvement is robust rather than within noise.

This is the cheapest guard against the search chasing noise. The right method is domain-specific (a Poisson-count likelihood, a classification metric, and an ODE misfit each call for different tests), which is why it is not prescribed in the core.

## Control optimizer noise

A good *structure* can score badly purely because its parameter fit failed to converge (bad init, poor scaling), making structural gains indistinguishable from optimizer luck. Phase 2 checks convergence for the baseline; carry the same discipline into the search:

- Fix and **log the random seed(s)** for each fit so runs are reproducible.
- Run a few optimizer restarts per candidate (multi-start, or a global method such as basin-hopping / differential evolution for stiff landscapes) and keep the best.
- Record a convergence diagnostic and flag non-converged fits, so they are not compared as if trustworthy.

Note that expensive global optimizers can dominate per-candidate cost and blow the loop budget estimated in Phase 3 — size the search accordingly.

## Log structured results and a leaderboard

`evaluation.md` files are free-form prose that the meta-agent must re-parse across every `meta_m/agent_s/attempt_i/`. If your metrics are stable and numeric, it is worth also emitting:

- A machine-readable `results.json` per attempt (fixed fields: structure, `n_params`/complexity, train/val metrics, runtime, memory, converged, seed).
- One append-only `leaderboard.csv` across all meta-iterations.

This makes selection, plotting, and failure-handling mechanical and auto-aggregatable. It is left optional because a schema that fits every domain (regression, PDE terms, neural architectures) is hard to fix in advance — define one that suits your problem.

## Record trivial baselines

Where it makes sense for the problem (it may not for, say, a PDE), also record simple reference points — predict-the-mean, a linear regression, or the incumbent domain-standard model. If the discovered model barely beats the mean, that is critical to surface, and it contextualizes every later improvement.
