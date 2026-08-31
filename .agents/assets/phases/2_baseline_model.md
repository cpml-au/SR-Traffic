---
name: baseline_model
description: Second phase. Implement a simple baseline model with a parameterized signature, test the full pipeline end-to-end using a sub-agent, and verify outputs are plausible before the iterative loop.
entry: `CONTEXT.md` present.
exit: Baseline model defined; end-to-end pipeline verified by sub-agent; CONTEXT.md updated.
---

# Phase 2 — Baseline Model

---

## Steps

### 1. Baseline model
Program a first, intentionally simple model in a `model.X` file (`X` = `.py`, `.m`, `.r`, etc.). The goal is a working skeleton, not a good model.

- Accept parameters as an explicit function argument with reasonable defaults — this makes it easy for the optimization routine to sweep parameter values without changing the call signature.
- Implement companion file(s) to train/calibrate and evaluate the model.

### 2. End-to-end test via sub-agent
**Use a sub-agent** to run the full pipeline on a small data subset:
- Train/calibrate the model using the optimization routine
- Generate predictions
- Log all metrics (including run-time and memory)

This step is critical: it confirms that sub-agents can read/write files and execute code *before* the iterative loop begins.

### 3. Sanity check
Verify:
- Model produces plausible outputs (outputs in expected range, no NaNs or infinities)
- Optimization problem is well-posed (convergence achieved, metrics not degenerate)
- Run-time and memory usage are compatible with the iterative loop planned in Phase 3

If the model fails to converge, diagnose and fix before proceeding.

---

## Context dump (end of phase)

Update `CONTEXT.md` with:
- Mock model description (structure, key equations or code functions)
- Baseline metrics on train and validation sets (or in line with previous choices e.g. cross-val)
- Confirmed run-time and memory per optimization run
- Any environment or permission findings from the sub-agent test

Then **verify `CONTEXT.md` is self-sufficient**: a fresh agent with no conversation history should be able to resume from it alone.

If the sub-agent test left large logs in the conversation, **suggest the user compact it** before proceeding to Phase 3 — Iterate.
