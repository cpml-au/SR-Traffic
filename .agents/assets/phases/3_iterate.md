---
name: iterate
description: Third and main phase. Orchestrate meta/inner agent loop. Guide user in choosing iteration parameters, spawn S parallel sub-agents per meta-iteration each running I sequential structural modifications, review results across agents, and repeat for M meta-iterations until user is satisfied with validation performance.
entry: CONTEXT.md exists; end-to-end pipeline verified for baseline model. If `meta_*/` directories already exist, resume from the highest completed meta-iteration rather than restarting.
exit: User satisfied with validation performance; CONTEXT.md updated with final summary and self-sufficient for Phase 4
---

# Phase 3 — Iterate

---

## Setup (once, with user)
Read `CONTEXT.md` to restore context. Then:

1. Use the confirmed run-time per optimization step and memory usage to estimate total compute for a full loop (account for sub-agents working in parallel, and for sub-agent thinking and writing overhead of a few seconds per step).
2. Guide the user in choosing:
   - **M** — number of meta-iterations
   - **S** — number of sub-agents per meta-iteration (run in parallel)
   - **I** — number of inner iterations per sub-agent (run sequentially)

---

## Execution — meta loop
For each meta-iteration `m = 1 … M`:

### 1. Spawn sub-agents
Launch **S sub-agents in parallel**. Each agent writes exclusively within its own `meta_m/agent_s/` directory to avoid file conflicts.

Provide each sub-agent with:

- If `m = 1`, the baseline model file; if `m > 1`, the best model from `meta_(m-1)/agent_s` + respective evaluation results;
- A distinct hypothesis or search direction (ensure agents don't duplicate effort);
- Any relevant material from the skill's `references/` folder, and the **same** evaluation protocol/script — train on `train.X`, evaluate on both `train.X` and `validation.X` (or what applies, e.g. cross-validation), log all metrics — so results are comparable across agents;
- The instruction to **run each fit and evaluation to completion and write its results before returning** — do not background a long job and end the turn, which leaves work unharvested and processes orphaned.

### 2. Sub-agent inner loop (I iterations each)
Each sub-agent, for `i = 1 … I`:

1. Plan a *structural* modification to the model — at the level of equations or code functions (e.g. new terms, different functional form, changed layers), based on the previous attempt's evaluation results, in particular where it fits worst (residual structure, worst region/subgroup);
2. Implement the modified model in `meta_m/agent_s/attempt_i/model.X`, making sure that it is commented in a clear and complete way, to document your planned choices;
3. Train/calibrate on `train.X` using the optimization routine;
4. Evaluate with the shared protocol; store results in `meta_m/agent_s/attempt_i/evaluation.md`;
5. Proceed to next iteration.

### 3. Meta-iteration review (no user involvement)
Once all S agents complete their I iterations:

- Discard any agent that returned without complete written results (crashed, wrote NaNs, or left a job unfinished) as a dead lineage — do not compare it — and clean up any orphaned processes it left behind;
- Review each surviving agent's `evaluation.md` to shortlist the best-performing models across all attempts and agents;
- Re-evaluate the shortlist yourself through the common evaluation protocol before declaring a winner: sub-agent point estimates from differing fit settings are not directly comparable, and small metric gaps may be within run-to-run noise (see `TIPS.md` for making this quantitative);
- Identify which structural changes improved performance and which didn't;
- Formulate hypotheses for the next meta-iteration; ensure next-iteration agent prompts cover non-overlapping search directions.

### 4. Context dump (each meta-iteration)
Update `CONTEXT.md`:

- Meta-iteration index and best validation metric achieved;
- Summary of structural changes that worked / didn't work;
- Hypotheses and search directions for the next meta-iteration (if an agent is stuck, prompt it to explore a different direction).

---

## Review with user (after M meta-iterations)
Present:

- Plots of training and validation performance of the best model found at each meta-iteration;
- Summary of structural changes that drove improvements.

Ask the user:

- **Continue iterating?** → return to Setup above, choosing new M/S/I; increment meta-iteration index to `M+1` to avoid overwriting previous logs. Ensure `CONTEXT.md` captures the state needed to plan the next batch; if the last batch returned bulky output into the conversation, **suggest compacting** before starting;
- **Satisfied with validation performance?** → proceed to Phase 4 — Finalize.

---

## Context dump (before handing off to Finalize)
Update `CONTEXT.md` with:

- Path to the best model file and its validation metrics;
- Final search summary: what was tried, what worked, rationale for stopping.

Finally, **verify `CONTEXT.md` is self-sufficient** — it must point to the best model file and carry the full search summary, so Phase 4 can run from it alone. If the search left bulky output in the conversation, **suggest the user compact it** before proceeding to Phase 4 — Finalize.
