---
name: setup
entry: No `CONTEXT.md` artifact in project root folder
exit: Set up completed and documented in CONTEXT.md.
description: First phase. Confirm goal and terminology, prepare data splits, define loss/metrics, and set up parameter optimization.
---

# Phase 1 — Setup

---

## Steps

### 0. Feasibility
- Confirm that subagents can be spawned and read, write, and execute files in the project directory (and subdirectories)
- Confirm with the user their intended language/runtime and that required libraries are available

### 1. Goal
Confirm with the user:
- Check for any useful info in the skill's `references` folder
- Context, intended goal, and scope of the model
- Domain lingo to use throughout (e.g. *loss function* vs. *cost function* / *features* vs. *covariates* / *training* vs. *calibration*)
- Whether the model needs to meet certain properties (e.g. differentiability, positivity, boundedness, some equivariance, adhere to some physical constraints, etc.)

### 2. Existing code
If code already exists, review it with the user and decide whether to reuse or rewrite. Use these skip signals: if train/val/test split files exist, skip to Metrics; if an optimization routine exists, skip to the context dump.

### 3. Data
Identify and confirm:
- Where the data lives and which files/tables are relevant
- Intended input and output variables
- Whether data is present, sufficient, and clean; flag any cleaning needs

### 4. Metrics
Define and confirm with the user:
- Objective/loss function for training/calibration
- Evaluation metrics to report per attempt and at the final check
- *Always include run-time and memory usage as metrics* — these gate the feasibility of the iterative loop
- If applicable (e.g. searching for equations, not neural architectures), include parsimony pressure — either within the loss in consultation with the user, or at least reported per attempt (e.g. AIC/BIC/MDL) — so the search does not silently drift toward ever more complex models

**Selection score.** Agree explicitly which score ranks candidates in Phase 3, since it determines whether a validation split is needed:
- **Not complexity-penalised** (RMSE, log-loss, misfit, …): must be measured out-of-sample, so a validation set or (stratified) *k*-fold cross-validation is required.
- **Complexity-penalised** (an information criterion such as AIC/BIC/MDL): measured in-sample, so no validation split is needed. Do not default to one anyway — ask the user whether they still want a held-out split, and recommend against it where the criterion is well-posed (a proper likelihood, enough data for its asymptotics), since withholding data buys nothing here.
- **User unsure**: lay out both options and what each implies for validation, and ask them to decide before continuing.

See `TIPS.md` in the skill root for optional practices best decided here: attaching uncertainty to metrics, controlling optimizer noise, and recording trivial baselines for context.

### 5. Splits and final check
Following the selection score agreed above, create the data files (e.g. `.csv`, `.mat`, `.rdata`) in a consistent location:
- `train.X` always; `validation.X` only if the selection score requires held-out data (for *k*-fold cross-validation, define the folds instead). If the score needs neither, fitting uses the full dataset.
- Agree an **external check that the search never touches**, for Phase 4. A held-out `test.X` is the standard form; where a holdout is unaffordable or meaningless (very small N, a single trajectory), replication on an independent dataset or constraints the form must satisfy outside the fitted range can serve instead. Something is needed even when the selection score penalises complexity: the criterion prices the complexity of the model it scores, not of the search that found it, so the winner over M×S×I candidates is optimistically biased.
- **Leakage.** Guard against it when splitting: group/entity leakage (e.g. repeat measurements from one patient landing in different splits), time-series autocorrelation across split boundaries, and duplicate rows.
- Record the selection score, the split strategy, and the external check in `CONTEXT.md`, since later phases score every candidate against them.

### 6. Optimization routine
Implement a parameter optimization routine, or confirm an existing one, e.g. gradient descent or a black-box method (PSO, Nelder-Mead). Prefer libraries over from-scratch implementations.

---

## Context dump (end of phase)

Create `CONTEXT.md` in the project root using `assets/CONTEXT.md` as a starting point. Fill in the Phase 1 sections:
- Goal and scope
- Language/runtime and key libraries
- Data location, variable names, selection score and resulting split strategy
- Chosen loss/cost/objective function and evaluation metrics
- Brief description of the optimization routine

Then **verify `CONTEXT.md` is self-sufficient**: a fresh agent with no conversation history should be able to resume from it alone. This is the real requirement — the durable state lives in `CONTEXT.md`, not in the conversation.

If this phase left large data or tool-output dumps in the conversation, **suggest the user compact it** (`/compact`, `/compress`, or whatever their harness provides — the agent cannot do this itself) before proceeding to Phase 2 — Baseline Model.
