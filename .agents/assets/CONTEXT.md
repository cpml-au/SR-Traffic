# <Project name> — project context

> **Living document.** Each phase appends new/updates sections.

---

<!-- ─── Phase 1 — Setup ───────────────────────────────────────────────── -->

## Goal
<One-to-three sentence description of what the model should do, why it matters, and any
constraints on form (interpretable, closed-form, mechanistic, etc.).>

## Problem definition
- **Inputs:** <list of input variables, units, and expected ranges>
- **Output:** <target variable, units, any hard constraints (e.g. must be ≥ 0, must be monotone)>
- **Fixed components (if any):** <code or equations that are held fixed; the model plugs into these>

## Data
- **File(s):** `<path/to/data>` — <brief description>
- **Key variables:** <which columns / fields map to inputs and outputs>
- **Cleaning:** <outlier handling, filtering, interpolation — reference the function name if in code>
- **Selection score:** <the score that ranks candidates in Phase 3, and what the user agreed it implies for validation — held-out set, (stratified) k-fold CV, or an in-sample information criterion (AIC/BIC/MDL) needing no validation split>
- **Splits:** <description of how the data is split, following from the selection score, e.g. "random 70/15/15 train/validation/test with seed=42", "5-fold CV on 85%, 15% test", "train on years 2010–2018, validate on 2019, test on 2020", or "none — full-dataset fit">
- **External check (Phase 4):** <what the search never touches — a held-out test set, replication on an independent dataset, or constraints the form must satisfy outside the fitted range>

## <Loss/Cost/Objective> / metrics
<rename these items appropriately based on domain terminology (e.g. pharmacology, aerospace, ...) e.g. "cost function" instead of "loss">
- **Training <loss/cost/objective>:** <formula or name, e.g. "mean squared error over a 200-point resampled grid">
- **Evaluation metrics:** <what is reported on val and test sets>
- **Runtime and memory:** <seconds per evaluation; peak memory in MB — these gate the loop budget>

## Parameter optimization routine
<Brief description: which optimizer (e.g. L-BFGS-B, Nelder-Mead, PSO), key settings,
convergence criterion, and where it lives in code.>

---

<!-- ─── Phase 2 — Baseline Model ──────────────────────────────────────── -->

## Mock model / baseline
- **File:** `<path/to/model.X>`
- **Form:** <brief description of the functional form, e.g. "sum of three Lorentzians in |v|">
- **Parameters:** <list of free parameters and their roles>
- **Baseline metrics:** <metrics, from train and validation losses/costs/objective values, to anything other defined with the user>
- **Runtime:** <seconds per evaluation>; peak memory: <MB>

---

<!-- ─── Phase 3 — Iterate ──────────────────────────────────────────────── -->

## Search plan
- **M × S × I:** <M> meta-iterations × <S> sub-agents × <I> inner iterations
- **Directory layout:** `meta_m/agent_s/attempt_i/{plan.md, model.X, training.md, evaluation.md}`
- **Diversification strategy:** <how agents are assigned different starting points or search directions, e.g. "agents 1–2 resume from baseline; agent 3 restarts from a simpler form">

## Hypotheses for meta_1
- <Hypothesis 1 — e.g. which region / bin / subgroup is hardest to fit and why>
- <Hypothesis 2 — e.g. which functional families or structural changes to explore>
- <Hypothesis 3 — e.g. any known symmetry, constraint, or physical prior to exploit>

## Meta_1 outcome (<date>)

| Source | Model family | <val loss/cost/objective> | <metric_2> | <metric_3> |
|---|---|---|---|---|
| baseline | <form> | <value> | … | … |
| agent_1/att_3 | <form> | <value> | … | … |
| agent_2/att_3 | <form> | <value> | … | … |
| agent_3/att_3 | <form> | <value> | … | … |

**Key findings:**
- <What structural changes consistently helped — the most transferable signal across agents>
- <What didn't work, and why (optimizer behavior, residual structure, saturation)>
- <Any diagnostic patterns worth noting (optimizer hitting bounds, loss surface shape, etc.)>

**Hypotheses for meta_2:**
- <Updated direction based on meta_1 findings>
- <New families or structural modifications to try next>

---
*(Repeat `## Meta_m outcome` block for each subsequent meta-iteration, incrementing m.)*

---

## Final model pointer (end of Phase 3)
- **Best model file:** `<meta_m/agent_s/attempt_i/model.X>`
- **Validation metrics:** <description featuring the loss/cost/objective value, and any other evaluation metrics defined with the user>
- **Why chosen:** <brief rationale — e.g. best val_loss, or best loss/complexity trade-off>
- **Search summary:** <what was tried, what worked, why iteration was stopped>

---

<!-- ─── Phase 4 — Finalize ─────────────────────────────────────────────── -->

## Final outcome
- **Decision:** <accept final model / return to iteration / do not use>
- **Selected model:** `<path/to/final/model.X>` — <concise description and why it was selected>
- **Usage:** <entry point or command, expected inputs/outputs, and required preprocessing>
- **Final validation metrics:** <all metrics defined with the user, including runtime and memory>
- **Final test metrics:** <all metrics defined with the user, including runtime and memory>
- **Generalization:** <brief interpretation of the validation-to-test difference>
- **Acceptance criteria:** <which goals and constraints passed or failed>
- **Key limitation:** <most important limitation or untested assumption>
- **Next step:** <most useful follow-up, if any>
