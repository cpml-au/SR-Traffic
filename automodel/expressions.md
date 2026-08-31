# Expressions evaluated during automodel search

All scores use the first 60% I80/prediction selection interval; test data were not accessed.

| Source | Expression | Nodes | Eligible baselines | Mean eligible fitness |
|---|---|---:|---:|---:|
| `meta_1/agent_1/attempt_1/results.json` | `1 + a * ((delta flat_left exp(b*rho)) *_1 exp(c*rho))` | 17 | 0/5 | — |
| `meta_1/agent_1/attempt_2/results.json` | `exp(a * ((delta flat_left exp(b*rho)) *_1 exp(c*rho)))` | 16 | 5/5 | 7.435213 |
| `meta_1/agent_2/attempt_1/results.json` | `1 + (delta flat_downwind/right exp(c1*rho)) *_1 exp(c2*rho)` | 15 | 2/5 | 7.281220 |
| `meta_1/agent_2/attempt_2/results.json` | `1 + (delta flat_downwind/right exp(c1*rho)) *_3 exp(c2*rho)` | 15 | 5/5 | 6.940513 |
| `meta_1/agent_3/attempt_1/results.json` | `exp(c0*rho + c1*((delta flat_right_P rho) *_1 exp(c2*rho)))` | 15 | 3/5 | 7.437046 |
| `meta_1/agent_3/attempt_2/results.json` | `exp(c0*rho + c1*square(delta flat_right_P rho))` | 11 | 5/5 | 7.198418 |
| `meta_2/agent_1/attempt_1/results.json` | `1 + a * ((delta flat_downwind/right exp(c1*rho)) *_3 exp(c2*rho))` | 17 | 5/5 | 7.295421 |
| `meta_2/agent_1/attempt_2/results.json` | `1 + a * ((delta flat_downwind/right rho) *_3 exp(c*rho))` | 14 | 5/5 | 7.277023 |
| `meta_2/agent_2/attempt_1/results.json` | `1 + a * ((delta flat_downwind/right rho) *_3 exp(b*rho))` | 14 | 5/5 | 7.351449 |
| `meta_2/agent_2/attempt_2/results.json` | `1 + a * ((delta flat_upwind/left rho) *_3 exp(b*rho))` | 14 | 5/5 | 7.527641 |
| `meta_2/agent_3/attempt_1/results.json` | `exp(a * ((delta flat_right_P exp(b*rho)) *_3 exp(c*rho)))` | 16 | 5/5 | 7.242966 |
| `meta_2/agent_3/attempt_2/results.json` | `exp(c0*rho) + ((delta flat_right_P exp(c1*rho)) *_3 exp(c2*rho))` | 18 | 5/5 | 7.192254 |
