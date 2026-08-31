# Frozen-model I80/prediction external check

Frozen training-selected models evaluated once on the final 40% of the I80 prediction interval; test metrics were not used for selection.

| Baseline | Expression | Parameters | Train fitness | Identity test E_data | Selected test E_data | Delta | Test rho rRMSE | Test v rRMSE | Test flow rRMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Greenshields | `exp(c0*rho) + (delta flat_downwind exp(c1*rho)) *_3 exp(c2*rho)` | `0.5, -5.957330756, -10` | 6.419024 | 9.230426 | 6.121604 | -3.108822 | 0.243317 | 0.251453 | 0.219032 |
| Weidmann | `1 + (delta flat_downwind exp(c1*rho)) *_3 exp(c2*rho)` | `0.2017540235, 8.151159515` | 6.364392 | 7.651073 | 6.706620 | -0.944453 | 0.242499 | 0.274457 | 0.273791 |
| Triangular | `1 + (delta flat_downwind exp(c1*rho)) *_3 exp(c2*rho)` | `0.1327074798, 8.704037661` | 7.648460 | 7.918859 | 6.816831 | -1.102028 | 0.245972 | 0.275381 | 0.286967 |
| IDM | `exp(a * ((delta flat_downwind exp(b*rho)) *_3 exp(c*rho)))` | `0.6140045959, -10, -0.4413702102` | 7.142832 | 6.957441 | 6.828271 | -0.129170 | 0.258415 | 0.264173 | 0.205784 |
| Del Castillo | `1 + (delta flat_downwind exp(c1*rho)) *_3 exp(c2*rho)` | `0.1300955079, 9.519560936` | 6.465031 | 7.196903 | 6.822161 | -0.374742 | 0.239879 | 0.280894 | 0.289676 |

## Summary

- Mean identity test E_data: 7.790941
- Mean selected test E_data: 6.659097
- Mean delta: -1.131843
- Per-baseline improvements: 5/5
- Triangular delta versus README SR test score: -1.257432
- Full-check runtime: 14.618 s
- Peak resident memory: 1589.5 MB
