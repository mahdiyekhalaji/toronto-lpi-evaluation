# Data dictionary

All effects are percentage changes in collisions after LPI installation compared to
what would have happened without it. Negative means fewer collisions.

| Column | Meaning |
|---|---|
| effect_percent | estimated change in collisions, in percent |
| ci95_lower_percent / ci95_upper_percent | 95% confidence interval around the effect |
| p_value | probability of an effect this large if the true effect were zero |
| p_value_holm_adjusted | p-value corrected for testing several hypotheses at once |
| years_since_lpi_installed | year relative to installation (-3 = three years before, +2 = two after) |
| specification / test_group | which model or robustness check the row comes from |
| n_collisions | how many collisions the estimate is based on |

Abbreviations used in row labels: LPI = leading pedestrian interval. KSI = killed or
seriously injured. ROW = right-of-way. DiD = difference-in-differences. EB = Empirical
Bayes. CMF = crash modification factor (0.89 means 11% fewer crashes than expected).
SPF = safety performance function. NIA = Neighbourhood Improvement Area.

## Files
- results_main.csv: every specification and robustness check
- results_primary_hypotheses.csv: the six pre-declared hypotheses with corrected p-values
- event_study_ksi.csv: effect by year relative to installation, serious injuries
- event_study_all_injury.csv: same, all-severity police-reported injuries
