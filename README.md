# simple_bank_corr
A simple script that loads stock data for banks and tracks the correlation over periods as a time series to measure systematic risk. Done for CPG FA2026 policy proposal "Lend Short &amp; Live Long: Testing Emergency Liquidity Readiness for US Banks".

Install dependencies with `.venv/bin/python -m pip install yfinance pandas numpy matplotlib scipy`, then run `.venv/bin/python bank_correlation_analysis.py` to write results to `bank_correlation_results/`.

Prices, daily returns, portfolio plots, and 90-trading-day rolling correlations cover January 1, 2015 through the latest available data. The full-period correlation matrix and heatmap are saved as `full_period_correlation_matrix.csv` and `full_period_correlation_heatmap.png`.

The annual mean and median correlation CSV tables and `annual_bank_correlation_tables.png` use complete calendar years starting in 2015 (2015–2025 when run in 2026). Rows are banks and columns are years. For each year, the script calculates Pearson correlations from that year's daily log returns, then takes each bank's mean and median correlation with the other nine banks, excluding itself. These are calculated separately from the 90-day rolling correlations. An undefined pairwise correlation produces `N/A` for the affected bank's annual summary.

`pearson_correlation_pvalues(correlations, observations)` returns two-sided p-values for testing zero population correlation, using the same exact beta null distribution as SciPy's `pearsonr`. Pass a correlation table and the number of complete paired daily observations used in every cell. The script exports:

- `full_period_correlation_pvalues.csv`: banks by banks, matching the full-period correlation matrix.
- `annual_pairwise_correlation_pvalues.csv`: 45 unique bank pairs by complete calendar year, using each year's actual sample size.
- `rolling_90day_correlation_pvalues.csv`: dates by bank pairs, matching the rolling correlation table with 90 observations per test.

Self-correlations and undefined correlations are reported as `N/A`; very small p-values use scientific notation. Extremely small probabilities can underflow to zero. These Pearson p-values are for individual pairs, not the mean or median of their correlations. These are unadjusted classical Pearson tests: they assume independent observations under a normal null model and do not account for serial dependence, overlapping rolling windows, or multiple comparisons. See the [SciPy Pearson test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html).

The original `annual_bank_correlation_tables.png` also includes two bottom panels with p-values for each bank's mean and median correlation. `annual_summary_correlation_pvalues()` calculates these using two-sided [SciPy permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), not averages of pairwise p-values. Within each year, each bank's daily returns are shuffled against the fixed joint returns of its peers, preserving relationships among the other nine banks. The null is random pairing between that bank and its peers, with the mean or median of the nine correlations as the test statistic. This assumes exchangeable trading days and does not correct for serial dependence or multiple comparisons.

Summary p-values are also saved to `annual_mean_correlation_pvalues.csv` and `annual_median_correlation_pvalues.csv`. The randomized tests use 9,999 permutations and seed 0; the smallest two-sided p-value reported is `0.0002`. Values at that limit do not distinguish how much smaller the true tail probability might be. Constant or incomplete return samples produce `N/A`.
