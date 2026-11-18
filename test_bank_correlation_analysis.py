from itertools import permutations
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import bank_correlation_analysis as analysis


class SummaryCorrelationPvalueTests(unittest.TestCase):
    def test_matches_exhaustive_permutations_of_one_bank_at_a_time(self):
        returns = pd.DataFrame(
            {"A": [1, 2, 4, 3], "B": [2, 4, 6, 8], "C": [4, 1, 3, 2], "D": [1, 3, 0, 2]},
            index=pd.bdate_range("2025-01-02", periods=4),
        )
        means, medians = analysis.annual_summary_correlation_pvalues(returns, n_resamples=100)
        self.assertEqual(means.columns.tolist(), [2025])
        self.assertEqual(means.index.tolist(), returns.columns.tolist())
        for bank in returns:
            observed_peers = returns.corr().loc[bank].drop(bank)
            observed = np.array([observed_peers.mean(), observed_peers.median()])
            null = []
            for permutation in permutations(returns[bank]):
                permuted = returns.copy()
                permuted[bank] = permutation
                peers = permuted.corr().loc[bank].drop(bank)
                null.append([peers.mean(), peers.median()])
            null = np.array(null)
            lower = (null <= observed + 1e-13).mean(axis=0)
            upper = (null >= observed - 1e-13).mean(axis=0)
            expected = np.minimum(1, 2 * np.minimum(lower, upper))
            np.testing.assert_allclose([means.loc[bank, 2025], medians.loc[bank, 2025]], expected)

    def test_reproducible_randomized_tests_and_undefined_summaries(self):
        returns = pd.DataFrame(
            np.random.default_rng(5).normal(size=(30, 3)), columns=["A", "B", "C"],
            index=pd.bdate_range("2025-01-02", periods=30),
        )
        first = analysis.annual_summary_correlation_pvalues(returns, n_resamples=49, seed=4)
        second = analysis.annual_summary_correlation_pvalues(returns, n_resamples=49, seed=4)
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left, right)
            self.assertTrue(((left >= 2 / 50) & (left <= 1)).all().all())
        returns["A"] = 0.0
        undefined = analysis.annual_summary_correlation_pvalues(returns, n_resamples=49)
        self.assertTrue(all(table.isna().all().all() for table in undefined))


class CorrelationPvalueTests(unittest.TestCase):
    def test_matches_pearsonr_and_preserves_labels(self):
        returns = pd.DataFrame({
            "A": [1, 2, 3, 4, 5, 6, 7],
            "B": [10, 9, 2.5, 6, 4, 3, 2],
            "C": [-1, -2, -3, -4, -5, -6, -7],
        })
        correlations = returns.corr()
        pvalues = analysis.pearson_correlation_pvalues(correlations, len(returns))
        pd.testing.assert_index_equal(pvalues.index, correlations.index)
        pd.testing.assert_index_equal(pvalues.columns, correlations.columns)
        for row in returns:
            for col in returns:
                expected = pearsonr(returns[row], returns[col]).pvalue
                self.assertAlmostEqual(pvalues.loc[row, col], expected, places=14)
        np.testing.assert_allclose(pvalues, pvalues.T)

    def test_sample_size_and_undefined_correlations(self):
        correlations = pd.DataFrame([[0.5, 0.0, 1.0, -1.0, np.nan]])
        small = analysis.pearson_correlation_pvalues(correlations, 10)
        large = analysis.pearson_correlation_pvalues(correlations, 100)
        self.assertLess(large.iloc[0, 0], small.iloc[0, 0])
        np.testing.assert_allclose(small.iloc[0, 1:4], [1, 0, 0])
        self.assertTrue(np.isnan(small.iloc[0, 4]))
        two = analysis.pearson_correlation_pvalues(correlations, 2)
        np.testing.assert_allclose(two.iloc[0, :4], 1)
        self.assertTrue(np.isnan(two.iloc[0, 4]))
        self.assertTrue(analysis.pearson_correlation_pvalues(correlations, 1).isna().all().all())


class AnnualCorrelationTests(unittest.TestCase):
    def test_correlations_are_recomputed_each_year_and_exclude_self(self):
        x = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        y = np.array([1, 1, -1, -1, 1, 1, -1, -1])
        first_year = pd.DataFrame(
            {"A": x, "B": x, "C": -x, "D": y},
            index=pd.bdate_range("2024-01-02", periods=8),
        )
        second_year = pd.DataFrame(
            {bank: x for bank in first_year.columns},
            index=pd.bdate_range("2025-01-02", periods=8),
        )
        means, medians = analysis.annual_bank_correlations(pd.concat([first_year, second_year]))
        self.assertEqual(means.columns.tolist(), [2024, 2025])
        self.assertEqual(means.index.tolist(), ["A", "B", "C", "D"])
        np.testing.assert_allclose(means[2024], [0, 0, -2 / 3, 0], atol=1e-15)
        np.testing.assert_allclose(medians[2024], [0, 0, -1, 0], atol=1e-15)
        np.testing.assert_allclose(means[2025], 1)
        np.testing.assert_allclose(medians[2025], 1)


class PortfolioReturnsTests(unittest.TestCase):
    def test_equal_weights_apply_to_simple_returns(self):
        # Offset gains/losses cancel in a rebalanced portfolio, unlike log averages.
        simple_returns = pd.DataFrame(
            [[0.10] * 5 + [-0.10] * 5, [0.02] * 10],
            index=pd.to_datetime(["2026-11-12", "2026-11-13"]),
            columns=analysis.TICKERS,
        )
        result = analysis.equal_weight_portfolio_log_returns(np.log1p(simple_returns))
        np.testing.assert_allclose(result, [0.0, np.log1p(0.02)], atol=1e-15)
        pd.testing.assert_index_equal(result.index, simple_returns.index)


class DownloadValidationTests(unittest.TestCase):
    def test_failed_ticker_is_reported_before_dropping_common_rows(self):
        dates = pd.bdate_range("2025-01-01", periods=110)
        valid = pd.DataFrame(100.0, index=dates, columns=analysis.TICKERS)
        for failure in ("absent", "nan", "infinite"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                prices = valid.copy()
                if failure == "absent":
                    prices = prices.drop(columns="BNY")
                else:
                    prices["BNY"] = np.nan if failure == "nan" else np.inf
                raw = pd.concat({"Close": prices}, axis=1)
                with patch.object(analysis.yf, "download", return_value=raw) as download, \
                        patch.object(analysis, "OUTDIR", Path(directory)):
                    with self.assertRaisesRegex(RuntimeError, "No usable prices.*BNY"):
                        analysis.main()
                    self.assertIn("BNY", download.call_args.kwargs["tickers"])
                    self.assertNotIn("BK", download.call_args.kwargs["tickers"])
                    self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
