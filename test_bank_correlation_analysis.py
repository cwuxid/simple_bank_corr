import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import bank_correlation_analysis as analysis


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
