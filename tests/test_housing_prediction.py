import importlib.util
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import housing_prediction as hp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "redfin_data.xlsx"
HAS_OPENPYXL = importlib.util.find_spec("openpyxl") is not None


class ParseNumericColumnTests(unittest.TestCase):
    def test_parse_numeric_column_handles_mixed_currency_percent_and_missing_values(self) -> None:
        series = pd.Series(["$423,000", "7.5%", "", None])

        parsed = hp._parse_numeric_column(series)

        self.assertEqual(parsed.iloc[0], 423000.0)
        self.assertAlmostEqual(parsed.iloc[1], 0.075)
        self.assertTrue(pd.isna(parsed.iloc[2]))
        self.assertTrue(pd.isna(parsed.iloc[3]))


class PrepareForecastingPanelTests(unittest.TestCase):
    def test_prepare_forecasting_panel_forward_fills_region_and_selects_exogenous_inputs(self) -> None:
        df = pd.DataFrame(
            {
                "Region": ["National", None, "Boston, MA metro area", None],
                "Month of Period End": ["January 2024", "February 2024", "January 2024", "February 2024"],
                "Median Sale Price": ["$400,000", "$405,000", "$600,000", "$606,000"],
                "Homes Sold": ["200", "210", "100", "105"],
                "New Listings": ["300", "320", "140", "145"],
                "Inventory": ["500", "510", "220", "225"],
                "Days on Market": ["40", "38", "25", "24"],
                "Average Sale To List": ["99.1%", "99.4%", "101.2%", "101.0%"],
            }
        )

        panel, exogenous_features = hp._prepare_forecasting_panel(
            df,
            target_col="Median Sale Price",
            month_col="Month of Period End",
            region_col="Region",
        )

        self.assertListEqual(
            exogenous_features,
            ["Homes Sold", "New Listings", "Inventory", "Days on Market", "Average Sale To List"],
        )
        self.assertListEqual(
            panel["region"].tolist(),
            ["Boston, MA metro area", "Boston, MA metro area", "National", "National"],
        )
        self.assertListEqual(
            panel["period"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"],
        )
        self.assertListEqual(
            panel["target"].tolist(),
            [600000.0, 606000.0, 400000.0, 405000.0],
        )
        self.assertAlmostEqual(panel.iloc[0]["Average Sale To List"], 1.012)


class ForecastTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not HAS_OPENPYXL:
            raise unittest.SkipTest("openpyxl is required for the Excel-backed training tests.")

        cls.result = hp.train_model(DATA_PATH)
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            hp.save_model_artifact(cls.result, model_path)
            with open(model_path, "rb") as file:
                cls.artifact = pickle.load(file)

    def test_model_quality_meets_thresholds(self) -> None:
        self.assertLess(
            self.result.metrics["mape"],
            0.10,
            f"Expected MAPE < 10%, got {self.result.metrics['mape']:.2%}.",
        )
        self.assertGreater(
            self.result.metrics["r2"],
            0.80,
            f"Expected R2 > 0.80, got {self.result.metrics['r2']:.4f}.",
        )

    def test_training_result_captures_regions_and_selected_features(self) -> None:
        self.assertGreaterEqual(len(self.result.regions), 5)
        self.assertEqual(self.result.feature_columns[:3], ["year", "month", "region"])
        self.assertListEqual(
            self.result.feature_columns[3:],
            ["Homes Sold", "New Listings", "Inventory", "Days on Market", "Average Sale To List"],
        )
        self.assertEqual(self.result.forecast_horizon, 12)

    def test_saved_artifact_can_predict_above_the_training_year(self) -> None:
        prediction_2026 = hp.predict_from_artifact(
            self.artifact,
            {"year": 2026, "month": 4, "region": "National"},
        )
        prediction_2028 = hp.predict_from_artifact(
            self.artifact,
            {"year": 2028, "month": 4, "region": "National"},
        )
        prediction_2030 = hp.predict_from_artifact(
            self.artifact,
            {"year": 2030, "month": 4, "region": "National"},
        )

        self.assertNotEqual(
            prediction_2026["predicted_median_sale_price"],
            prediction_2028["predicted_median_sale_price"],
        )
        self.assertNotEqual(
            prediction_2028["predicted_median_sale_price"],
            prediction_2030["predicted_median_sale_price"],
        )
        self.assertEqual(prediction_2030["prediction_mode"], "forecast")

    def test_artifact_uses_default_region_aliases_and_optional_exogenous_overrides(self) -> None:
        default_prediction = hp.predict_from_artifact(
            self.artifact,
            {"year": 2027, "month": 4, "region": "United States"},
        )
        stressed_prediction = hp.predict_from_artifact(
            self.artifact,
            {
                "year": 2027,
                "month": 4,
                "region": "United States",
                "Inventory": 1_000_000,
                "Days on Market": 70,
            },
        )

        self.assertEqual(default_prediction["region_used"], "National")
        self.assertNotEqual(
            default_prediction["predicted_median_sale_price"],
            stressed_prediction["predicted_median_sale_price"],
        )

    def test_format_training_summary_includes_holdout_details_and_pass_status(self) -> None:
        summary = hp.format_training_summary(self.result, hp.MODEL_QUALITY_THRESHOLDS)

        self.assertIn("Training Summary", summary)
        self.assertIn("Holdout", summary)
        self.assertIn("Metrics", summary)
        self.assertIn("PASS", summary)


if __name__ == "__main__":
    unittest.main()
