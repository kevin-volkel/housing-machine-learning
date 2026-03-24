import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import housing_prediction as hp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "redfin_data.xlsx"


class ParseNumericColumnTests(unittest.TestCase):
    def test_parse_numeric_column_handles_mixed_currency_percent_and_missing_values(self) -> None:
        series = pd.Series(["$423,000", "7.5%", "", None])

        parsed = hp._parse_numeric_column(series)

        self.assertEqual(parsed.iloc[0], 423000.0)
        self.assertAlmostEqual(parsed.iloc[1], 0.075)
        self.assertTrue(pd.isna(parsed.iloc[2]))
        self.assertTrue(pd.isna(parsed.iloc[3]))


class PrepareFeaturesTests(unittest.TestCase):
    def test_prepare_features_extracts_expected_model_inputs(self) -> None:
        df = pd.DataFrame(
            {
                "Month of Period End": ["January 2024", "February 2024"],
                "Region": ["Phoenix", "Tucson"],
                "Median Sale Price": ["$423,000", "$350,000"],
                "Homes Sold": ["250", "175"],
                "Inventory YoY": ["5.0%", "10.0%"],
                "Other Sale Price Signal": ["$410,000", "$340,000"],
            }
        )

        X, y = hp._prepare_features(
            df,
            target_col="Median Sale Price",
            month_col="Month of Period End",
            region_col="Region",
        )

        self.assertListEqual(X.columns.tolist(), ["year", "month", "region", "Homes Sold", "Inventory YoY"])
        self.assertEqual(X.loc[0, "year"], 2024.0)
        self.assertEqual(X.loc[0, "month"], 1.0)
        self.assertEqual(X.loc[0, "region"], "Phoenix")
        self.assertEqual(X.loc[0, "Homes Sold"], 250.0)
        self.assertAlmostEqual(X.loc[0, "Inventory YoY"], 0.05)
        self.assertListEqual(y.tolist(), [423000.0, 350000.0])


class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = hp.train_model(DATA_PATH)

    @classmethod
    def tearDownClass(cls) -> None:
        print()
        print(hp.format_training_summary(cls.result, hp.MODEL_QUALITY_THRESHOLDS))

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

    def test_saved_artifact_can_predict_on_holdout_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            hp.save_model_artifact(self.result, model_path)

            with open(model_path, "rb") as file:
                artifact = pickle.load(file)

        self.assertSetEqual(
            set(artifact.keys()),
            {"pipeline", "target_column", "feature_columns", "metrics"},
        )
        self.assertEqual(artifact["target_column"], self.result.target_column)
        self.assertListEqual(artifact["feature_columns"], self.result.feature_columns)

        sample = self.result.X_test.head(5)
        predictions = artifact["pipeline"].predict(sample)
        self.assertEqual(len(predictions), len(sample))
        self.assertTrue(pd.notna(predictions).all())

    def test_format_training_summary_includes_metrics_and_threshold_status(self) -> None:
        summary = hp.format_training_summary(self.result, hp.MODEL_QUALITY_THRESHOLDS)

        self.assertIn("Training Summary", summary)
        self.assertIn("Metrics", summary)
        self.assertIn("MAPE", summary)
        self.assertIn("R2", summary)
        self.assertIn("PASS", summary)


if __name__ == "__main__":
    unittest.main()
