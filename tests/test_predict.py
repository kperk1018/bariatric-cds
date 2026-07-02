"""Tests for src/predict.py.

All tests mock joblib.load so no real CSV or trained artifacts are needed.
"""
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.config import MODEL_PERFORMANCE


def _make_dummy_meta(n_features: int = 5):
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median").fit([[0] * n_features])
    return {
        "imputer": imp,
        "columns": [f"f{i}" for i in range(n_features)],
        "n_train": 100,
    }


def _make_dummy_model():
    m = MagicMock()
    m.predict.return_value = np.array([20.0])
    return m


def _make_dummy_scaler():
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit([[0] * 5, [1] * 5])
    return sc


class TestPredictTrajectory(unittest.TestCase):

    def _run_predict(self, patient=None):
        """Helper: run predict_trajectory with mocked artifacts."""
        if patient is None:
            patient = {f"f{i}": 1.0 for i in range(5)}
            patient.update({"Sex": "Female", "Race": "White", "Surgery_Type": "RYGB"})

        dummy_meta = _make_dummy_meta()
        dummy_model = _make_dummy_model()
        dummy_scaler = _make_dummy_scaler()

        def _fake_load(path):
            name = str(path)
            if "meta" in name:
                return dummy_meta
            if "scaler" in name:
                return dummy_scaler
            if "background" in name:
                return np.zeros((5, 5))
            return dummy_model

        import src.predict as pred_module
        pred_module._MODEL_CACHE.clear()

        with patch("src.predict._load_artifact", side_effect=_fake_load):
            from src.predict import predict_trajectory
            return predict_trajectory(patient)

    def test_returns_both_outcomes_and_six_years(self):
        result = self._run_predict()
        self.assertIn("TBWL", result)
        self.assertIn("FML", result)
        for outcome in ["TBWL", "FML"]:
            self.assertEqual(set(result[outcome].keys()), set(range(1, 7)))

    def test_red_years_have_point_none(self):
        result = self._run_predict()
        # Red years per config: TBWL yr6, FML yr1, yr5, yr6
        red_pairs = [
            ("TBWL", 6), ("FML", 1), ("FML", 5), ("FML", 6)
        ]
        for outcome, yr in red_pairs:
            self.assertIsNone(
                result[outcome][yr]["point"],
                f"Expected point=None for red year {outcome} yr{yr}",
            )

    def test_uncertainty_band_width_matches_1_96_rmse(self):
        result = self._run_predict()
        # TBWL yr2 is green, predicted point=20.0 (mocked)
        d = result["TBWL"][2]
        rmse = MODEL_PERFORMANCE["TBWL"][2]["rmse"]
        expected_half = 1.96 * rmse
        self.assertAlmostEqual(d["hi"] - d["point"], expected_half, places=1)
        self.assertAlmostEqual(d["point"] - d["lo"], expected_half, places=1)

    def test_gate_dict_embedded_in_every_year(self):
        result = self._run_predict()
        required_keys = {"tier", "r2", "rmse", "allow_point_prediction", "message"}
        for outcome in ["TBWL", "FML"]:
            for yr in range(1, 7):
                self.assertTrue(
                    required_keys.issubset(result[outcome][yr].keys()),
                    f"Gate keys missing for {outcome} yr{yr}",
                )

    def test_missing_artifact_raises_file_not_found(self):
        from unittest.mock import patch
        from pathlib import Path
        import src.predict as pred_module
        pred_module._MODEL_CACHE.clear()
        from src.predict import predict_trajectory
        patient = {
            "Age": 45, "Sex": "Female", "Race": "White",
            "Height": 165, "Initial_BMI": 45.0, "Initial_Weight": 120.0,
            "Initial_BMR": 1700, "Initial_VF": 15.0, "Initial_FATpct": 45.0,
            "Initial_FATMASS": 55.0, "Initial_FFM": 65.0, "Time_to_Surgery": 6,
            "Surgery_Type": "RYGB", "Preop_BMI": 43.0, "Preop_TBWL": 8.0,
        }
        # Simulate missing artifacts by patching Path.exists to always return False
        with patch.object(Path, "exists", return_value=False):
            pred_module._MODEL_CACHE.clear()
            with self.assertRaises(FileNotFoundError) as ctx:
                predict_trajectory(patient)
        self.assertIn("reproduce_models", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
