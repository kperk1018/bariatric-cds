"""Tests for src/predict.py.

All tests mock joblib.load so no real CSV or trained artifacts are needed.
"""
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


def _make_dummy_direct_meta(n_features: int = 5):
    from sklearn.impute import SimpleImputer
    cols = [f"f{i}" for i in range(n_features)]
    imp = SimpleImputer(strategy="median").fit([[0] * n_features])
    return {
        "columns": cols,
        "imputer": imp,
        "num_encoded": cols,
        "scaler": None,
        "feature_cols": cols,
        "estimator": "RF",
        "n_train": 100,
    }


def _make_dummy_scaler():
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit([[0] * 5, [1] * 5])
    return sc


class TestPredictTrajectory(unittest.TestCase):

    def _run_predict(self, patient=None, calibration=None, postop_tbwl=None):
        """Helper: run predict_trajectory with mocked artifacts.

        calibration=None mocks the no-calibration fallback (±1.96×RMSE bands);
        pass a dict to exercise the calibrated-band path.
        """
        if patient is None:
            patient = {f"f{i}": 1.0 for i in range(5)}
            patient.update({"Sex": "Female", "Race": "White", "Surgery_Type": "RYGB"})

        dummy_meta = _make_dummy_meta()
        dummy_model = _make_dummy_model()
        dummy_scaler = _make_dummy_scaler()

        def _fake_load(path):
            name = str(path)
            if "direct_meta" in name:
                return _make_dummy_direct_meta()
            if "meta" in name:
                return dummy_meta
            if "scaler" in name:
                return dummy_scaler
            if "background" in name:
                return np.zeros((5, 5))
            return dummy_model

        import src.predict as pred_module
        pred_module._MODEL_CACHE.clear()

        with patch("src.predict._load_artifact", side_effect=_fake_load), \
             patch("src.predict._load_calibration", return_value=calibration):
            from src.predict import predict_trajectory
            return predict_trajectory(patient, postop_tbwl=postop_tbwl)

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

    def test_calibrated_band_used_in_cascade_mode(self):
        # yr2+ without postop data = cascade mode -> empirical OOF quantiles
        calib = {
            "min_calib": 30,
            "per_year": {
                ("TBWL", 2): {
                    "n_oof": 399, "cascade_r2": 0.179,
                    "resid_q025": -20.0, "resid_q975": 15.0,
                },
            },
        }
        result = self._run_predict(calibration=calib)
        d = result["TBWL"][2]
        self.assertEqual(d["mode"], "cascade")
        self.assertEqual(d["band_source"], "calibrated_oof")
        self.assertAlmostEqual(d["lo"], d["point"] - 20.0, places=1)
        self.assertAlmostEqual(d["hi"], d["point"] + 15.0, places=1)
        self.assertEqual(d["cascade_r2"], 0.179)

    def test_calibration_skipped_when_n_oof_too_small(self):
        calib = {
            "min_calib": 30,
            "per_year": {
                ("TBWL", 2): {
                    "n_oof": 12, "cascade_r2": 0.1,
                    "resid_q025": -20.0, "resid_q975": 15.0,
                },
            },
        }
        result = self._run_predict(calibration=calib)
        d = result["TBWL"][2]
        self.assertEqual(d["band_source"], "s5_rmse")
        rmse = MODEL_PERFORMANCE["TBWL"][2]["rmse"]
        self.assertAlmostEqual(d["hi"] - d["point"], 1.96 * rmse, places=1)

    def test_conditioned_mode_with_actual_postop_lag(self):
        # yr1 actual supplied -> yr2 runs under training-equivalent conditions
        result = self._run_predict(postop_tbwl={1: 22.5})
        self.assertEqual(result["TBWL"][1]["mode"], "conditioned")  # yr1 has no lag
        self.assertEqual(result["TBWL"][2]["mode"], "conditioned")
        self.assertEqual(result["TBWL"][2]["band_source"], "s5_rmse")
        self.assertIsNone(result["TBWL"][2]["cascade_r2"])
        # yr3's lag is the *predicted* yr2 -> back to cascade
        self.assertEqual(result["TBWL"][3]["mode"], "cascade")

    def test_direct_model_adopted_when_it_beat_cascade_oof(self):
        calib = {
            "min_calib": 30,
            "per_year": {
                ("TBWL", 2): {
                    "n_oof": 399, "cascade_r2": 0.179,
                    "resid_q025": -20.0, "resid_q975": 15.0,
                },
            },
            "direct_per_year": {
                ("TBWL", 2): {
                    "n_oof": 399, "direct_r2": 0.259, "direct_rmse": 9.99,
                    "resid_q025": -18.0, "resid_q975": 14.0,
                    "config": "Fexp+RF",
                },
            },
        }
        result = self._run_predict(calibration=calib)
        d = result["TBWL"][2]
        self.assertEqual(d["mode"], "direct")
        self.assertEqual(d["band_source"], "direct_oof")
        self.assertAlmostEqual(d["lo"], d["point"] - 18.0, places=1)
        self.assertAlmostEqual(d["hi"], d["point"] + 14.0, places=1)
        self.assertEqual(d["cascade_r2"], 0.259)

    def test_direct_model_rejected_when_cascade_was_better(self):
        # e.g. TBWL yr5 / FML yr3: direct did NOT beat cascade OOF -> keep cascade
        calib = {
            "min_calib": 30,
            "per_year": {
                ("TBWL", 2): {
                    "n_oof": 399, "cascade_r2": 0.179,
                    "resid_q025": -20.0, "resid_q975": 15.0,
                },
            },
            "direct_per_year": {
                ("TBWL", 2): {
                    "n_oof": 399, "direct_r2": 0.150, "direct_rmse": 11.0,
                    "resid_q025": -22.0, "resid_q975": 16.0,
                    "config": "F15+RF",
                },
            },
        }
        result = self._run_predict(calibration=calib)
        d = result["TBWL"][2]
        self.assertEqual(d["mode"], "cascade")
        self.assertEqual(d["band_source"], "calibrated_oof")

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
