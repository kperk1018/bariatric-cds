"""Tests for src/what_if.py — uses mocked predict_trajectory."""
import unittest
from unittest.mock import patch


def _fake_traj(patient: dict) -> dict:
    """Returns a fixed fake trajectory; varies TBWL yr2 by Preop_TBWL value."""
    preop = patient.get("Preop_TBWL", 8.0)
    return {
        "TBWL": {
            1: {"point": 20.0 + preop * 0.5, "lo": 10.0, "hi": 30.0,
                "tier": "amber", "r2": 0.262, "rmse": 8.29,
                "allow_point_prediction": True, "message": ""},
            2: {"point": 25.0 + preop * 0.5, "lo": 15.0, "hi": 35.0,
                "tier": "green", "r2": 0.524, "rmse": 7.88,
                "allow_point_prediction": True, "message": ""},
            3: {"point": 27.0, "lo": 17.0, "hi": 37.0,
                "tier": "green", "r2": 0.522, "rmse": 8.27,
                "allow_point_prediction": True, "message": ""},
            4: {"point": 26.0, "lo": 16.0, "hi": 36.0,
                "tier": "green", "r2": 0.459, "rmse": 8.73,
                "allow_point_prediction": True, "message": ""},
            5: {"point": 22.0, "lo": 10.0, "hi": 34.0,
                "tier": "amber", "r2": 0.259, "rmse": 11.70,
                "allow_point_prediction": True, "message": ""},
            6: {"point": None, "lo": None, "hi": None,
                "tier": "red", "r2": 0.102, "rmse": 9.83,
                "allow_point_prediction": False, "message": ""},
        },
        "FML": {yr: {"point": None, "lo": None, "hi": None,
                     "tier": "red", "r2": 0.0, "rmse": 0.0,
                     "allow_point_prediction": False, "message": ""}
                for yr in range(1, 7)},
    }


_PATIENT = {
    "Age": 48, "Sex": "Female", "Race": "White", "Height": 163,
    "Initial_BMI": 41.0, "Initial_Weight": 109.0, "Initial_BMR": 1650,
    "Initial_VF": 12.0, "Initial_FATpct": 48.0, "Initial_FATMASS": 52.0,
    "Initial_FFM": 57.0, "Time_to_Surgery": 8, "Surgery_Type": "Sleeve",
    "Preop_BMI": 39.5, "Preop_TBWL": 7.8,
}


class TestWhatIfAnalysis(unittest.TestCase):

    @patch("src.what_if.predict_trajectory", side_effect=_fake_traj)
    def test_delta_nonzero_when_preop_tbwl_changed(self, _):
        from src.what_if import what_if_analysis
        result = what_if_analysis(_PATIENT, {"Preop_TBWL": 11.5})
        # TBWL yr1 point changes because _fake_traj uses Preop_TBWL
        delta_yr1 = result["deltas"]["TBWL"][1]["delta_point"]
        self.assertIsNotNone(delta_yr1)
        self.assertNotAlmostEqual(delta_yr1, 0.0, places=3)

    @patch("src.what_if.predict_trajectory", side_effect=_fake_traj)
    def test_original_dict_not_mutated(self, _):
        from src.what_if import what_if_analysis
        original_preop = _PATIENT["Preop_TBWL"]
        what_if_analysis(_PATIENT, {"Preop_TBWL": 15.0})
        self.assertEqual(_PATIENT["Preop_TBWL"], original_preop)

    @patch("src.what_if.predict_trajectory", side_effect=_fake_traj)
    def test_red_year_delta_none(self, _):
        from src.what_if import what_if_analysis
        result = what_if_analysis(_PATIENT, {"Preop_TBWL": 11.5})
        self.assertIsNone(result["deltas"]["TBWL"][6]["delta_point"])

    @patch("src.what_if.predict_trajectory", side_effect=_fake_traj)
    def test_modified_features_echoed(self, _):
        from src.what_if import what_if_analysis
        mods = {"Preop_TBWL": 11.5, "Age": 50}
        result = what_if_analysis(_PATIENT, mods)
        self.assertEqual(result["modified_features"], mods)

    @patch("src.what_if.predict_trajectory", side_effect=_fake_traj)
    def test_result_has_original_and_modified(self, _):
        from src.what_if import what_if_analysis
        result = what_if_analysis(_PATIENT, {"Preop_TBWL": 11.5})
        self.assertIn("original", result)
        self.assertIn("modified", result)
        self.assertIn("deltas", result)


if __name__ == "__main__":
    unittest.main()
