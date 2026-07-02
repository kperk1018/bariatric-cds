"""Tests for src/explain.py — no real artifacts required (mocked)."""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np


def _make_dummy_meta(n: int = 5):
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median").fit([[0] * n])
    return {"imputer": imp, "columns": [f"f{i}" for i in range(n)], "n_train": 50}


def _make_dummy_model():
    m = MagicMock()
    m.predict.return_value = np.array([20.0])
    return m


class TestExplainRedYearRefused(unittest.TestCase):

    def test_tbwl_yr6_raises_value_error(self):
        from src.explain import explain_with_shap
        patient = {f"f{i}": 1.0 for i in range(5)}
        with self.assertRaises(ValueError) as ctx:
            explain_with_shap(patient, "TBWL", 6)
        self.assertIn("RED", str(ctx.exception))

    def test_fml_yr1_raises_value_error(self):
        from src.explain import explain_with_shap
        patient = {f"f{i}": 1.0 for i in range(5)}
        with self.assertRaises(ValueError):
            explain_with_shap(patient, "FML", 1)

    def test_fml_yr5_raises_value_error(self):
        from src.explain import explain_with_shap
        patient = {f"f{i}": 1.0 for i in range(5)}
        with self.assertRaises(ValueError):
            explain_with_shap(patient, "FML", 5)


class TestExplainStructure(unittest.TestCase):

    def _run_explain(self, outcome="TBWL", year=2, top_n=5):
        from src.explain import explain_with_shap
        import src.explain as expl_module
        expl_module._CACHE.clear()

        patient = {f"f{i}": 1.0 for i in range(5)}
        dummy_meta = _make_dummy_meta()
        dummy_model = _make_dummy_model()

        shap_vals = np.array([[1.5, -0.8, 0.3, -0.2, 0.1]])

        mock_tree_explainer = MagicMock()
        mock_tree_explainer.shap_values.return_value = shap_vals
        mock_tree_explainer.expected_value = 18.0

        def _fake_load(filename):
            if "meta" in str(filename):
                return dummy_meta
            if "scaler" in str(filename):
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler().fit([[0]*5])
                return sc
            if "background" in str(filename):
                return np.zeros((5, 5))
            return dummy_model

        with patch("src.explain._load", side_effect=_fake_load), \
             patch("shap.TreeExplainer", return_value=mock_tree_explainer):
            return explain_with_shap(patient, outcome, year, top_n=top_n)

    def test_returns_required_keys(self):
        result = self._run_explain()
        required = {"outcome", "year", "tier", "base_value", "top_positive", "top_negative"}
        self.assertTrue(required.issubset(result.keys()))

    def test_top_n_respected(self):
        result = self._run_explain(top_n=2)
        self.assertLessEqual(len(result["top_positive"]), 2)
        self.assertLessEqual(len(result["top_negative"]), 2)

    def test_magnitudes_nonnegative(self):
        result = self._run_explain()
        for d in result["top_positive"] + result["top_negative"]:
            self.assertGreaterEqual(d["magnitude"], 0.0)

    def test_direction_consistent_with_shap_value(self):
        result = self._run_explain()
        for d in result["top_positive"]:
            self.assertGreaterEqual(d["shap_value"], 0.0)
        for d in result["top_negative"]:
            self.assertLess(d["shap_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
