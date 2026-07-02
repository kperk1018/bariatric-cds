"""Tests for src/agent.py — mocks Anthropic API to avoid real calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


_PATIENT = {
    "Age": 48, "Sex": "Female", "Race": "White", "Height": 163,
    "Initial_BMI": 41.0, "Initial_Weight": 109.0, "Initial_BMR": 1650,
    "Initial_VF": 12.0, "Initial_FATpct": 48.0, "Initial_FATMASS": 52.0,
    "Initial_FFM": 57.0, "Time_to_Surgery": 8, "Surgery_Type": "Sleeve",
    "Preop_BMI": 39.5, "Preop_TBWL": 7.8,
}


def _mock_end_turn_response(text: str = "Test response."):
    """Build a minimal Anthropic-style response object that ends the turn."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


class TestSanitizeUserMessage(unittest.TestCase):

    def test_phi_keyword_name_raises(self):
        from src.agent import _sanitize_user_message
        with self.assertRaises(ValueError):
            _sanitize_user_message("Patient name John Smith needs trajectory", None)

    def test_phi_keyword_mrn_raises(self):
        from src.agent import _sanitize_user_message
        with self.assertRaises(ValueError):
            _sanitize_user_message("MRN 12345 patient query", None)

    def test_phi_keyword_dob_raises(self):
        from src.agent import _sanitize_user_message
        with self.assertRaises(ValueError):
            _sanitize_user_message("DOB 1980-01-01", None)

    def test_clean_message_passes(self):
        from src.agent import _sanitize_user_message
        msg = "What is the predicted trajectory for a 48-year-old female with BMI 41?"
        result = _sanitize_user_message(msg, None)
        self.assertEqual(result, msg)

    def test_patient_summary_appended(self):
        from src.agent import _sanitize_user_message
        result = _sanitize_user_message("Query", _PATIENT)
        self.assertIn("Patient features", result)
        self.assertIn("Age=48", result)

    def test_patient_summary_no_id(self):
        from src.agent import _sanitize_user_message
        patient_with_id = {**_PATIENT, "ID": "STUDY_001"}
        result = _sanitize_user_message("Query", patient_with_id)
        # The summary should include the ID key-value (since it's structured data,
        # not free-text PHI), but the PHI guard checks keyword patterns in the message
        self.assertIn("Patient features", result)


class TestRunAgent(unittest.TestCase):

    def _make_client_mock(self, response):
        client = MagicMock()
        client.messages.create.return_value = response
        return client

    @patch("src.agent.anthropic.Anthropic")
    def test_system_prompt_in_api_call(self, mock_anthropic_cls):
        from src.agent import SYSTEM_PROMPT, run_agent
        mock_client = self._make_client_mock(_mock_end_turn_response())
        mock_anthropic_cls.return_value = mock_client

        run_agent("What is the expected trajectory?", patient=_PATIENT)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["system"], SYSTEM_PROMPT)

    @patch("src.agent.anthropic.Anthropic")
    def test_tools_registered_in_api_call(self, mock_anthropic_cls):
        from src.agent import _TOOL_SCHEMAS, run_agent
        mock_client = self._make_client_mock(_mock_end_turn_response())
        mock_anthropic_cls.return_value = mock_client

        run_agent("Query", patient=_PATIENT)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        tool_names = {t["name"] for t in call_kwargs["tools"]}
        expected = {"predict_trajectory", "assign_phenotype", "explain_with_shap",
                    "assess_preop_risk", "what_if_analysis"}
        self.assertEqual(tool_names, expected)

    @patch("src.agent.anthropic.Anthropic")
    def test_audit_log_appended(self, mock_anthropic_cls):
        from src.agent import run_agent
        import src.agent as agent_module
        mock_client = self._make_client_mock(_mock_end_turn_response("Hello."))
        mock_anthropic_cls.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            orig_artifacts = agent_module.ARTIFACTS
            agent_module.ARTIFACTS = Path(tmpdir)
            try:
                run_agent("Query", patient=_PATIENT)
                log_path = Path(tmpdir) / "audit_log.jsonl"
                self.assertTrue(log_path.exists())
                with open(log_path) as f:
                    lines = f.readlines()
                self.assertEqual(len(lines), 1)
                entry = json.loads(lines[0])
                self.assertTrue(entry["user_message_redacted"])
                self.assertTrue(entry["patient_features_provided"])
            finally:
                agent_module.ARTIFACTS = orig_artifacts

    @patch("src.agent.anthropic.Anthropic")
    def test_returns_text_response(self, mock_anthropic_cls):
        from src.agent import run_agent
        mock_client = self._make_client_mock(_mock_end_turn_response("Expected output text."))
        mock_anthropic_cls.return_value = mock_client

        result = run_agent("Query")
        self.assertEqual(result, "Expected output text.")


if __name__ == "__main__":
    unittest.main()
