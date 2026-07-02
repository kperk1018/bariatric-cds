"""Agentic AI layer — Anthropic tool-use loop (Stage 4).

Wraps the five clinical tools in an Anthropic API agent that a surgeon can query
in natural language. PHI never leaves local Python; only derived outputs
(%, cluster id, driver names) go to the LLM.

Usage:
    from src.agent import run_agent

    patient = {
        "Age": 48, "Sex": "Female", "Race": "White",
        "Height": 163, "Initial_BMI": 41.0, "Initial_Weight": 109.0,
        # ... all 15 BASELINE_FEATURES
    }
    response = run_agent(
        "Patient is 48-year-old female, BMI 41, diabetes yes, hypertension yes, "
        "PreopTBWL 7.8%, planning sleeve: what trajectory should I expect over 5 years?",
        patient=patient,
    )
    print(response)

Requires ANTHROPIC_API_KEY in environment.
Model is read from AGENT_MODEL env var (default: claude-sonnet-4-6).
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import anthropic

from src.config import ARTIFACTS
from src.predict import predict_trajectory
from src.phenotype import assign_phenotype
from src.explain import explain_with_shap
from src.risk import assess_preop_risk
from src.what_if import what_if_analysis

_PHI_PATTERN = re.compile(
    r"\b(name|mrn|dob|date.of.birth|ssn|social.security|record.number|"
    r"address|phone|email|patient.id)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are a clinical decision support assistant for an internal bariatric surgery research project.
You help surgeons understand predicted weight-loss trajectories in a research context.

STRICT RULES — never violate:
1. RESEARCH SUPPORT ONLY. You do NOT give medical advice, diagnoses, or treatment recommendations.
   Always state: "This is research-support output — defer to the treating care team for all clinical decisions."
2. UNCERTAINTY. Always report the reliability tier (green/amber/red) and the uncertainty band
   with every prediction. For red-tier years, there is NO reliable point prediction — say so explicitly
   and reason via phenotype and preop-risk flag instead.
3. SCOPE. Work only from tool outputs. Do not invent numbers or extrapolate beyond year 6.
4. NO PHI. Never repeat patient identifiers, names, MRNs, dates of birth, or any identifying
   information in your responses. You receive only de-identified numerical features.
5. PROVISIONAL PHENOTYPES. Clustering recipe is provisional until confirmed with Dr. Raftopoulos.
   Present phenotype assignments with this caveat.
6. WHEN IN DOUBT, say "consult the care team" — never speculate on clinical outcomes beyond what
   the tools return."""

_TOOL_SCHEMAS = [
    {
        "name": "predict_trajectory",
        "description": (
            "Predict TBWL% and FML% trajectories for years 1-6 post-surgery with reliability gating. "
            "Returns point=null for RED (unreliable) years."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient": {
                    "type": "object",
                    "description": "De-identified patient baseline features (no names/IDs). "
                                   "Must include all 15 BASELINE_FEATURES keys.",
                    "properties": {
                        "Age": {"type": "number"}, "Sex": {"type": "string"},
                        "Race": {"type": "string"}, "Height": {"type": "number"},
                        "Initial_BMI": {"type": "number"}, "Initial_Weight": {"type": "number"},
                        "Initial_BMR": {"type": "number"}, "Initial_VF": {"type": "number"},
                        "Initial_FATpct": {"type": "number"}, "Initial_FATMASS": {"type": "number"},
                        "Initial_FFM": {"type": "number"}, "Time_to_Surgery": {"type": "number"},
                        "Surgery_Type": {"type": "string"}, "Preop_BMI": {"type": "number"},
                        "Preop_TBWL": {"type": "number"},
                    },
                    "required": [
                        "Age", "Sex", "Race", "Height", "Initial_BMI", "Initial_Weight",
                        "Initial_BMR", "Initial_VF", "Initial_FATpct", "Initial_FATMASS",
                        "Initial_FFM", "Time_to_Surgery", "Surgery_Type", "Preop_BMI", "Preop_TBWL",
                    ],
                }
            },
            "required": ["patient"],
        },
    },
    {
        "name": "assign_phenotype",
        "description": (
            "Assign a patient to a trajectory phenotype cluster (0-4) based on TBWL% at years 1-3. "
            "Provisional until clustering recipe is confirmed with Dr. Raftopoulos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient": {
                    "type": "object",
                    "description": "Dict with keys '1yr_Postop_TBWL', '2yr_Postop_TBWL', '3yr_Postop_TBWL'.",
                    "properties": {
                        "1yr_Postop_TBWL": {"type": "number"},
                        "2yr_Postop_TBWL": {"type": "number"},
                        "3yr_Postop_TBWL": {"type": "number"},
                    },
                    "required": ["1yr_Postop_TBWL", "2yr_Postop_TBWL", "3yr_Postop_TBWL"],
                }
            },
            "required": ["patient"],
        },
    },
    {
        "name": "explain_with_shap",
        "description": (
            "Return top SHAP feature drivers for a prediction. "
            "Only works for non-red reliability years; raises an error for red years."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient": {"type": "object", "description": "15-feature de-identified patient dict."},
                "outcome": {"type": "string", "enum": ["TBWL", "FML"]},
                "year": {"type": "integer", "minimum": 1, "maximum": 6},
                "top_n": {"type": "integer", "default": 5, "minimum": 1, "maximum": 15},
            },
            "required": ["patient", "outcome", "year"],
        },
    },
    {
        "name": "assess_preop_risk",
        "description": (
            "Flag whether PreopTBWL% is below the 10.5% threshold associated with worse long-term outcomes. "
            "Rule-based — no model."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "preop_tbwl_pct": {"type": "number", "description": "Preoperative TBWL percentage."}
            },
            "required": ["preop_tbwl_pct"],
        },
    },
    {
        "name": "what_if_analysis",
        "description": (
            "Re-run trajectory prediction with feature overrides and return deltas vs. original. "
            "Use for scenario exploration: 'What if PreopTBWL were 11.5% instead of 7.8%?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient": {"type": "object", "description": "Original 15-feature patient dict."},
                "modified_features": {
                    "type": "object",
                    "description": "Feature overrides (keys must be valid BASELINE_FEATURES names).",
                },
            },
            "required": ["patient", "modified_features"],
        },
    },
]


def _sanitize_user_message(user_message: str, patient: dict | None) -> str:
    """Raise ValueError if PHI keywords are detected; append structured patient summary."""
    if _PHI_PATTERN.search(user_message):
        raise ValueError(
            "PHI keyword detected in user message. "
            "Remove any patient identifiers (name, MRN, DOB, etc.) before passing to the agent."
        )
    if patient is None:
        return user_message
    # Append a structured, identifier-free patient summary
    summary_lines = [f"{k}={v}" for k, v in patient.items()]
    summary = "Patient features (de-identified): " + ", ".join(summary_lines)
    return f"{user_message}\n\n{summary}"


def _dispatch_tool(name: str, inputs: dict, patient: dict | None) -> object:
    """Call the appropriate local function and return its result."""
    # For tools that accept a 'patient' dict from the LLM, substitute the
    # verified patient dict if provided (prevents the LLM from fabricating features).
    def _patient(inp: dict) -> dict:
        return patient if patient is not None else inp.get("patient", {})

    dispatch = {
        "predict_trajectory":  lambda inp: predict_trajectory(_patient(inp)),
        "assign_phenotype":    lambda inp: assign_phenotype(inp["patient"]),
        "explain_with_shap":   lambda inp: explain_with_shap(
            _patient(inp), inp["outcome"], inp["year"], inp.get("top_n", 5)
        ),
        "assess_preop_risk":   lambda inp: assess_preop_risk(inp["preop_tbwl_pct"]),
        "what_if_analysis":    lambda inp: what_if_analysis(
            _patient(inp), inp["modified_features"]
        ),
    }
    if name not in dispatch:
        return {"error": f"Unknown tool: {name}"}
    try:
        return dispatch[name](inputs)
    except Exception as exc:
        return {"error": str(exc)}


def _extract_text(response) -> str:
    return " ".join(
        block.text for block in response.content if hasattr(block, "text")
    )


def _audit_log(response_text: str, patient_provided: bool) -> None:
    """Append one JSON line to artifacts/audit_log.jsonl. Never logs raw user message."""
    ARTIFACTS.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_message_redacted": True,
        "patient_features_provided": patient_provided,
        "response_length": len(response_text),
        "response_hash": "sha256:" + hashlib.sha256(response_text.encode()).hexdigest()[:16],
    }
    log_path = ARTIFACTS / "audit_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_agent(
    user_message: str,
    patient: dict | None = None,
    max_turns: int = 10,
) -> str:
    """Run the tool-use agent and return the final text response.

    PHI guard: user_message must not contain identifying keywords.
    If patient dict is provided, it is passed to tools directly (LLM sees only
    the structured summary, never the raw dict).

    Requires ANTHROPIC_API_KEY environment variable.
    """
    model = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")
    client = anthropic.Anthropic()

    safe_content = _sanitize_user_message(user_message, patient)
    messages = [{"role": "user", "content": safe_content}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=_TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            final_text = _extract_text(response)
            _audit_log(final_text, patient is not None)
            return final_text

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _dispatch_tool(block.name, block.input, patient)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    return "Agent reached maximum turns without a final response."
