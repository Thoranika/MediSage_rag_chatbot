"""Medical safety boundaries, emergency triage detection, and educational disclaimers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EDUCATIONAL_DISCLAIMER: str = (
    "Educational Information Only: MediSage is an evidence-grounded reference assistant "
    "based solely on your uploaded documents. It does not provide medical diagnoses, "
    "prescriptions, or personalized treatment advice, and is not a substitute for a "
    "qualified healthcare professional."
)

_EMERGENCY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "chest_pain_or_cardiac",
        re.compile(
            r"\b(crushing chest pain|chest pain|chest tightness|heart attack|"
            r"pain radiating to (?:my |the )?(?:left )?arm)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "severe_breathing",
        re.compile(
            r"\b(can'?t breathe|cannot breathe|difficulty breathing|severe shortness of breath|"
            r"struggling to breathe|turning blue|cyanosis)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "stroke_symptoms",
        re.compile(
            r"\b(face drooping|slurred speech|sudden numbness|sudden weakness on one side|"
            r"having a stroke|stroke symptoms right now)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "anaphylaxis_or_bleeding",
        re.compile(
            r"\b(anaphylaxis|throat swelling|tongue swelling|severe bleeding|won'?t stop bleeding|"
            r"unconscious|unresponsive|seizure right now)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "overdose_or_self_harm",
        re.compile(
            r"\b(overdose|took too many pills|suicidal|kill myself|end my life|self[- ]harm)\b",
            re.IGNORECASE,
        ),
    ),
]

_DIAGNOSIS_PATTERN = re.compile(
    r"\b(diagnose me|do i have|what disease do i have|am i suffering from|"
    r"tell me if i have|confirm my diagnosis)\b",
    re.IGNORECASE,
)

_PERSONAL_DOSAGE_PATTERN = re.compile(
    r"\b(what dose should i take|how much should i take|my dosage|prescribe me|"
    r"can i take \d+|should i increase my dose|should i stop taking|"
    r"can i stop my medication|stop taking my)\b",
    re.IGNORECASE,
)

_PERSONAL_TREATMENT_PATTERN = re.compile(
    r"\b(what medication should i start|which drug is best for me|"
    r"treat my condition|what should i take for my)\b",
    re.IGNORECASE,
)


@dataclass
class SafetyAssessment:
    """Result of evaluating a user question against medical safety guardrails."""

    is_emergency: bool = False
    requires_safety_boundary: bool = False
    flags: list[str] = field(default_factory=list)
    override_response: str = ""
    system_guidance: str = ""


def assess_query_safety(query: str) -> SafetyAssessment:
    """Assess user query for emergencies, diagnosis requests, or personalized dosing."""
    text = (query or "").strip()
    if not text:
        return SafetyAssessment()

    for label, pattern in _EMERGENCY_PATTERNS:
        if pattern.search(text):
            return SafetyAssessment(
                is_emergency=True,
                requires_safety_boundary=True,
                flags=["emergency", label],
                override_response=(
                    "### Urgent Medical Safety Notice\n\n"
                    "If you or someone nearby is experiencing symptoms such as chest pain, "
                    "difficulty breathing, stroke warning signs, severe allergic reaction, "
                    "uncontrolled bleeding, overdose, or thoughts of self-harm, **please contact "
                    "your local emergency services (e.g., 911 / 112 / 999) or go to the nearest "
                    "emergency department immediately.**\n\n"
                    "MediSage is an educational document-reference assistant and cannot assess "
                    "emergencies or provide urgent medical care."
                ),
            )

    flags: list[str] = []
    directives: list[str] = []

    if _DIAGNOSIS_PATTERN.search(text):
        flags.append("personal_diagnosis_request")
        directives.append(
            "The user appears to be asking for a personal medical diagnosis. Explicitly state "
            "that MediSage cannot diagnose medical conditions and advise consulting a qualified "
            "healthcare professional for evaluation."
        )

    if _PERSONAL_DOSAGE_PATTERN.search(text):
        flags.append("personalized_dosage_or_cessation_request")
        directives.append(
            "The user appears to be asking for personalized dosing, a prescription, or whether "
            "to start/stop medication. Explicitly decline to prescribe, recommend a personal dose, "
            "or advise stopping/starting treatment, and advise consulting their clinician or pharmacist."
        )

    if _PERSONAL_TREATMENT_PATTERN.search(text):
        flags.append("personalized_treatment_request")
        directives.append(
            "The user appears to be asking for a personalized treatment choice. Explain clearly "
            "that treatment selection depends on an individual clinical assessment by a licensed clinician."
        )

    if flags:
        guidance = (
            "IMPORTANT SAFETY BOUNDARY:\n"
            + "\n".join(f"- {d}" for d in directives)
            + "\n- Only if general educational facts exist in the retrieved context, summarize them "
            "neutrally as general reference information with [Source: filename, page N] citations; "
            "never tailor recommendations to an individual patient."
        )
        return SafetyAssessment(
            is_emergency=False,
            requires_safety_boundary=True,
            flags=flags,
            system_guidance=guidance,
        )

    return SafetyAssessment()


def build_safety_footer() -> str:
    """Standard concise disclaimer appended or rendered alongside responses."""
    return (
        "*Disclaimer: Educational reference only based on uploaded documents — not a medical "
        "diagnosis, personalized treatment recommendation, or substitute for professional clinical care.*"
    )
