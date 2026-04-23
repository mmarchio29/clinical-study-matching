"""
chat_intake.py — Conversational RAG with full agent eligibility loop.
"""

import os
import json
from dotenv import load_dotenv
from retriever import (
    extract_patient_fields,
    run_agent,
    chat_search_and_answer,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REQUIRED_FIELDS = {
    "age": "your age",
    "conditions": "any health conditions you have (or let us know if you're a healthy volunteer)",
    "sex": "your sex",
}

def get_missing_fields(profile: dict) -> list[str]:
    missing = []
    if not profile.get("age"):
        missing.append(REQUIRED_FIELDS["age"])
    if not profile.get("conditions") and not profile.get("healthy_volunteer"):
        missing.append(REQUIRED_FIELDS["conditions"])
    if not profile.get("sex"):
        missing.append(REQUIRED_FIELDS["sex"])
    return missing

def merge_profile(existing: dict, new_fields: dict) -> dict:
    """Merge extracted fields into running patient profile, never overwriting with None."""
    for k, v in new_fields.items():
        if v is None:
            continue
        if isinstance(v, list) and not v:
            continue
        if isinstance(v, dict):
            existing[k] = {**existing.get(k, {}), **{kk: vv for kk, vv in v.items() if vv is not None}}
        else:
            existing[k] = v
    return existing

SIMPLIFY_SYSTEM = """You are a friendly clinical trial intake assistant.
Convert this list of clinical missing information into simple, conversational
questions a non-medical person can answer. 

Rules:
- Use plain English, no medical jargon
- Only include questions the person can reasonably answer themselves
- Skip anything requiring a doctor to measure (liver function, renal panels,
  SCID, LSAS, HDRS, HAM scores, MINI assessments, olfactory tests, etc)
- Skip anything about willingness, consent, or ability to attend
- Max 3 questions
- Return JSON only: {"questions": ["question1", "question2", ...]}

Examples:
  "HAM-A Total Score" -> skip
  "SCID-5-CT confirmation" -> skip
  "Willingness to provide consent" -> skip
  "Risk for suicidal ideation" -> "Have you ever had thoughts of self-harm?"
  "Confirmation of GAD diagnosis" -> "Have you ever been formally diagnosed with an anxiety disorder by a doctor?"
  "Insulin use" -> "Are you currently taking insulin?"
  "Pregnancy or breastfeeding status" -> "Are you currently pregnant or breastfeeding?"
  "English language proficiency" -> skip
"""

def simplify_missing_fields(missing_items: list[str]) -> list[str]:
    if not missing_items:
        return []
    from retriever import client
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SIMPLIFY_SYSTEM},
            {"role": "user", "content": json.dumps(missing_items)},
        ],
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content).get("questions", [])

def run_chat() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    print("Clinical Trial Matcher")
    print("Tell me about yourself — condition, age, sex, and any preferences.")
    print("Type 'exit' to quit.\n")

    conversation_turns: list[str] = []
    patient_profile: dict = {}
    agent_run = False
    answered_questions: set[str] = set()

    while True:
        user_msg = input("You: ").strip()
        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            break

        conversation_turns.append(f"User: {user_msg}")
        context = "\n".join(conversation_turns[-8:])

        # Extract and accumulate structured fields
        extracted = extract_patient_fields(user_msg, context)
        patient_profile = merge_profile(patient_profile, extracted)

        missing = get_missing_fields(patient_profile)

        if missing:
            # Still missing required fields — ask for all of them at once
            ask = ", ".join(missing[:-1]) + (f" and {missing[-1]}" if len(missing) > 1 else missing[0])
            print(f"\nAssistant: To find your best trial matches I also need {ask}.\n")

        elif not agent_run:
            # Have everything — run full agent
            conditions = patient_profile.get("conditions", [])
            primary_diagnosis = ", ".join(conditions) if conditions else "Healthy volunteer"

            structured_patient = {
                "patient_id":          "CHAT_USER",
                "age":                 patient_profile.get("age"),
                "sex":                 patient_profile.get("sex", "Unknown"),
                "primary_diagnosis":   primary_diagnosis,
                "lab_values":          patient_profile.get("lab_values", {}),
                "current_medications": patient_profile.get("medications", []),
                "prior_treatments":    patient_profile.get("prior_treatments", []),
                "comorbidities":       [],
                "exclusion_flags":     {},
            }

            result = run_agent(structured_patient, verbose=False)
            agent_run = True

            verdicts = result.get("verdicts", [])
            eligible = [v for v in verdicts if v["verdict"] == "eligible"]
            uncertain = [v for v in verdicts if v["verdict"] == "uncertain"]

            if eligible:
                print("\nAssistant: Based on what you've shared, here are trials you may qualify for:\n")
                for v in eligible:
                    print(f"  ✓ {v['trial_title']} ({v['nct_id']})")
                    print(f"    {v['summary']}\n")
            elif uncertain:
                print("\nAssistant: I found some potential matches based on your profile:\n")
                for v in uncertain[:3]:
                    print(f"  ? {v['trial_title']} ({v['nct_id']})")
                    print(f"    {v['summary']}\n")
                agent_run = False

            elif not eligible and not uncertain:
                print("\nAssistant: I wasn't able to find trials you clearly qualify for. "
                    "You may want to consult a physician or search clinicaltrials.gov directly.\n")
                
        conversation_turns.append(f"Assistant: [response given]")

if __name__ == "__main__":
    run_chat()