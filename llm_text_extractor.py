"""
Module 4a — Text LLM Extractor
SHRI Project · RISHA Lab · IIT Tirupati

Uses Groq API (free) with Llama 3.1 70B to:
  1. Identify which story the page belongs to
  2. Confirm and match character names to scaffold character_ids
  3. Extract scene-specific fields: narrative_role, posture, clothing_override
  4. Extract events: action_type, action, action_modifier, duration, participants

Get free Groq API key at: https://console.groq.com
No credit card required.

The LLM never invents character attributes — it only extracts what is
in the text and matches to scaffold entries.
"""

import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL   = "llama-3.3-70b-versatile"


def extract_from_text(
    raw_text: str,
    candidate_names: list,
    index_summary: list
) -> dict:
    """
    Main LLM text extraction function.

    Args:
        raw_text: cleaned OCR text from page
        candidate_names: character names found by rule-based extractor
        index_summary: list of known stories from scaffold index

    Returns:
        {
            "story_id": str,
            "skandha": str,
            "characters": [
                {
                    "name": str,
                    "character_id": str,
                    "narrative_role": str,
                    "posture": str,
                    "clothing_override": str or null
                }
            ],
            "events": [
                {
                    "id": str,
                    "sequence_order": int,
                    "source_text": str,
                    "action_type": str,
                    "action": str,
                    "action_modifier": str,
                    "duration": str,
                    "participants": [
                        { "character": str, "action": str }
                    ]
                }
            ]
        }
    """

    stories_str    = json.dumps(index_summary, indent=2)
    candidates_str = ", ".join(candidate_names) if candidate_names else "none detected"

    prompt = f"""You are processing a page from Srimad Bhagavatam.

OCR extracted text from the page:
\"\"\"{raw_text}\"\"\"

Character names detected by rule-based NER: {candidates_str}

Known stories in the scaffold library:
{stories_str}

Return ONLY valid JSON matching this exact structure.
Do NOT invent any values. Only extract what is present in the text.
Do NOT generate character physical attributes.

{{
  "story_id": "<match to one of the story_ids above>",
  "skandha": "<skandha number as string>",
  "characters": [
    {{
      "name": "<exact name as it appears in text>",
      "character_id": "<match to scaffold character_id, snake_case if unknown>",
      "narrative_role": "<protagonist | antagonist | supporting>",
      "posture": "<what the character is physically doing in this scene>",
      "clothing_override": "<only if text explicitly mentions clothing, else null>"
    }}
  ],
  "events": [
    {{
      "id": "event_1",
      "sequence_order": 1,
      "source_text": "<exact sentence this event comes from>",
      "action_type": "<locomotion | reaction | interaction | gesture | transformation>",
      "action": "<core action description>",
      "action_modifier": "<tone or manner of action>",
      "duration": "<continuous | instantaneous | gradual>",
      "participants": [
        {{
          "character": "<character_id>",
          "action": "<what this character specifically does>"
        }}
      ]
    }}
  ]
}}"""

    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a structured data extractor for Hindu scripture text. Return only valid JSON, no explanation, no markdown fences."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=1500
    )

    raw_response = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them
    if raw_response.startswith("```"):
        raw_response = raw_response.split("```")[1]
        if raw_response.startswith("json"):
            raw_response = raw_response[4:]
    raw_response = raw_response.strip()

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nResponse: {raw_response}")

    return result


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from scaffold_loader import get_index_summary
    from rule_extractor import extract_candidates

    sample_text = """
    Long ago, Princess Devaki and her brother Prince Kansa lived in Mathura.
    Devaki was kind while Kansa was cruel. Soon, Devaki was married to a
    nobleman, Vasudeva. One day, while Kansa was taking Devaki and Vasudeva
    to the palace, a divine voice said, Kansa will be killed by Devaki's
    eighth son. Kansa was terrified and drew his sword.
    """

    candidates = extract_candidates(sample_text)
    index      = get_index_summary()
    result     = extract_from_text(sample_text, candidates, index)
    print(json.dumps(result, indent=2))
