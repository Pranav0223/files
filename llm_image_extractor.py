"""
Module 4b — Image LLM Extractor
SHRI Project · RISHA Lab · IIT Tirupati

Uses Google Gemini Vision API (free tier) to analyse
the illustration present on the page.

Only runs when has_image = True.

Extracts:
  - posture override per character (confirms or corrects text LLM)
  - clothing override per character
  - additional characters visible but not named in text
  - additional events visible but not described in text
"""

import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL   = "gemini-2.0-flash"


def extract_from_image(
    image_path: str,
    known_characters: list,
    story_name: str
) -> dict:
    """
    Main image LLM extraction function.

    Args:
        image_path: path to the scanned page image
        known_characters: list of character_ids already found by text pipeline
        story_name: story name for context

    Returns:
        {
            "character_overrides": [
                {
                    "character_id": str,
                    "posture_override": str or null,
                    "clothing_override": str or null
                }
            ],
            "additional_characters": [
                {
                    "description": str,
                    "posture": str,
                    "clothing": str
                }
            ],
            "additional_events": [
                {
                    "action_type": str,
                    "action": str,
                    "action_modifier": str,
                    "duration": str,
                    "participants": [ { "character": str, "action": str } ]
                }
            ]
        }
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    known_str = ", ".join(known_characters) if known_characters else "none"

    prompt = f"""This is an illustration from a Hindu scripture story: {story_name}.

Known characters already identified from text: {known_str}

Analyse this image and return ONLY valid JSON, no explanation:

{{
  "character_overrides": [
    {{
      "character_id": "<match to known character or describe if unknown>",
      "posture_override": "<what the character is physically doing — only if clearly visible>",
      "clothing_override": "<clothing description — only if clearly visible and specific>"
    }}
  ],
  "additional_characters": [
    {{
      "description": "<describe the character type e.g. soldier, horse, guard>",
      "posture": "<what they are doing>",
      "clothing": "<what they are wearing>"
    }}
  ],
  "additional_events": [
    {{
      "action_type": "<locomotion | reaction | interaction | gesture | transformation>",
      "action": "<what is happening in the image not described in text>",
      "action_modifier": "<tone or manner>",
      "duration": "<continuous | instantaneous | gradual>",
      "participants": [
        {{ "character": "<character description>", "action": "<their specific action>" }}
      ]
    }}
  ]
}}

Only include fields where something is clearly visible. Use empty arrays if nothing to add."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt
        ]
    )
    raw_response = response.text.strip()

    # Strip markdown fences if present
    if raw_response.startswith("```"):
        raw_response = raw_response.split("```")[1]
        if raw_response.startswith("json"):
            raw_response = raw_response[4:]
    raw_response = raw_response.strip()

    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as e:
        # Return empty structure if parsing fails
        print(f"Warning: Image LLM returned invalid JSON: {e}")
        return {
            "character_overrides": [],
            "additional_characters": [],
            "additional_events": []
        }

    return result


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python llm_image_extractor.py <image_path>")
        sys.exit(1)

    result = extract_from_image(
        sys.argv[1],
        known_characters=["kansa", "devaki", "vasudeva"],
        story_name="Krishna Janma Katha"
    )
    print(json.dumps(result, indent=2))
