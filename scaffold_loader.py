"""
Module 2 — Scaffold Loader
SHRI Project · RISHA Lab · IIT Tirupati

Loads pre-built scaffold files for stories and characters.
These scaffolds contain pre-loaded attributes from:
  - Shilpa Shastra (iconographic features)
  - Pancharatra Agama (being type, manifestation)
  - Srimad Bhagavatam 12th Skandha (yuga, scale)
"""

import json
import os


# Path to scaffold directory — adjust as needed
SCAFFOLD_DIR = os.path.join(os.path.dirname(__file__), "scaffolds")


def load_index() -> dict:
    """
    Load the scaffold index file.
    Contains all known stories and their key terms for identification.
    """
    index_path = os.path.join(SCAFFOLD_DIR, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Scaffold index not found at {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_story(story_id: str) -> dict:
    """
    Load the story scaffold for a given story_id.

    Args:
        story_id: e.g. "gajendra_moksham", "vamana_charitam"

    Returns:
        Full story scaffold dict
    """
    story_path = os.path.join(SCAFFOLD_DIR, "stories", f"{story_id}.json")
    if not os.path.exists(story_path):
        raise FileNotFoundError(f"Story scaffold not found: {story_id}")
    with open(story_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_character(character_id: str) -> dict:
    """
    Load the character scaffold for a given character_id.

    Args:
        character_id: e.g. "vishnu", "gajendra", "vamana"

    Returns:
        Full character scaffold dict
    """
    char_path = os.path.join(SCAFFOLD_DIR, "characters", f"{character_id}.json")
    if not os.path.exists(char_path):
        return None  # Character not in scaffold — will be handled gracefully
    with open(char_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_for_story(story_id: str) -> dict:
    """
    Load story scaffold + all character scaffolds for that story.

    Args:
        story_id: story identifier

    Returns:
        {
            "story": story scaffold dict,
            "characters": { character_id: character scaffold dict }
        }
    """
    story = load_story(story_id)

    characters = {}
    for char_id in story.get("characters", []):
        char_data = load_character(char_id)
        if char_data:
            characters[char_id] = char_data

    return {
        "story": story,
        "characters": characters
    }


def get_index_summary() -> list:
    """
    Returns a simplified list of stories for LLM identification prompt.
    """
    index = load_index()
    summary = []
    for s in index.get("stories", []):
        summary.append({
            "story_id": s["story_id"],
            "story_name": s["story_name"],
            "skandha": s["skandha"],
            "key_terms": s["key_terms"]
        })
    return summary


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = get_index_summary()
    print("Known stories:")
    for s in summary:
        print(f"  {s['story_id']} — skandha {s['skandha']} — terms: {s['key_terms']}")

    print("\nLoading Gajendra Moksham scaffold:")
    data = load_all_for_story("gajendra_moksham")
    print(f"  Story: {data['story']['story_name']}")
    print(f"  Characters loaded: {list(data['characters'].keys())}")
