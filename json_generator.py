"""
Module 5 — JSON Generator
SHRI Project · RISHA Lab · IIT Tirupati

Merges scaffold data + LLM text extraction + LLM image extraction
into two final JSON files:
  - graph1_3d_generation.json
  - graph2_animation.json

Priority order for field values:
  scaffold < text_llm_override < image_llm_override
"""

import json
import os


def generate(
    story_id: str,
    scaffold_data: dict,
    text_extraction: dict,
    image_extraction: dict = None,
    output_dir: str = "."
) -> dict:
    """
    Main JSON generation function.

    Args:
        story_id: identified story id
        scaffold_data: loaded from scaffold_loader.load_all_for_story()
        text_extraction: output from llm_text_extractor.extract_from_text()
        image_extraction: output from llm_image_extractor (optional)
        output_dir: where to write the output JSON files

    Returns:
        { "graph1_path": str, "graph2_path": str }
    """
    story_scaffold = scaffold_data.get("story", {})
    char_scaffolds = scaffold_data.get("characters", {})

    graph1 = _build_graph1(
        story_id,
        story_scaffold,
        char_scaffolds,
        text_extraction,
        image_extraction
    )

    graph2 = _build_graph2(
        story_id,
        story_scaffold,
        text_extraction,
        image_extraction
    )

    # Write files
    os.makedirs(output_dir, exist_ok=True)
    g1_path = os.path.join(output_dir, "graph1_3d_generation.json")
    g2_path = os.path.join(output_dir, "graph2_animation.json")

    with open(g1_path, "w", encoding="utf-8") as f:
        json.dump(graph1, f, indent=2, ensure_ascii=False)

    with open(g2_path, "w", encoding="utf-8") as f:
        json.dump(graph2, f, indent=2, ensure_ascii=False)

    return {"graph1_path": g1_path, "graph2_path": g2_path}


def _build_2d_prompt(char_entry: dict) -> str:
    """
    Auto-generate the 2D image generation prompt from character attributes.
    This prompt is used to generate the reference image for 3D reconstruction.
    Built entirely from scaffold + LLM merged fields — no extra LLM call.
    """
    parts = ["Full body T-pose, front-facing"]

    if char_entry.get("skin_color"):
        parts.append(char_entry["skin_color"])

    if char_entry.get("build"):
        parts.append(char_entry["build"])

    if char_entry.get("crown") and char_entry["crown"] != "none":
        parts.append(char_entry["crown"])

    if char_entry.get("clothing"):
        parts.append(char_entry["clothing"])

    ornaments = char_entry.get("ornaments", [])
    if ornaments:
        parts.append(", ".join(ornaments))

    if char_entry.get("right_hand"):
        parts.append(f"{char_entry['right_hand']} in right hand")

    if char_entry.get("left_hand"):
        parts.append(f"{char_entry['left_hand']} in left hand")

    if char_entry.get("posture"):
        parts.append(char_entry["posture"])

    yuga = char_entry.get("yuga", "")
    if yuga:
        parts.append(f"realistic Hindu {yuga} style")

    parts.extend([
        "neutral background",
        "symmetrical anatomy",
        "no perspective distortion",
        "game-ready topology"
    ])

    return ", ".join(parts)


def _build_graph1(
    story_id, story_scaffold, char_scaffolds,
    text_extraction, image_extraction
) -> dict:
    """Build Graph 1 — 3D model generation JSON."""

    characters = []

    for char_info in text_extraction.get("characters", []):
        char_id = char_info.get("character_id", "")
        scaffold = char_scaffolds.get(char_id, {})
        physical = scaffold.get("physical", {})
        scale = scaffold.get("scale", {})

        # Start with scaffold values
        char_entry = {
            "id": char_id,
            "name": char_info.get("name", scaffold.get("name", "")),
            "aliases": scaffold.get("aliases", []),
            "being_type": scaffold.get("being_type", "unknown"),
            "narrative_role": char_info.get("narrative_role", "supporting"),
            "skin_color": physical.get("skin_color", ""),
            "num_arms": 2,
            "right_hand": physical.get("right_hand", ""),
            "left_hand": physical.get("left_hand", ""),
            "crown": physical.get("crown", ""),
            "clothing": physical.get("clothing", ""),
            "ornaments": physical.get("ornaments", []),
            "posture": char_info.get("posture", physical.get("posture", "")),
            "build": physical.get("build", ""),
            "yuga": scale.get("yuga", story_scaffold.get("yuga", "")),
            "scale": scale.get("mythological_scale", None)
        }

        # Apply clothing override from text LLM
        if char_info.get("clothing_override"):
            char_entry["clothing"] = char_info["clothing_override"]

        # Apply overrides from image LLM
        if image_extraction:
            for override in image_extraction.get("character_overrides", []):
                if override.get("character_id") == char_id:
                    if override.get("posture_override"):
                        char_entry["posture"] = override["posture_override"]
                    if override.get("clothing_override"):
                        char_entry["clothing"] = override["clothing_override"]

        # Generate 2D image prompt from merged attributes
        char_entry["image_generation_prompt"] = _build_2d_prompt(char_entry)
        char_entry["output_file"] = f"{char_id}.glb"

        characters.append(char_entry)

    # Add additional characters from image (unnamed in text)
    if image_extraction:
        for i, add_char in enumerate(image_extraction.get("additional_characters", [])):
            char_entry = {
                "id": f"unnamed_{i+1}",
                "name": add_char.get("description", "unnamed character"),
                "aliases": [],
                "being_type": "unknown",
                "narrative_role": "supporting",
                "skin_color": "",
                "num_arms": 2,
                "right_hand": "",
                "left_hand": "",
                "crown": "",
                "clothing": add_char.get("clothing", ""),
                "ornaments": [],
                "posture": add_char.get("posture", ""),
                "build": "",
                "yuga": story_scaffold.get("yuga", ""),
                "scale": None
            }
            char_entry["image_generation_prompt"] = _build_2d_prompt(char_entry)
            char_entry["output_file"] = f"unnamed_{i+1}.glb"
            characters.append(char_entry)

    return {
        "graph": "3d_model_generation",
        "story": story_scaffold.get("story_name", story_id),
        "skandha": text_extraction.get("skandha", story_scaffold.get("skandha", "")),
        "has_image": image_extraction is not None,
        "characters": characters
    }


def _build_graph2(
    story_id, story_scaffold,
    text_extraction, image_extraction
) -> dict:
    """Build Graph 2 — Animation JSON."""

    events = []

    for event in text_extraction.get("events", []):
        events.append({
            "id": event.get("id", ""),
            "sequence_order": event.get("sequence_order", 0),
            "source_text": event.get("source_text", ""),
            "action_type": event.get("action_type", ""),
            "action": event.get("action", ""),
            "action_modifier": event.get("action_modifier", ""),
            "duration": event.get("duration", ""),
            "participants": event.get("participants", [])
        })

    # Add additional events from image LLM
    if image_extraction:
        start_order = len(events) + 1
        for i, add_event in enumerate(image_extraction.get("additional_events", [])):
            add_event["id"] = f"event_{start_order + i}"
            add_event["sequence_order"] = start_order + i
            add_event["source_text"] = "extracted from image"
            events.append(add_event)

    return {
        "graph": "animation",
        "story": story_scaffold.get("story_name", story_id),
        "skandha": text_extraction.get("skandha", story_scaffold.get("skandha", "")),
        "has_image": image_extraction is not None,
        "events": events
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("json_generator.py — import and call generate() to use")
