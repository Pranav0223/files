"""
SHRI Pipeline — Main Entry Point
RISHA Lab · IIT Tirupati

Full pipeline:
  image_path
      → Module 1: OCR → raw_text + has_image
      → Module 3: rule_extractor → candidate_names
      → Module 4a: llm_text_extractor → story_id, characters, events
      → Module 2: scaffold_loader → story + character scaffolds
      → [if has_image] Module 4b: llm_image_extractor → overrides
      → Module 5: json_generator → graph1.json + graph2.json

Usage:
  python pipeline.py <image_path> [output_dir]
"""

import sys
import os
import json

from ocr import extract_text
from rule_extractor import extract_candidates
from scaffold_loader import get_index_summary, load_all_for_story
from llm_text_extractor import extract_from_text
from llm_image_extractor import extract_from_image
from json_generator import generate


def run_pipeline(image_path: str, output_dir: str = "output") -> dict:
    """
    Run the full SHRI pipeline on a single page image.

    Args:
        image_path: path to scanned page image
        output_dir: directory to write output JSON files

    Returns:
        {
            "graph1_path": str,
            "graph2_path": str,
            "story_id": str,
            "ocr_confidence": float
        }
    """
    print(f"\n{'='*50}")
    print(f"SHRI Pipeline — processing: {os.path.basename(image_path)}")
    print(f"{'='*50}")

    # ── Step 1: OCR ──────────────────────────────────────────────────────────
    print("\n[1/5] Running OCR...")
    ocr_result = extract_text(image_path)
    print(f"      Confidence: {ocr_result['confidence']}%")
    print(f"      Has image:  {ocr_result['has_image']}")
    print(f"      Text length: {len(ocr_result['raw_text'])} chars")

    if ocr_result["confidence"] < 40:
        print("      WARNING: Low OCR confidence. Results may be inaccurate.")

    # ── Step 2: Rule-based NER ───────────────────────────────────────────────
    print("\n[2/5] Running rule-based character extraction...")
    candidate_names = extract_candidates(ocr_result["raw_text"])
    print(f"      Candidates found: {candidate_names}")

    # ── Step 3: Load scaffold index ──────────────────────────────────────────
    print("\n[3/5] Loading scaffold index...")
    index_summary = get_index_summary()
    print(f"      Known stories: {[s['story_id'] for s in index_summary]}")

    # ── Step 4a: Text LLM extraction ─────────────────────────────────────────
    print("\n[4a/5] Running text LLM extraction...")
    text_extraction = extract_from_text(
        ocr_result["raw_text"],
        candidate_names,
        index_summary
    )
    story_id = text_extraction.get("story_id", "unknown")
    print(f"       Story identified: {story_id}")
    print(f"       Characters: {[c['character_id'] for c in text_extraction.get('characters', [])]}")
    print(f"       Events: {len(text_extraction.get('events', []))}")

    # ── Step 4b: Load story scaffold ─────────────────────────────────────────
    print(f"\n[4b/5] Loading scaffold for: {story_id}...")
    try:
        scaffold_data = load_all_for_story(story_id)
        print(f"       Characters in scaffold: {list(scaffold_data['characters'].keys())}")
    except FileNotFoundError:
        print(f"       WARNING: No scaffold found for {story_id}. Using empty scaffold.")
        scaffold_data = {"story": {}, "characters": {}}

    # ── Step 4c: Image LLM extraction (if image present) ────────────────────
    image_extraction = None
    if ocr_result["has_image"]:
        print("\n[4c/5] Running image LLM extraction...")
        known_chars = [c["character_id"] for c in text_extraction.get("characters", [])]
        story_name = scaffold_data["story"].get("story_name", story_id)
        try:
            image_extraction = extract_from_image(
                image_path,
                known_chars,
                story_name
            )
            print(f"       Overrides: {len(image_extraction.get('character_overrides', []))}")
            print(f"       Additional characters: {len(image_extraction.get('additional_characters', []))}")
            print(f"       Additional events: {len(image_extraction.get('additional_events', []))}")
        except Exception as e:
            print(f"       WARNING: Image LLM failed: {e}. Skipping image analysis.")
    else:
        print("\n[4c/5] No image detected — skipping image LLM.")

    # ── Step 5: Generate JSON files ──────────────────────────────────────────
    print(f"\n[5/5] Generating JSON files...")
    result = generate(
        story_id=story_id,
        scaffold_data=scaffold_data,
        text_extraction=text_extraction,
        image_extraction=image_extraction,
        output_dir=output_dir
    )
    print(f"       Graph 1 (3D): {result['graph1_path']}")
    print(f"       Graph 2 (animation): {result['graph2_path']}")

    print(f"\n{'='*50}")
    print("Pipeline complete.")
    print(f"{'='*50}\n")

    return {
        "graph1_path": result["graph1_path"],
        "graph2_path": result["graph2_path"],
        "story_id": story_id,
        "ocr_confidence": ocr_result["confidence"]
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path> [output_dir]")
        print("Example: python pipeline.py page.jpg output/")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    result = run_pipeline(image_path, output_dir)
    print(json.dumps(result, indent=2))
