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
      → Module 5: json_generator → graph1.json
      → Module 6: beat_extractor → graph2.json + graph3.json

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
from beat_extractor import extract_beats


def run_pipeline(image_path: str, output_dir: str = "output") -> dict:
    print(f"\n{'='*50}")
    print(f"SHRI Pipeline — processing: {os.path.basename(image_path)}")
    print(f"{'='*50}")

    # Step 1: OCR
    print("\n[1/6] Running OCR...")
    ocr_result = extract_text(image_path)
    print(f"      Confidence: {ocr_result['confidence']}%")
    print(f"      Has image:  {ocr_result['has_image']}")

    # Step 2: Rule-based NER
    print("\n[2/6] Rule-based character extraction...")
    candidate_names = extract_candidates(ocr_result["raw_text"])
    print(f"      Candidates: {candidate_names}")

    # Step 3: Scaffold index
    print("\n[3/6] Loading scaffold index...")
    index_summary = get_index_summary()

    # Step 4a: Text LLM
    print("\n[4a/6] Text LLM extraction...")
    text_extraction = extract_from_text(
        ocr_result["raw_text"],
        candidate_names,
        index_summary
    )
    story_id = text_extraction.get("story_id", "unknown")
    print(f"       Story: {story_id}")

    # Step 4b: Load scaffold
    print(f"\n[4b/6] Loading scaffold for: {story_id}...")
    try:
        scaffold_data = load_all_for_story(story_id)
    except FileNotFoundError:
        print(f"       WARNING: No scaffold for {story_id}. Using empty.")
        scaffold_data = {"story": {}, "characters": {}}

    # Step 4c: Image LLM
    image_extraction = None
    if ocr_result["has_image"]:
        print("\n[4c/6] Image LLM extraction...")
        known_chars = [c["character_id"] for c in text_extraction.get("characters", [])]
        story_name  = scaffold_data["story"].get("story_name", story_id)
        try:
            image_extraction = extract_from_image(image_path, known_chars, story_name)
        except Exception as e:
            print(f"       WARNING: Image LLM failed: {e}")
    else:
        print("\n[4c/6] No image — skipping image LLM.")

    # Step 5: Graph 1
    print(f"\n[5/6] Generating Graph 1 — 3D model generation...")
    os.makedirs(output_dir, exist_ok=True)
    result = generate(
        story_id=story_id,
        scaffold_data=scaffold_data,
        text_extraction=text_extraction,
        image_extraction=image_extraction,
        output_dir=output_dir
    )
    print(f"       Graph 1: {result['graph1_path']}")

    # Step 6: Graph 2 + Graph 3
    print(f"\n[6/6] Extracting beats — Graph 2 + Graph 3...")
    story_name = scaffold_data["story"].get("story_name", story_id)
    beats = extract_beats(
        raw_text=ocr_result["raw_text"],
        story=story_name,
        skandha=text_extraction.get("skandha", "")
    )

    g2_path = os.path.join(output_dir, "graph2_animation_generation.json")
    g3_path = os.path.join(output_dir, "graph3_scene_composition.json")

    with open(g2_path, "w") as f:
        json.dump(beats["graph2"], f, indent=2)
    with open(g3_path, "w") as f:
        json.dump(beats["graph3"], f, indent=2)

    print(f"       Graph 2: {g2_path}")
    print(f"       Graph 3: {g3_path}")

    print(f"\n{'='*50}")
    print("Pipeline complete.")
    print(f"{'='*50}\n")

    return {
        "graph1_path"   : result["graph1_path"],
        "graph2_path"   : g2_path,
        "graph3_path"   : g3_path,
        "story_id"      : story_id,
        "ocr_confidence": ocr_result["confidence"]
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    result     = run_pipeline(image_path, output_dir)
    print(json.dumps(result, indent=2))
