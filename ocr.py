"""
Module 1 — OCR
SHRI Project · RISHA Lab · IIT Tirupati

Uses Groq Llama 3.2 Vision (free tier) to extract text from scanned pages.
Handles all complex layouts:
  - Text wrapping around illustrations
  - Fun fact / Did you know boxes
  - Captions, titles, page numbers
  - Decorative bordered text boxes

Extracts ONLY the main story narrative text.

Uses same GROQ_API_KEY as llm_text_extractor.py
Get free key at: https://console.groq.com
"""

import os
import base64
from PIL import Image
from groq import Groq


from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"

PROMPT = """This is a page from a children's Hindu scripture storybook.

Extract ONLY the main story narrative text from this page.

Ignore completely:
- Fun fact boxes or "Did you know?" boxes
- Text inside decorative borders or coloured or shaded boxes
- Captions on illustrations
- Chapter numbers and titles
- Page numbers
- Any text that appears inside a bordered or highlighted region
- Labels on images

Return only the flowing story text in correct reading order.
Do not add any explanation or commentary — just the extracted text."""


def _encode_image(image_path: str) -> tuple:
    """Encode image to base64 for Groq API."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    # Detect mime type
    ext = image_path.lower().split(".")[-1]
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    return data, mime


def _fix_rotation(img: Image.Image) -> Image.Image:
    """Fix image rotation using EXIF data."""
    try:
        from PIL.ExifTags import TAGS
        exif_data = img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                if TAGS.get(tag) == "Orientation":
                    if value == 3:
                        img = img.rotate(180, expand=True)
                    elif value == 6:
                        img = img.rotate(270, expand=True)
                    elif value == 8:
                        img = img.rotate(90, expand=True)
                    break
    except Exception:
        pass
    # If wider than tall — rotate to portrait
    w, h = img.size
    if w > h:
        img = img.rotate(90, expand=True)
    return img


def extract_text(image_path: str) -> dict:
    """
    Main OCR function using Groq Llama Vision.

    Args:
        image_path: path to scanned page image

    Returns:
        {
            "raw_text": str,
            "has_image": bool,
            "confidence": float
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Fix rotation
    img = Image.open(image_path)
    img = _fix_rotation(img)

    # Save rotated image to temp file
    temp_path = image_path + "_temp.jpg"
    img.save(temp_path, "JPEG", quality=95)

    # Encode image
    img_data, mime_type = _encode_image(temp_path)

    # Clean up temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass

    # Call Groq Llama Vision
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": PROMPT
                    }
                ]
            }
        ],
        max_tokens=2000,
        temperature=0.1
    )

    raw_text = response.choices[0].message.content.strip()

    return {
        "raw_text": raw_text
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_path>")
        sys.exit(1)

    try:
        result = extract_text(sys.argv[1])
        print(f"Text length: {len(result['raw_text'])} chars")
        print(f"\nExtracted text:\n{'-'*40}\n{result['raw_text']}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)