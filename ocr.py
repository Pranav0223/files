"""
Module 1 — OCR
SHRI Project · RISHA Lab · IIT Tirupati

Takes an image file path.
Returns extracted raw text string.
Also detects if the page contains an image (non-text region).
"""

import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import os


def preprocess_image(image_path: str) -> Image.Image:
    """
    Preprocess the image before OCR.
    Improves accuracy on printed book pages.
    """
    img = Image.open(image_path)

    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Convert to grayscale
    img = img.convert("L")

    # Boost contrast — helps with faded or uneven print
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    # Boost sharpness — helps with slightly blurred scans
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)

    return img


def detect_image_presence(image_path: str) -> bool:
    """
    Detect whether the page contains an illustration/image
    in addition to text.

    Simple heuristic: if OCR confidence is low on large regions
    of the page, those regions likely contain images not text.
    We use Tesseract's detailed output to check confidence scores.
    """
    try:
        img = preprocess_image(image_path)
        data = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DICT,
            config="--psm 6"
        )

        # Count words with very low confidence — likely image regions
        confidences = [
            int(c) for c in data["conf"]
            if c != "-1" and c != -1
        ]

        if not confidences:
            return False

        low_conf_count = sum(1 for c in confidences if c < 30)
        total_count = len(confidences)

        # If more than 20% of detected regions have very low confidence
        # the page likely has significant image content
        has_image = (low_conf_count / total_count) > 0.2 if total_count > 0 else False
        return has_image

    except Exception:
        return False


def extract_text(image_path: str) -> dict:
    """
    Main OCR function.

    Args:
        image_path: path to scanned page image (jpg, png, etc.)

    Returns:
        {
            "raw_text": str,       — extracted text
            "has_image": bool,     — whether page contains illustration
            "confidence": float    — average OCR confidence 0-100
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = preprocess_image(image_path)

    # PSM 6 — assume uniform block of text — best for book pages
    config = "--psm 6 --oem 1"

    # Get text
    raw_text = pytesseract.image_to_string(img, config=config, lang="eng")

    # Get confidence data
    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT,
        config=config
    )

    confidences = [
        int(c) for c in data["conf"]
        if str(c) != "-1" and c != -1 and int(c) > 0
    ]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Clean up extracted text
    cleaned_text = _clean_text(raw_text)

    # Detect image presence
    has_image = detect_image_presence(image_path)

    return {
        "raw_text": cleaned_text,
        "has_image": has_image,
        "confidence": round(avg_confidence, 1)
    }


def _clean_text(text: str) -> str:
    """
    Clean raw OCR output.
    Removes excessive whitespace and common OCR artifacts.
    """
    import re

    # Replace form feeds with newlines
    text = text.replace("\f", "\n")

    # Collapse multiple spaces to single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Collapse more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are just whitespace
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    return "\n".join(lines).strip()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_path>")
        sys.exit(1)

    result = extract_text(sys.argv[1])
    print(f"Confidence  : {result['confidence']}%")
    print(f"Has image   : {result['has_image']}")
    print(f"Extracted text:\n{'-'*40}\n{result['raw_text']}")
