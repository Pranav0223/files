"""
Module 3 — Rule-Based Extractor
SHRI Project · RISHA Lab · IIT Tirupati

Identifies candidate character names from OCR text
using rule-based NLP patterns. No ML model required.

Rules derived from:
  - Title-prefix NER (Stein & Glenn 1979 — protagonist identification)
  - POS-based proper noun detection
  - Sanskrit name morphology patterns
"""

import re
import spacy
from spacy.lang.en import English


# ── Title words that precede character names in Hindu scripture translations ──
HINDU_TITLES = [
    # Royalty
    "King", "Queen", "Prince", "Princess", "Emperor", "Empress",
    "Maharaja", "Maharani",
    # Religious
    "Lord", "God", "Goddess", "Sage", "Rishi", "Muni", "Swami",
    "Brahmana", "Saint", "Devotee",
    # Demonic
    "Demon", "Asura", "Daitya", "Danava",
    # Other
    "Noble", "Great", "Divine", "Holy"
]

# ── Sanskrit name patterns ─────────────────────────────────────────────────
# Sanskrit-origin names in English translations follow patterns:
# - Compound words ending in: -a, -i, -u, -ra, -na, -va, -sha
# - Starting with capitals
# - Often 2+ syllables
SANSKRIT_NAME_PATTERN = re.compile(
    r"\b[A-Z][a-z]{2,}(?:deva|muni|pati|nath|raya|raja|vati|devi|hari|"
    r"krishna|vishnu|shiva|rama|sita|devi|lakshmi|radha|arjuna|"
    r"pura|loka|giri|naga|asura|daitya)?\b"
)

# ── Intentional action verbs — character is doing something ───────────────
INTENTIONAL_VERBS = [
    "said", "spoke", "replied", "asked", "told", "declared", "prayed",
    "went", "came", "arrived", "entered", "left", "returned",
    "took", "gave", "offered", "received", "granted", "refused",
    "fought", "attacked", "defended", "protected", "saved",
    "looked", "saw", "heard", "felt", "thought", "decided",
    "ordered", "commanded", "blessed", "cursed", "worshipped"
]


def extract_candidates(raw_text: str) -> list:
    """
    Main rule-based extraction function.

    Args:
        raw_text: cleaned OCR text from page

    Returns:
        List of candidate character name strings found in text
    """
    candidates = set()

    # Rule 1: Title-prefix pattern
    title_candidates = _extract_by_title_prefix(raw_text)
    candidates.update(title_candidates)

    # Rule 2: Proper noun as subject of intentional verb
    verb_candidates = _extract_by_verb_subject(raw_text)
    candidates.update(verb_candidates)

    # Rule 3: Repeated capitalised words (appears 2+ times)
    repeat_candidates = _extract_repeated_proper_nouns(raw_text)
    candidates.update(repeat_candidates)

    # Clean and deduplicate
    cleaned = _clean_candidates(candidates, raw_text)

    return cleaned


def _extract_by_title_prefix(text: str) -> set:
    """
    Rule 1: Title prefix pattern.
    Finds "Lord Vishnu", "King Bali", "Sage Shukracharya" etc.
    """
    found = set()
    for title in HINDU_TITLES:
        pattern = re.compile(rf"\b{title}\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b")
        matches = pattern.findall(text)
        for match in matches:
            # Take only first two words max
            name = " ".join(match.split()[:2])
            found.add(name)
    return found


def _extract_by_verb_subject(text: str) -> set:
    """
    Rule 2: Capitalised word followed by intentional verb.
    "Gajendra prayed", "Kansa ordered", "Vamana asked"
    """
    found = set()
    for verb in INTENTIONAL_VERBS:
        # Pattern: CapitalisedWord + optional space + verb
        pattern = re.compile(
            rf"\b([A-Z][a-zA-Z]{{2,}}(?:\s+[A-Z][a-zA-Z]+)?)\s+{verb}\b"
        )
        matches = pattern.findall(text)
        for match in matches:
            name = match.strip()
            if len(name) > 2:
                found.add(name)
    return found


def _extract_repeated_proper_nouns(text: str, min_occurrences: int = 2) -> set:
    """
    Rule 3: Capitalised words appearing 2+ times are likely character names.
    Filters out common English words that start sentences.
    """
    # Common English words that start sentences — not character names
    STOP_WORDS = {
        "The", "This", "That", "These", "Those", "Then", "When", "While",
        "Once", "One", "Soon", "After", "Before", "Long", "Just", "Now",
        "But", "And", "So", "He", "She", "They", "His", "Her", "Their",
        "Its", "It", "We", "You", "In", "On", "At", "To", "For", "Of",
        "With", "From", "By", "As", "An", "A"
    }

    # Find all capitalised words not at sentence start
    words = re.findall(r"(?<=\s)([A-Z][a-z]{2,})", text)
    all_caps = re.findall(r"\b([A-Z][a-zA-Z]{2,})\b", text)

    from collections import Counter
    counts = Counter(all_caps)

    found = set()
    for word, count in counts.items():
        if count >= min_occurrences and word not in STOP_WORDS:
            found.add(word)

    return found


def _clean_candidates(candidates: set, text: str) -> list:
    """
    Remove duplicates, single words that are subsets of longer names,
    and words not actually present in the text.
    """
    # Remove empty strings
    candidates = {c.strip() for c in candidates if c.strip()}

    # Remove candidates not found in text (case-insensitive check)
    text_lower = text.lower()
    candidates = {c for c in candidates if c.lower() in text_lower}

    # If "Lord Vishnu" and "Vishnu" both found, keep "Vishnu" as canonical
    # but note the full form — deduplicate by last word
    canonical = {}
    for name in candidates:
        last_word = name.split()[-1]
        if last_word not in canonical:
            canonical[last_word] = name
        else:
            # Keep the longer form
            if len(name) > len(canonical[last_word]):
                canonical[last_word] = name

    result = sorted(set(canonical.values()))
    return result


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_text = """
    Long ago, Princess Devaki and her brother Prince Kansa lived in Mathura.
    Devaki was kind while Kansa was cruel. Soon, Devaki was married to a
    nobleman, Vasudeva. One day, while Kansa was taking Devaki and Vasudeva
    to the palace, a divine voice said, Kansa will be killed by Devaki's
    eighth son. Kansa was terrified and drew his sword.
    """

    candidates = extract_candidates(sample_text)
    print("Extracted character candidates:")
    for name in candidates:
        print(f"  {name}")
