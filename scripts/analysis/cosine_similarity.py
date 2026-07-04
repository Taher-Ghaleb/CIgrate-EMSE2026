"""
Cosine similarity utilities.

- Tokenization mirrors the Java implementation:
    - Skip lines that start with '#'
    - Split by a single space character
    - For each token: trim, lowercase
    - Skip tokens equal to ':'
    - Replace '-', "\n", single quote, and double quote with empty string
    - Count non-empty tokens
- Cosine similarity returns 2.0 when either vector has zero norm (invalid),
    exactly like the Java code.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence, List, Dict


def textToMap(text_or_lines: Sequence[str] | str) -> Dict[str, float]:
    """Build term-frequency map exactly like the Java implementation.

    Accepts a string (multi-line) or a list of lines.
    """
    textLines = text_or_lines.splitlines() if isinstance(text_or_lines, str) else list(text_or_lines)
    map: Dict[str, float] = {}
    for text in textLines:
        if text.startswith('#'):
            continue
        for word in text.split(" "):
            word = word.strip().lower()
            if word == ":":
                continue
            word = word.replace("-", "").replace("\n", "").replace("'", "").replace('"', "")
            if word:
                map[word] = map.get(word, 0.0) + 1.0
    return map


def cosineSimilarity(v1: Mapping[str, float], v2: Mapping[str, float]) -> float:
    """Compute cosine similarity exactly like the Java implementation.

    Returns 2.0 if either norm is zero; else dot / (|a| |b|).
    """
    # Common keys (both)
    both = set(v1.keys()) & set(v2.keys())

    # Dot product over common keys
    dotProduct = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for k in both:
        val1 = float(v1.get(k, 0.0))
        val2 = float(v2.get(k, 0.0))
        dotProduct += val1 * val2

    # For each key in v1
    for k in v1.keys():
        # Get the value (default 0.0 if missing) and add to norm1
        val = float(v1.get(k, 0.0))
        norm1 += val * val
    # For each key in v2
    for k in v2.keys():
        # Get the value (default 0.0 if missing) and add to norm2
        val = float(v2.get(k, 0.0))
        norm2 += val * val
    # Return the cosine similarity or 2.0 if invalid
    if norm1 == 0.0 or norm2 == 0.0:
        return 2.0
    return dotProduct / (math.sqrt(norm1) * math.sqrt(norm2))


__all__ = [
    "textToMap",
    "cosineSimilarity",
]
