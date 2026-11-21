"""
TechText AI - NLP Text Cleaning & Normalization Pipeline
Implements cleaning techniques covered in Workshop 3:
- HTML tag stripping
- Noise token removal (e.g. corruption markers ^^, %, $, @)
- Unicode normalization and smart quote conversion
- Emoji and control character filtering
- Whitespace normalization
- Deduplication and minimum length filtering
"""

import re
import html
import unicodedata
from typing import List, Optional


class TextCleaner:
    def __init__(
        self,
        strip_html: bool = True,
        remove_noise_tokens: bool = True,
        normalize_unicode: bool = True,
        remove_emojis: bool = True,
        min_line_length: int = 15,
    ):
        self.strip_html = strip_html
        self.remove_noise_tokens = remove_noise_tokens
        self.normalize_unicode = normalize_unicode
        self.remove_emojis = remove_emojis
        self.min_line_length = min_line_length

        # Regex patterns compiled for maximum speed
        self._html_tag_pattern = re.compile(r"<[^>]+>")
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+")
        # Corrupt markers commonly injected in corpora: ^^, %, $, @ surrounded by spaces
        self._noise_token_pattern = re.compile(r"(?<=\s)[\^%@$]+(?=\s)|^[\^%@$]+(?=\s)|(?<=\s)[\^%@$]+$")
        self._isolated_symbols = re.compile(r"^\s*[\^%@$#*&/\\|]+\s*$", re.MULTILINE)
        self._extra_spaces_pattern = re.compile(r"[ \t]+")
        self._extra_newlines_pattern = re.compile(r"\n{3,}")

    def clean_text(self, text: str) -> str:
        """Applies the full cleaning transformation to the input string."""
        if not text:
            return ""

        # 1. Unescape HTML entities (e.g. &amp; -> &, &quot; -> ")
        text = html.unescape(text)

        # 2. Strip HTML tags
        if self.strip_html:
            text = self._html_tag_pattern.sub(" ", text)

        # 3. Strip URLs
        text = self._url_pattern.sub("", text)

        # 4. Normalize Unicode (standardize accents, NFC form, replace fancy quotes)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
            # Standardize typographic quotes and dashes
            text = (
                text.replace("“", '"')
                .replace("”", '"')
                .replace("‘", "'")
                .replace("’", "'")
                .replace("—", " - ")
                .replace("–", " - ")
            )

        # 5. Remove corruption symbols and standalone noise tokens
        if self.remove_noise_tokens:
            # First remove lines that only consist of noise symbols
            text = self._isolated_symbols.sub("", text)
            # Next remove noise tokens flanked by whitespace
            text = self._noise_token_pattern.sub("", text)

        # 6. Remove Emojis and obscure non-ASCII symbols if configured
        if self.remove_emojis:
            # Strip emojis and pictographs
            text = re.sub(
                r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]",
                "",
                text,
                flags=re.UNICODE,
            )
            # Standardize math and special punctuation to clean ASCII equivalents
            char_map = {
                "−": "-", "‐": "-", "—": " - ", "–": " - ",
                "×": "x", "∗": "*", "⁄": "/", "‖": " ",
                "≤": "<=", "≥": ">=", "≠": "!=", "±": "+/-",
                "\u2060": "", "\u2061": "", "′": "'", "″": '"',
                "“": '"', "”": '"', "‘": "'", "’": "'",
            }
            for orig, repl in char_map.items():
                text = text.replace(orig, repl)
            # Remove obscure math operators and unprintable control characters
            text = re.sub(r"[^\x00-\x7F]+", " ", text)

        # 7. Normalize whitespace
        text = self._extra_spaces_pattern.sub(" ", text)
        lines = [line.strip() for line in text.split("\n")]

        # 8. Filter empty or noise-only lines
        cleaned_lines: List[str] = []
        seen_lines = set()

        for line in lines:
            line_str = line.strip()
            # Remove any trailing / leading artifacts
            line_str = re.sub(r"^[\^%@$#\s]+|[\^%@$#\s]+$", "", line_str)
            if len(line_str) >= self.min_line_length:
                # Deduplication for consecutive or identical lines
                if line_str not in seen_lines:
                    seen_lines.add(line_str)
                    cleaned_lines.append(line_str)

        cleaned_text = "\n".join(cleaned_lines)
        cleaned_text = self._extra_newlines_pattern.sub("\n\n", cleaned_text)
        return cleaned_text.strip()
