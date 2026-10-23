"""Light text cleaning shared by the questions. Tokenisation itself stays in each question."""

import html
import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")


def normalize(text: str, lower: bool = False, form: str = "NFC") -> str:
    """Unicode-normalise, collapse whitespace and optionally lowercase.

    NFC matters for Hindi: the same word can be typed as different code point
    sequences (e.g. with or without a precomposed nukta), and without NFC they end
    up as separate vocabulary entries.
    """
    text = unicodedata.normalize(form, text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text.lower() if lower else text


def strip_html_and_urls(text: str, url_replacement: str = " ") -> str:
    """Drop HTML tags (Amazon reviews are full of <br />), unescape entities and remove URLs."""
    text = _HTML_TAG.sub(" ", text)
    text = html.unescape(text)
    text = _URL.sub(url_replacement, text)
    return _WHITESPACE.sub(" ", text).strip()
