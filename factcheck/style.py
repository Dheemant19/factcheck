"""Strip a post's surface style, so a model has to read the words (owner M1, T59).

Why: the COVID models score 0.95-0.98 partly by reading how a post is written, not what it says.
In the COVID training split, real posts carry a link 69% of the time (fake 34%), a hashtag 50%
(fake 20%), an @mention 28% (fake 8%), emoji 8% (fake 1%) and "RT" 8% (fake 0.7%); fake posts
use "!" 4x as often and are the only ones that open with a fact-checker's "Says ..." (3%). A
WHO-style tweet with its facts swapped still reads as true (T58 probes).

``strip_style`` removes that surface layer and keeps the words:

    links, @mentions (and the HTTPURL / @USER placeholders), "RT", emoji and other symbols,
    a pile of two or more hashtags at the end of a post, the "#" of any other hashtag (the
    word stays, "_" becomes a space), "Says" / "Claims" at the start, "!" and "?", quotes,
    repeated punctuation, and capitals (everything is lower-cased)

What it cannot remove is wording: an official post still says "cases have been reported".
Removing wording would remove the content too, so whatever score is left after stripping is
what the models get from the words themselves.

Switched on per run with ``features: strip_style: true`` (``factcheck.inputs``); the app strips
a pasted claim the same way for those runs (``factcheck.serve``).
"""

from __future__ import annotations

import re
import unicodedata

from factcheck.data.cleaning import MENTION, URL, URL_TOKEN, USER_TOKEN

PLACEHOLDERS = re.compile(rf"{re.escape(URL_TOKEN)}|{re.escape(USER_TOKEN)}", re.IGNORECASE)
RETWEET = re.compile(r"\bRT\b:?", re.IGNORECASE)
TRAILING_HASHTAGS = re.compile(r"(?:\s*#\w+){2,}[\s.!?]*$")  # a pile of two or more
HASH = re.compile(r"#(\w+)")
OPENER = re.compile(r"^\s*(?:says|claims?)\b[\s:,-]*", re.IGNORECASE)
BANG = re.compile(r"[!?¡¿]+")
QUOTES = re.compile(r"[\"“”„«»`´]|(?<!\w)'|'(?!\w)")
REPEATS = re.compile(r"([.,;:\-])\1+")
SPACE_BEFORE = re.compile(r"\s+([.,;:])")
WHITESPACE = re.compile(r"\s+")


def is_symbol(char: str) -> bool:
    """Emoji, flags, arrows and other pictographs (Unicode symbols, private use, joiners)."""
    category = unicodedata.category(char)
    return category in ("So", "Sk", "Cs", "Co", "Cf") or "\U0001f1e6" <= char <= "\U0001f1ff"


def strip_style(text: str) -> str:
    """One post with its surface style removed, its words kept.

    >>> strip_style("RT @WHO: NEW #COVID19 cases in #Nagpur!! 🇮🇳 https://t.co/x #Corona #Update")
    'new covid19 cases in nagpur'
    >>> strip_style('Says "the virus" was MADE in a lab???')
    'the virus was made in a lab'
    """
    if not isinstance(text, str):
        return ""
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = PLACEHOLDERS.sub(" ", text)
    text = RETWEET.sub(" ", text)
    text = "".join(" " if is_symbol(c) else c for c in text)
    text = TRAILING_HASHTAGS.sub("", text.strip())
    text = HASH.sub(lambda m: m.group(1).replace("_", " "), text)
    text = OPENER.sub("", text)
    text = BANG.sub(" ", text)
    text = QUOTES.sub("", text)
    text = REPEATS.sub(r"\1", text)
    text = WHITESPACE.sub(" ", text).strip()
    text = SPACE_BEFORE.sub(r"\1", text)
    return text.lower().strip(" .,;:-")
