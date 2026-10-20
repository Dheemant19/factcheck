"""Light text cleaning shared by every dataset (owner M3, T03; plan Section 7).

Pre-trained transformers want text close to the original, so there is no lower-casing,
stop-word removal or stemming here. ``basic_clean`` only repairs the text: broken encodings,
HTML, and messy whitespace. Social-media masking (URLs and @mentions) is a flag, because the
shortcut study (T13, E7) needs a copy that keeps them.
"""

from __future__ import annotations

import html
import re

import ftfy

HTML_TAG = re.compile(r"</?[A-Za-z][^<>]*>")  # <p>, </a>, <br/>; leaves "<3" and "a < b" alone
URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION = re.compile(r"(?<![\w@])@\w+")  # @CDCgov, but not the @ inside an e-mail address
WHITESPACE = re.compile(r"\s+")

URL_TOKEN = "HTTPURL"  # same placeholders as COVID-Twitter-BERT and BERTweet
USER_TOKEN = "@USER"


def basic_clean(text: str, mask_social: bool = False) -> str:
    """Repair one text without changing its meaning.

    Fixes mojibake with ftfy, unescapes HTML entities, strips HTML tags and collapses all
    whitespace to single spaces. With ``mask_social=True`` URLs become ``HTTPURL`` and user
    mentions become ``@USER`` (used for the COVID posts).

    >>> basic_clean("CafÃ© &amp; <b>bar</b>\\n\\n  ok")
    'Café & bar ok'
    >>> basic_clean("@WHO says see https://t.co/x1", mask_social=True)
    '@USER says see HTTPURL'
    """
    if not isinstance(text, str):
        return ""
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = HTML_TAG.sub(" ", text)
    if mask_social:
        text = URL.sub(URL_TOKEN, text)
        text = MENTION.sub(USER_TOKEN, text)
    return WHITESPACE.sub(" ", text).strip()
