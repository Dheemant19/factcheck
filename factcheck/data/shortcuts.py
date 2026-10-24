"""Shortcut removal for the ``clean`` variant of ISOT (owner M3, T13; plan Section 7, E7).

ISOT's real articles are Reuters wire copy and its fake articles come from other websites, so
each class carries fingerprints of *where it was collected*, not of whether it is true: the
Reuters dateline, image credits, embedded tweets, links, site boilerplate, even stripped
apostrophes. A simple word-count model scores about 99% from these. ``remove_shortcuts``
deletes them, so the clean variant measures what a model learns from the content.

Every rule was found empirically with ``python scripts/find_shortcuts.py`` (TF-IDF + logistic
regression, then the highest-weighted words of each class) and has a name, so the leakage
report can say how many articles each rule changed. Rules only delete or normalise text; they
never insert a placeholder, because a placeholder that mostly one class gets is a new shortcut.

Deliberately kept: vocabulary and writing style (Reuters writes "said", "U.S." and weekdays;
fake sites use more capitals and exclamation marks). These differ between the classes too, but
they are how the text is written, not how it was collected, so a model using them is reading
the text. Clean accuracy therefore stays high; that is the honest finding, not a bug.
"""

from __future__ import annotations

import re

MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)

# name -> (pattern, replacement), applied in this order. Shares are of all ISOT articles.
# fmt: off  (one rule per block, regex pieces aligned for reading)
SHORTCUT_RULES: dict[str, tuple[re.Pattern[str], str]] = {
    # "WASHINGTON (Reuters) - " opens 98.5% of real articles and no fake one. The text starts
    # with the title, so the dateline follows ". "; the city is in capitals.
    # Also "SOCHI, Russia (Reuters) - " and "COX S BAZAR, Bangladesh/YANGON (Reuters) - ".
    "agency_dateline": (
        re.compile(
            r"(?:(?<=\. )|^)(?:[A-Z][A-Z.'/&-]*(?:\s+[A-Z][A-Z.'/&-]*){0,3}"
            r"(?:,\s+[A-Z][\w.'/-]*(?:\s+[A-Z][\w.'/-]*){0,2})?\s+)?"
            r"\((?:Reuters|AP|AFP)\)\s*[-–—]+\s*"
        ),
        "",
    ),
    # Any other "(Reuters) - " opening, e.g. after a byline "By Alison Frankel (Reuters) - ".
    "agency_tag_dash": (re.compile(r"\((?:Reuters|AP|AFP)\)\s*[-–—]+\s*"), " "),
    # Any other "(Reuters)" and the agency's name: in 99.8% of real, 1.2% of fake articles.
    "agency_name": (re.compile(r"\((?:Reuters|AP|AFP)\)|\bReuters\b"), " "),
    # 21st Century Wire sign-offs: "READ MORE SYRIA NEWS AT: 21st Century Wire SYRIA Files" and
    # "SUPPORT OUR WORK BY SUBSCRIBING & BECOMING A MEMBER @21WIRE.TV".
    "site_boilerplate": (
        re.compile(
            r"READ MORE[^:]{0,60}AT:\s*21st Century Wire[^.]{0,60}?Files|"
            r"SUPPORT (?:OUR WORK|21WIRE)[^@]{0,80}@\s?21WIRE\.TV|"
            r"21st Century Wire(?: says)?|21WIRE(?:\.TV)?"
        ),
        " ",
    ),
    # Repost credits of fake sites: "Continue this story at The Guardian", "Via: Breitbart News"
    # (the "Via:" form is in 21% of fake articles; "via" in a sentence is left alone).
    "repost_credit": (
        re.compile(
            r"Continue (?:this|the) (?:story|article|report)[^.]{0,10}? at "
            r"[A-Z][\w.]*(?: [A-Z][\w.]*){0,3}|"
            r"(?:(?<![A-Za-z])[Vv]ia|(?<=[a-z])Via|VIA)\s*:\s*(?:[A-Z][\w.'&-]*\s?){0,4}"
        ),
        " ",
    ),
    # Embedded tweet signature "(@realDonaldTrump) December 23, 2017": 15% of fake articles.
    "tweet_signature": (
        re.compile(rf"(?:[—–]\s*)?\(@\w{{1,30}}\)\s*{MONTH}\s+\d{{1,2}},\s+\d{{4}}"),
        " ",
    ),
    # Links: 14% of fake articles, almost no real ones (a few bit.ly links in tweet logs).
    "link": (re.compile(r"(?:https?://|www\.|pic\.twitter\.com/|bit\.ly/)\S+", re.I), " "),
    # Remaining @handles: 27% of fake, 1% of real.
    "handle": (re.compile(r"(?<![\w@])@\s?\w+"), " "),
    # Image credits close 48% of fake articles: "Featured image via Win McNamee/Getty Images".
    "image_credit": (
        re.compile(
            r"(?:Featured|Header|Top|Feature)\s+(?:image|photo)[sd]?\b[^.!?]{0,150}[.!?]?|"
            r"(?:Photo|Image|Screenshot|Screen\s?shot|Screen\s?grab|"
            r"Video\s?screen\s?(?:shot|grab))\s*(?:via|by|credit|courtesy of|from|:)"
            r"[^.!?]{0,150}[.!?]?",
            re.I,
        ),
        " ",
    ),
    "getty_images": (re.compile(r"\b(?:Getty\s+Images|Getty|screengrab|screenshot)\b", re.I), " "),
    # Title tags of fake sites: "[VIDEO]", "(TWEETS)": 27% of fake, no real article.
    "media_tag": (
        re.compile(
            r"[\[(]\s*(?:VIDEOS?|IMAGES?|TWEETS?|WATCH|DETAILS|BREAKING|GRAPHIC|PHOTOS?|"
            r"AUDIO|SCREENSHOTS?|MAP|LIVE|UPDATE[DS]?)\s*[\])]",
            re.I,
        ),
        " ",
    ),
    "title_prefix": (
        re.compile(
            r"^(?:BREAKING(?: NEWS)?|WATCH|JUST IN|EXCLUSIVE|LOL|WOW|BOOM|WHOA)\s*[:!-]+\s*"
        ),
        "",
    ),
    # Reuters house style: "(1600 GMT)" conversions and "[1814 EST]" stamps in tweet logs.
    "time_stamp": (re.compile(r"[\[(]\d{3,4}\s+(?:GMT|EST|EDT)[\])]"), " "),
    # Fake articles were collected with apostrophes stripped ("didn t", "Trump s"): 61% of fake
    # vs 7% of real. Restoring them stops "didn" and "t" fragments separating the classes.
    "stripped_nt": (re.compile(r"\b(\w+n) t\b(?!-)"), r"\1't"),
    "stripped_s": (re.compile(r"\b(\w+) s\b(?![-'])"), r"\1's"),
    "stripped_contraction": (
        re.compile(
            r"\b(I|we|you|they|he|she|it|that|there|who|what|would|should|could|might|"
            r"must) (m|re|ve|ll|d)\b",
            re.I,
        ),
        r"\1'\2",
    ),
}
# fmt: on

SPACES = re.compile(r"\s+")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")  # "Again ." left by a removed tag


def remove_shortcuts(text: str, rules: dict = SHORTCUT_RULES) -> tuple[str, list[str]]:
    """Return the text without source fingerprints and the names of the rules that fired."""
    fired = []
    for name, (pattern, replacement) in rules.items():
        text, n = pattern.subn(replacement, text)
        if n:
            fired.append(name)
    return SPACE_BEFORE_PUNCT.sub(r"\1", SPACES.sub(" ", text)).strip(), fired


def clean_shortcuts(text: str) -> str:
    """``remove_shortcuts`` without the report, for ``Series.map``."""
    return remove_shortcuts(text)[0]
