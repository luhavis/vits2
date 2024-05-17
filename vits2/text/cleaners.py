import re

from phonemizer import phonemize
from unidecode import unidecode

from vits2.text import korean

# Regular expression matching whitespace
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations
_abbreviations = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text: str):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


# def expand_numbers(text: str):
#     return normalize_numbers(text: str)


def lowercase(text: str):
    return text.lower()


def collapse_whitespace(text: str):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text: str):
    return unidecode(text)


def basic_cleaners(text: str):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text: str):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text: str):
    """Pipeline for English text, including abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language="en-us", backend="espeak", strip=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_cleaners2(text: str):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def korean_cleaners(text: str):
    """Pipeline for Korean text"""
    text = korean.latin_to_hangul(text)
    text = korean.number_to_hangul(text)
    text = korean.divide_hangul(text)
    if re.match("[\u3131-\u3163]", text[-1]):
        text += "."
    return text
