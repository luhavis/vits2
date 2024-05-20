""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.
"""
pad = ["_"]
# _punctuation = ';:,.!?¡¿—…"«»“” '
punctuation = ["!", "?", "…", ",", ".", "'", "-", "¿", "¡", " "]
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
others = ["SP", "UNK"]

korean_symbols = ["(", ")", "~", "\\", "[", "]", "/", "^", ":", "ㄸ", "*"]

_jamo_leads = [chr(_) for _ in range(0x1100, 0x1113)]
_jamo_vowels = [chr(_) for _ in range(0x1161, 0x1176)]
_jamo_tails = [chr(_) for _ in range(0x11A8, 0x11C3)]

_new_korean_sybols = [
    "ㄱ",
    "ㄴ",
    "ㄷ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅅ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
    "ㄲ",
    "ㄸ",
    "ㅃ",
    "ㅆ",
    "ㅉ",
    "ㅏ",
    "ㅓ",
    "ㅗ",
    "ㅜ",
    "ㅡ",
    "ㅣ",
    "ㅐ",
    "ㅔ",
]

# Export all symbols
symbols = (
    pad
    + list(punctuation)
    + list(letters)
    + list(letters_ipa)
    + korean_symbols
    + _jamo_leads
    + _jamo_vowels
    + _jamo_tails
    + _new_korean_sybols
)

# Special symbol ids
SPACE_ID = symbols.index(" ")
