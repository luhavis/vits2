from typing import List

from torchtext.vocab import Vocab


def tokenizer(text: str, vocab: Vocab, cleaner_names: List[str], language="ko-kr", cleaned_text=False) -> List[int]:

    if not cleaned_text:
        return _clean_text()
    else:
        return list(map(int, text.split("\t")))

def detokenizer(sequence: List[int], vocab: Vocab) -> str:
    return "".join(vocab.lookup_tokens(sequence))

def _clean_text(text: str, vocab: Vocab, cleaner_names: List[str], language="ko-kr") -> str:
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        assert callable(cleaner), f"Unknown cleaner: {name}"
        text = cleaner(text, vocab=vocab, language=language)
    return text

if __name__ == "__main__":
    from utils.task import load_vocab

    vocab = load_vocab("datasets/ljs_base/vocab.txt")
    cleaner_names = ["phonemize_text", "add_spaces", "tokenize_text", "delete_unks", "add_bos_eos", "detokenize_sequence"]
    text = "Well, I like pizza. <laugh> You know … Who doesn't like pizza? <laugh>"
    print(tokenizer(text, vocab, cleaner_names, language="en-us", cleaned_text=False))