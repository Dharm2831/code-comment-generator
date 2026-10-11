"""Dataset loading, cleaning and tokenisation.

Each training example is a pair:
    input  : "Summarize Python: <function code without its docstring>"
    target : "<one-line description of what the function does>"
"""

import re

import torch
from torch.utils.data import Dataset

from src import config

LANG_LABEL = {"python": "Python", "java": "Java"}

_PY_DOCSTRING = re.compile(r'[rRuUbB]{0,2}("""|\'\'\')[\s\S]*?\1')
_DOC_TAGS = re.compile(
    r"(@param|@return|@throws|@see|@since|@author|@deprecated|@exception"
    r"|:param|:return|:rtype|:type|:raises|Args:|Arguments:|Parameters\b|Returns:|Returns\n|Example|>>>)"
)


# ------------------------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------------------------
def strip_python_docstring(code: str) -> str:
    """Remove the first triple-quoted string (the docstring) from a Python function.

    Without this the model could simply copy the answer out of its input.
    """
    return _PY_DOCSTRING.sub("", code, count=1)


def clean_code(code: str, lang: str) -> str:
    """Remove the docstring (Python) and squeeze all whitespace to single spaces."""
    if lang == "python":
        code = strip_python_docstring(code)
    return " ".join(code.split())


def clean_comment(doc: str) -> str:
    """Keep only the short summary part of a docstring / Javadoc."""
    doc = doc.strip()
    doc = _DOC_TAGS.split(doc, maxsplit=1)[0]           # drop @param, :return:, Args: ...
    doc = re.split(r"\n\s*\n", doc, maxsplit=1)[0]      # first paragraph only
    doc = re.sub(r"\{@\w+\s+([^}]*)\}", r"\1", doc)     # {@link Foo} -> Foo
    doc = re.sub(r"<[^>]+>", " ", doc)                  # HTML tags
    doc = " ".join(doc.split())
    doc = re.sub(r"\s+([,.;:])", r"\1", doc)            # no space before punctuation
    match = re.search(r"(?<=[a-z0-9\)])\.\s", doc + " ")  # first sentence only
    if match:
        doc = doc[: match.start() + 1]
    return doc.strip()


def is_good_pair(code: str, comment: str) -> bool:
    """Filter out examples that would teach the model bad habits."""
    words = comment.split()
    if not (3 <= len(words) <= 40):
        return False
    if not comment.isascii():                      # non-English comments
        return False
    low = comment.lower()
    if "http" in low or "todo" in low or "fixme" in low:
        return False
    if len(code) < 30:
        return False
    if comment[:40] in code:                       # comment still visible in the code (leak)
        return False
    return True


def make_source(code: str, lang: str) -> str:
    """Build the exact text that is fed to the model."""
    return f"Summarize {LANG_LABEL[lang]}: {clean_code(code, lang)}"


# ------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------
def load_raw_split(lang: str, split: str):
    """Download (first time) and return one split of one language."""
    from datasets import load_dataset

    return load_dataset(config.DATASET_NAME, lang, split=split, cache_dir=config.DATA_DIR)


def build_pairs(lang: str, split: str, limit: int, seed: int = config.SEED):
    """Return up to `limit` cleaned examples as dicts: {lang, source, target}."""
    raw = load_raw_split(lang, split).shuffle(seed=seed)
    pairs, seen = [], set()
    for row in raw:
        comment = clean_comment(row["docstring"])
        code = clean_code(row["code"], lang)
        if not is_good_pair(code, comment):
            continue
        if code in seen:                           # duplicate removal
            continue
        seen.add(code)
        pairs.append({"lang": lang, "source": make_source(row["code"], lang), "target": comment})
        if len(pairs) >= limit:
            break
    return pairs


def build_all_pairs(split: str, per_lang: int):
    """Cleaned examples for every language in config.LANGUAGES, mixed together."""
    pairs = []
    for lang in config.LANGUAGES:
        pairs.extend(build_pairs(lang, split, per_lang))
    return pairs


# ------------------------------------------------------------------------------
# Tokenisation
# ------------------------------------------------------------------------------
class CommentDataset(Dataset):
    """Turns text pairs into token ids once, up front."""

    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        sources = [p["source"] for p in pairs]
        targets = [p["target"] for p in pairs]
        self.input_ids = tokenizer(sources, max_length=config.MAX_SOURCE_LEN, truncation=True)["input_ids"]
        self.labels = tokenizer(targets, max_length=config.MAX_TARGET_LEN, truncation=True)["input_ids"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def make_collate_fn(pad_token_id: int):
    """Pads every example in a batch to the length of the longest one."""

    def collate(batch):
        max_in = max(len(b["input_ids"]) for b in batch)
        max_out = max(len(b["labels"]) for b in batch)
        input_ids = torch.full((len(batch), max_in), pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(batch), max_in), dtype=torch.long)
        labels = torch.full((len(batch), max_out), -100, dtype=torch.long)  # -100 = ignored by the loss
        for i, b in enumerate(batch):
            input_ids[i, : len(b["input_ids"])] = torch.tensor(b["input_ids"])
            attention[i, : len(b["input_ids"])] = 1
            labels[i, : len(b["labels"])] = torch.tensor(b["labels"])
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

    return collate
