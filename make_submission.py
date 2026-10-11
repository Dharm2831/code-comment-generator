"""Build the Part II submission:  Group_<N>_code.zip

Run from the project folder (with .venv active), giving your group number:
    python make_submission.py 7

It creates
    Group_7_code/
        Group_7_1.py               main file: runs the app, tests the model on data/test, or trains
        data/train/python/sample_1.json ... sample_500.json
        data/train/java/  sample_1.json ... sample_500.json
        data/validate/python/ and java/   100 samples each
        data/test/python/ and java/       100 samples each
        data/README.txt            where the data comes from
        src/, app.py, check_setup.py, requirements.txt, README.md
and zips it to Group_7_code.zip.

The samples are cleaned exactly like the training data, taken from the dataset already
downloaded into the dataset/ folder. The test and validation samples are the first 100 per
language of the same test and validation functions used in the report's evaluation.
The trained model (about 240 MB) is not included; it stays in models/ on the demo laptop.
"""

import json
import os
import shutil
import sys

from src import config
from src.preprocessing import clean_code, clean_comment, is_good_pair, load_raw_split, make_source, strip_python_docstring

COUNTS = {"train": 500, "validate": 100, "test": 100}          # per language
HF_SPLIT = {"train": "train", "validate": "validation", "test": "test"}
COPY = ["src", "app.py", "check_setup.py", "requirements.txt", "README.md"]


def sample_pairs(lang, split, limit):
    """Same order and filters as src/preprocessing.build_pairs, but keeps readable code."""
    raw = load_raw_split(lang, split).shuffle(seed=config.SEED)
    out, seen = [], set()
    for row in raw:
        comment = clean_comment(row["docstring"])
        cleaned = clean_code(row["code"], lang)
        if not is_good_pair(cleaned, comment) or cleaned in seen:
            continue
        seen.add(cleaned)
        code = strip_python_docstring(row["code"]) if lang == "python" else row["code"]
        out.append({
            "language": lang,
            "function_name": row.get("func_name", ""),
            "code": code.strip(),                       # the function, without its docstring
            "comment": comment,                         # target: first sentence of the docstring
            "model_input": make_source(row["code"], lang),
            "source_url": row.get("url", ""),
        })
        if len(out) >= limit:
            break
    return out


MAIN_TEMPLATE = r'''"""Group __N__ - AI-Based Code Comment Generator (Deep Learning mini project)

Usage (from this folder, with the project's Python environment active):
    python Group___N___1.py            start the web app  -> http://127.0.0.1:7860
    python Group___N___1.py test       test the trained model on data/test and print BLEU / chrF
    python Group___N___1.py train      fine-tune CodeT5-small (downloads the full dataset, ~3.5 h on GPU)

The trained model must be in models/comment-generator (copy it from the training folder),
or set the MODEL_ID environment variable to the Hugging Face model id.
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
# use the trained model from the parent project folder if it was not copied here
_parent_model = os.path.join(HERE, "..", "models", "comment-generator")
if not os.path.isdir(os.path.join("models", "comment-generator")) and os.path.isdir(_parent_model):
    os.environ.setdefault("MODEL_ID", os.path.abspath(_parent_model))


def load_split(split):
    """Read data/<split>/<language>/sample_*.json into a list of dicts."""
    rows = []
    for path in sorted(glob.glob(os.path.join("data", split, "*", "sample_*.json")),
                       key=lambda p: (p.split(os.sep)[-2], int(p.split("_")[-1].split(".")[0]))):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def test():
    import sacrebleu
    import torch
    from src import config
    from src.prediction import CommentGenerator

    rows = load_split("test")
    if not rows:
        sys.exit("No samples found in data/test")
    gen = CommentGenerator()
    preds = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            batch = [r["model_input"] for r in rows[i:i + 32]]
            enc = gen.tokenizer(batch, max_length=config.MAX_SOURCE_LEN, truncation=True,
                                padding=True, return_tensors="pt").to(gen.device)
            out = gen.model.generate(**enc, num_beams=config.NUM_BEAMS, max_new_tokens=config.MAX_TARGET_LEN,
                                     no_repeat_ngram_size=3, early_stopping=True)
            preds += [t.strip() for t in gen.tokenizer.batch_decode(out, skip_special_tokens=True)]
    refs = [r["comment"] for r in rows]
    print(f"Test samples: {len(rows)}")
    print(f"BLEU: {sacrebleu.corpus_bleu(preds, [refs]).score:.2f}   chrF: {sacrebleu.corpus_chrf(preds, [refs]).score:.2f}")
    for lang in ("python", "java"):
        idx = [i for i, r in enumerate(rows) if r["language"] == lang]
        if idx:
            p, r = [preds[i] for i in idx], [refs[i] for i in idx]
            print(f"  {lang:6s}  BLEU {sacrebleu.corpus_bleu(p, [r]).score:.2f}   chrF {sacrebleu.corpus_chrf(p, [r]).score:.2f}")
    print("\nExamples:")
    for r, p in list(zip(rows, preds))[:3] + list(zip(rows, preds))[-3:]:
        print(f"- [{r['language']}] {r['function_name']}\n    human : {r['comment']}\n    model : {p}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "app"
    if cmd == "app":
        import app
        app.demo.launch()
    elif cmd == "test":
        test()
    elif cmd == "train":
        from src import training
        sys.argv = ["training"]
        training.main()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
'''

DATA_README = """Data sample for Group {n} - AI-Based Code Comment Generator

Source: CodeXGLUE Code-to-Text dataset (Python and Java subsets), built from CodeSearchNet.
        https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text  (licence: C-UDA)

This folder holds a small sample of the data used in the project; the full training set
(406,121 cleaned pairs, about 480 MB download) is downloaded automatically by "python Group_{n}_1.py train".

Structure:  data/<train|validate|test>/<python|java>/sample_<k>.json
Per language: train {train}, validate {validate}, test {test} samples.

Each JSON file contains:
  language       python or java
  function_name  name of the function in the original repository
  code           the function source code, with its docstring removed
  comment        the target: first sentence of the original documentation comment
  model_input    the exact text given to the model ("Summarize Python: ..." / "Summarize Java: ...")
  source_url     link to the function on GitHub

The samples were cleaned exactly as in training (docstring removed, first sentence kept,
3-40 word ASCII comments, no URLs/TODO, duplicates removed) and taken in the same
shuffled order (seed 42) as the experiments, so the test and validate samples are the
first 100 per language of the evaluation and validation sets used in the report.
"""


def main():
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        sys.exit("Usage: python make_submission.py <group number>     e.g. python make_submission.py 7")
    n = sys.argv[1]
    out = f"Group_{n}_code"
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    total = 0
    for split, count in COUNTS.items():
        for lang in config.LANGUAGES:
            rows = sample_pairs(lang, HF_SPLIT[split], count)
            folder = os.path.join(out, "data", split, lang)
            os.makedirs(folder)
            for k, row in enumerate(rows, 1):
                with open(os.path.join(folder, f"sample_{k}.json"), "w", encoding="utf-8") as f:
                    json.dump(row, f, ensure_ascii=False, indent=2)
            total += len(rows)
            print(f"  data/{split}/{lang}: {len(rows)} samples")
    with open(os.path.join(out, "data", "README.txt"), "w", encoding="utf-8") as f:
        f.write(DATA_README.format(n=n, **COUNTS))

    with open(os.path.join(out, f"Group_{n}_1.py"), "w", encoding="utf-8") as f:
        f.write(MAIN_TEMPLATE.replace("__N__", n))
    for item in COPY:
        if os.path.isdir(item):
            shutil.copytree(item, os.path.join(out, item), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif os.path.exists(item):
            shutil.copy2(item, out)

    archive = shutil.make_archive(out, "zip", root_dir=".", base_dir=out)
    size = os.path.getsize(archive) / 1024 / 1024
    print(f"\nDone: {total} samples. Folder {out}/ and {archive} ({size:.1f} MB)")
    print(f"Check it with:  cd {out}  then  python Group_{n}_1.py test")


if __name__ == "__main__":
    main()
