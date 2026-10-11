"""Test the trained model on functions it has never seen.

Run after training:
    python -m src.evaluation
    python -m src.evaluation --quick     (small test set, for a smoke test)

Outputs (all real measurements, ready for the report):
    results/test_metrics.json            BLEU, chrF, loss, token accuracy, inference time,
                                         per-language scores, and the same scores for the
                                         model BEFORE fine-tuning (model comparison)
    results/sample_predictions.csv       code, human comment, generated comment
"""

import argparse
import csv
import json
import os
import time

import sacrebleu
import torch
from torch.utils.data import DataLoader

from src import config
from src.model import get_device, load_model
from src.preprocessing import CommentDataset, build_all_pairs, make_collate_fn
from src.training import evaluate_loss


@torch.no_grad()
def generate_all(tokenizer, model, pairs, device, batch_size=32):
    """Generate a comment for every test example. Returns (predictions, seconds per function)."""
    model.eval()
    predictions, start = [], time.time()
    for i in range(0, len(pairs), batch_size):
        sources = [p["source"] for p in pairs[i:i + batch_size]]
        enc = tokenizer(sources, max_length=config.MAX_SOURCE_LEN, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        out = model.generate(**enc, num_beams=config.NUM_BEAMS, max_new_tokens=config.MAX_TARGET_LEN,
                             no_repeat_ngram_size=3, early_stopping=True)
        predictions.extend(t.strip() for t in tokenizer.batch_decode(out, skip_special_tokens=True))
        print(f"  generated {len(predictions)}/{len(pairs)}", flush=True)
    return predictions, (time.time() - start) / max(len(pairs), 1)


def text_scores(predictions, references):
    """BLEU and chrF: how close the generated comments are to the human-written ones (0-100)."""
    return {
        "bleu": round(sacrebleu.corpus_bleu(predictions, [references]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(predictions, [references]).score, 2),
    }


def score_model(name_or_path, pairs, device):
    tokenizer, model = load_model(name_or_path, device)
    loader = DataLoader(CommentDataset(pairs, tokenizer), batch_size=32,
                        collate_fn=make_collate_fn(tokenizer.pad_token_id))
    loss, accuracy = evaluate_loss(model, loader, device)
    predictions, seconds = generate_all(tokenizer, model, pairs, device)
    references = [p["target"] for p in pairs]
    result = {"test_loss": round(loss, 4), "token_accuracy": round(accuracy, 4),
              **text_scores(predictions, references),
              "inference_seconds_per_function": round(seconds, 4), "per_language": {}}
    for lang in config.LANGUAGES:
        idx = [i for i, p in enumerate(pairs) if p["lang"] == lang]
        if idx:
            result["per_language"][lang] = text_scores([predictions[i] for i in idx],
                                                       [references[i] for i in idx])
    return result, predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained comment generator")
    parser.add_argument("--samples", type=int, default=config.TEST_PER_LANG, help="test examples per language")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true", help="do not score the model before fine-tuning")
    args = parser.parse_args()
    if args.quick:
        args.samples = 50

    device = get_device()
    pairs = build_all_pairs("test", args.samples)
    print(f"Test examples: {len(pairs)}")

    print("Scoring the fine-tuned model...")
    tuned, predictions = score_model(config.MODEL_DIR, pairs, device)
    metrics = {"test_samples": len(pairs), "device": str(device), "fine_tuned_model": tuned}

    if not args.skip_baseline:
        print("Scoring the original model before fine-tuning (for comparison)...")
        metrics["before_fine_tuning"], _ = score_model(config.BASE_MODEL, pairs, device)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Sample predictions: first 10 Python + first 10 Java test functions (not hand-picked)
    with open(os.path.join(config.RESULTS_DIR, "sample_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "code", "human_comment", "generated_comment"])
        for lang in config.LANGUAGES:
            rows = [(p, pred) for p, pred in zip(pairs, predictions) if p["lang"] == lang][:10]
            for p, pred in rows:
                writer.writerow([lang, p["source"].split(": ", 1)[1], p["target"], pred])

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved to '{config.RESULTS_DIR}/test_metrics.json' and '{config.RESULTS_DIR}/sample_predictions.csv'")


if __name__ == "__main__":
    main()
