"""Fine-tune CodeT5-small to write comments for Python and Java functions.

Run from the project folder:
    python -m src.training --quick      (2-3 minute smoke test, tiny data)
    python -m src.training              (real training run)

Outputs:
    models/comment-generator/           best model (lowest validation loss)
    results/training_history.json       loss / accuracy per epoch
    results/training_curves.png         graph for the report
    results/run_config.json             every setting + dataset sizes + model facts
"""

import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src import config
from src.model import describe_model, get_device, load_model
from src.preprocessing import CommentDataset, build_all_pairs, make_collate_fn


def autocast(device):
    """Mixed precision on the GPU (faster, less memory). bfloat16 is stable for T5 models."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


def count_correct(logits, labels):
    """How many target tokens the model predicted correctly (ignores padding)."""
    mask = labels != -100
    correct = ((logits.argmax(dim=-1) == labels) & mask).sum().item()
    return correct, mask.sum().item()


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """Average cross-entropy loss and token accuracy over a data loader."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device):
            out = model(**batch)
        c, n = count_correct(out.logits, batch["labels"])
        loss_sum += out.loss.item() * n
        correct += c
        total += n
    return loss_sum / max(total, 1), correct / max(total, 1)


def plot_curves(history, path):
    """Training vs validation loss and accuracy, one panel each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    blue, orange, ink, grid = "#2a78d6", "#eb6834", "#52514e", "#e1e0d9"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [("Loss (cross-entropy)", "train_loss", "val_loss"),
              ("Token accuracy", "train_accuracy", "val_accuracy")]
    for ax, (title, train_key, val_key) in zip(axes, panels):
        ax.plot(epochs, [h[train_key] for h in history], color=blue, lw=2, marker="o", label="Training")
        ax.plot(epochs, [h[val_key] for h in history], color=orange, lw=2, marker="s", ls="--", label="Validation")
        ax.set_title(title, loc="left", fontsize=12, color="#0b0b0b")
        ax.set_xlabel("Epoch", color=ink)
        ax.set_xticks(epochs)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=ink)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the code comment generator")
    parser.add_argument("--samples", type=int, default=config.TRAIN_PER_LANG, help="training examples per language")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    args = parser.parse_args()

    val_per_lang = config.VAL_PER_LANG
    if args.quick:
        args.samples, args.epochs, val_per_lang = 400, 1, 100

    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = get_device()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type != "cuda":
        print("WARNING: no GPU found. Training on CPU will be very slow. Run check_setup.py.")

    # 1. Data ------------------------------------------------------------------
    print("Loading and cleaning the dataset (first run downloads it)...")
    train_pairs = build_all_pairs("train", args.samples)
    val_pairs = build_all_pairs("validation", val_per_lang)
    random.shuffle(train_pairs)
    print(f"Training examples: {len(train_pairs)}   Validation examples: {len(val_pairs)}")
    print("Example input :", train_pairs[0]["source"][:150], "...")
    print("Example target:", train_pairs[0]["target"])

    # 2. Model -----------------------------------------------------------------
    tokenizer, model = load_model(config.BASE_MODEL, device)
    collate = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(CommentDataset(train_pairs, tokenizer), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(CommentDataset(val_pairs, tokenizer), batch_size=args.batch_size * 2,
                            collate_fn=collate)

    # 3. Optimiser and learning-rate schedule ------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config.WARMUP_RATIO * total_steps), total_steps)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    history, best_val = [], float("inf")
    start = time.time()

    # 4. Training loop -----------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device):
                out = model(**batch)                 # forward pass (loss = cross-entropy)
            optimizer.zero_grad()
            out.loss.backward()                      # backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()                         # weight update
            scheduler.step()

            c, n = count_correct(out.logits.detach(), batch["labels"])
            loss_sum += out.loss.item() * n
            correct += c
            total += n
            if step % 100 == 0 or step == len(train_loader):
                print(f"  epoch {epoch}  step {step}/{len(train_loader)}  loss {loss_sum / total:.4f}", flush=True)

        val_loss, val_acc = evaluate_loss(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total,
               "val_loss": val_loss, "val_accuracy": val_acc, "minutes_elapsed": (time.time() - start) / 60}
        history.append(row)
        print(f"Epoch {epoch}: train loss {row['train_loss']:.4f}  val loss {val_loss:.4f}  "
              f"train acc {row['train_accuracy']:.3f}  val acc {val_acc:.3f}")

        if val_loss < best_val:                      # checkpoint: keep only the best model
            best_val = val_loss
            model.save_pretrained(config.MODEL_DIR)
            tokenizer.save_pretrained(config.MODEL_DIR)
            print(f"  saved best model to {config.MODEL_DIR}")

    # 5. Save everything the report needs ----------------------------------------
    minutes = (time.time() - start) / 60
    with open(os.path.join(config.RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_curves(history, os.path.join(config.RESULTS_DIR, "training_curves.png"))
    run_config = {
        "base_model": config.BASE_MODEL,
        "dataset": config.DATASET_NAME,
        "languages": config.LANGUAGES,
        "training_samples": len(train_pairs),
        "validation_samples": len(val_pairs),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "optimizer": f"AdamW (weight decay {config.WEIGHT_DECAY})",
        "lr_schedule": f"linear warmup ({config.WARMUP_RATIO:.0%}) then linear decay",
        "loss_function": "cross-entropy over the vocabulary",
        "max_input_tokens": config.MAX_SOURCE_LEN,
        "max_output_tokens": config.MAX_TARGET_LEN,
        "gradient_clipping": config.GRAD_CLIP,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "training_time_minutes": round(minutes, 1),
        "best_validation_loss": best_val,
        "model": describe_model(model),
    }
    with open(os.path.join(config.RESULTS_DIR, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"\nDone in {minutes:.1f} minutes. Graph and numbers are in the '{config.RESULTS_DIR}' folder.")
    print("Next: python -m src.evaluation")


if __name__ == "__main__":
    main()
