"""Model loading and a description of its architecture.

The model is CodeT5-small: a Transformer encoder-decoder (sequence-to-sequence).
The encoder reads the code tokens; the decoder writes the comment one token at a time.
"""

import json
import os

import torch
from huggingface_hub import hf_hub_download
from transformers import AddedToken, AutoModelForSeq2SeqLM, AutoTokenizer

from src import config


def get_device() -> torch.device:
    """Use the NVIDIA GPU when PyTorch can see it, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(name_or_path: str):
    """Load the tokenizer.

    The original codet5-small files are in an old format: its 100 extra special tokens
    (<extra_id_0> ... <extra_id_99>) are stored as JSON objects, which transformers 5.x
    refuses with "TypeError: Input must be a List[Union[str, AddedToken]]". When that
    happens we read those tokens ourselves and hand them over in the form it expects.
    """
    try:
        return AutoTokenizer.from_pretrained(name_or_path)
    except TypeError:
        if os.path.isdir(name_or_path):
            path = os.path.join(name_or_path, "special_tokens_map.json")
        else:
            path = hf_hub_download(name_or_path, "special_tokens_map.json")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f).get("additional_special_tokens", [])
        extra = [
            AddedToken(t["content"], lstrip=t.get("lstrip", False), rstrip=t.get("rstrip", False),
                       single_word=t.get("single_word", False), normalized=t.get("normalized", True),
                       special=True) if isinstance(t, dict) else t
            for t in raw
        ]
        return AutoTokenizer.from_pretrained(name_or_path, extra_special_tokens=extra)


def load_model(name_or_path: str = config.BASE_MODEL, device: torch.device | None = None):
    """Load tokenizer + model from the Hugging Face Hub or from a local folder."""
    device = device or get_device()
    tokenizer = load_tokenizer(name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path)
    model.to(device)
    return tokenizer, model


def describe_model(model) -> dict:
    """Facts about the network, for the 'Model Architecture / Configuration' tables in the report."""
    cfg = model.config
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "architecture": "Transformer encoder-decoder (T5 family)",
        "encoder_layers": cfg.num_layers,
        "decoder_layers": cfg.num_decoder_layers,
        "hidden_size_d_model": cfg.d_model,
        "feed_forward_size_d_ff": cfg.d_ff,
        "attention_heads": cfg.num_heads,
        "activation_function": getattr(cfg, "feed_forward_proj", "relu"),
        "output_activation": "softmax over the vocabulary",
        "dropout": cfg.dropout_rate,
        "vocabulary_size_output_classes": cfg.vocab_size,
        "total_parameters": total,
        "trainable_parameters": trainable,
    }


if __name__ == "__main__":
    # Run:  python -m src.model   -> prints the architecture summary
    import json

    _, m = load_model()
    print(json.dumps(describe_model(m), indent=2))
    print(m)
