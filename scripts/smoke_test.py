"""Smoke test: load the dev model, run one forward pass on MPS, print the loss."""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import math

import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
TEXT = "The quick brown fox jumps over the lazy dog."


def main() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    info = model_info(MODEL)
    print(f"model={MODEL} revision={info.sha} license={info.card_data.license if info.card_data else 'unknown'}")
    print(f"device={device} torch={torch.__version__}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()

    batch = tok(TEXT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**batch, labels=batch["input_ids"])
    loss = out.loss.item()
    print(f"loss={loss:.4f} ppl={math.exp(loss):.2f}")
    assert math.isfinite(loss), "loss is not finite"


if __name__ == "__main__":
    main()
