"""Mean token negative log-likelihood. Mean is over non-padding tokens, never over sequences."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class NLLResult:
    nll: float
    n_tokens: int

    @property
    def perplexity(self) -> float:
        return math.exp(self.nll)


def nll(model, batches: Iterable, device=None) -> NLLResult:
    """`batches` yields either a LongTensor [B, L] or a dict with input_ids / attention_mask.

    The first token of each sequence is context only (no prediction target). Eval mode and
    no_grad are enforced; the model's previous train/eval mode is restored afterwards.
    """
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        with torch.no_grad():
            for batch in batches:
                if isinstance(batch, dict):
                    ids, mask = batch["input_ids"], batch.get("attention_mask")
                else:
                    ids, mask = batch, None
                if mask is None:
                    mask = torch.ones_like(ids)
                if device is not None:
                    ids, mask = ids.to(device), mask.to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
                targets = ids[:, 1:]
                valid = mask[:, 1:].bool()
                loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
                total += loss[valid].sum().item()
                count += int(valid.sum().item())
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("no non-padding target tokens to evaluate")
    return NLLResult(nll=total / count, n_tokens=count)
