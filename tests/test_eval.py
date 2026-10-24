import math
import subprocess
import sys
import textwrap

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from commitgate.counters import ComputeCounter
from commitgate.eval.nll import nll
from commitgate.seed import set_seed

VOCAB = 100


def tiny_model(seed: int = 0) -> GPT2LMHeadModel:
    set_seed(seed)
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=32, n_layer=1, n_head=2)
    return GPT2LMHeadModel(cfg)


def data(seed: int = 1, n: int = 4, length: int = 16) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n, length), generator=g)


def test_deterministic():
    model, ids = tiny_model(), data()
    a, b = nll(model, [ids]), nll(model, [ids])
    assert a.nll == b.nll and a.n_tokens == b.n_tokens


def test_padding_does_not_change_result():
    model, ids = tiny_model(), data()
    plain = nll(model, [ids])
    padded_ids = torch.cat([ids, torch.zeros(4, 5, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones_like(ids), torch.zeros(4, 5, dtype=torch.long)], dim=1)
    padded = nll(model, [{"input_ids": padded_ids, "attention_mask": mask}])
    assert padded.n_tokens == plain.n_tokens
    assert padded.nll == pytest.approx(plain.nll, abs=1e-5)


def test_random_init_close_to_log_vocab():
    res = nll(tiny_model(), [data(n=32, length=32)])
    assert res.nll == pytest.approx(math.log(VOCAB), abs=0.2)


def test_perplexity_is_exp_nll():
    res = nll(tiny_model(), [data()])
    assert res.perplexity == pytest.approx(math.exp(res.nll))


def test_token_mean_not_sequence_mean():
    model = tiny_model()
    short, long = data(n=1, length=4), data(seed=2, n=1, length=40)
    both = nll(model, [short, long])
    s, l = nll(model, [short]), nll(model, [long])
    expected = (s.nll * s.n_tokens + l.nll * l.n_tokens) / (s.n_tokens + l.n_tokens)
    assert both.nll == pytest.approx(expected, abs=1e-6)


def test_restores_train_mode():
    model = tiny_model().train()
    nll(model, [data()])
    assert model.training
    model.eval()
    nll(model, [data()])
    assert not model.training


def test_identical_across_processes():
    code = textwrap.dedent(
        """
        from tests.test_eval import tiny_model, data
        from commitgate.eval.nll import nll
        print(repr(nll(tiny_model(), [data()]).nll))
        """
    )
    outs = [
        subprocess.check_output([sys.executable, "-c", code], text=True).strip().splitlines()[-1]
        for _ in range(2)
    ]
    assert outs[0] == outs[1]


def test_compute_counter():
    c = ComputeCounter()
    c.add("train", 100)
    c.add("gate", 20)
    c.add("oracle", 1000)
    assert c.method_cost() == 120
    assert c.get("oracle") == 1000
    with pytest.raises(KeyError):
        c.add("bogus", 1)
