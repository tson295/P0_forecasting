from pathlib import Path
import random

import numpy as np
import pytest
import torch

from src.config import Config, TrainConfig
from src.models import build_model, DEFAULTS
from src.training.checkpoint import (from_pretrained, capture_rng, restore_rng,
    save_training_checkpoint, load_training_state)
from src.training.trainer import make_loader


@pytest.mark.parametrize("name", DEFAULTS)
@pytest.mark.parametrize("history", [49, 64, 98, 146])
def test_shapes_history_and_finite(name, history):
    # Tiny architecture variants keep edge-shape unit tests bounded.
    kw = {}
    if name == "moderntcn": kw = dict(dims=[8, 8, 8, 8])
    if name == "patchtst": kw = dict(d_model=16, heads=2, layers=1, ffn_dim=32)
    channels = 20 if name == "ofi_lstm" else 38 if name == "hfformer" else 40
    model = build_model(name, history, channels, **kw).eval()
    x = torch.randn(2, history, 2, 10, 2) if name == "lit" else torch.randn(2, history, channels)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 3) and torch.isfinite(y).all()
    if name == "e0": assert (y == 0).all()


def test_hfformer_spike_gradients_and_batch_independence():
    model = build_model("hfformer", 49, 38).eval()
    x = torch.randn(2, 49, 38)
    y = model(x)
    torch.testing.assert_close(model(x[:1]), y[:1], rtol=1e-5, atol=1e-6)
    y.square().sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_moderntcn_singleton_final_minibatch():
    model = build_model("moderntcn", 49, 40, dims=[8, 8, 8, 8]).train()
    y = model(torch.randn(1, 49, 40))
    assert y.shape == (1, 3) and torch.isfinite(y).all()
    y.square().sum().backward()


def test_named_attention_and_freeze(tmp_path):
    for name in ("hfformer", "patchtst", "lit"):
        model = build_model(name, 49, 38 if name == "hfformer" else 40)
        modules = dict(model.named_modules())
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert any(n.endswith(proj) for n in modules)
        model.freeze_backbone(); model.train()
        assert all(p.requires_grad == n.startswith("head.") for n, p in model.named_parameters())
        assert all(not child.training for n, child in model.named_children() if n != "head")
        model.unfreeze()
        assert all(p.requires_grad for p in model.parameters())


def test_full_training_state_roundtrip_without_training(tmp_path):
    model = build_model("ofi_lstm", 49, 20)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator().manual_seed(42)
    state = dict(epoch=-1, global_step=0, best_validation_mse=1., head_only=False,
                 scaler=scaler.state_dict(), rng=capture_rng(generator))
    metadata = {"experiment": Config().to_dict()}
    path = tmp_path/"last"
    save_training_checkpoint(model, path, metadata, optimizer, scheduler, state)
    expected = (random.random(), np.random.rand(), torch.rand(2), torch.randperm(10, generator=generator))
    restored = from_pretrained(path)
    opt = torch.optim.AdamW(restored.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    loaded = load_training_state(path, opt, sched, scaler, generator)
    assert loaded["global_step"] == 0
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(2), expected[2], rtol=0, atol=0)
    torch.testing.assert_close(torch.randperm(10, generator=generator), expected[3], rtol=0, atol=0)
    assert opt.state_dict() == optimizer.state_dict()
    assert sched.state_dict() == scheduler.state_dict()
    save_training_checkpoint(model, path, metadata, optimizer, scheduler, state)
    assert not path.with_name("last.previous").exists()


def test_loader_permutation_resume_and_zero_workers():
    ds = torch.utils.data.TensorDataset(torch.arange(100))
    c = TrainConfig(batch_size=16, num_workers=0)
    g = torch.Generator().manual_seed(9)
    loader = make_loader(ds, c, torch.device("cpu"), True, g)
    list(loader)
    state = capture_rng(g)
    expected = torch.cat([x[0] for x in loader])
    restore_rng(state, g)
    recreated = make_loader(ds, c, torch.device("cpu"), True, g)
    torch.testing.assert_close(torch.cat([x[0] for x in recreated]), expected)
