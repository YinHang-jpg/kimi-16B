#!/usr/bin/env python3
"""NPU-oriented runner based on MoonshotAI/Moonlight examples/toy_train.py.

Upstream (unmodified copy):
  third_party/MoonshotAI-Moonlight/examples/toy_train.py

PATCHES vs upstream (only these; optimizer math is unchanged):
  1. Optional `import torch_npu` for Ascend.
  2. Device selection: npu > cuda > cpu.
  3. `--disable-torch-compile` (default on NPU) because Ascend often lacks
     full torch.compile support for the Newton-Schulz kernel path.
  4. `--local-data-dir` / `--tokenizer-path` for offline caches.
  5. `torch.load(..., weights_only=False)` for newer PyTorch compatibility
     when loading the token cache `.bin` produced by this script.

Official CLI (from upstream README) still applies, e.g.:
  python examples/toy_train_npu.py --model qwen --optimizer muon \\
      --dataset openwebtext-100k --hidden_size 896 --lr 1e-3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from loguru import logger
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Below: Muon / dataset / train loop copied from upstream toy_train.py with
# only the patches listed in the module docstring.
# ---------------------------------------------------------------------------


class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512, cache_dir="."):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        cache_path = os.path.join(self.cache_dir, f"{self.dataset_name}.bin")
        if os.path.exists(cache_path):
            self.tokens = torch.load(cache_path, weights_only=False)
        else:
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(self.tokens, cache_path)

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * (self.max_length)
        end_idx = start_idx + (self.max_length)
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data


def build_zeropower(disable_torch_compile: bool):
    def zeropower_via_newtonschulz5(G, steps):
        """Newton-Schulz iteration (from upstream / KellerJordan Muon)."""
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        if G.size(0) > G.size(1):
            X = X.T
        X = X / (X.norm() + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X

    if disable_torch_compile:
        return zeropower_via_newtonschulz5
    try:
        return torch.compile(zeropower_via_newtonschulz5)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"torch.compile unavailable ({exc}); running eager Newton-Schulz")
        return zeropower_via_newtonschulz5


class Muon(torch.optim.Optimizer):
    """Upstream MoonshotAI Muon (see third_party copy for original)."""

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        zeropower_fn=None,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        self.zeropower_fn = zeropower_fn
        for p in muon_params:
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = self.zeropower_fn(g, steps=group["ns_steps"])
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def pick_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model_and_dataloader(
    model_name,
    dataset_name,
    hidden_size,
    local_data_dir: str | None,
    tokenizer_path: str | None,
    cache_dir: str,
):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    ds_name = name2path[dataset_name]
    if local_data_dir and os.path.isdir(local_data_dir):
        train_dataset = load_from_disk(local_data_dir)
    else:
        train_dataset = load_dataset(ds_name, trust_remote_code=True)

    if model_name == "qwen":
        tok_src = tokenizer_path or "Qwen/Qwen2.5-0.5B"
        tokenizer = Qwen2Tokenizer.from_pretrained(tok_src, trust_remote_code=True)
    else:
        assert 0, f"model {model_name} not supported"

    train_dataset = MoonDataset(
        dataset_name, train_dataset, tokenizer, cache_dir=cache_dir
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    if model_name == "qwen":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=4864,
            max_position_embeddings=513,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=16,
            num_hidden_layers=12,
            num_key_value_heads=16,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
        model = Qwen2ForCausalLM(config)
    else:
        assert 0, f"model {model_name} not supported"
    return model, train_loader


def get_optimizer(optimizer_name, model, lr=1e-3, wd=0.1, zeropower_fn=None):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    if optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]
        return Muon(
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
            zeropower_fn=zeropower_fn,
        )
    assert 0, "optimizer not supported"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--optimizer", type=str, default="muon")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=896)
    parser.add_argument(
        "--local-data-dir",
        type=str,
        default="",
        help="load_from_disk path produced by scripts/prefetch_official_toy_data.py",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help="local Qwen2.5-0.5B tokenizer dir; default Hub id",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="toy_cache",
        help="directory for tokenized .bin cache",
    )
    parser.add_argument(
        "--disable-torch-compile",
        action="store_true",
        default=True,
        help="default True on this NPU runner",
    )
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="try torch.compile for Newton-Schulz (may fail on Ascend)",
    )
    args = parser.parse_args()

    disable_compile = not args.enable_torch_compile
    zeropower_fn = build_zeropower(disable_compile)

    Path("logs").mkdir(exist_ok=True)
    logger.add(f"logs/train_{args.model}_{args.optimizer}_lr{args.lr}.log")

    model, train_loader = get_model_and_dataloader(
        args.model,
        args.dataset,
        args.hidden_size,
        args.local_data_dir or None,
        args.tokenizer_path or None,
        args.cache_dir,
    )
    optimizer = get_optimizer(
        args.optimizer, model, lr=args.lr, wd=args.wd, zeropower_fn=zeropower_fn
    )

    device = pick_device()
    logger.info(f"Using device: {device}")
    model.to(device)
    model.train()

    epoch = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epoch,
        num_cycles=0.5,
    )
    for epoch_i in range(epoch):
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            logger.info(
                f"Epoch: {epoch_i} Step: {step} LR: {optimizer.param_groups[0]['lr']} Training loss: {loss.item()}"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"TRAIN_FAIL: {exc}", file=sys.stderr)
        raise
