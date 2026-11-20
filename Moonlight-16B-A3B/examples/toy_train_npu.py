#!/usr/bin/env python3
"""NPU-oriented runner based on MoonshotAI/Moonlight examples/toy_train.py.

Upstream (unmodified copy):
  third_party/MoonshotAI-Moonlight/examples/toy_train.py

PATCHES vs upstream (optimizer math unchanged):
  1. Optional `import torch_npu` for Ascend.
  2. Device selection: npu > cuda > cpu.
  3. Disable torch.compile by default on NPU.
  4. `--local-data-dir` / `--tokenizer-path` for offline caches.
  5. Safer logging, dtype cast, cache-off while training, `--max-steps`.
  6. Absolute log/cache paths under this package directory.
  7. NPU default dtype=float16 (many Ascend chips lack bf16 Embedding kernels).
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import sys
import traceback
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

faulthandler.enable(all_threads=True)


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
            logger.info(f"Loading token cache: {cache_path}")
            self.tokens = torch.load(cache_path, weights_only=False)
            logger.info(f"Loaded {len(self.tokens):,} tokens from cache")
            return

        texts = self.texts
        total = len(texts)
        use_tqdm = sys.stderr.isatty()
        logger.info(f"Tokenizing {total} texts into {cache_path}")
        if use_tqdm:
            iterator = tqdm(
                texts,
                desc="Tokenizing texts",
                total=total,
                dynamic_ncols=True,
                mininterval=1.0,
                file=sys.stderr,
            )
            for text in iterator:
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
        else:
            step = max(1, total // 100)
            for i, text in enumerate(texts):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
                done = i + 1
                if done % step == 0 or done == total:
                    pct = 100.0 * done / total
                    logger.info(f"Tokenizing texts: {done}/{total} ({pct:.1f}%)")

        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save(self.tokens, cache_path)
        logger.info(f"Saved token cache ({len(self.tokens):,} tokens) -> {cache_path}")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length
        token_slice = self.tokens[start_idx:end_idx]
        return torch.tensor(token_slice, dtype=torch.long)


def resolve_train_dtype(name: str, device: torch.device) -> torch.dtype:
    """Pick weight dtype. Ascend Embedding often lacks bf16 kernels (err 561103)."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name != "auto":
        if name not in mapping:
            raise ValueError(f"unsupported --dtype {name!r}")
        return mapping[name]
    # auto: prefer fp16 on NPU to avoid GatherV2AiCore/bf16 Embedding failures
    if device.type == "npu":
        return torch.float16
    return torch.bfloat16


def build_zeropower(disable_torch_compile: bool, ns_dtype: torch.dtype):
    def zeropower_via_newtonschulz5(G, steps):
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        # Upstream uses bf16; on NPU fall back to fp16/fp32 when bf16 kernels are missing.
        X = G.to(dtype=ns_dtype)
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
        if zeropower_fn is None:
            raise ValueError("zeropower_fn is required")
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


def pick_device(prefer: str = "auto") -> torch.device:
    prefer = (prefer or "auto").lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "npu":
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise RuntimeError("--device npu requested but torch.npu is unavailable")
        return torch.device("npu:0")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda:0")
    # auto
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def probe_npu_embedding(dtype: torch.dtype) -> tuple[bool, str]:
    """Return (ok, detail). Isolates GatherV2/Embedding kernel health."""
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return False, "npu unavailable"
    try:
        w = torch.randn(128, 32, device="npu:0", dtype=dtype)
        for idx_dtype in (torch.long, torch.int32):
            idx = torch.randint(0, 128, (2, 8), device="npu:0", dtype=idx_dtype)
            out = torch.nn.functional.embedding(idx, w)
            torch.npu.synchronize()
            _ = float(out.float().sum().cpu())
        return True, f"ok dtype={dtype}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def enable_cpu_embedding(model: torch.nn.Module) -> None:
    """Keep embed_tokens on CPU; untie lm_head so the rest can stay on NPU."""
    embed = model.get_input_embeddings()
    if getattr(model.config, "tie_word_embeddings", False):
        # Break weight tying so lm_head can remain on accelerator.
        with torch.no_grad():
            model.lm_head.weight = torch.nn.Parameter(
                embed.weight.detach().clone()
            )
        model.config.tie_word_embeddings = False
        if hasattr(model, "tie_weights"):
            # Prevent transformers from re-tying later.
            model.tie_weights = lambda: None  # type: ignore[method-assign]
        logger.info("Untied lm_head from embed_tokens for CPU embedding path")
    embed.to("cpu")
    logger.info("embed_tokens moved to CPU (NPU Embedding kernel workaround)")


def forward_with_optional_cpu_embed(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    train_dtype: torch.dtype,
    embed_on_cpu: bool,
):
    if not embed_on_cpu:
        batch_dev = batch.to(device)
        return model(input_ids=batch_dev, labels=batch_dev)

    # Embedding on CPU (works around broken aclnnEmbedding / GatherV2AiCore).
    labels = batch.to(device)
    embed = model.get_input_embeddings()
    inputs_embeds = embed(batch.cpu()).to(device=device, dtype=train_dtype)
    return model(inputs_embeds=inputs_embeds, labels=labels)


def synchronize(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def device_mem_summary(device: torch.device) -> str:
    try:
        if device.type == "npu" and hasattr(torch, "npu"):
            free, total = torch.npu.mem_get_info(device)
            return f"npu mem free={free/1e9:.2f}GB total={total/1e9:.2f}GB"
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            return f"cuda mem free={free/1e9:.2f}GB total={total/1e9:.2f}GB"
    except Exception as exc:  # noqa: BLE001
        return f"mem info unavailable ({exc})"
    return "cpu"


def resolve_path(path: str | None, repo_root: Path) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = repo_root / p
    return str(p)


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
        logger.info(f"Loading dataset from disk: {local_data_dir}")
        train_dataset = load_from_disk(local_data_dir)
    else:
        logger.info(f"Loading dataset from Hub: {ds_name}")
        train_dataset = load_dataset(ds_name, trust_remote_code=True)

    if model_name == "qwen":
        tok_src = tokenizer_path or "Qwen/Qwen2.5-0.5B"
        logger.info(f"Loading tokenizer: {tok_src}")
        tokenizer = Qwen2Tokenizer.from_pretrained(tok_src, trust_remote_code=True)
    else:
        assert 0, f"model {model_name} not supported"

    train_dataset = MoonDataset(
        dataset_name, train_dataset, tokenizer, cache_dir=cache_dir
    )
    logger.info(
        f"Dataset ready: {len(train_dataset)} sequences "
        f"(max_length={train_dataset.max_length}, tokens={len(train_dataset.tokens):,})"
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Token dataset is empty; cannot train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

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
            use_cache=False,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
        logger.info(
            f"Building Qwen2ForCausalLM hidden_size={hidden_size} "
            f"layers={config.num_hidden_layers}"
        )
        model = Qwen2ForCausalLM(config)
    else:
        assert 0, f"model {model_name} not supported"
    return model, train_loader, tokenizer


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    tokenizer,
    meta: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # HF save prefers CPU tensors; keep training device intact via state copy.
    to_save = model.module if hasattr(model, "module") else model
    cpu_model = to_save.to("cpu")
    cpu_model.config.use_cache = True
    cpu_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    meta_path = output_dir / "train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(f"Saved checkpoint -> {output_dir}")


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
        logger.info(
            f"Muon param tensors={len(muon_params)}, AdamW param tensors={len(adamw_params)}"
        )
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
    parser.add_argument("--local-data-dir", type=str, default="")
    parser.add_argument("--tokenizer-path", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default="toy_cache")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="stop after N optimizer steps; -1 means one full epoch",
    )
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="try torch.compile for Newton-Schulz (may fail on Ascend)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="model weight dtype; auto=float16 on NPU, bfloat16 elsewhere",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "npu", "cuda", "cpu"],
        help="force device; use cpu if NPU kernels are broken",
    )
    parser.add_argument(
        "--embed-on-cpu",
        action="store_true",
        help="run Embedding on CPU and pass inputs_embeds (bypass broken NPU Embedding)",
    )
    parser.add_argument(
        "--no-embed-fallback",
        action="store_true",
        help="do not auto-enable --embed-on-cpu when NPU Embedding probe fails",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/toy_muon",
        help="HF-format checkpoint directory (config + weights + tokenizer)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="also save intermediate ckpt every N steps; 0 = only final",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    log_dir = repo_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{args.model}_{args.optimizer}_lr{args.lr}.log"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="INFO", enqueue=False, buffering=1)
    logger.info(f"Logging to {log_path}")
    logger.info(f"Python {sys.version}")
    logger.info(f"torch={torch.__version__} torch_npu={getattr(torch_npu, '__version__', None)}")
    logger.info(f"cwd={os.getcwd()} repo_root={repo_root}")
    logger.info(
        f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '')!r} "
        f"ASCEND_OPP_PATH={os.environ.get('ASCEND_OPP_PATH', '')!r}"
    )

    device = pick_device(args.device)
    train_dtype = resolve_train_dtype(args.dtype, device)
    # Newton-Schulz: match train dtype on NPU; keep bf16 elsewhere (upstream).
    ns_dtype = train_dtype if device.type == "npu" else torch.bfloat16
    disable_compile = not args.enable_torch_compile
    zeropower_fn = build_zeropower(disable_compile, ns_dtype=ns_dtype)
    logger.info(f"Newton-Schulz torch.compile disabled={disable_compile} ns_dtype={ns_dtype}")

    embed_on_cpu = bool(args.embed_on_cpu)
    if device.type == "npu":
        ok, detail = probe_npu_embedding(train_dtype)
        logger.info(f"NPU Embedding probe ({train_dtype}): ok={ok} detail={detail}")
        if not ok:
            # Probe other dtypes for diagnosis
            for alt in (torch.float16, torch.float32, torch.bfloat16):
                if alt == train_dtype:
                    continue
                ok_alt, detail_alt = probe_npu_embedding(alt)
                logger.info(f"NPU Embedding probe ({alt}): ok={ok_alt} detail={detail_alt}")
            if not args.no_embed_fallback:
                logger.warning(
                    "NPU Embedding kernel appears broken (CANN 561103 / GatherV2AiCore). "
                    "Auto-enabling --embed-on-cpu. Install matching Ascend-cann-kernels "
                    "for a proper fix; or use --device cpu."
                )
                embed_on_cpu = True

    cache_dir = resolve_path(args.cache_dir, repo_root) or str(repo_root / "toy_cache")
    local_data_dir = resolve_path(args.local_data_dir, repo_root)
    tokenizer_path = resolve_path(args.tokenizer_path, repo_root)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    model, train_loader, tokenizer = get_model_and_dataloader(
        args.model,
        args.dataset,
        args.hidden_size,
        local_data_dir,
        tokenizer_path,
        cache_dir,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    output_dir = Path(
        resolve_path(args.output_dir, repo_root) or str(repo_root / "checkpoints" / "toy_muon")
    )
    logger.info(f"Checkpoints will be saved under {output_dir}")

    logger.info(f"Using device: {device} embed_on_cpu={embed_on_cpu}")
    logger.info(device_mem_summary(device))

    # Config torch_dtype alone does not cast freshly constructed weights.
    logger.info(f"Casting model to {train_dtype} (--dtype={args.dtype})")
    model = model.to(dtype=train_dtype)
    model.config.use_cache = False

    logger.info(f"Moving model to {device}")
    model = model.to(device)
    if embed_on_cpu and device.type != "cpu":
        enable_cpu_embedding(model)
    # Optimizer after embed untie so new lm_head.weight is included.
    optimizer = get_optimizer(
        args.optimizer, model, lr=args.lr, wd=args.wd, zeropower_fn=zeropower_fn
    )
    synchronize(device)
    logger.info(f"Model is on device: {next(model.parameters()).device}")
    logger.info(
        f"embed_tokens device={model.get_input_embeddings().weight.device} "
        f"lm_head device={model.lm_head.weight.device}"
    )
    logger.info(device_mem_summary(device))
    model.train()

    steps_per_epoch = len(train_loader)
    max_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch
    max_steps = min(max_steps, steps_per_epoch)
    logger.info(f"steps_per_epoch={steps_per_epoch} max_steps={max_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=min(100, max_steps),
        num_training_steps=max_steps,
        num_cycles=0.5,
    )

    logger.info("Entering training loop")
    last_loss = None
    for step, batch in enumerate(train_loader):
        if step >= max_steps:
            break
        try:
            if step == 0:
                logger.info(f"step0: batch shape={tuple(batch.shape)} dtype={batch.dtype}")
            if step == 0:
                logger.info("step0: forward")
            outputs = forward_with_optional_cpu_embed(
                model, batch, device, train_dtype, embed_on_cpu
            )
            loss = outputs.loss
            last_loss = float(loss.detach().cpu())
            if step == 0:
                logger.info(f"step0: loss={last_loss} ; backward")
            loss.backward()
            if step == 0:
                logger.info("step0: optimizer.step")
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "npu":
                synchronize(device)
            logger.info(
                f"Epoch: 0 Step: {step}/{max_steps} "
                f"LR: {optimizer.param_groups[0]['lr']} "
                f"Training loss: {last_loss}"
            )
            if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
                step_dir = output_dir / f"step_{step + 1}"
                save_checkpoint(
                    step_dir,
                    model,
                    tokenizer,
                    meta={
                        "step": step + 1,
                        "max_steps": max_steps,
                        "loss": last_loss,
                        "optimizer": args.optimizer,
                        "lr": args.lr,
                        "dtype": str(train_dtype),
                        "device": str(device),
                        "hidden_size": args.hidden_size,
                        "n_params": n_params,
                    },
                )
                # save_checkpoint moved weights to CPU; put model back for training
                model = model.to(device=device, dtype=train_dtype)
                if embed_on_cpu and device.type != "cpu":
                    enable_cpu_embedding(model)
        except Exception:
            logger.error(f"Failed at step={step}\n{traceback.format_exc()}")
            raise

    save_checkpoint(
        output_dir / "final",
        model,
        tokenizer,
        meta={
            "step": max_steps,
            "max_steps": max_steps,
            "loss": last_loss,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "dtype": str(train_dtype),
            "device": str(device),
            "hidden_size": args.hidden_size,
            "n_params": n_params,
            "status": "TRAIN_OK",
        },
    )
    logger.info("TRAIN_OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        # Ensure failure is visible even if loguru file sink had issues.
        print(f"TRAIN_FAIL: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
