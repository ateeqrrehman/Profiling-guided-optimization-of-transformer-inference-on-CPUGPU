from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from utils.prompts import SimpleWhitespaceTokenizer

try:
    from torch.ao.quantization import quantize_dynamic
except ImportError:
    from torch.quantization import quantize_dynamic


@dataclass
class LoadedArtifacts:
    model: torch.nn.Module
    tokenizer: object
    model_name: str
    model_source: str
    precision: str
    compiled: bool
    notes: list[str] = field(default_factory=list)


def precision_supported(device: str, precision: str) -> bool:
    if precision == "fp32":
        return True
    if precision == "int8_dynamic":
        return device == "cpu"
    if precision == "fp16":
        return device == "cuda" and torch.cuda.is_available()
    if precision == "bf16":
        return (
            device == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
    return False


def _build_random_fallback(max_seq_length: int) -> tuple[torch.nn.Module, object]:
    vocab_size = 4096
    tokenizer = SimpleWhitespaceTokenizer(vocab_size=vocab_size)
    model_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max(512, max_seq_length + 32),
        n_ctx=max(512, max_seq_length + 32),
        n_embd=128,
        n_layer=4,
        n_head=4,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    model = GPT2LMHeadModel(model_config)
    return model, tokenizer


def _load_pretrained(model_name: str) -> tuple[torch.nn.Module, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer


def _apply_precision(model: torch.nn.Module, device: str, precision: str) -> torch.nn.Module:
    if precision == "fp32":
        return model.float()
    if precision == "fp16":
        return model.to(dtype=torch.float16)
    if precision == "bf16":
        return model.to(dtype=torch.bfloat16)
    if precision == "int8_dynamic":
        return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    raise ValueError(f"Unsupported precision mode: {precision}")


def _maybe_compile(model: torch.nn.Module, enabled: bool, backend: str | None) -> tuple[torch.nn.Module, bool, str | None]:
    if not enabled:
        return model, False, None
    if not hasattr(torch, "compile"):
        return model, False, "torch.compile is not available in this PyTorch build."

    try:
        compiled_model = torch.compile(model, backend=backend) if backend else torch.compile(model)
        return compiled_model, True, None
    except Exception as exc:
        return model, False, f"torch.compile failed: {exc}"


def load_artifacts(
    settings,
    device: str,
    precision: str,
    compiled: bool,
) -> LoadedArtifacts:
    if not precision_supported(device, precision):
        raise ValueError(f"Precision {precision} is not supported on {device}.")

    notes: list[str] = []
    if settings.model_name == "random-fallback":
        model, tokenizer = _build_random_fallback(max(settings.sequence_lengths.values()))
        model_source = "random_fallback"
        model_name = settings.model_name
        notes.append("Using the local random fallback model by request.")
    else:
        try:
            model, tokenizer = _load_pretrained(settings.model_name)
            model_source = "huggingface"
            model_name = settings.model_name
        except Exception as exc:
            if not settings.allow_random_fallback:
                raise
            model, tokenizer = _build_random_fallback(max(settings.sequence_lengths.values()))
            model_source = "random_fallback"
            model_name = f"{settings.model_name} (fallback)"
            notes.append(f"Pretrained model load failed; using random fallback. Reason: {exc}")

    model.eval()
    model = _apply_precision(model, device, precision)
    model = model.to(device)

    compiled_model, compile_enabled, compile_note = _maybe_compile(
        model, enabled=compiled, backend=settings.compile_backend
    )
    if compiled and not compile_enabled and compile_note:
        notes.append(compile_note)

    return LoadedArtifacts(
        model=compiled_model,
        tokenizer=tokenizer,
        model_name=model_name,
        model_source=model_source,
        precision=precision,
        compiled=compile_enabled,
        notes=notes,
    )
