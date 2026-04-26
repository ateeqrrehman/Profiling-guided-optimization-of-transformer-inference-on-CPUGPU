from __future__ import annotations

import math
import statistics

import torch


def evaluate_quality(
    model: torch.nn.Module,
    tokenizer,
    device: str,
    prompts: list[str],
    seq_length: int,
) -> dict[str, float]:
    losses: list[float] = []

    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs, labels=inputs["input_ids"])

        losses.append(float(outputs.loss.detach().cpu()))

    mean_loss = statistics.mean(losses) if losses else math.nan
    perplexity = math.exp(min(mean_loss, 20.0)) if not math.isnan(mean_loss) else math.nan
    return {
        "quality_loss": mean_loss,
        "quality_perplexity": perplexity,
    }
