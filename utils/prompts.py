from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SimpleWhitespaceTokenizer:
    vocab_size: int = 4096
    pad_token_id: int = 0
    eos_token_id: int = 1

    @property
    def pad_token(self) -> str:
        return "<pad>"

    @property
    def eos_token(self) -> str:
        return "<eos>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = []
        for word in text.strip().split():
            token_id = 2 + (abs(hash(word)) % max(2, self.vocab_size - 2))
            tokens.append(token_id)
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens or [self.eos_token_id]

    def decode(self, token_ids: list[int]) -> str:
        visible = [
            f"tok{token_id}"
            for token_id in token_ids
            if token_id not in {self.pad_token_id, self.eos_token_id}
        ]
        return " ".join(visible)

    def __call__(
        self,
        texts: str | list[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        encoded = [self.encode(text, add_special_tokens=True) for text in texts]
        if max_length is not None and truncation:
            encoded = [tokens[:max_length] for tokens in encoded]

        target_length = max(len(tokens) for tokens in encoded)
        if max_length is not None and padding:
            target_length = max_length

        padded = []
        masks = []
        for tokens in encoded:
            current = list(tokens)
            if padding and len(current) < target_length:
                current.extend([self.pad_token_id] * (target_length - len(current)))
            padded.append(current)
            masks.append([1 if token != self.pad_token_id else 0 for token in current])

        if return_tensors != "pt":
            raise ValueError("SimpleWhitespaceTokenizer only supports return_tensors='pt'.")

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def build_prompt_text(tokenizer, seq_length: int, prefix: str = "") -> str:
    base = (
        "transformers process token sequences with attention layers and matrix multiplications "
        "while system performance depends on memory bandwidth compute throughput and runtime overhead "
    )
    text = (prefix + " " + base * (seq_length // 8 + 8)).strip()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    trimmed = token_ids[: max(seq_length - 1, 1)]
    return tokenizer.decode(trimmed)


def build_inputs(tokenizer, batch_size: int, seq_length: int, device: str) -> dict[str, torch.Tensor]:
    prompts = [
        build_prompt_text(tokenizer, seq_length, prefix=f"sample {index}")
        for index in range(batch_size)
    ]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_length,
    )
    return {key: value.to(device) for key, value in encoded.items()}
