"""
Shared text generation helpers used across the learning notebooks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def generate_sample(
    model,
    tokenizer,
    prompt: str,
    max_len: int = 100,
    temperature: float = 0.8,
    device: str | torch.device = "cpu",
    max_seq_len: int | None = None,
    eos_token_id: int | None = None,
    repetition_penalty: float = 1.0,
):
    """Generate text from a prompt using autoregressive sampling."""
    model.eval()

    if eos_token_id is None and hasattr(tokenizer, "get_endoftext_token"):
        eos_token_id = tokenizer.get_endoftext_token()

    if max_seq_len is None and hasattr(model, "pos_encoding") and hasattr(model.pos_encoding, "pe"):
        max_seq_len = int(model.pos_encoding.pe.size(0))

    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            current_tokens = input_ids
            if max_seq_len is not None and input_ids.size(1) > max_seq_len:
                current_tokens = input_ids[:, -max_seq_len:]

            logits = model(current_tokens)
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-6)
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    next_token_logits[token_id] /= repetition_penalty
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_tokens = input_ids[0].detach().cpu()
    return tokenizer.decode(generated_tokens)
