"""
Shared tokenizer helpers used across the learning notebooks.

This module intentionally supports both:
- a simple character-level tokenizer for the original notebook 6 flow
- a lightweight educational BPE tokenizer for improved word/subword handling
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable
import re

import torch


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class BaseTokenizer:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    tokenizer_type = "base"

    def __init__(self, vocab_size: int = 5000) -> None:
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}

    def decode(self, token_ids: Iterable[int] | torch.Tensor) -> str:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def get_endoftext_token(self) -> int:
        return self.vocab[self.EOS_TOKEN]

    def get_pad_token(self) -> int:
        return self.vocab[self.PAD_TOKEN]

    def __len__(self) -> int:
        return len(self.vocab)

    def _init_special_tokens(self) -> None:
        self.vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def get_state(self) -> dict:
        return {
            "tokenizer_type": self.tokenizer_type,
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "inverse_vocab": self.inverse_vocab,
        }

    def load_state(self, state: dict) -> None:
        self.vocab_size = state.get("vocab_size", self.vocab_size)
        self.vocab = state["vocab"]
        if "inverse_vocab" in state:
            self.inverse_vocab = {
                int(idx): token for idx, token in state["inverse_vocab"].items()
            }
        else:
            self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}


class CharacterTokenizer(BaseTokenizer):
    """Simple character-level tokenizer used by the original notebook flow."""

    tokenizer_type = "char"

    def train(self, text: str, vocab_size: int | None = None) -> None:
        if vocab_size is not None:
            self.vocab_size = vocab_size

        self._init_special_tokens()
        for char in sorted(set(text)):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(char, self.vocab[self.UNK_TOKEN]) for char in text]

    def decode(self, token_ids: Iterable[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        chars = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, self.UNK_TOKEN)
            if token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
                continue
            chars.append(token)
        return "".join(chars)


class BPETokenizer(BaseTokenizer):
    """
    Lightweight educational BPE tokenizer.

    This implementation starts from character pieces inside words, repeatedly
    merges the most common adjacent pair, and then encodes words greedily using
    the learned subword vocabulary.
    """

    tokenizer_type = "bpe"

    def __init__(self, vocab_size: int = 2000, lowercase: bool = False) -> None:
        super().__init__(vocab_size=vocab_size)
        self.merges: list[tuple[str, str]] = []
        self.lowercase = lowercase

    def _word_to_symbols(self, word: str) -> list[str]:
        return list(word) + ["</w>"]

    def _split_text(self, text: str) -> list[str]:
        return WORD_PATTERN.findall(text)

    def _normalize_text(self, text: str) -> str:
        if self.lowercase:
            return text.lower()
        return text

    def train(self, text: str, vocab_size: int | None = None) -> None:
        if vocab_size is not None:
            self.vocab_size = vocab_size

        self._init_special_tokens()

        words = self._split_text(self._normalize_text(text))
        word_freq = Counter(words)
        corpus = {tuple(self._word_to_symbols(word)): freq for word, freq in word_freq.items()}

        base_symbols = set()
        for symbols in corpus:
            base_symbols.update(symbols)

        for symbol in sorted(base_symbols):
            if symbol not in self.vocab:
                self.vocab[symbol] = len(self.vocab)

        target_vocab = max(self.vocab_size, len(self.vocab))

        while len(self.vocab) < target_vocab:
            pair_counts = Counter()
            for symbols, freq in corpus.items():
                for idx in range(len(symbols) - 1):
                    pair_counts[(symbols[idx], symbols[idx + 1])] += freq

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            merged = "".join(best_pair)
            self.merges.append(best_pair)

            updated_corpus = {}
            for symbols, freq in corpus.items():
                new_symbols = []
                idx = 0
                while idx < len(symbols):
                    if (
                        idx < len(symbols) - 1
                        and symbols[idx] == best_pair[0]
                        and symbols[idx + 1] == best_pair[1]
                    ):
                        new_symbols.append(merged)
                        idx += 2
                    else:
                        new_symbols.append(symbols[idx])
                        idx += 1
                updated_corpus[tuple(new_symbols)] = freq

            corpus = updated_corpus
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        for left, right in self.merges:
            merged = left + right
            updated = []
            idx = 0
            while idx < len(symbols):
                if (
                    idx < len(symbols) - 1
                    and symbols[idx] == left
                    and symbols[idx + 1] == right
                ):
                    updated.append(merged)
                    idx += 2
                else:
                    updated.append(symbols[idx])
                    idx += 1
            symbols = updated
        return symbols

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        for piece in self._split_text(self._normalize_text(text)):
            symbols = self._word_to_symbols(piece)
            symbols = self._apply_merges(symbols)
            for symbol in symbols:
                tokens.append(self.vocab.get(symbol, self.vocab[self.UNK_TOKEN]))
        return tokens

    def decode(self, token_ids: Iterable[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        pieces: list[str] = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, self.UNK_TOKEN)
            if token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
                continue
            if token == self.UNK_TOKEN:
                pieces.append(token)
                continue

            if token.endswith("</w>"):
                pieces.append(token[:-4])
                pieces.append(" ")
            else:
                pieces.append(token)

        text = "".join(pieces).strip()
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text

    def get_state(self) -> dict:
        state = super().get_state()
        state["merges"] = self.merges
        state["lowercase"] = self.lowercase
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.merges = [tuple(pair) for pair in state.get("merges", [])]
        self.lowercase = state.get("lowercase", False)


def build_tokenizer(
    tokenizer_type: str = "char",
    vocab_size: int = 5000,
    lowercase: bool = False,
) -> BaseTokenizer:
    if tokenizer_type == "char":
        return CharacterTokenizer(vocab_size=vocab_size)
    if tokenizer_type == "bpe":
        return BPETokenizer(vocab_size=vocab_size, lowercase=lowercase)
    raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")


def tokenizer_from_checkpoint(checkpoint: dict, default_vocab_size: int = 5000) -> BaseTokenizer:
    state = checkpoint.get("tokenizer_state")
    tokenizer_type = checkpoint.get("tokenizer_type", "char")

    if state is None:
        # Backward compatibility for older checkpoints.
        tokenizer = build_tokenizer(tokenizer_type="char", vocab_size=default_vocab_size)
        tokenizer.load_state(
            {
                "tokenizer_type": "char",
                "vocab_size": len(checkpoint["tokenizer_vocab"]),
                "vocab": checkpoint["tokenizer_vocab"],
                "inverse_vocab": checkpoint["tokenizer_inverse_vocab"],
            }
        )
        return tokenizer

    tokenizer = build_tokenizer(
        tokenizer_type=state.get("tokenizer_type", tokenizer_type),
        vocab_size=state.get("vocab_size", default_vocab_size),
    )
    tokenizer.load_state(state)
    return tokenizer


# Backward-compatible alias used by the existing notebooks.
ShakespeareBPETokenizer = CharacterTokenizer
