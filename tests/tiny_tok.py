"""Small test doubles for the T5 driver: a WordLevel `tokenizers.Tokenizer` over a synthetic vocab, and a fake model
with the reference forward's interface (`__call__(tokens [1, s], start_pos) -> (ids [1], logits fp32 [1, V], None)`,
`max_seq_len`, `temperature`) that records how it was called."""
from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Tuple

import torch


def make_wordlevel_tokenizer(vocab_size: int, eos_token: str = "<eos>", out_dir: Optional[str] = None):
    """Token i = "t{i}" except id 1 = `eos_token` (a special token). Writes tokenizer.json + tokenizer_config.json to
    `out_dir` when given (so `Generator(model, out_dir)` reads them like a checkpoint root)."""
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    vocab = {f"t{i}": i for i in range(vocab_size) if i != 1}
    vocab[eos_token] = 1
    tok = Tokenizer(WordLevel(vocab, unk_token="t0"))
    tok.pre_tokenizer = WhitespaceSplit()
    tok.add_special_tokens([AddedToken(eos_token, special=True)])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tok.save(os.path.join(out_dir, "tokenizer.json"))
        with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
            json.dump({"eos_token": {"__type": "AddedToken", "content": eos_token}}, f)
    return tok


class FakeModel:
    """Deterministic logits from (position, fed token); optional EOS forced at a given generated index."""

    def __init__(self, vocab: int = 64, max_seq_len: int = 4096, eos_id: int = 1, eos_at_pos: Optional[int] = None):
        self.vocab, self.max_seq_len, self.eos_id, self.eos_at_pos = vocab, max_seq_len, eos_id, eos_at_pos
        self.temperature = 1.0
        self.calls: List[Tuple[Tuple[int, ...], int]] = []

    def parameters(self):
        return iter(())

    def logits_for(self, pos: int, last_tok: int) -> torch.Tensor:
        v = torch.arange(self.vocab, dtype=torch.float64)
        x = torch.cos(v * 0.37 + pos * 1.3 + last_tok * 0.11) * 5.0
        x[self.eos_id] = -50.0
        if self.eos_at_pos is not None and pos == self.eos_at_pos:
            x[self.eos_id] = 50.0
        return x.to(torch.float32)

    def __call__(self, tokens: torch.Tensor, start_pos: int):
        assert tokens.dim() == 2 and tokens.size(0) == 1 and tokens.dtype == torch.long
        self.calls.append((tuple(tokens.shape), int(start_pos)))
        pos = start_pos + tokens.size(1)  # position of the token being predicted
        logits = self.logits_for(pos, int(tokens[0, -1])).unsqueeze(0)
        ids = logits.argmax(dim=-1) if self.temperature == 0 else torch.full((1,), -7)  # -7: "sampled", must not leak
        return ids, logits, None


def log_softmax_ref(x: torch.Tensor) -> List[float]:
    """Independent fp32 log-softmax for checking (max-shifted, float64 accumulate)."""
    xs = [float(v) for v in x.reshape(-1)]
    m = max(xs)
    s = math.log(sum(math.exp(v - m) for v in xs))
    return [v - m - s for v in xs]
