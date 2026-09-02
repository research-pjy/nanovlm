"""Byte-level BPE tokenizer, trained once on the training split's captions
and reused unmodified by both encoder strategies (DESIGN_DECISIONS.md §5).
"""

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def train_tokenizer(texts: list[str], vocab_size: int, out_dir: str) -> ByteLevelBPETokenizer:
    """Trains a byte-level BPE tokenizer on `texts` and saves it to
    `out_dir` (vocab.json + merges.txt). Call this exactly once on the
    training split's captions (see DESIGN_DECISIONS.md §5 — never on
    validation or held-out captions, to keep held-out evaluation clean).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    corpus_file = out_path / "_tokenizer_train_corpus.txt"
    corpus_file.write_text("\n".join(texts), encoding="utf-8")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(corpus_file)],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(str(out_path))
    corpus_file.unlink()
    return tokenizer


def load_tokenizer(tokenizer_dir: str) -> ByteLevelBPETokenizer:
    tdir = Path(tokenizer_dir)
    tokenizer = ByteLevelBPETokenizer(str(tdir / "vocab.json"), str(tdir / "merges.txt"))
    return tokenizer


class NanoVLMTokenizer:
    """Thin wrapper adding pad/bos/eos id lookups and encode/decode
    helpers the training and eval scripts use directly.
    """

    def __init__(self, tokenizer: ByteLevelBPETokenizer):
        self._tok = tokenizer
        self.pad_id = tokenizer.token_to_id(PAD_TOKEN)
        self.bos_id = tokenizer.token_to_id(BOS_TOKEN)
        self.eos_id = tokenizer.token_to_id(EOS_TOKEN)
        self.unk_id = tokenizer.token_to_id(UNK_TOKEN)
        assert None not in (self.pad_id, self.bos_id, self.eos_id, self.unk_id), (
            "special tokens missing from tokenizer vocab"
        )

    @classmethod
    def from_dir(cls, tokenizer_dir: str) -> "NanoVLMTokenizer":
        return cls(load_tokenizer(tokenizer_dir))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if skip_special:
            special_ids = {self.pad_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in special_ids]
        return self._tok.decode(ids)
