from __future__ import annotations

import math
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


BUILTIN_CORPUS = """
Mira found a brass key beneath the old pear tree. The key hummed whenever the
moon rose, so she followed its song through the sleeping village. At the edge
of the river stood a door with no wall around it. Mira turned the key, opened
the door, and discovered a library where every unwritten story waited quietly.

Theo built a small boat from blue paper. Rain filled the gutters and carried
the boat past gardens, bridges, and bright windows. A baker rescued it near the
market and wrote a message on its sail. By sunset the boat returned to Theo
with greetings from half the town.

The observatory clock stopped at midnight. Lina climbed the hill with a lantern
and found that the stars had rearranged themselves into a map. She copied the
map carefully, and the next morning a new path appeared through the woods.
""".strip()


class ByteTokenizer:
    vocab_size = 256

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        fragments: list[str] = []
        byte_buffer: list[int] = []
        for token_id in token_ids:
            token = int(token_id)
            if 0 <= token < 256:
                byte_buffer.append(token)
                continue
            if byte_buffer:
                fragments.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer = []
            fragments.append(f"<tok:{token}>")
        if byte_buffer:
            fragments.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
        return "".join(fragments)


class ByteNgramTokenizer:
    """Greedy byte-level subword tokenizer with byte fallback for exact round-trips."""

    def __init__(self, ngram_tokens: list[bytes]):
        self.id_to_bytes = [bytes([token_id]) for token_id in range(ByteTokenizer.vocab_size)]
        self.id_to_bytes.extend(ngram_tokens)
        self.bytes_to_id = {
            token: ByteTokenizer.vocab_size + index for index, token in enumerate(ngram_tokens)
        }
        self.max_ngram = max((len(token) for token in ngram_tokens), default=1)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_bytes)

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        max_ngram: int = 8,
        train_bytes: int | None = None,
        min_frequency: int = 2,
    ) -> "ByteNgramTokenizer":
        if vocab_size < ByteTokenizer.vocab_size:
            raise ValueError("Subword vocab_size must be at least 256 for byte fallback.")
        data = text.encode("utf-8")
        if train_bytes is not None:
            data = data[: int(train_bytes)]
        max_ngram = max(2, int(max_ngram))
        min_frequency = max(1, int(min_frequency))

        counts: Counter[bytes] = Counter()
        max_width = min(max_ngram, len(data))
        for width in range(2, max_width + 1):
            for start in range(0, len(data) - width + 1):
                counts[data[start : start + width]] += 1

        needed = vocab_size - ByteTokenizer.vocab_size
        candidates = [
            (count * (len(token) - 1), count, len(token), token)
            for token, count in counts.items()
            if count >= min_frequency
        ]
        candidates.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
        return cls([token for _, _, _, token in candidates[:needed]])

    @classmethod
    def from_file(cls, path: str | Path) -> "ByteNgramTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "byte_ngram":
            raise ValueError(f"Unsupported tokenizer payload: {payload.get('type')}")
        return cls([bytes(token) for token in payload["tokens"]])

    def to_file(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "byte_ngram",
            "vocab_size": self.vocab_size,
            "tokens": [list(token) for token in self.id_to_bytes[ByteTokenizer.vocab_size :]],
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        tokens: list[int] = []
        cursor = 0
        while cursor < len(data):
            match_id = None
            match_width = 1
            max_width = min(self.max_ngram, len(data) - cursor)
            for width in range(max_width, 1, -1):
                token_id = self.bytes_to_id.get(data[cursor : cursor + width])
                if token_id is not None:
                    match_id = token_id
                    match_width = width
                    break
            if match_id is None:
                tokens.append(data[cursor])
                cursor += 1
            else:
                tokens.append(match_id)
                cursor += match_width
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        pieces: list[bytes] = []
        for token_id in token_ids:
            token = int(token_id)
            if 0 <= token < len(self.id_to_bytes):
                pieces.append(self.id_to_bytes[token])
            else:
                pieces.append(f"<tok:{token}>".encode("utf-8"))
        return b"".join(pieces).decode("utf-8", errors="replace")


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, tokens: list[int], seq_len: int):
        usable = max(0, (len(tokens) - 1) // seq_len * seq_len)
        if usable < seq_len:
            raise ValueError("Corpus is too short for the configured sequence length.")
        self.tokens = torch.tensor(tokens[: usable + 1], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.seq_len
        block = self.tokens[start : start + self.seq_len + 1]
        return block[:-1], block[1:]


class EpochReshuffledDataLoader:
    """A thin wrapper that creates a fresh permutation each time a new epoch starts.

    This makes the repeated-order confound explicit and controllable instead of
    depending on the default PyTorch generator semantics of a long-lived loader.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        seed: int,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            generator=generator,
        )
        return iter(loader)

    def __len__(self) -> int:
        dataset_len = len(self.dataset)
        if self.drop_last:
            return dataset_len // self.batch_size
        return math.ceil(dataset_len / self.batch_size)


def load_corpus(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    return (BUILTIN_CORPUS + "\n") * 256


def _limited_split_text(split, limit: int | None) -> str:
    if limit is not None:
        split = split.select(range(min(int(limit), len(split))))
    return "\n\n".join(split["text"])


def _load_text_splits(data_config: dict) -> tuple[str, str]:
    train_path = data_config.get("train_path")
    validation_path = data_config.get("validation_path")
    if train_path or validation_path:
        if not train_path or not validation_path:
            raise ValueError("data.train_path and data.validation_path must be configured together.")
        return Path(train_path).read_text(encoding="utf-8"), Path(validation_path).read_text(
            encoding="utf-8"
        )

    path_value = data_config.get("path")
    if path_value and Path(path_value).is_dir():
        try:
            from datasets import DatasetDict, load_from_disk
        except ImportError as error:
            raise ImportError(
                "Reading a Hugging Face dataset directory requires the 'datasets' package."
            ) from error
        dataset = load_from_disk(path_value)
        if not isinstance(dataset, DatasetDict) or "train" not in dataset or "validation" not in dataset:
            raise ValueError(
                f"Expected a DatasetDict with train and validation splits at {path_value}."
            )
        train_text = _limited_split_text(dataset["train"], data_config.get("max_train_examples"))
        validation_text = _limited_split_text(
            dataset["validation"], data_config.get("max_validation_examples")
        )
        return train_text, validation_text

    corpus = load_corpus(path_value)
    split = int(len(corpus) * float(data_config["train_fraction"]))
    return corpus[:split], corpus[split:]


def _build_subword_tokenizer(data_config: dict, train_text: str) -> ByteNgramTokenizer:
    requested_vocab_size = int(data_config.get("tokenizer_vocab_size", 1024))
    cache_path = data_config.get("tokenizer_cache_path")
    if cache_path:
        path = Path(cache_path)
        if path.exists():
            tokenizer = ByteNgramTokenizer.from_file(path)
            if tokenizer.vocab_size == requested_vocab_size:
                return tokenizer

    tokenizer = ByteNgramTokenizer.train(
        train_text,
        vocab_size=requested_vocab_size,
        max_ngram=int(data_config.get("tokenizer_max_ngram", 8)),
        train_bytes=data_config.get("tokenizer_train_bytes"),
        min_frequency=int(data_config.get("tokenizer_min_frequency", 2)),
    )
    if cache_path:
        tokenizer.to_file(cache_path)
    return tokenizer


def load_token_splits(data_config: dict) -> tuple[list[int], list[int]]:
    train_text, validation_text = _load_text_splits(data_config)
    tokenizer_type = data_config.get("tokenizer", "byte")
    if tokenizer_type == "byte":
        return ByteTokenizer.encode(train_text), ByteTokenizer.encode(validation_text)
    if tokenizer_type == "byte_ngram":
        tokenizer = _build_subword_tokenizer(data_config, train_text)
        return tokenizer.encode(train_text), tokenizer.encode(validation_text)
    raise ValueError("data.tokenizer must be one of: byte, byte_ngram")


def build_dataloaders(
    data_config: dict,
    batch_size: int,
    seed: int,
    reshuffle_each_epoch: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_tokens, validation_tokens = load_token_splits(data_config)
    seq_len = int(data_config["seq_len"])
    train_dataset = TokenBlockDataset(train_tokens, seq_len)
    validation_dataset = TokenBlockDataset(validation_tokens, seq_len)

    generator = torch.Generator().manual_seed(seed)
    if reshuffle_each_epoch:
        train_loader = EpochReshuffledDataLoader(
            train_dataset,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            drop_last=True,
        )
        validation_loader = EpochReshuffledDataLoader(
            validation_dataset,
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
            drop_last=False,
        )
        return train_loader, validation_loader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, validation_loader
