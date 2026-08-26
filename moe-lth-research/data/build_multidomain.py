from __future__ import annotations

import argparse
import json
from pathlib import Path


def _sample_evenly(text: str, target_chars: int, chunk_chars: int) -> str:
    if target_chars >= len(text):
        return text
    chunk_count = max(1, target_chars // chunk_chars)
    usable_chars = chunk_count * chunk_chars
    max_start = len(text) - chunk_chars
    starts = [
        round(index * max_start / max(1, chunk_count - 1))
        for index in range(chunk_count)
    ]
    return "".join(text[start : start + chunk_chars] for start in starts)[:usable_chars]


def _interleave(first: str, second: str, chunk_chars: int) -> str:
    chunks = []
    limit = max(len(first), len(second))
    for start in range(0, limit, chunk_chars):
        if start < len(first):
            chunks.append(first[start : start + chunk_chars])
        if start < len(second):
            chunks.append(second[start : start + chunk_chars])
    return "\n\n".join(chunks)


def build_multidomain(
    tiny_path: Path,
    wiki_path: Path,
    output_path: Path,
    chunk_chars: int,
) -> dict:
    tiny = tiny_path.read_text(encoding="utf-8")
    wiki = wiki_path.read_text(encoding="utf-8")
    target_chars = min(len(tiny), len(wiki))
    tiny_sample = _sample_evenly(tiny, target_chars, chunk_chars)
    wiki_sample = _sample_evenly(wiki, target_chars, chunk_chars)
    combined = _interleave(tiny_sample, wiki_sample, chunk_chars)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined, encoding="utf-8")
    return {
        "output_path": str(output_path),
        "tiny_path": str(tiny_path),
        "wiki_path": str(wiki_path),
        "tiny_chars": len(tiny_sample),
        "wiki_chars": len(wiki_sample),
        "output_chars": len(combined),
        "chunk_chars": chunk_chars,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced TinyStories/WikiText corpus.")
    parser.add_argument("--tiny-train", default="data/processed/tinystories_train_50k.txt")
    parser.add_argument("--tiny-validation", default="data/processed/tinystories_validation_2k.txt")
    parser.add_argument("--wiki-train", default="data/wikitext103_subset/wikitext103_train.txt")
    parser.add_argument("--wiki-validation", default="data/wikitext103_subset/wikitext103_validation.txt")
    parser.add_argument("--output-dir", default="data/multidomain_balanced")
    parser.add_argument("--chunk-chars", type=int, default=16384)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train = build_multidomain(
        Path(args.tiny_train),
        Path(args.wiki_train),
        output_dir / "multidomain_train.txt",
        args.chunk_chars,
    )
    validation = build_multidomain(
        Path(args.tiny_validation),
        Path(args.wiki_validation),
        output_dir / "multidomain_validation.txt",
        args.chunk_chars,
    )
    metadata = {"train": train, "validation": validation}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
