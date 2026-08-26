#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def download_wikitext103(
    output_dir: Path,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    limits = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }

    for split, limit in limits.items():
        if limit <= 0:
            continue

        print(f"Downloading {split} split ({limit:,} rows)...")

        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split=f"{split}[:{limit}]",
        )

        output_file = output_dir / f"wikitext103_{split}.txt"

        with output_file.open("w", encoding="utf-8") as file:
            for text in dataset["text"]:
                if text.strip():
                    file.write(text.rstrip())
                    file.write("\n")

        print(f"Saved: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a subset of WikiText-103 next to this script."
    )
    parser.add_argument("--train-rows", type=int, default=10_000)
    parser.add_argument("--validation-rows", type=int, default=1_000)
    parser.add_argument("--test-rows", type=int, default=1_000)
    parser.add_argument(
        "--directory",
        default="wikitext103_subset",
        help="Output directory relative to this script.",
    )
    args = parser.parse_args()

    script_directory = Path(__file__).resolve().parent
    output_directory = script_directory / args.directory

    download_wikitext103(
        output_dir=output_directory,
        train_rows=args.train_rows,
        validation_rows=args.validation_rows,
        test_rows=args.test_rows,
    )


if __name__ == "__main__":
    main()