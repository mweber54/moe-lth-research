from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk


def export_split(dataset, split: str, limit: int, destination: Path) -> None:
    selected = dataset[split].select(range(min(limit, len(dataset[split]))))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(selected["text"]), encoding="utf-8")
    print(f"{split}: {len(selected)} stories -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic TinyStories text subsets.")
    parser.add_argument("--dataset-dir", default="data/TinyStories")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-examples", type=int, default=50000)
    parser.add_argument("--validation-examples", type=int, default=2000)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_dir)
    output_dir = Path(args.output_dir)
    export_split(dataset, "train", args.train_examples, output_dir / "tinystories_train_50k.txt")
    export_split(
        dataset,
        "validation",
        args.validation_examples,
        output_dir / "tinystories_validation_2k.txt",
    )


if __name__ == "__main__":
    main()
