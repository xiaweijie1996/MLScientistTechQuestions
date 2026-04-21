"""
Teaching script: generate 4 JSON shard files with different sample counts.

Each JSON file is a list of objects like:
    {"index": 0, "text": "..."}

Run this once before using the data loaders:
    python generate_data.py
"""

import json
import random

SHARDS = [
    ("shard_0.json", 5),
    ("shard_1.json", 8),
    ("shard_2.json", 3),
    ("shard_3.json", 6),
]

WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "machine", "learning", "data", "model", "train", "batch", "epoch",
    "gradient", "loss", "weight", "bias", "neuron", "layer", "deep",
]


def make_sentence(length: int = 8) -> str:
    return " ".join(random.choices(WORDS, k=length))


def main() -> None:
    global_index = 0
    for filename, num_samples in SHARDS:
        samples = []
        for _ in range(num_samples):
            samples.append({"index": global_index, "text": make_sentence()})
            global_index += 1
        with open(filename, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Wrote {num_samples} samples -> {filename}")

    print(f"\nTotal samples across all shards: {global_index}")


if __name__ == "__main__":
    main()
