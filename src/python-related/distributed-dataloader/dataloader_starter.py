"""
Starter template: distributed-aware data loader using Python __iter__ with yield.

Your tasks
----------
1. Complete `load_shards` so it reads every JSON file and returns a flat list.
2. Complete `ShardedDataLoader.__init__` to assign each rank its slice.
3. Implement `__len__` and `__iter__` (using yield) to make the class iterable.
   No __next__ or cursor variable needed — yield handles that automatically.
4. Run the demo with different --rank values to verify each worker sees a
   different, non-overlapping subset of samples.

Run after generate_data.py:
    python dataloader_starter.py --rank 0 --world_size 4
    python dataloader_starter.py --rank 1 --world_size 4
"""

import argparse
import json
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Task 1 — load JSON shard files
# ---------------------------------------------------------------------------

def load_shards(file_paths: list[str]) -> list[dict]:
    """
    Read every JSON shard file and return a single flat list of samples.

    Each file contains a JSON array of objects like:
        [{"index": 0, "text": "..."}, ...]

    Hint: open each file, json.load it, then extend a running list.
    """
    all_samples: list[dict] = []
    for path in file_paths:
        # TODO: open `path`, load the JSON, extend all_samples
        raise NotImplementedError("load_shards: read and merge shard files")
    return all_samples


# ---------------------------------------------------------------------------
# Task 2 & 3 — iterator class
# ---------------------------------------------------------------------------

class ShardedDataLoader:
    """
    Iterates over the subset of samples assigned to this worker (rank).

    Parameters
    ----------
    file_paths  : paths to JSON shard files
    rank        : index of this process (0-based)
    world_size  : total number of parallel processes
    drop_last   : drop samples that don't divide evenly (see solution for details)
    """

    def __init__(
        self,
        file_paths: list[str],
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        assert 0 <= rank < world_size, "rank must be in [0, world_size)"

        all_samples = load_shards(file_paths)

        # TODO: if drop_last is True, truncate all_samples to the largest
        #       multiple of world_size.

        # TODO: assign self.samples — the slice belonging to this rank.
        #       Hint: use Python slice stepping (list[start::step]).
        self.samples: list[dict] = []   # replace this

        self.rank = rank
        self.world_size = world_size
        raise NotImplementedError("ShardedDataLoader.__init__: assign self.samples")

    def __len__(self) -> int:
        # TODO: return the number of samples this worker owns
        raise NotImplementedError("__len__")

    def __iter__(self) -> Iterator[dict]:
        """
        Yield one sample at a time.

        Hint: a simple `for sample in self.samples: yield sample` is all you
        need. Because __iter__ contains yield, Python makes it a generator —
        each for-loop call gets a fresh generator starting from index 0
        automatically. No __next__ or cursor variable required.
        """
        # TODO: yield each sample in self.samples
        raise NotImplementedError("__iter__")


# ---------------------------------------------------------------------------
# Demo — do not modify
# ---------------------------------------------------------------------------

def main(rank: int, world_size: int) -> None:
    shard_dir = Path(__file__).parent
    file_paths = sorted(str(p) for p in shard_dir.glob("shard_*.json"))

    if not file_paths:
        print("No shard files found. Run generate_data.py first.")
        return

    loader = ShardedDataLoader(file_paths, rank=rank, world_size=world_size)

    print(f"\n[rank {rank}/{world_size}] Assigned {len(loader)} samples:")
    for sample in loader:
        print(f"  index={sample['index']:>2}  text='{sample['text']}'")

    print(f"\n[rank {rank}/{world_size}] Second pass (should reset cleanly):")
    count = sum(1 for _ in loader)
    print(f"  Counted {count} samples, should match {len(loader)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()
    main(args.rank, args.world_size)
