"""
Teaching solution: distributed-aware data loader using Python __iter__ with yield.

Concepts covered
----------------
1. Loading and merging samples from multiple JSON shard files.
2. Implementing __iter__ as a generator (yield) — no __next__ or cursor needed.
3. Splitting data across workers with `rank` and `world_size` so each
   process only iterates over its own slice — the core idea behind
   PyTorch's DistributedSampler.

Why yield instead of __next__?
  When __iter__ contains a yield statement, Python turns it into a generator
  function. Each call to __iter__ creates a fresh generator object, so the
  loader resets automatically on every new for-loop — no _index bookkeeping.

Run after generate_data.py:
    python dataloader_solution.py

To simulate multi-process behaviour without actually spawning processes,
pass different rank values on the command line:
    python dataloader_solution.py --rank 0 --world_size 4
    python dataloader_solution.py --rank 1 --world_size 4
    python dataloader_solution.py --rank 2 --world_size 4
    python dataloader_solution.py --rank 3 --world_size 4
"""

import argparse
import json
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Step 1 — load JSON shard files
# ---------------------------------------------------------------------------

def load_shards(file_paths: list[str]) -> list[dict]:
    """Read every JSON shard and return a flat list of sample dicts."""
    all_samples: list[dict] = []
    for path in file_paths:
        with open(path) as f:
            shard = json.load(f)            # each file is a list of dicts
        all_samples.extend(shard)
        print(f"  Loaded {len(shard):>3} samples from {path}")
    return all_samples


# ---------------------------------------------------------------------------
# Step 2 — iterator class using yield inside __iter__
# ---------------------------------------------------------------------------

class ShardedDataLoader:
    """
    Iterates over the subset of samples assigned to this worker.

    Parameters
    ----------
    file_paths  : paths to JSON shard files (created by generate_data.py)
    rank        : index of this process, 0-based   (default 0)
    world_size  : total number of parallel processes (default 1)
    drop_last   : if True, discard samples that don't divide evenly
                  across workers; if False, some workers get one extra
    """

    def __init__(
        self,
        file_paths: list[str],
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        assert 0 <= rank < world_size, "rank must be in [0, world_size)"

        print(f"\n[rank {rank}/{world_size}] Loading shards...")
        all_samples = load_shards(file_paths)
        total = len(all_samples)

        # --- distribute samples across workers ----------------------------
        # Slice with step = world_size: deterministic, no communication.
        #   rank 0 -> indices 0, world_size,   2*world_size, ...
        #   rank 1 -> indices 1, world_size+1, 2*world_size+1, ...
        if drop_last:
            keep = (total // world_size) * world_size
            all_samples = all_samples[:keep]

        self.samples: list[dict] = all_samples[rank::world_size]
        self.rank = rank
        self.world_size = world_size

    # ------------------------------------------------------------------
    # iterator protocol — yield makes __iter__ a generator
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[dict]:
        """
        Yield one sample at a time.

        Because this method contains `yield`, Python treats it as a generator
        function. Every `for sample in loader` call creates a brand-new
        generator object starting from index 0 — no manual reset required.
        """
        for sample in self.samples:
            yield sample


# ---------------------------------------------------------------------------
# Step 3 — demonstration
# ---------------------------------------------------------------------------

def main(rank: int, world_size: int) -> None:
    shard_dir = Path(__file__).parent
    file_paths = sorted(str(p) for p in shard_dir.glob("shard_*.json"))

    if not file_paths:
        print("No shard files found. Run generate_data.py first.")
        return

    loader = ShardedDataLoader(
        file_paths,
        rank=rank,
        world_size=world_size,
        drop_last=False,
    )

    print(f"\n[rank {rank}/{world_size}] Assigned {len(loader)} samples:")
    for sample in loader:           # __iter__ returns a fresh generator each time
        print(f"  index={sample['index']:>2}  text='{sample['text']}'")

    # Second loop creates another fresh generator automatically.
    print(f"\n[rank {rank}/{world_size}] Second pass (auto-resets via yield):")
    count = sum(1 for _ in loader)
    print(f"  Counted {count} samples, matches len()={len(loader)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()
    main(args.rank, args.world_size)
