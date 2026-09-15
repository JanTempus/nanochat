"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import random
import time
import requests
import pyarrow.parquet as pq
from contextlib import ExitStack, closing
from functools import lru_cache
from itertools import islice
from filelock import FileLock
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()

def configure_dataset(fineweb=False):
    global DATA_DIR
    DATA_DIR = os.path.join(base_dir, "base_data_fineweb" if fineweb else "base_data_climbmix")

configure_dataset()

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir

    # Legacy-supporting code due to the upgrade from FinewebEdu-100B to ClimbMix-400B
    # This code will eventually be deleted.
    if data_dir == os.path.join(base_dir, "base_data_climbmix") and not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  WARNING: DATASET UPGRADE REQUIRED")
            print("=" * 80)
            print()
            print(f"  Could not find: {data_dir}")
            print()
            print("  nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.")
            print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
            print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
            print()
            print("    python -m nanochat.dataset -n 170     # download ~170 shards, enough for GPT-2, adjust as desired")
            print("    python -m scripts.tok_train           # re-train tokenizer on new ClimbMix data")
            print()
            print("  For now, falling back to your old FinewebEdu-100B dataset...")
            print("=" * 80)
            print()
        # attempt a fallback to the legacy data directory
        data_dir = os.path.join(base_dir, "base_data")

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _fineweb_configs(repo_id):
    """Read the small dataset card once, without resolving every data file."""
    from huggingface_hub import DatasetCard

    print(f"Reading source metadata: {repo_id}", flush=True)
    configs = DatasetCard.load(repo_id, repo_type="dataset").data.configs
    return {config["config_name"]: config["data_files"] for config in configs}


def _fineweb_texts(repo_id, data_files, split, english=False):
    """Open only the requested split, and only when its first text is needed."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    remote_split = "train" if english or split == "train" else "test"
    for entry in data_files:
        if entry["split"] != remote_split:
            continue
        paths = entry["path"]
        paths = [paths] if isinstance(paths, str) else paths
        for path in paths:
            print(f"Streaming {repo_id}/{path} ({split})", flush=True)
            files = sorted(fs.glob(f"datasets/{repo_id}/{path}"))
            for filename in files:
                if not filename.endswith(".parquet"):
                    continue
                # Bound each source's read cache and avoid background Arrow
                # readers remaining active when a partially read stream closes.
                with fs.open(filename, "rb", block_size=256 * 1024) as handle:
                    with pq.ParquetFile(handle) as parquet:
                        for batch in parquet.iter_batches(batch_size=32, columns=["text"], use_threads=False):
                            yield from batch.column("text").to_pylist()


def _fineweb_source(repo_id, data_files, split, english=False):
    # Restart smaller languages to retain equal sampling by document count.
    while True:
        nonempty = False
        cached = []
        with closing(_fineweb_texts(repo_id, data_files, split, english)) as texts:
            selected = texts
            if english:
                selected = islice(texts, 1024, None) if split == "train" else islice(texts, 1024)
            for text in selected:
                if isinstance(text, str) and text:
                    nonempty = True
                    if cached is not None:
                        if len(cached) < 32:
                            cached.append(text)
                        else:
                            cached = None
                    yield text
        if not nonempty:
            print(f"Skipping empty {split} source in {repo_id}: {data_files}", flush=True)
            return
        # Some language splits contain only a handful of documents. Reuse those
        # texts instead of fetching the same remote file many times per batch.
        if cached is not None:
            while True:
                yield from cached


def _fineweb_batches(split):
    """Mix equal blocks of documents from English and every FineWeb2 language."""
    english_configs = _fineweb_configs("HuggingFaceFW/fineweb")
    # Resolve one crawl at a time, rather than globbing the entire English corpus.
    english_files = [
        entry
        for name in sorted(english_configs) if name.startswith("CC-MAIN-")
        for entry in english_configs[name]
    ]
    if not english_files:
        raise RuntimeError("No FineWeb crawl configurations found")
    configs = _fineweb_configs("HuggingFaceFW/fineweb-2")
    if not configs:
        raise RuntimeError("No FineWeb2 language configurations found")
    sources = [_fineweb_source("HuggingFaceFW/fineweb", english_files, split, english=True)]
    sources.extend(_fineweb_source("HuggingFaceFW/fineweb-2", configs[name], split) for name in sorted(configs))
    print(f"Preparing {split}: English + {len(configs)} FineWeb2 languages", flush=True)
    rng = random.Random(42)
    rng.shuffle(sources)
    # A small equal-sized block per source avoids opening every language before
    # writing the first batch. Shuffle each output batch to mix its languages.
    with ExitStack() as stack:
        for source in sources:
            stack.enter_context(closing(source))
        batch = []
        while sources:
            active_sources = []
            for source in sources:
                texts = list(islice(source, 32))
                if not texts:
                    continue
                active_sources.append(source)
                batch.extend(texts)
                if len(batch) == 1024:
                    rng.shuffle(batch)
                    yield batch
                    batch = []
            sources = active_sources
        raise RuntimeError(f"No usable {split} documents in FineWeb or FineWeb2")


def download_fineweb(num_train_shards, chars_per_shard=250_000_000):
    """Prepare mixed local shards, with a separate, fixed validation shard."""
    import pyarrow as pa

    if not 0 <= num_train_shards <= MAX_SHARD or chars_per_shard <= 0:
        raise ValueError("Invalid number of training shards or characters per shard")
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Preparing {num_train_shards} training shards and 1 validation shard in {DATA_DIR}", flush=True)
    with FileLock(os.path.join(DATA_DIR, ".download.lock")):
        schema = pa.schema([("text", pa.string())])
        for split, indices in [("val", [MAX_SHARD]), ("train", range(num_train_shards))]:
            # Check actual Parquet footers. An empty directory or interrupted file
            # must never count as a completed download.
            completed = {}
            for index in indices:
                filepath = os.path.join(DATA_DIR, index_to_filename(index))
                if os.path.isfile(filepath):
                    try:
                        with pq.ParquetFile(filepath) as parquet:
                            if parquet.metadata.num_rows > 0 and "text" in parquet.schema_arrow.names:
                                completed[index] = parquet.metadata.num_rows
                    except (OSError, pa.ArrowInvalid):
                        pass
            if len(completed) == len(indices):
                print(f"All {split} shards already exist; skipping", flush=True)
                continue
            last_missing = max(index for index in indices if index not in completed)
            with closing(_fineweb_batches(split)) as batches:
                for index in indices:
                    if index > last_missing:
                        break
                    filepath = os.path.join(DATA_DIR, index_to_filename(index))
                    if index in completed:
                        # Advance the deterministic stream so a resumed shard does not
                        # duplicate documents from earlier shards.
                        print(f"Skipping {filepath}; advancing past {completed[index]:,} documents", flush=True)
                        remaining = completed[index]
                        while remaining > 0:
                            remaining -= len(next(batches))
                        if remaining != 0:
                            raise RuntimeError(f"Cannot resume {filepath}: incompatible batch boundaries")
                        continue
                    temp_path = filepath + ".tmp"
                    nchars = ndocs = 0
                    last_progress = time.monotonic()
                    print(f"Writing {temp_path}", flush=True)
                    try:
                        with pq.ParquetWriter(temp_path, schema, compression="zstd", compression_level=3,
                                              use_dictionary=False, write_statistics=False) as writer:
                            while nchars < chars_per_shard:
                                batch = next(batches)
                                if not batch or not any(batch):
                                    raise RuntimeError(f"Empty batch while preparing {split}")
                                writer.write_table(pa.table({"text": batch}, schema=schema), row_group_size=1024)
                                ndocs += len(batch)
                                nchars += sum(map(len, batch))
                                now = time.monotonic()
                                if ndocs == len(batch) or now - last_progress >= 10:
                                    print(f"  {split} shard {index:05d}: {ndocs:,} documents, {nchars:,}/{chars_per_shard:,} characters", flush=True)
                                    last_progress = now
                        os.replace(temp_path, filepath)
                    except BaseException:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise
                    print(f"Wrote {filepath} ({ndocs:,} documents, {nchars:,} characters)", flush=True)
    print(f"Done! FineWeb shards are ready in {DATA_DIR}", flush=True)


def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    parser.add_argument("--fineweb", action="store_true", help="Prepare an equal-document mix of FineWeb and all FineWeb2 languages")
    parser.add_argument("--chars-per-shard", type=int, default=250_000_000, help="Characters per prepared FineWeb shard (default: 250000000)")
    args = parser.parse_args()
    if args.num_files < -1 or args.chars_per_shard <= 0:
        parser.error("--num-files must be -1 or nonnegative; --chars-per-shard must be positive")
    configure_dataset(args.fineweb)

    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    if args.fineweb:
        download_fineweb(num_train_shards, args.chars_per_shard)
        raise SystemExit

    # Prepare the output directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD) # always download the validation shard

    # Download the shards
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
