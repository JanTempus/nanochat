"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import hashlib
import json
import random
import re
import tempfile
import time
import requests
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from email.utils import parsedate_to_datetime
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

FINEWEB2_LANGUAGES = (
    "arb_Arab",  # Arabic
    "cmn_Hani",  # Mandarin Chinese
    "fra_Latn",  # French
    "hin_Deva",  # Hindi
    "rus_Cyrl",  # Russian
    "swh_Latn",  # Swahili
    "tel_Telu",  # Telugu
    "tha_Thai",  # Thai
    "tur_Latn",  # Turkish
)
FINEWEB_CHECKPOINT_KEY = b"nanochat.fineweb.checkpoint"


def _atomic_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=os.path.dirname(path), delete=False) as handle:
            temp_path = handle.name
            json.dump(value, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)


def _read_json(path):
    with open(path) as handle:
        return json.load(handle)

def configure_dataset(fineweb=False):
    global DATA_DIR
    # Keep the ten-language mix separate from previously prepared all-language shards.
    DATA_DIR = os.path.join(base_dir, "base_data_fineweb_10lang" if fineweb else "base_data_climbmix")

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
def _hf_retry(operation, *args, **kwargs):
    """Retry rate-limited Hub metadata calls without restarting document streams."""
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            return operation(*args, **kwargs)
        except (HfHubHTTPError, LocalEntryNotFoundError) as error:
            # DatasetCard.load can wrap a failed HEAD request in a cache miss.
            cause = error.__cause__ if isinstance(error, LocalEntryNotFoundError) else error
            response = cause.response if isinstance(cause, HfHubHTTPError) else None
            if response is None or response.status_code != 429:
                raise
            if attempt == 0:
                print("Hugging Face rate limit reached. Ensure HF_TOKEN or a saved `hf auth login` "
                      "is available inside the download job, and HF_HUB_DISABLE_IMPLICIT_TOKEN is unset.",
                      flush=True)
            if attempt == max_attempts - 1:
                raise
            # Older huggingface_hub versions do not retry these API calls.
            # Respect both HTTP Retry-After and the Hub's RateLimit reset time.
            delays = []
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delays.append(float(retry_after))
                except ValueError:
                    try:
                        delays.append(parsedate_to_datetime(retry_after).timestamp() - time.time())
                    except (TypeError, ValueError, OverflowError):
                        pass
            resets = re.findall(r'(?:^|;)\s*t=(\d+)', response.headers.get("RateLimit", ""))
            delays.extend(float(reset) for reset in resets)
            delays = [delay for delay in delays if 0 <= delay < float("inf")]
            wait_time = max(delays) + 1 if delays else min(60 * 2 ** attempt, 300)
            print(f"Hugging Face HTTP 429; retrying in {wait_time:.0f}s "
                  f"(attempt {attempt + 2}/{max_attempts})", flush=True)
            time.sleep(wait_time)


@lru_cache(maxsize=2)
def _fineweb_configs(repo_id, revision=None):
    """Read the small dataset card once, without resolving every data file."""
    from huggingface_hub import DatasetCard

    print(f"Reading source metadata: {repo_id}", flush=True)
    card_path = _fineweb_local_file(repo_id, revision, "README.md") if revision else repo_id
    configs = _hf_retry(DatasetCard.load, card_path, repo_type="dataset").data.configs
    return {config["config_name"]: config["data_files"] for config in configs}


def _fineweb_local_file(repo_id, revision, filename):
    """Use complete cached files offline; let the Hub resume unfinished downloads."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    kwargs = dict(repo_type="dataset", revision=revision, cache_dir=os.path.join(base_dir, "fineweb_source_cache"))
    cached = try_to_load_from_cache(repo_id, filename, **kwargs)
    if isinstance(cached, str):
        return cached
    print(f"Downloading source file: {repo_id}/{filename}", flush=True)
    return _hf_retry(hf_hub_download, repo_id, filename, **kwargs)


@lru_cache(maxsize=128)
def _fineweb_files(repo_id, revision, pattern, cache_dir):
    """Persist file listings for immutable source revisions, one crawl at a time."""
    from huggingface_hub import HfFileSystem

    key = hashlib.sha256(json.dumps([repo_id, revision, pattern]).encode()).hexdigest()
    path = os.path.join(cache_dir, "listings", key + ".json")
    if os.path.isfile(path):
        return _read_json(path)
    fs = HfFileSystem()
    paths = _hf_retry(fs.glob, f"datasets/{repo_id}@{revision}/{pattern}")
    files = [fs.resolve_path(name).path_in_repo for name in sorted(paths) if name.endswith(".parquet")]
    _atomic_json(path, files)
    return files


def _fineweb_catalog():
    from huggingface_hub import HfApi

    path = os.path.join(DATA_DIR, ".sources.json")
    if os.path.isfile(path):
        return _read_json(path)
    api = HfApi()
    revisions = {repo: _hf_retry(api.dataset_info, repo).sha
                 for repo in ("HuggingFaceFW/fineweb", "HuggingFaceFW/fineweb-2")}
    english_configs = _fineweb_configs("HuggingFaceFW/fineweb", revisions["HuggingFaceFW/fineweb"])
    # Resolve one crawl at a time, rather than globbing the entire English corpus.
    english_files = [
        entry
        for name in sorted(english_configs) if name.startswith("CC-MAIN-")
        for entry in english_configs[name]
    ]
    if not english_files:
        raise RuntimeError("No FineWeb crawl configurations found")
    configs = _fineweb_configs("HuggingFaceFW/fineweb-2", revisions["HuggingFaceFW/fineweb-2"])
    missing = sorted(set(FINEWEB2_LANGUAGES) - configs.keys())
    if missing:
        raise RuntimeError(f"Missing requested FineWeb2 language configurations: {', '.join(missing)}")
    sources = [{"repo_id": "HuggingFaceFW/fineweb", "language": "english", "data_files": english_files}]
    sources.extend({"repo_id": "HuggingFaceFW/fineweb-2", "language": name, "data_files": configs[name]}
                   for name in sorted(FINEWEB2_LANGUAGES))
    for source in sources:
        source["revision"] = revisions[source["repo_id"]]
    _atomic_json(path, sources)
    return sources


class _FinewebReader:
    """A local Parquet reader with a serializable position within a language."""

    def __init__(self, spec, split, state=None):
        self.spec, self.split = spec, split
        self.english = spec["language"] == "english"
        remote_split = "train" if self.english or split == "train" else "test"
        self.patterns = [path for entry in spec["data_files"] if entry["split"] == remote_split
                         for path in ([entry["path"]] if isinstance(entry["path"], str) else entry["path"])]
        self.state = dict(state) if state is not None else self._initial_state()
        self.parquet = self.batches = None
        self.buffer = []
        self.buffer_index = 0

    @staticmethod
    def _initial_state():
        return dict(pattern=0, file=0, row_group=0, row=0, raw_seen=0, emitted=0, exhausted=False)

    def close(self):
        self.batches = None
        self.buffer = []
        self.buffer_index = 0
        if self.parquet is not None:
            self.parquet.close()
            self.parquet = None

    def _raw_text(self):
        state = self.state
        while state["pattern"] < len(self.patterns):
            if self.english and self.split == "val" and state["raw_seen"] >= 1024:
                return None
            if self.parquet is None:
                files = _fineweb_files(self.spec["repo_id"], self.spec["revision"], self.patterns[state["pattern"]],
                                       os.path.join(base_dir, "fineweb_source_cache"))
                if state["file"] >= len(files):
                    state.update(pattern=state["pattern"] + 1, file=0, row_group=0, row=0)
                    continue
                path = _fineweb_local_file(self.spec["repo_id"], self.spec["revision"], files[state["file"]])
                self.parquet = pq.ParquetFile(path)
            if state["row_group"] >= self.parquet.num_row_groups:
                self.close()
                state.update(file=state["file"] + 1, row_group=0, row=0)
                continue
            if state["row"] >= self.parquet.metadata.row_group(state["row_group"]).num_rows:
                state.update(row_group=state["row_group"] + 1, row=0)
                self.batches = None
                self.buffer = []
                self.buffer_index = 0
                continue
            if self.buffer_index == len(self.buffer):
                skip = 0
                if self.batches is None:
                    self.batches = self.parquet.iter_batches(batch_size=1024, row_groups=[state["row_group"]],
                                                            columns=["text"], use_threads=False)
                    skip = state["row"]
                # Resume skips whole row groups using metadata and decodes only
                # the current row group's prefix, without rereading older files.
                batch = next(self.batches)
                while skip >= len(batch):
                    skip -= len(batch)
                    batch = next(self.batches)
                self.buffer = batch.column("text").slice(skip).to_pylist()
                self.buffer_index = 0
            text = self.buffer[self.buffer_index]
            self.buffer_index += 1
            state["row"] += 1
            state["raw_seen"] += 1
            # Return a tuple so null text is distinct from end of source.
            return (text,)
        return None

    def read_block(self):
        texts = []
        while len(texts) < 32 and not self.state["exhausted"]:
            row = self._raw_text()
            if row is None:
                emitted = self.state["emitted"]
                self.close()
                self.state = self._initial_state()
                if not emitted:
                    self.state["exhausted"] = True
                    print(f"Skipping empty {self.split} source: {self.spec['language']}", flush=True)
                continue
            if self.english and self.split == "train" and self.state["raw_seen"] <= 1024:
                continue
            text = row[0]
            if isinstance(text, str) and text:
                texts.append(text)
                self.state["emitted"] += 1
        return texts, dict(self.state)


class _FinewebMixer:
    """Prefetch one block per language; commit positions in deterministic order."""

    def __init__(self, split, num_workers=4, state=None):
        if num_workers < 1:
            raise ValueError("num_workers must be positive")
        if state is not None and (state["version"] != 1 or state["split"] != split):
            raise ValueError("Incompatible FineWeb checkpoint")
        self.split = split
        self.catalog = state["catalog"] if state is not None else _fineweb_catalog()
        if [spec["language"] for spec in self.catalog] != ["english", *sorted(FINEWEB2_LANGUAGES)]:
            raise ValueError("FineWeb source catalog does not match the selected languages")
        self.rng = random.Random(42)
        self.order = list(range(len(self.catalog)))
        self.rng.shuffle(self.order)
        self.next_source = 0
        self.committed = [_FinewebReader._initial_state() for _ in self.catalog]
        if state is not None:
            self.order = list(state["order"])
            self.next_source = state["next_source"]
            rng_state = state["rng"]
            self.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
            self.committed = deepcopy(state["readers"])
        self.readers = [_FinewebReader(spec, split, position) for spec, position in zip(self.catalog, self.committed)]
        self.executor = ThreadPoolExecutor(max_workers=min(num_workers, len(self.catalog)), thread_name_prefix="fineweb")
        self.pending = {}
        self.closed = False
        print(f"Preparing {split}: English + {len(FINEWEB2_LANGUAGES)} FineWeb2 languages, "
              f"{min(num_workers, len(self.catalog))} download/read workers", flush=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        if not self.pending:
            for index in self.order:
                self.pending[index] = self.executor.submit(self.readers[index].read_block)
        batch = []
        while self.order:
            index = self.order[self.next_source]
            texts, position = self.pending.pop(index).result()
            self.committed[index] = position
            if not texts:
                self.order.pop(self.next_source)
                if self.order:
                    self.next_source %= len(self.order)
                continue
            self.pending[index] = self.executor.submit(self.readers[index].read_block)
            self.next_source = (self.next_source + 1) % len(self.order)
            batch.extend(texts)
            if len(batch) == 1024:
                self.rng.shuffle(batch)
                return batch
        raise RuntimeError(f"No usable {self.split} documents in FineWeb or FineWeb2")

    def state_dict(self):
        return deepcopy(dict(version=1, split=self.split, catalog=self.catalog, order=self.order,
                             next_source=self.next_source, rng=self.rng.getstate(), readers=self.committed))

    def close(self):
        if not self.closed:
            self.closed = True
            self.executor.shutdown(wait=True, cancel_futures=True)
            for reader in self.readers:
                reader.close()


def _fineweb_batches(split, num_workers=4, state=None):
    return _FinewebMixer(split, num_workers, state)


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
    parser.add_argument("--fineweb", action="store_true", help="Prepare an equal-document mix of English FineWeb and nine selected FineWeb2 languages")
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
