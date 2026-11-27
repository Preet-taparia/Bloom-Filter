# Bloom Filters and Counting Bloom Filters

Implementation, experiments and a simulated URL filter workload for standard, counting and layered Bloom filters.

## Layout

* src/bloom: hashing, standard filter, counting filter, layered filter, theory formulas, validation
* src/Experiments: parameter studies, cache and URL workloads, CSV export and plots
* main.py: command line entry point

## Usage

Requires Python 3.11 or newer. Set up and run from the project root:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python main.py

To choose an output directory:

    python main.py output

You can also run the experiment runner directly:

    cd src/Experiments
    ../../.venv/bin/python Runall.py

Or as a module from the project root:

    python -m src.Experiments.Runall

All entry points run the full experiment suite and write CSV tables, PNG plots
and summary.txt. The default output is the project's results directory; an
explicit relative output directory is resolved from the current directory.

## Design notes

* Hash positions come from one blake2b digest split into two 64 bit values and combined by double hashing. Each layer or trial uses a different seed as the blake2b salt.
* The counting filter uses saturating counters. A saturated counter is never decremented, which keeps the no false negative guarantee at the cost of a few stuck counters.
* Layered filters split the error budget with split_error_budget. Under a fixed total memory, layering does not beat one larger filter, and the experiments show this.
* Counting filter memory is counter_bits times a standard filter of the same length, reported by memory_bits.

## Configurable hashing and comparisons

Select `murmur3`, `xxhash`, `fnv1a`, `md5`, or the original `blake2b`:

```bash
python main.py results/md5 --hash md5
python main.py results/all-hashes --hash all
```

`all` runs the full existing experiment suite separately for each of the five
algorithms (including the original BLAKE2b), then creates a matched benchmark. Runs are sequential to avoid
CPU contention affecting timing. Each full suite has its own algorithm folder.
The original no-argument command retains BLAKE2b behavior.

For just the hash comparison (much faster):

```bash
python main.py results/hash-comparison --hash all --compare-only
python main.py results/hash-comparison --hash all --compare-only --items 500 --queries 2000 --trials 3
```

The benchmark sweeps 4, 8, 12 and 16 bits per item using identical inputs and
seeds for each algorithm, with optimal hash counts and shuffled algorithm order.
`csv/hash_trials.csv` stores individual trials; `csv/hash_summary.csv` stores
means and standard deviations. Only two metrics are saved and plotted:
false positive rate and execution time in milliseconds. Execution time covers
inserting all items, querying every inserted item, and querying all absent
probes. Input generation, filter allocation and output writing are excluded.
Run configuration columns remain in the CSV files for reproducibility.
Metadata and implementation details are saved in `hash_metadata.json`.

Variants: pure-Python MurmurHash3 x86-32, pure-Python xxHash32, pure-Python
FNV-1a 64, and pure-Python MD5 and BLAKE2b. No hash implementation calls a native hashing library. Timing compares these
implementations, not native-library performance rankings. BLAKE2b retains
its original seed salt and hash behavior.

Programmatic selection works for standard and counting filters; layered filters
use the algorithms configured on their member filters:

```python
from src.bloom.standard import BloomFilter
bloom = BloomFilter.from_capacity(1000, 0.01, hash_algorithm="xxhash")
```

Tests: `python -m unittest discover -s tests -v`.

## Pure-Python comparison (preserves previous results)

All five hash implementations now execute their rounds in Python. `hashlib`
is used only as an independent test oracle, never in the benchmark.
MD5 follows [RFC 1321](https://www.rfc-editor.org/rfc/rfc1321);
BLAKE2b compression follows [RFC 7693](https://www.rfc-editor.org/rfc/rfc7693),
with the original salt and 16-byte digest configuration preserved.

Use a separate destination to keep the earlier mixed-implementation graphs:

```bash
python main.py results/hash-comparison-python --hash all --compare-only --items 2000 --queries 10000 --trials 3
```

Previous CSVs and graphs remain in `results/hash-comparison/`.
New CSVs, graphs and metadata are in `results/hash-comparison-python/`.
The workload matches the saved earlier run: 2000 inserted items, 10000 absent
queries, 3 trials, and four bits-per-item settings. Only false positive rate
and execution time are compared. Timings still measure these particular
implementations, not an implementation-independent ranking of algorithms.

## Library-backed hashing (current implementation)

The earlier pure-Python comparisons above describe historical runs. The current
`HashAlgorithms.py` uses native-backed libraries for all five algorithms:

- `mmh3.hash(..., signed=False)` for MurmurHash3 x86-32.
- `xxhash.xxh32_intdigest` for unsigned xxHash32.
- `fnv-c` / `fnv_c.fnv1a_64` for native unsigned FNV-1a 64 via CFFI.
- `hashlib.md5(..., usedforsecurity=False)` and `hashlib.blake2b` for byte digests.

Install with `python -m pip install -r requirements.txt`. Building `fnv-c` from
source may require a C compiler. Its native implementation is documented at
https://github.com/botify-labs/fnv-c. No silent pure-Python fallback is used, so
missing packages fail explicitly instead of mixing implementation types in timings.

The seed wrapping, algorithm variants, BLAKE2b digest size and salt, hash-pair
construction and index generation are preserved. Compatibility fixtures in
`tests/hash_compatibility_vectors.json` were captured from the handwritten
implementations before removal. Run `python -m unittest discover -s tests -v`
to check those outputs, known vectors and Bloom membership against the libraries.
No hash-position changes are intended; any failed compatibility check must be
resolved before reusing populated filters.

Use a separate output directory for new experiments:

```bash
python main.py results/hash-comparison-libraries --hash all --compare-only
```

Benchmark and plotting code is intentionally unchanged. Its hard-coded
“pure Python” graph titles and implementation metadata are now historical labels
and do not describe the current backend; interpret new runs using this section.
During this migration, package installation in the agent environment was blocked
by DNS resolution of pypi.org, so full library compatibility and membership tests
could not yet run there.

## Counting Bloom filter hash comparison

```bash
python main.py results/counting-hash-comparison-libraries --hash all --compare-only --filter counting
```

This runs the same five library-backed hashes with a counting Bloom filter,
using 4-bit counters by default (`--counter-bits 1` through `--counter-bits 8`).
It saves `counting_hash_trials.csv`, `counting_hash_summary.csv`, two
`counting_hash_*.png` graphs, and `counting_hash_metadata.json`.
Standard-filter outputs retain their existing names and are not overwritten.

The x-axis measures **counter slots per inserted item**, not bits per item:
4, 8, 12 and 16 slots are tested. A 4-bit counter needs four logical bits per
slot. Inputs, seeds, slot counts and hash counts match the standard comparison.
Execution time covers insertion and present/absent membership queries; deletion
is not included, so the workload stays comparable. Only false positive rate
and execution time are plotted. Titles identify the counting Bloom filter and
library implementation. `--filter` selects the hash comparison filter only;
the full experiment suite still runs its existing standard and counting studies.
