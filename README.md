# Bloom Filters and Counting Bloom Filters

Implementation, experiments and a simulated URL filter workload for standard, counting and layered Bloom filters.

## Layout

* src/bloom: hashing, standard filter, counting filter, layered filter, theory formulas, validation
* src/simulation: experiments and the web cache workload
* src/reporting: tables, plots and the report builder
* src/cli: command line interface
* tests: unit tests

## Usage

Run from the project root.

    python main.py demo
    python main.py fpr --n 2000
    python main.py optimal --bits 8
    python main.py layered --layers 1 2 3 4
    python main.py counting
    python main.py workload
    python main.py report --out output
    python main.py report --quick --out output

The report command writes CSV tables, PNG plots and results.md into the output folder.

## Tests

    python -m unittest discover tests

## Design notes

* Hash positions come from one blake2b digest split into two 64 bit values and combined by double hashing. Each layer or trial uses a different seed as the blake2b salt.
* The counting filter uses saturating counters. A saturated counter is never decremented, which keeps the no false negative guarantee at the cost of a few stuck counters.
* Layered filters split the error budget with split_error_budget. Under a fixed total memory, layering does not beat one larger filter, and the experiments show this.
* Counting filter memory is counter_bits times a standard filter of the same length, reported by memory_bits.
