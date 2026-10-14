import json
import math
import os
import random
import sys
import time
import tracemalloc
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bloom import (
    BloomFilter,
    CountingBloomFilter,
    exact_fpr,
    optimal_k,
    optimal_m,
    theoretical_fpr,
)

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 19


def make_urls(n, rng, tag="u"):
    seen = set()
    out = []
    while len(out) < n:
        host = rng.randrange(2000)
        depth = rng.randrange(1, 5)
        path = "/".join(f"{rng.randrange(10**6):x}" for _ in range(depth))
        url = f"http://site{host}.example.org/{tag}/{path}.html"
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def experiment_fpr():
    rng = random.Random(SEED)
    n = 20000
    members = make_urls(n, rng, "in")
    probes = make_urls(200000, rng, "out")
    rows = []
    for bits_per_item in (4, 6, 8, 10, 12, 16):
        m = bits_per_item * n
        k = optimal_k(m, n)
        for scheme in ("double", "salted"):
            bf = BloomFilter(m, k, scheme)
            for u in members:
                bf.add(u)
            fp = sum(1 for u in probes if u in bf)
            rows.append(
                {
                    "bits_per_item": bits_per_item,
                    "m": m,
                    "n": n,
                    "k": k,
                    "scheme": scheme,
                    "observed": fp / len(probes),
                    "theory": theoretical_fpr(m, n, k),
                    "exact": exact_fpr(m, n, k),
                    "fill": bf.fill_ratio(),
                    "fp_count": fp,
                }
            )
    return rows


def experiment_k_sweep():
    rng = random.Random(SEED + 1)
    n = 50000
    m = 10 * n
    members = make_urls(n, rng, "in")
    probes = make_urls(100000, rng, "out")
    hits = members[:50000]
    rows = []
    for k in range(1, 21):
        for scheme in ("double", "salted"):
            bf = BloomFilter(m, k, scheme)
            for u in members:
                bf.add(u)
            fp = sum(1 for u in probes if u in bf)
            t0 = time.perf_counter()
            for u in probes:
                u in bf
            neg_time = (time.perf_counter() - t0) / len(probes)
            t0 = time.perf_counter()
            for u in hits:
                u in bf
            pos_time = (time.perf_counter() - t0) / len(hits)
            rows.append(
                {
                    "k": k,
                    "scheme": scheme,
                    "observed": fp / len(probes),
                    "theory": theoretical_fpr(m, n, k),
                    "neg_query_us": neg_time * 1e6,
                    "pos_query_us": pos_time * 1e6,
                }
            )
    return {"n": n, "m": m, "rows": rows, "k_opt": optimal_k(m, n)}


def experiment_memory():
    rng = random.Random(SEED + 2)
    n = 100000
    urls = make_urls(n, rng, "mem")
    avg_len = sum(len(u) for u in urls) / n
    string_bytes = sum(sys.getsizeof(u) for u in urls)

    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    container = set(urls)
    set_container = tracemalloc.get_traced_memory()[0] - base
    tracemalloc.stop()

    fingerprints = None
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    fingerprints = {hash(u) & 0xFFFFFFFFFFFFFFFF for u in urls}
    fp_set = tracemalloc.get_traced_memory()[0] - base
    tracemalloc.stop()

    rows = []
    for p in (0.1, 0.01, 0.001, 0.0001):
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        bf = BloomFilter.for_capacity(n, p)
        for u in urls:
            bf.add(u)
        measured = tracemalloc.get_traced_memory()[0] - base
        tracemalloc.stop()
        cbf = CountingBloomFilter.for_capacity(n, p, counter_bits=4)
        for u in urls:
            cbf.add(u)
        probes = make_urls(100000, rng, "pr")
        obs = sum(1 for u in probes if u in bf) / len(probes)
        rows.append(
            {
                "target_p": p,
                "m": bf.m,
                "k": bf.k,
                "bits_per_item": bf.m / n,
                "ideal_lower_bound_bytes": math.ceil(n * math.log2(1 / p) / 8),
                "theoretical_bytes": math.ceil(bf.m / 8),
                "measured_bytes": measured,
                "counting_bytes": cbf.memory_bytes(),
                "observed_fpr": obs,
            }
        )
    return {
        "n": n,
        "avg_url_len": avg_len,
        "set_container_bytes": set_container,
        "string_bytes": string_bytes,
        "set_total_bytes": set_container + string_bytes,
        "fingerprint_set_bytes": fp_set,
        "theoretical_64bit_fingerprints_bytes": n * 8,
        "rows": rows,
    }


def experiment_deletions():
    rng = random.Random(SEED + 3)
    result = {}

    n = 20000
    items = make_urls(n, rng, "d")
    absent = make_urls(100000, rng, "abs")
    cbf = CountingBloomFilter.for_capacity(n, 0.01, counter_bits=4)
    for u in items:
        cbf.add(u)
    removed, kept = items[: n // 2], items[n // 2 :]
    all_ok = all(cbf.remove(u) for u in removed)
    false_neg = sum(1 for u in kept if u not in cbf)
    still_present = sum(1 for u in removed if u in cbf)
    fp_absent = sum(1 for u in absent if u in cbf) / len(absent)
    std = BloomFilter.for_capacity(n, 0.01)
    for u in kept:
        std.add(u)
    std_fp_absent = sum(1 for u in absent if u in std) / len(absent)
    ref = BloomFilter.for_capacity(n, 0.01)
    for u in items:
        ref.add(u)
    removed_in_standard = sum(1 for u in removed if u in ref)
    result["basic"] = {
        "n": n,
        "m": cbf.m,
        "k": cbf.k,
        "deleted": len(removed),
        "all_removals_accepted": all_ok,
        "false_negatives_among_kept": false_neg,
        "deleted_still_reported_present": still_present,
        "deleted_still_present_rate": still_present / len(removed),
        "fpr_on_absent_after_deletion": fp_absent,
        "fpr_on_absent_fresh_filter_of_kept": std_fp_absent,
        "standard_filter_deleted_still_present": removed_in_standard,
    }

    n2 = 20000
    items2 = make_urls(n2, rng, "o")
    cbf2 = CountingBloomFilter.for_capacity(n2, 0.01, counter_bits=4)
    for u in items2:
        cbf2.add(u)
    result["counter_distribution"] = {
        "histogram": cbf2.counter_histogram(),
        "max_counter": cbf2.max_counter(),
        "overflow_events": cbf2.overflow_events,
    }

    overflow_rows = []
    for bits in (2, 4, 8):
        for sticky in (True, False):
            m = 4000
            k = 3
            f = CountingBloomFilter(m, k, counter_bits=bits, sticky=sticky)
            pool = make_urls(2000, random.Random(SEED + 40 + bits), "ov")
            for u in pool:
                f.add(u)
            for u in pool[:1000]:
                f.remove(u, guard=False)
            fn = sum(1 for u in pool[1000:] if u not in f)
            overflow_rows.append(
                {
                    "counter_bits": bits,
                    "sticky": sticky,
                    "overflow_events": f.overflow_events,
                    "false_negatives": fn,
                    "kept": 1000,
                }
            )
    result["overflow"] = overflow_rows

    analytic = []
    for j in (4, 8, 12, 16):
        for mm in (10**5, 10**6):
            bound = mm * (math.e * math.log(2) / j) ** j
            analytic.append({"j": j, "m": mm, "bound": bound})
    result["overflow_bound"] = analytic

    pool = make_urls(10000, random.Random(SEED + 5), "rd")
    rep_rows = []
    for guard in (True, False):
        f = CountingBloomFilter(60000, 4, counter_bits=4)
        for u in pool:
            f.add(u)
        victims = pool[:200]
        for u in victims:
            f.remove(u, guard=guard)
        accepted = 0
        for _ in range(5):
            for u in victims:
                accepted += 1 if f.remove(u, guard=guard) else 0
        fn = sum(1 for u in pool[200:] if u not in f)
        rep_rows.append(
            {
                "guard": guard,
                "repeat_removals_accepted": accepted,
                "repeat_removals_attempted": 5 * len(victims),
                "false_negatives_among_kept": fn,
                "kept": len(pool) - 200,
            }
        )
    result["repeated_deletion"] = rep_rows

    pool = make_urls(10000, random.Random(SEED + 6), "nm")
    strangers = make_urls(5000, random.Random(SEED + 7), "st")
    stranger_rows = []
    for m_factor in (6, 10, 16):
        f = CountingBloomFilter(m_factor * len(pool), optimal_k(m_factor * len(pool), len(pool)))
        for u in pool:
            f.add(u)
        accepted = [u for u in strangers if f.remove(u, guard=True)]
        fn = sum(1 for u in pool if u not in f)
        stranger_rows.append(
            {
                "bits_per_item": m_factor,
                "strangers_tried": len(strangers),
                "accepted_by_guard": len(accepted),
                "false_negatives_caused": fn,
            }
        )
    result["stranger_deletion"] = stranger_rows

    per_bits = []
    for bits in (2, 4, 8):
        f = CountingBloomFilter(optimal_m(20000, 0.01), 7, counter_bits=bits)
        pool = make_urls(20000, random.Random(SEED + 8), "pb")
        for u in pool:
            f.add(u)
        per_bits.append(
            {
                "counter_bits": bits,
                "memory_bytes": f.memory_bytes(),
                "overflow_events": f.overflow_events,
                "max_counter": f.max_counter(),
            }
        )
    result["counter_width"] = per_bits
    return result


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = OrderedDict()

    def lookup(self, url):
        if url in self.data:
            self.data.move_to_end(url)
            return True
        return False

    def insert(self, url):
        evicted = None
        self.data[url] = True
        if len(self.data) > self.capacity:
            evicted, _ = self.data.popitem(last=False)
        return evicted


def zipf_stream(universe, length, alpha, rng):
    weights = [1.0 / (i + 1) ** alpha for i in range(len(universe))]
    return rng.choices(universe, weights=weights, k=length)


def simulate_cache(strategy, stream, capacity, bits_per_entry, rebuild_every, counter_bits=4):
    cache = LRUCache(capacity)
    m = bits_per_entry * capacity
    k = optimal_k(m, capacity)
    if strategy == "counting":
        flt = CountingBloomFilter(m, k, counter_bits=counter_bits)
    elif strategy in ("standard", "standard_rebuild"):
        flt = BloomFilter(m, k)
    else:
        flt = None
    stats = {
        "requests": len(stream),
        "disk_lookups": 0,
        "avoided_lookups": 0,
        "wasted_lookups": 0,
        "true_hits": 0,
        "origin_fetches": 0,
        "false_negatives": 0,
        "negatives_total": 0,
        "negatives_filter_positive": 0,
        "rebuilds": 0,
    }
    for r, url in enumerate(stream, 1):
        in_cache = url in cache.data
        if flt is None:
            predicted = True
        else:
            predicted = url in flt
            if not in_cache:
                stats["negatives_total"] += 1
                if predicted:
                    stats["negatives_filter_positive"] += 1
            if in_cache and not predicted:
                stats["false_negatives"] += 1
        if predicted:
            stats["disk_lookups"] += 1
            if cache.lookup(url):
                stats["true_hits"] += 1
            else:
                stats["wasted_lookups"] += 1
                stats["origin_fetches"] += 1
                evicted = cache.insert(url)
                if flt is not None:
                    flt.add(url)
                    if evicted is not None and strategy == "counting":
                        flt.remove(evicted)
        else:
            stats["avoided_lookups"] += 1
            stats["origin_fetches"] += 1
            evicted = cache.insert(url)
            flt.add(url)
            if evicted is not None and strategy == "counting":
                flt.remove(evicted)
        if strategy == "standard_rebuild" and r % rebuild_every == 0:
            flt.clear()
            for u in cache.data:
                flt.add(u)
            stats["rebuilds"] += 1
    if flt is not None:
        fill = flt.fill_ratio() if isinstance(flt, BloomFilter) else flt.nonzero_ratio()
    else:
        fill = None
    stats["final_fill_ratio"] = fill
    stats["hit_ratio"] = stats["true_hits"] / stats["requests"]
    stats["avoided_fraction"] = stats["avoided_lookups"] / stats["requests"]
    stats["wasted_fraction_of_lookups"] = (
        stats["wasted_lookups"] / stats["disk_lookups"] if stats["disk_lookups"] else 0.0
    )
    stats["observed_fpr"] = (
        stats["negatives_filter_positive"] / stats["negatives_total"]
        if stats["negatives_total"]
        else None
    )
    stats["m"] = m
    stats["k"] = k
    if flt is not None:
        stats["filter_bytes"] = flt.memory_bytes()
    return stats


def experiment_web_cache():
    rng = random.Random(SEED + 9)
    universe = make_urls(100000, rng, "web")
    stream = zipf_stream(universe, 300000, 0.8, rng)
    capacity = 10000
    main = {}
    for strategy in ("none", "standard", "standard_rebuild", "counting"):
        main[strategy] = simulate_cache(strategy, stream, capacity, 10, 20000)
    sweep = []
    for bits in (4, 6, 8, 10, 12, 16):
        s = simulate_cache("counting", stream, capacity, bits, 20000)
        s["bits_per_entry"] = bits
        sweep.append(s)
    cap_sweep = []
    for cap in (2000, 5000, 10000, 20000, 40000):
        none = simulate_cache("none", stream, cap, 10, 20000)
        cnt = simulate_cache("counting", stream, cap, 10, 20000)
        cap_sweep.append(
            {
                "capacity": cap,
                "hit_ratio": cnt["hit_ratio"],
                "avoided_fraction": cnt["avoided_fraction"],
                "wasted_lookups": cnt["wasted_lookups"],
                "baseline_lookups": none["disk_lookups"],
                "counting_lookups": cnt["disk_lookups"],
                "observed_fpr": cnt["observed_fpr"],
            }
        )
    width = []
    for bits in (2, 4, 8):
        s = simulate_cache("counting", stream, capacity, 10, 20000, counter_bits=bits)
        s["counter_bits"] = bits
        width.append(s)
    return {
        "universe": len(universe),
        "requests": len(stream),
        "capacity": capacity,
        "alpha": 0.8,
        "main": main,
        "bits_sweep": sweep,
        "capacity_sweep": cap_sweep,
        "counter_width": width,
    }


def make_figures(res):
    rows = res["fpr"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for scheme, marker in (("double", "o"), ("salted", "s")):
        sub = [r for r in rows if r["scheme"] == scheme]
        ax.plot(
            [r["bits_per_item"] for r in sub],
            [r["observed"] for r in sub],
            marker=marker,
            linestyle="none",
            label=f"observed ({scheme})",
        )
    th = [r for r in rows if r["scheme"] == "double"]
    ax.plot([r["bits_per_item"] for r in th], [r["theory"] for r in th], label="theory")
    ax.set_yscale("log")
    ax.set_xlabel("bits per item (m/n)")
    ax.set_ylabel("false positive rate")
    ax.set_title("Observed vs theoretical false positive rate, optimal k")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_fpr.png"), dpi=150)
    plt.close(fig)

    ks = res["k_sweep"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for scheme, marker in (("double", "o"), ("salted", "s")):
        sub = [r for r in ks if r["scheme"] == scheme]
        axes[0].plot(
            [r["k"] for r in sub],
            [r["observed"] for r in sub],
            marker=marker,
            linestyle="none",
            label=f"observed ({scheme})",
        )
        axes[1].plot(
            [r["k"] for r in sub],
            [r["neg_query_us"] for r in sub],
            marker=marker,
            label=f"absent items ({scheme})",
        )
        axes[1].plot(
            [r["k"] for r in sub],
            [r["pos_query_us"] for r in sub],
            marker=marker,
            linestyle="--",
            label=f"present items ({scheme})",
        )
    sub = [r for r in ks if r["scheme"] == "double"]
    axes[0].plot([r["k"] for r in sub], [r["theory"] for r in sub], label="theory")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of hash functions k")
    axes[0].set_ylabel("false positive rate")
    axes[0].set_title("False positive rate vs k (m/n = 10)")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("number of hash functions k")
    axes[1].set_ylabel("query time (microseconds)")
    axes[1].set_title("Query time vs k")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_k_sweep.png"), dpi=150)
    plt.close(fig)

    mem = res["memory"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = [f"p={r['target_p']}" for r in mem["rows"]]
    xs = range(len(labels))
    ax.bar([x - 0.27 for x in xs], [r["measured_bytes"] / 1024 for r in mem["rows"]], 0.27, label="Bloom filter")
    ax.bar([x for x in xs], [r["counting_bytes"] / 1024 for r in mem["rows"]], 0.27, label="Counting (4 bit)")
    ax.bar(
        [x + 0.27 for x in xs],
        [mem["fingerprint_set_bytes"] / 1024] * len(labels),
        0.27,
        label="Set of 64 bit hashes",
    )
    ax.axhline(mem["set_total_bytes"] / 1024, color="red", linestyle=":", label="Python set of URLs")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("memory (KiB, log scale)")
    ax.set_title(f"Memory for {mem['n']} URLs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_memory.png"), dpi=150)
    plt.close(fig)

    wc = res["web_cache"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["none", "standard", "standard_rebuild", "counting"]
    axes[0].bar(names, [wc["main"][n]["disk_lookups"] / 1000 for n in names])
    axes[0].set_ylabel("disk lookups (thousands)")
    axes[0].set_title("Disk lookups by strategy, 300000 requests")
    axes[0].tick_params(axis="x", labelsize=8)
    sweep = wc["bits_sweep"]
    axes[1].plot(
        [s["bits_per_entry"] for s in sweep],
        [s["wasted_lookups"] / 1000 for s in sweep],
        marker="o",
    )
    axes[1].set_xlabel("bits per cache entry")
    axes[1].set_ylabel("wasted lookups from false positives (thousands)")
    axes[1].set_title("Counting filter: wasted lookups vs filter size")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_web_cache.png"), dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    res = {}
    print("false positive rate")
    res["fpr"] = experiment_fpr()
    print("k sweep")
    res["k_sweep"] = experiment_k_sweep()
    print("memory")
    res["memory"] = experiment_memory()
    print("deletions")
    res["deletions"] = experiment_deletions()
    print("web cache")
    res["web_cache"] = experiment_web_cache()
    make_figures(res)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("done in", round(time.time() - t0, 1), "seconds")


if __name__ == "__main__":
    main()
