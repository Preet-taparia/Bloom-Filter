# P19: Bloom Filters and Counting Bloom Filters

Probabilistic and Streaming Structures. Implementation and Study Report

## Abstract

This report studies space efficient probabilistic set membership filters and their use in avoiding unnecessary disk and network queries. I implemented a standard Bloom filter and a counting Bloom filter in Python, both with a configurable number of hash functions and two selectable hash schemes (double hashing and independently salted hashes). Using synthetic URL workloads, I measured the false positive rate across filter sizes and values of k, compared the results with the classical analysis of Bloom (1970) and Broder and Mitzenmacher (2004), measured real memory use against a hash set, tested counting filter deletions including counter overflow and repeated deletion, and simulated a web proxy cache driven by a Zipf request stream. The headline results are that observed false positive rates track the theoretical formula to within sampling error (for example 0.84 percent observed against 0.82 percent predicted at 10 bits per item), that a 1 percent Bloom filter for 100,000 URLs needs about 120 KB against 13.3 MB for a Python set of the same URLs (roughly 111 times smaller), and that in the cache simulation a counting filter removed 99.2 percent of the unnecessary disk lookups while a standard filter that cannot delete removed only 22.7 percent. I also report two failure modes of counting filters that are easy to miss: saturated counters must be made sticky, and the deletion guard is only as reliable as the filter's own false positive rate.

## 1. Introduction

Many systems must answer the question "is this item in a large set?" before doing something expensive, such as reading a disk, querying a database, or contacting another cache over a network. An exact answer requires storing the set, which can cost more memory than the saved work is worth. A Bloom filter trades a small, tunable chance of a wrong "yes" for a large reduction in memory, and never gives a wrong "no". Bloom (1970) introduced the structure for hash coding with allowable errors, and Broder and Mitzenmacher (2004) surveyed its mathematics and its use in networking. Fan, Cao, Almeida and Broder (2000) showed that a counting variant makes the structure usable for sets that change over time, in the Summary Cache protocol for sharing web caches.

The project goals were to implement both filters with configurable hash functions, quantify false positive rates across parameter choices, and test the structures on a simulated web cache or URL filter workload. Section 2 summarizes the theory, Section 3 describes the implementation, Section 4 the experimental method, Section 5 the results, and Sections 6 to 8 discuss findings, limitations and reproduction.

## 2. Background and theory

### 2.1 The standard Bloom filter

A Bloom filter is a bit array of m bits, all initially zero, together with k hash functions that each map an item to a position in the array. To insert an item, set the k positions to one. To query an item, check the k positions: if any is zero the item is definitely absent, and if all are one the item is reported present. Because bits are only ever set, an inserted item always finds its own bits set, so there are no false negatives. A false positive occurs when an absent item happens to have all k of its positions set by other items.

### 2.2 False positive probability

After n insertions, the probability that a given bit is still zero is approximately e raised to the power of (kn/m) with a negative sign on the exponent, that is, e to the minus kn/m. The false positive probability is therefore approximately

```
p = (1 minus e^(minus kn/m))^k
```

where m is the number of bits, n the number of inserted items and k the number of hash functions. This is the formula given in the project brief. The exact expression replaces the exponential with (1 minus 1/m)^(kn); for the sizes used here the two agree to at least five decimal places.

For fixed m and n, p is minimized at k = (m/n) ln 2, about 0.693 m/n, where each bit is set with probability one half. At that point p is about 0.6185^(m/n), so each extra bit per item multiplies the error rate by roughly 0.62. Equivalently, a target error p needs m = n ln(1/p) / (ln 2)^2 bits, about 1.44 n log2(1/p). The information theoretic lower bound for any structure with the same error guarantee is n log2(1/p) bits, so an optimally tuned Bloom filter is within a factor of 1.44 of optimal (Broder and Mitzenmacher, 2004).

### 2.3 Choosing and implementing the hash functions

Using k fully independent hash functions is expensive. Double hashing derives all k indexes from two base hash values, as in index\_i = (h1 + i times h2) mod m, and Broder and Mitzenmacher discuss why this preserves the asymptotic false positive rate. I implemented both this scheme and a slower baseline that computes k independently salted hashes, and compared them experimentally.

### 2.4 Counting Bloom filters

A standard filter cannot delete, because clearing a bit may remove evidence for other items that share it. Fan et al. (2000) replace each bit with a small counter. Insertion increments the k counters and deletion decrements them. An item is present if all k counters are nonzero. They observe that 4 bit counters are sufficient in practice, because with optimally chosen k the probability that any counter reaches 16 is negligible. The bound they use is that the probability that the maximum counter value reaches j is at most m times (e ln 2 / j)^j, where e is Euler's number. Counter overflow is still possible in principle, so a correct implementation must define what happens at the maximum value. The 2019 analysis of counting filters in the Electronics journal studies false positive behavior when counters are used for thresholds, and motivates measuring counter behavior directly rather than assuming it.

## 3. Implementation

The code is in the files bloom.py (library), test\_bloom.py (12 unit tests, all passing) and experiments.py (all experiments and figures). The only dependency beyond the Python standard library is matplotlib for plotting.

### 3.1 Library design

| Component | Purpose |
| --- | --- |
| optimal\_m, optimal\_k, theoretical\_fpr, exact\_fpr | Parameter selection and the two false positive formulas |
| HashFamily | Produces k indexes per item. Schemes: double (one BLAKE2b digest split into two 64 bit values, with h2 forced odd) and salted (k separate BLAKE2b hashes with different salts) |
| BloomFilter(m, k, scheme) | Bit array in a bytearray. Methods: add, membership test, fill\_ratio, estimated\_fpr, expected\_fpr, memory\_bytes, clear. A for\_capacity(n, p) constructor picks m and k |
| CountingBloomFilter(m, k, counter\_bits, scheme, sticky) | Counters of 1, 2, 4 or 8 bits packed into bytes, so a 4 bit filter really uses m/2 bytes. Methods: add, membership test, remove, max\_counter, counter\_histogram, overflow and refusal statistics, to\_bloom |

The number of hash functions k, the filter size m, the hash scheme and the counter width are all configurable, which satisfies the project requirement for configurable hash functions.

### 3.2 Counting filter semantics

Two design decisions matter for correctness.

First, increments saturate at the maximum counter value, and the number of lost increments is recorded as an overflow event. If a saturated counter were later decremented, it would undercount and could reach zero while other items still depend on it, creating false negatives. With the sticky option enabled (the default), a counter that has reached its maximum is never decremented. The cost is that such a counter can never return to zero, which can only add false positives, never remove true members.

Second, remove accepts a guard flag. With the guard on, removal is refused if any of the item's counters is already zero, since the item cannot have been inserted. This prevents counters from underflowing. The guard cannot detect an absent item whose counters are all nonzero because of other items, which is exactly a false positive. Section 5.4 quantifies the consequence.

## 4. Experimental method

All workloads use synthetic URLs of the form http://siteN.example.org/tag/path.html with random hexadecimal path segments (average length about 50 characters), generated from fixed random seeds so results are reproducible. Member and probe sets are generated separately with different tags and verified unique, so every probe is a true negative. Parameter choices follow the five experiments in the project brief.

| Experiment | Setup |
| --- | --- |
| 1. False positive rate | n = 20,000 members, 200,000 absent probes, 4 to 16 bits per item, optimal k, both hash schemes |
| 2. Hash count trade off | n = 50,000, m = 500,000 (10 bits per item), k from 1 to 20, 100,000 absent probes, per query time measured separately for absent and present items |
| 3. Memory | n = 100,000 URLs. Python set of URLs against Bloom and counting filters at target error rates from 10 percent down to 0.01 percent, measured with tracemalloc |
| 4. Counting deletions | Deletion correctness, counter distribution, overflow with 2, 4 and 8 bit counters, repeated deletion, and deletion of items never inserted |
| 5. Web cache | Universe of 100,000 URLs, 300,000 requests with Zipf popularity (exponent 0.8), LRU cache of 10,000 entries, filter of 10 bits per entry. Four strategies compared |

With 200,000 probes and an error rate near 1 percent, the binomial standard error of an observed rate is about 0.02 percentage points, so differences smaller than that are noise.

## 5. Results

### 5.1 False positive rate against theory

Table 1 compares observed and predicted rates when k is set to its optimum for each size. Both hash schemes agree with the formula. Observed fill ratios were between 0.49 and 0.53, matching the expectation that an optimally tuned filter is about half full.

| Bits per item | k | Observed (double hashing) | Observed (salted) | Theory |
| --- | --- | --- | --- | --- |
| 4 | 3 | 0.1463 | 0.1469 | 0.1469 |
| 6 | 4 | 0.0552 | 0.0555 | 0.0561 |
| 8 | 6 | 0.0214 | 0.0217 | 0.0216 |
| 10 | 7 | 0.0084 | 0.0081 | 0.0082 |
| 12 | 8 | 0.0031 | 0.0033 | 0.0031 |
| 16 | 11 | 0.0004 | 0.0005 | 0.0005 |

Table 1. False positive rate with n = 20,000 and 200,000 absent probes. Figure: fig\_fpr.png.

The largest absolute gap is 0.0006 and all gaps are within about two standard errors. Double hashing, which computes one digest instead of k, is statistically indistinguishable from independent hashing at these sizes.

### 5.2 The hash function trade off

Holding m and n fixed and varying k, the false positive rate follows a U shape (Figure: fig\_k\_sweep.png). With too few hash functions each item sets few bits and the filter discriminates poorly. With too many, the array fills and almost every position is set.

| k | Observed | Theory | Absent query time (microseconds), double | Absent query time, salted |
| --- | --- | --- | --- | --- |
| 1 | 0.0960 | 0.0952 | 1.7 | 1.5 |
| 3 | 0.0180 | 0.0174 | 2.2 | 3.4 |
| 5 | 0.00945 | 0.00943 | 2.4 | 5.1 |
| 7 | 0.0086 | 0.0082 | 2.6 | 6.6 |
| 10 | 0.0104 | 0.0102 | 3.3 | 9.7 |
| 14 | 0.0189 | 0.0190 | 4.3 | 13.6 |
| 20 | 0.0551 | 0.0546 | 5.2 | 19.3 |

Table 2. Effect of k at 10 bits per item (n = 50,000). The predicted optimum is k = 6.93, and the observed minimum is at k = 7. Timings come from a pure Python implementation and are noisy, so only the trends should be read from them.

Three conclusions follow. The minimum sits at the predicted k, and the curve is flat near the optimum: k = 5 to 9 are all within roughly 15 percent of the best rate, so practitioners may choose a slightly smaller k to save time. Query cost grows with k, and the growth is steep with salted hashes (about 19 microseconds at k = 20) because each hash is a separate digest, while double hashing grows slowly (about 5 microseconds) because only the cheap index arithmetic scales with k. Finally, absent queries are slightly cheaper than present ones because they can stop at the first zero bit.

### 5.3 Memory comparison

For 100,000 URLs averaging 50.2 characters, a Python set costs 4,194,520 bytes for the hash table plus 9,124,544 bytes for the string objects, 13,319,064 bytes in total. Even a set that stores only 64 bit hash values uses 7,769,720 bytes in CPython because of per object overhead, against a theoretical 800,000 bytes if packed.

| Target error | Bits per item | k | Ideal lower bound (bytes) | Theoretical Bloom (bytes) | Measured Bloom (bytes) | Counting, 4 bit (bytes) | Observed error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 percent | 4.79 | 3 | 41,525 | 59,907 | 60,404 | 239,627 | 0.1026 |
| 1 percent | 9.59 | 7 | 83,049 | 119,814 | 120,215 | 479,253 | 0.0103 |
| 0.1 percent | 14.38 | 10 | 124,573 | 179,720 | 180,121 | 718,880 | 0.0011 |
| 0.01 percent | 19.17 | 13 | 166,097 | 239,627 | 240,028 | 958,506 | 0.0001 |

Table 3. Memory for 100,000 URLs. Figure: fig\_memory.png.

At a 1 percent error rate the Bloom filter uses 120,215 bytes, about 111 times less than the Python set of URLs, and even the 4 bit counting filter (479,253 bytes) is about 28 times smaller. Measured memory exceeds the theoretical value by only about 400 bytes of allocator overhead. The ratio of the theoretical Bloom size to the information theoretic bound is 1.44 at every error rate, as predicted. The counting filter costs four times a standard filter at 4 bit counters, which is the price of supporting deletion.

A caveat on fairness: the set provides exact answers and can enumerate its members, which a Bloom filter cannot. The comparison is about membership testing only.

### 5.4 Counting filter deletions

**Basic correctness.** I inserted 20,000 URLs into a counting filter sized for a 1 percent error rate (m = 191,702, k = 7), then deleted 10,000 of them. All 10,000 removals were accepted, and none of the 10,000 remaining items became a false negative. Of the deleted items, only 1 (0.01 percent) was still reported present, and the false positive rate on 100,000 absent items was 0.026 percent, identical to a fresh filter built from only the kept items. A standard filter holding all 20,000 items continued to report all 10,000 deleted items as present, since it cannot remove them.

**Counter distribution.** After the 20,000 insertions the counters had the following distribution, which matches a Poisson distribution with mean kn/m = 0.73.

| Counter value | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 and above |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Number of counters | 92,449 | 67,354 | 24,537 | 6,087 | 1,098 | 148 | 24 | 5 | 0 |

The maximum observed was 7, so a 4 bit counter has headroom of more than twice the largest value seen. The analytical bound m (e ln 2 / j)^j gives about 1.4 in one billion for j = 16 at m = one million, so overflow of 4 bit counters is essentially impossible under normal loading. The same bound is vacuous for small j (it exceeds 1 at j = 8 for m = 100,000), so it is loose in that range and the empirical distribution is the more informative evidence.

**Counter overflow.** To force overflow, I used a deliberately overloaded filter (m = 4,000, k = 3, 2,000 items), inserted all items, deleted 1,000 and checked whether the other 1,000 were still present.

| Counter width | Policy | Overflow events | False negatives among 1,000 kept items |
| --- | --- | --- | --- |
| 2 bit | Sticky | 360 | 0 |
| 2 bit | Not sticky | 360 | 113 |
| 4 bit | Sticky | 0 | 0 |
| 4 bit | Not sticky | 0 | 0 |
| 8 bit | Sticky | 0 | 0 |
| 8 bit | Not sticky | 0 | 0 |

Table 4. Overflow behavior. Without the sticky rule, 2 bit counters lose increments at saturation and later decrement to zero too early, causing 11.3 percent of the retained items to disappear. The sticky rule eliminates all false negatives.

In the realistic setting (n = 20,000 at 9.6 bits per item), 2 bit counters overflowed 1,522 times and 4 bit and 8 bit counters never did, so 4 bits is the smallest width that is safe in practice, in agreement with Fan et al. (2000).

**Repeated deletion.** I built a filter of 10,000 items (m = 60,000, k = 4), removed 200 items once, then tried to remove the same 200 items five more times (1,000 attempts).

| Guard | Repeat removals accepted | False negatives among the 9,800 kept items |
| --- | --- | --- |
| On | 14 of 1,000 | 43 |
| Off | 1,000 of 1,000 | 499 |

Table 5. Repeated deletion. The guard refuses nearly all repeats because the removed item's counters have usually dropped to zero. It is not perfect: the 14 repeats it accepted were removed items that still looked present through counters shared with other items, and each such removal damaged other items.

**Deleting items that were never inserted.** I attempted to remove 5,000 strangers from a filter of 10,000 items.

| Bits per item | Strangers accepted by the guard | False negatives caused |
| --- | --- | --- |
| 6 | 250 (5.0 percent) | 681 |
| 10 | 46 (0.92 percent) | 215 |
| 16 | 3 (0.06 percent) | 19 |

Table 6. The acceptance rate equals the filter's own false positive rate, as expected. This is an inherent limit of counting filters: they must only be told to delete items that are known to be present, for example from the cache directory that the filter summarizes.

### 5.5 Simulated web cache

**Model.** A proxy holds an LRU cache of 10,000 documents on disk. Each request is for one of 100,000 URLs with Zipf popularity (exponent 0.8), 300,000 requests in total. Checking whether a document is on disk costs a disk lookup. A filter summarizing the cache contents is consulted first. If it says "absent" the proxy skips the lookup and goes straight to the origin server, which is always correct. If it says "present" the proxy performs the lookup, and if the document is not there (a false positive) the lookup was wasted. On a miss the document is fetched, inserted into the cache and the filter, and the least recently used document is evicted. The strategies are no filter, a standard Bloom filter that never deletes, a standard filter rebuilt from the cache every 20,000 requests, and a counting filter that removes evicted items. All filters use 10 bits per cache entry (k = 7).

| Strategy | Disk lookups | Lookups avoided | Wasted lookups | Observed false positive rate on uncached URLs | Filter size (bytes) | False negatives |
| --- | --- | --- | --- | --- | --- | --- |
| No filter | 300,000 | 0 | 160,666 | not applicable | 0 | not applicable |
| Standard, no deletion | 263,537 | 36,463 (12.2 percent) | 124,203 | 0.773 | 12,500 | 0 |
| Standard, periodic rebuild | 164,481 | 135,519 (45.2 percent) | 25,147 | 0.157 | 12,500 | 0 |
| Counting, 4 bit | 140,627 | 159,373 (53.1 percent) | 1,293 | 0.0080 | 50,000 | 0 |

Table 7. Cache simulation, 300,000 requests. In every strategy the cache hit ratio is identical (0.4644), so the filter changes cost but never changes correctness. Figure: fig\_web\_cache.png.

Without a filter, all 160,666 requests for uncached documents cause an unnecessary disk lookup. The counting filter prevents 159,373 of them (99.2 percent) and wastes only 1,293, a false positive rate of 0.80 percent that matches the theoretical 0.82 percent for n = 10,000, m = 100,000, k = 7. The standard filter without deletion degrades as evictions accumulate: since evicted URLs remain in the filter, its fill ratio reaches 99.3 percent and it falsely accepts 77 percent of uncached requests, avoiding only 22.7 percent of unnecessary lookups. Rebuilding periodically recovers most of the benefit (84.4 percent avoided), but the filter is stale between rebuilds, its average false positive rate is 15.7 percent, and each rebuild costs a full pass over the cache directory. The counting filter pays four times the memory (50,000 against 12,500 bytes) to stay accurate continuously.

**Effect of filter size.** Sweeping the number of bits per cache entry for the counting filter:

| Bits per entry | Disk lookups | Wasted lookups | Observed false positive rate | Counter memory (bytes) |
| --- | --- | --- | --- | --- |
| 4 | 161,916 | 22,582 | 0.1406 | 20,000 |
| 6 | 147,817 | 8,483 | 0.0528 | 30,000 |
| 8 | 142,607 | 3,273 | 0.0204 | 40,000 |
| 10 | 140,627 | 1,293 | 0.0080 | 50,000 |
| 12 | 139,782 | 448 | 0.0028 | 60,000 |
| 16 | 139,411 | 77 | 0.0005 | 80,000 |

Table 8. Filter size against wasted work. The floor is 139,334 lookups, which are the genuine cache hits. Beyond about 10 to 12 bits per entry the remaining waste is below 1 percent of lookups, so additional memory buys little.

**Effect of cache size.** With the counting filter at 10 bits per entry, the share of requests that skip the disk falls as the cache grows, because more requests are real hits that need the lookup: 72.7 percent at 2,000 entries, 62.6 percent at 5,000, 53.1 percent at 10,000, 42.0 percent at 20,000 and 30.0 percent at 40,000 entries. The false positive rate stays between 0.55 and 0.82 percent throughout. The filter is therefore most valuable for caches with a low hit ratio, which is the common case for the wide area proxy caches studied by Fan et al. (2000).

**Counter width in the cache.** 4 bit and 8 bit counters gave identical results, while 2 bit counters raised wasted lookups from 1,293 to 10,867 because saturated counters stuck at nonzero values and slowly polluted the filter. There were no false negatives in any run, confirming the sticky rule.

## 6. Discussion

**Theory holds in practice.** Every measured false positive rate matched the closed form to within sampling error, for both hash schemes, across sizes and k values. This supports using p = (1 minus e^(minus kn/m))^k for sizing real systems, and supports double hashing as a cheaper substitute for independent hashing.

**Parameter guidance.** Choose m from the target error rate using about 1.44 log2(1/p) bits per item, then set k near 0.69 m/n. Because the optimum is flat, a k slightly below the optimum trades a small increase in error for a measurable gain in query time, which matters when the filter is queried on every request.

**When to pay for counters.** A standard filter is the right choice for a static or rarely changing set, or one that can be rebuilt cheaply. When items leave the set continuously, as with cache evictions, the counting filter keeps accuracy constant at a fixed four times memory cost, and the cache experiment shows the alternative (accumulating stale bits) is much worse.

**Operational hazards.** The experiments show two real hazards. Counters that saturate must be sticky, otherwise deletions silently create false negatives, which breaks the one guarantee Bloom filters offer. And deletion must be restricted to items known to be members, because the filter cannot tell a false positive from a member and a mistaken removal corrupts other items.

## 7. Limitations

1. Workloads are synthetic. Real URL sets have shared prefixes and skewed popularity that I model only through Zipf requests, though a good hash makes the false positive rate insensitive to key structure.
2. Timings are from pure Python on a shared machine and show trends but not absolute performance. A C implementation would be faster and its cost per k different.
3. Each configuration was run once with a fixed seed, with no repetitions across seeds. The sample sizes keep standard errors small, but I did not compute confidence intervals.
4. The cache simulation counts disk lookups as the cost and does not model latency, concurrency or the multi proxy network exchange of Summary Cache. Extending it to several proxies that exchange filter updates would be the natural next step.
5. The bound on counter overflow is loose for small counter values, so the empirical distribution should be preferred.

## 8. Reproducing the results

Files: bloom.py, test\_bloom.py, experiments.py, results.json (all raw numbers) and figures fig\_fpr.png, fig\_k\_sweep.png, fig\_memory.png, fig\_web\_cache.png. Run python3 test\_bloom.py to execute the 12 unit tests, and python3 experiments.py to regenerate every result and figure (about three minutes). All random seeds are fixed in experiments.py.

## 9. Conclusion

The implementation confirms the classical theory: Bloom filters give predictable error rates at roughly 10 bits per item, they are over 100 times smaller than a hash set of URLs at 1 percent error, and a counting variant with 4 bit sticky counters supports safe deletion at four times the memory. In a simulated web cache, the counting filter eliminated 99.2 percent of the unnecessary disk lookups, with the filter size, counter width and hash count each having measurable and explainable effects.

## References

1. Bloom, B. H. (1970). Space/Time Trade offs in Hash Coding with Allowable Errors. Communications of the ACM, 13(7), pp. 422 to 426. https://doi.org/10.1145/362686.362692
2. Broder, A., and Mitzenmacher, M. (2004). Network Applications of Bloom Filters: A Survey. Internet Mathematics, 1(4), pp. 485 to 509. https://doi.org/10.1080/15427951.2004.10129096
3. Fan, L., Cao, P., Almeida, J., and Broder, A. (2000). Summary Cache: A Scalable Wide Area Web Cache Sharing Protocol. IEEE/ACM Transactions on Networking, 8(3), pp. 281 to 293. https://doi.org/10.1109/90.851975
4. Analysis of Counting Bloom Filters Used for Count Thresholding. (2019). Electronics, 8(7), 779. https://www.mdpi.com/2079-9292/8/7/779
