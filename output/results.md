# Bloom filter experiment results

## False positive rate against array size

Items inserted: 5000. Hash count is the rounded optimum for every size.

```
bits_per_item  size   hash_count  theory   measured
=============  =====  ==========  =======  ========
            4  20000           3  0.14689   0.14811
            6  30000           4  0.05606   0.05618
            8  40000           6  0.02158   0.02104
           10  50000           7  0.00819   0.00799
           12  60000           8  0.00314   0.00325
           16  80000          11  0.00046   0.00038
```

## Optimal number of hash functions

Array holds 8 bits per item. The formula is k = ln(2) * m / n.

```
hash_count  theory   measured
==========  =======  ========
         1  0.11750   0.11878
         2  0.04893   0.04960
         3  0.03058   0.03059
         4  0.02397   0.02375
         5  0.02168   0.02138
         6  0.02158   0.02104
         7  0.02293   0.02273
         8  0.02549   0.02494
         9  0.02922   0.02932
        10  0.03419   0.03457
        11  0.04051   0.04048
        12  0.04833   0.04784
        13  0.05780   0.05708
        14  0.06909   0.06925
        15  0.08232   0.08142
        16  0.09763   0.09765

bits_per_item      8.000
formula_k          5.545
formula_k_rounded  6
best_theory_k      6
best_measured_k    6
best_measured_fpr  0.02104
```

## False positive rate against inserted items

Array size m = 100000, k = 7 fixed while the item count grows.

```
item_count  load     theory     measured
==========  =======  =========  ========
      2500  0.02500  0.0000027     0.000
      5000  0.05000    0.00020   0.00018
     10000  0.10000    0.00819   0.00868
     15000  0.15000    0.04906   0.04956
     20000  0.20000    0.13778   0.13824
     30000  0.30000    0.40076   0.40109
     40000  0.40000    0.64458   0.64411
```

## Multiple filters with the same total memory

The total bit budget is split across layers that must all agree.

```
layers  layer_size  hash_count  memory_bits  theory   measured
======  ==========  ==========  ===========  =======  ========
     1       60000           8        60000  0.00314   0.00319
     2       30000           4        60000  0.00314   0.00363
     3       20000           3        60000  0.00317   0.00304
     4       15000           2        60000  0.00314   0.00342
```

## Multiple filters aiming at a one percent error target

The error budget is split so the product of layer rates meets the target.

```
layers  hash_count  memory_bits  target   theory   measured
======  ==========  ===========  =======  =======  ========
     1           7        47926  0.01000  0.01004   0.01005
     2           3        47926  0.01000  0.01014   0.00987
     3           2        47928  0.01000  0.01014   0.01027
     4           2        47928  0.01000  0.01052   0.01037
```

## Counting filter deletion

Half of the inserted items are removed from a counting filter. A standard filter keeps reporting them.

```
inserted                         5000
removed                          2500
removals_succeeded               2500
false_negatives_among_remaining  0
counting_removed_still_present   0.00040
standard_removed_still_present   1.000
counting_fpr_on_fresh_keys       0.00040
theory_fpr_after_removal         0.00025
standard_memory_bits             47926
counting_memory_bits             191704
```

## Counter width overhead

Memory relative to a standard filter of the same length equals the counter width.

```
counter_bits  memory_bits  memory_vs_standard  overflow_bound  avg_overflow_events  avg_max_counter  avg_saturated_counters
============  ===========  ==================  ==============  ===================  ===============  ======================
           2       100000               2.000           1.000                  336                3                1719.400
           3       150000               3.000         0.51214                    0            6.400                 0.40000
           4       200000               4.000       0.0000000                    0            6.400                       0
           8       400000               8.000           0.000                    0            6.400                       0
```

## URL filter workload

Universe 40000 URLs, 8000 stored, 20000 Zipf queries per epoch, 5 epochs, 30 percent churn per epoch.

```
strategy          queries  disk_lookups  wasted_lookups  lookups_avoided  avoided_pct  false_negatives  memory_bits  maintenance_ops
================  =======  ============  ==============  ===============  ===========  ===============  ===========  ===============
       no filter   100000        100000           79641                0        0.000                0            0                0
  standard stale   100000         35253           14894            64747       64.747                0        80000            20000
standard rebuilt   100000         21022             663            78978       78.978                0        80000            48000
        counting   100000         20955             596            79045       79.045                0       320000            32000
```
