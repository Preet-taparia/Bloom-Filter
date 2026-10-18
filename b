Count-Min Sketch

bloom filter with int instead of bits, allows to increment an item and queery an upper bound

count items approximately [incr(x), upper_bound(x)]

number of times element is added, take min of all hash arrays value

A Counting Bloom Filter and a Count-Min Sketch (CMS) are both probabilistic data structures used to summarize large datasets with limited memory. The key difference is that a counting Bloom filter tracks approximate membership and supports deletions, while a Count-Min Sketch estimates how frequently items occur.

heavy hitter problem -> priority queue + count min sketch

HyperLogLog

estimates number of unique items

track max number of trailing zeroes

maintain multiple registers, take harmonic mean of all registers