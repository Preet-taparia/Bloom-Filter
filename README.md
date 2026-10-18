Bloom Filters & Counting Bloom Filters — Details & Scope

Project Idea: Explore space-efficient probabilistic set-membership filters and their role in minimizing disk and network queries.

Expectations: Implement standard and counting Bloom filters with configurable hash functions. Quantify false positive rates across parameter choices and test on a simulated web cache or URL filter workload.

Suggested References: Bloom 1970; Broder & Mitzenmacher 2004.

https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p422-bloom.pdf

https://www.cs.cornell.edu/courses/cs619/2004fa/documents/BloomFilterSurvey.pdf


space efficient probablistic data structure, used to answer is "this element" in the "set" with firm no, or probably yes, gives false positive, no false negatives, less memory -> less accuracy, cant remove item it never forgets. need to be space constrained, tolerate false positive, querry for membership in the set

bloom filters are used in network router, databases, web browsers, LSM Trees, caching, password validotr, content relevance, malicous url, username, spell checker 

uses k hash functions, uses buckets

lower false +ve rate

optimal number of hash functions, k = ln(2) * (array size / estimated number of inserted objects)

increase array size

use multiple bloom filters (cross checking )

counting bloom filters uses counters instead of bits so that we can remove element