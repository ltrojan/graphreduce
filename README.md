# GraphReduce

GraphReduce is a graph-based text summarizer for easier consumption of product reviews.

It is based on Opinosis by Kavita Ganesan (http://kavita-ganesan.com/opinosis)

The main differences with the Opinosis implementation are:

 1) Before stitching the sentences to an anchor we use k-means to cluster their sentiment

 2) We use the Oxford Language Model API to evaluate whether a sentence is readable or not
 

---

www.matteotomassetti.com
