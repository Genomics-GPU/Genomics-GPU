# gene-sequence-clustering
accurate and fast gene greedy clustering tool

compile:
> make

run:
> ./a.out i testData.fasta g 200 b 20

Two new run flags:
* `g`: Grid size, how many blocks should be launched at a time (where 1 block = 1 iteration). 
* `b`: Buffer size, how big each of the buffer's sub-arrays should be. 

Note: increasing `g` and `b` may improve performance, but it will also increase storage requirements. If you encounter memory errors, attempt different combinations of these.

