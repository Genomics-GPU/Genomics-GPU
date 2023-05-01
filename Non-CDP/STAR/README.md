# CMSA2
Improved Center Star Algorithm for Multiple Sequences Alignment (DNA/RNA/Protein) Based on CUDA

Original project: https://github.com/wangvsa/CMSA  
Spark/Hadoop project: https://github.com/ShixiangWan/HAlign2.0  
MPI project:https://github.com/ShixiangWan/MPI-MSA  

#### Introduction
CMSA is a robust and efficient MSA system for large-scale datasets on the heterogeneous CPU/GPU and MIC platform. It performs and optimizes multiple sequence alignment automatically for users’ submitted sequences without any assumptions. CMSA adopts the co-run computation model so that both CPU and GPU devices are fully utilized. Moreover, CMSA proposes an improved center star strategy that reduces the time complexity of its center sequence selection process from O(mn^2) to O(mn).


#### Compilation

```
bash build.sh
```

Note: CUDA and C++11 environment need to be supported.


#### Usage:

```
./cmsa2 [options] input_path output_path

Options:
	-d	: DNA/RNA alignment (default)
	-p	: PROTEIN alignment
	-m	: specify the score matrix of PROTEIN alignment (default use BLOSUM62)
	-g	: use GPU only (default use both GPU and CPU)
	-c	: use CPU only (default use both GPU and CPU)
	-w <int>	: specify the workload ratio of CPU / CPU
	-b <int>	: specify the number of blocks per grid
	-t <int>	: specify the number of threads per block
	-n <int>	: specify the number of GPU devices should be used
```

