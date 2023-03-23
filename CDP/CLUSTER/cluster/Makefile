NVCC = -arch=sm_50 -Wno-deprecated-gpu-targets
MPI = -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi

all:
	@#nvcc $(NVCC) src/generateIndex.cpp -o generateIndex
	@nvcc $(NVCC) $(MPI) src/main.cu src/func.cu -o cluster
	@#nvcc $(NVCC) $(MPI) zeroTest.cu
gen:
	./generateIndex -i current_NCBI_gg16S_unaligned.fasta
run:
	mpirun -n 2 ./cluster
