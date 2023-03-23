all:
	nvcc -arch=sm_70 -dc main.cu func.cu
	nvcc -arch=sm_70 -dlink main.o func.o -o link.o 
	g++ main.o func.o link.o -lcudadevrt -lcudart -L/usr/local/cuda/lib64

clean:
	rm -rf cluster_result