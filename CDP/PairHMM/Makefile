all:
	nvcc  -arch=sm_60 --use_fast_math -maxrregcount=48 -rdc=true -lcudadevrt --ptxas-options=-v improved_warp_based.cu  -O3  -o streams
