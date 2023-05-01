all:
	nvcc  -arch=sm_35 --use_fast_math -lineinfo  --maxrregcount=32 --ptxas-options=-v tile_1.cu  -O3  -o streams
