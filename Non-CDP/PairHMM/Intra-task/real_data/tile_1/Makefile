all:
	nvcc  -arch=sm_35 --use_fast_math --maxrregcount=32  -lineinfo  --ptxas-options=-v tile_1.cu  -O3  -o streams
