all:
	nvcc  -arch=sm_35 --use_fast_math --maxrregcount=72  --ptxas-options=-v   -lineinfo   tile_4.cu  -O3  -o streams_new

