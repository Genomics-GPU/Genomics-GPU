all:
	nvcc  -arch=sm_35 --use_fast_math -maxrregcount=48  --ptxas-options=-v improved_warp_based.cu  -O3  -o streams
