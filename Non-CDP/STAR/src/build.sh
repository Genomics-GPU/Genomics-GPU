nvcc main.cu cuda-nw.cu center-star.cc util.cu nw.cc global.cc load-matrix.cc -lcuda -Wno-deprecated-gpu-targets -Xcompiler -fopenmp -std=c++11 -o cmsa2
#nvcc fastautil.cc util.cu global.cc -o fastautil.out
#nvcc sort.cc util.cu global.cc -o sort.out
