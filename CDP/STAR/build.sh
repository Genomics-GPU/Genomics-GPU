nvcc main.cu cuda-nw.cu center-star.cc util.cu nw.cc global.cc load-matrix.cc -std=c++11 -lcuda -Wno-deprecated-gpu-targets -Xcompiler -fopenmp -rdc=true -lcudadevrt -o cmsa2
#nvcc fastautil.cc util.cu global.cc -o fastautil.out
#nvcc sort.cc util.cu global.cc -o sort.out
