icpc  -O3 main.cc center-star.cc util.cc nw.cc global.cc -qopenmp -o msa.out
#icc main.cc center-star.cc util.cc nw.cc global.cc -fopenmp -o msa.out
icpc  fastautil.cc util.cc global.cc -o fastautil.out
#nvcc sort.cc util.cu global.cc -o sort.out
