C = nvcc
NVCCFLAGS = -arch=sm_70 
CFLAGS = -std=c++11 -rdc=true -lcudadevrt

all: align

align: main.cu  
	$(C) $(NVCCFLAGS) $(CFLAGS) -o align main.cu 

clean:
	rm -f align *.dat
