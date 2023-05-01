CC=nvcc
CFLAGS= -g -G -lcuda -lcudart -Xptxas="-v"
OBJS= main.o unixtimer.o

all: alignCuda

alignCuda: $(OBJS) 
	$(CC) $(OBJS) $(CFLAGS) -o $@

%.o : %.cu
	$(CC) -c $(CFLAGS) $<

clean:
	rm -rf  alignCuda

