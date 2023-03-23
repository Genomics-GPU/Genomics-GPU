nGIA is a precise and fast gene sequence clustering tool that supports multi nodes.

## Installation
RabbitTClust version 1.0

### Compile manually 
```bash
nvcc src/generateIndex.cpp -o generateIndex
cluster4 nvcc -I<mpi include path> -L<mpi lib path> -lmpi src/main.cu src/func.cu -o cluster
```

## Usage
```bash
usage: ./generateIndex --input=string [options] ...
options:
  -i, --input    input file (string)
  -?, --help     print this message
 
usage: ./cluster [options] ...
options:
  -s, --similarity    similarity 0.8-0.99 (float [=0.95])
  -?, --help          print this message
```

## Bug Report
All bug reports, comments and suggestions are welcome.
