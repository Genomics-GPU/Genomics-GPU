#include <iostream>
#include "timer.h"

Timer timer;
__global__ void sayHello(int *data) {
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  data[index] += 1;
}

int main() {
  int *data;
  cudaMallocManaged(&data, sizeof(int)*10);
  for (int i=0; i<10; i++) {
    data[i] = i;
  }
  sayHello<<<2, 5>>>(data);
  cudaDeviceSynchronize();
  timer.start();
  sayHello<<<2, 5>>>(data);
  cudaDeviceSynchronize();
  timer.getDuration();
  int sum = 0;
  for (int i=0; i<10; i++) {
    sum += data[i];
  }
  std::cout << sum << " done\n";
}