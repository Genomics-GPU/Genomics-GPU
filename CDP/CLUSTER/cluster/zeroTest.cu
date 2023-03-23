#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "src/func.h"


// 检查显卡错误
void checkError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }
}

struct Read {
  unsigned int age;
  double height;
  char name[10];
  int ceshi;
};

void createType (Bench &bench) {
  int lengths[4] = {1, 1, 10, 1};
  Read dummyRead;
  MPI_Aint begin;
  MPI_Aint offsets[4];  // MPI_Aint 类似unsigned long
  MPI_Get_address(&dummyRead, &begin);
  MPI_Get_address(&dummyRead.age, &offsets[0]);
  MPI_Get_address(&dummyRead.height, &offsets[1]);
  MPI_Get_address(&dummyRead.name[0], &offsets[2]);
  MPI_Get_address(&dummyRead.ceshi, &offsets[3]);
  offsets[0] = MPI_Aint_diff(offsets[0], begin);  // 绝对差值
  offsets[1] = MPI_Aint_diff(offsets[1], begin);
  offsets[2] = MPI_Aint_diff(offsets[2], begin);
  offsets[3] = MPI_Aint_diff(offsets[3], begin);

  // MPI_Datatype ReadMpi;
  MPI_Datatype types[4] = {MPI_UNSIGNED, MPI_DOUBLE, MPI_CHAR, MPI_INT};
  MPI_Type_create_struct(4, lengths, offsets, types, &bench.ReadMpi);
  MPI_Type_commit(&bench.ReadMpi);
}

__global__ void test(Read *buffer) {
  for (int i=0; i<10; i++) {
    printf("%d: ",i);
    printf("%c\n", buffer[i].name[0]);
  }
}

int main() {
  MPI_Init(NULL, NULL);
  Bench bench;
  createType(bench);  // 注册MPI类型
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Read *send;  // 发送数据
  Read *buffer;  // 收集数据
  Read *recv;  // 接收数据
  cudaMallocHost(&send, 1*sizeof(Read));
  cudaMallocManaged(&buffer, 2*sizeof(Read));
  cudaMallocHost(&recv, 1*sizeof(Read));
  send[0].age = rank;
  send[0].ceshi = rank;
  // if (rank == 0) send[0].name[3] = 'a';
  // if (rank == 1) send[0].name[3] = 'b';
  MPI_Gather(send, 1, bench.ReadMpi, buffer,
    1, bench.ReadMpi, 0, MPI_COMM_WORLD);
  int order = 4;  // 那个中选
  for (int i=0; i<2; i++) {
    if (buffer[i].age == 1) {  // 找到了目标
      order = i;
      break;
    }
  }
  MPI_Bcast(&order, 1, MPI_INT, 0, MPI_COMM_WORLD);
  recv[0] = send[0];
  MPI_Bcast(&recv[0], 1, bench.ReadMpi, order, MPI_COMM_WORLD);
  std::cout << rank << " " << recv[0].ceshi << std::endl;
  // std::cout << rank << " " << recv[0].name[3] << std::endl;
  checkError();  // 检查错误
  MPI_Finalize();
}
