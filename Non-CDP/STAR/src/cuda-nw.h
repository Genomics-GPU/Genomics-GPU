#ifndef _CUDA_NW_H_
#define _CUDA_NW_H_
#include <vector>
#include <string>
#include <cuda.h>

typedef struct DPCell_t {
    short score;
    short x_gap;
    short y_gap;
} DPCell;


typedef struct GPUData_t {
    int totalWorkload;                  // 此设备所需要计算的总量

    char *h_seqs;
    int *h_seqsSize;

    char *d_centerSeq;                  // 中心串

    char *d_seqs;                       // 要处理的串
    int *d_seqsSize;                    // 要出列的串的长度


    short *d_space;                     // 中心串的空格信息
    short *d_spaceForOther;             // 其他串的空格信息
    cudaPitchedPtr matrix3DPtr;         // 3D DP Matrix

    cudaStream_t stream;
} GPUData;

void multi_gpu_msa(int workCount, std::string centerSeq, std::vector<std::string> seqs, int maxLength, short *space, short *spaceForOther);

//void cuda_msa(int workCount, std::string centerSeq, std::vector<std::string> seqs, int maxLength, short *space, short *spaceForOther);

#endif
