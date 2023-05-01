#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include "util.h"
#include "omp.h"
#include "cuda-nw.h"
#include "global.h"
using namespace std;


#define get_tid (threadIdx.x+blockIdx.x*blockDim.x)

/**
  * 打印矩阵
  * m 行, n 列
  */
__device__
void printMatrix(short *matrix, int m, int n) {
    for(int i=0;i<m;i++) {
        for(int j=0;j<n;j++)
            printf("%d ", matrix[i*m+j]);
        printf("\n");
    }
}

__device__
short max(short v1, short v2) {
    return v1 > v2 ? v1 : v2;
}
__device__
short max(short v1, short v2, short v3) {
    return max(max(v1, v2), v3);
}

__device__
int cuda_strlen(char *str) {
    int count = 0;
    while(str[count]!='\0')
        count++;
    return count;
}




/**
  * m               in, 中心串长度
  * n               in, 对比串长度
  * centerSeq       in, 中心串
  * seqs            in, 其他n-1条串
  * seqIdx          in, 要被计算的串的编号
  * matrix          out, 需要计算的DP矩阵
  * 此函数没有被使用
  */
__device__
void cuda_nw(int m, int n, char *centerSeq, char *seq, short*matrix, int width) {
    // 初始化矩阵, DP矩阵m+1行,n+1列
    for(int i=0;i<=m;i++)
        matrix[i*width+0] = i * MISMATCH;   // matrix[i][0]
    for(int j=0;j<=n;j++)
        matrix[0*width+j] = j * MISMATCH;   // matrix[0][j];

    for(int i=1;i<=m;i++) {
        for(int j=1;j<=n;j++) {
            short up = matrix[(i-1)*width+j] + GAP;           // matrix[i-1][j]
            short left = matrix[i*width+j-1] + GAP;           // matrix[i][j-1]
            short diag = matrix[(i-1)*width+j-1] + ((centerSeq[i-1]==seq[j-1])?MATCH:MISMATCH);      // matrix[i-1][j-1]
            matrix[i*width+j] = max(up, left, diag);
        }
    }
}

#define COL_STEP 12
__device__
void cuda_nw_3d(int m, int n, char *centerSeq, char *seq, cudaPitchedPtr matrix3DPtr) {
    size_t slicePitch = matrix3DPtr.pitch * (m+1);
    char *slice = (char *)matrix3DPtr.ptr + get_tid * slicePitch;

    // 初始化矩阵, DP矩阵m+1行,n+1列
    DPCell *matrixRow;
    for(int i=0;i<=m;i++) {
        matrixRow = (DPCell *)(slice + i * matrix3DPtr.pitch);
        matrixRow[0].score = MIN_SCORE;   // matrix[i][0]
        matrixRow[0].x_gap = MIN_SCORE;
        matrixRow[0].y_gap = GAP_START + i * GAP_EXTEND;
    }
    matrixRow = (DPCell *)(slice + 0 * matrix3DPtr.pitch);
    for(int j=0;j<=n;j++) {
        matrixRow[j].score = MIN_SCORE;   // matrix[0][j];
        matrixRow[j].x_gap = GAP_START + j * GAP_EXTEND;
        matrixRow[j].y_gap = MIN_SCORE;
    }
    matrixRow[0].score = 0;             // matrix[0][0]


    /**
      * 参照这篇论文：
      * [IPDPS-2009]An Efficient Implementation Of Smith Waterman Algorithm On Gpu Using Cuda, For Massively Parallel Scanning Of Sequence Databases
      * 横向计算，每次计算COL_STEP列，理论上讲COL_STEP越大越好，取决与register per block的限制
      * 这样左侧依赖数据，以及一列（COL_STEP个cell）内的上侧依赖数据就可以存储在register中
      * 有效减少global memory访问次数。
      * TODO: 1. 对角线的global memory访问也可以节省掉
      *       2. 如果中心串的长度不能被COL_STEP整除怎么处理
      */
    short upScore, upYGap, diagScore;
    for(int i=1;i<=m;i+=COL_STEP) {
        // 直接这样生命没有把所有元素初始化为MIN_SCORE
        //short leftScore[COL_STEP] = {MIN_SCORE}, leftXGap[COL_STEP] = {MIN_SCORE};
        short leftScore[COL_STEP], leftXGap[COL_STEP];
        for(int tmp=0;tmp<COL_STEP;tmp++) {
            leftScore[tmp] = MIN_SCORE;
            leftXGap[tmp] = MIN_SCORE;
        }

        for(int j=1;j<=n;j++) {
            for(int k=0;k<COL_STEP;k++) {
                if(i+k>m) break;
                DPCell *matrixRow = (DPCell *)(slice + (i+k) * matrix3DPtr.pitch);
                DPCell *matrixLastRow = (DPCell *)(slice + (i-1+k) * matrix3DPtr.pitch);

                if(k==0) {
                    upScore = matrixLastRow[j].score;
                    upYGap = matrixLastRow[j].y_gap;
                    diagScore = matrixLastRow[j-1].score;
                }
                DPCell cell;            // 当前计算的cell

                cell.x_gap = max(GAP_SE+leftScore[k], GAP_EXTEND+leftXGap[k]);
                cell.y_gap = max(GAP_SE+upScore, GAP_EXTEND+upYGap);
                cell.score = diagScore + ((centerSeq[i+k-1]==seq[j-1])?MATCH:MISMATCH);               // matrix[i-1][j-1]
                cell.score = max(cell.x_gap, cell.y_gap, cell.score);

                // 更新当前列下一行cell计算所需要的数据
                upScore = cell.score;
                upYGap = cell.y_gap;
                diagScore = leftScore[k];
                // 更新当前行下一列cell计算所需要的数据
                leftScore[k] = cell.score;
                leftXGap[k] = cell.x_gap;

                matrixRow[j] = cell;    // 写入当前cell到Global Memory
            }
        }
    }
}

/**
  * m               in, 中心串长度, m 行
  * n               in, 对比串长度, n 列
  * seqIdx          in, 要被计算的串的编号
  * matrix          in, 本次匹配得到的DP矩阵
  * space           out, 需要计算的本次匹配给中心串引入的空格
  * spaceForOther   out, 需要计算的本次匹配给当前串引入的空格
  * 此函数没有被使用
  */
__device__
void cuda_backtrack(int m, int n, short* matrix, short *spaceRow, short *spaceForOtherRow, int width) {
    // 从(m, n) 遍历到 (0, 0)
    // DP矩阵的纬度是m+1, n+1
    int i = m, j = n;
    while(i!=0 || j!=0) {
        int score = matrix[i*width+j];                              // matrix[i][j]
        //printf("%d,%d:  %d\n", i, j, score);
        if(i > 0 && matrix[(i-1)*width+j] + GAP == score) {         // matrix[i-1][j]
            spaceForOtherRow[j]++;                                  // spaceForOther[seqIdx][j]
            i--;
        } else if(j > 0 && matrix[i*width+j-1] + GAP == score) {    // matrix[i][j-1]
            spaceRow[i]++;                                          // space[seqIdx][i]
            j--;
        } else {
            i--;
            j--;
        }
    }
}
__device__
void cuda_backtrack_3d(int m, int n, char *centerSeq, char *seq, cudaPitchedPtr matrix3DPtr, short *spaceRow, short *spaceForOtherRow) {
    size_t slicePitch = matrix3DPtr.pitch * (m+1);
    char *slice = (char *)matrix3DPtr.ptr + get_tid * slicePitch;

    int i = m, j = n;
    while(i!=0 || j!=0) {
        DPCell *matrixRow = (DPCell *)(slice + i * matrix3DPtr.pitch);
        DPCell *matrixLastRow = (DPCell *)(slice + (i-1) * matrix3DPtr.pitch);
        int score = (centerSeq[i-1] == seq[j-1]) ? MATCH : MISMATCH;
        if(i>0 && j>0 && score+matrixLastRow[j-1].score == matrixRow[j].score) {
            i--;
            j--;
        } else {
            int k = 1;
            while(true) {
                DPCell *matrixLastKRow = (DPCell *)(slice + (i-k) * matrix3DPtr.pitch);
                if(i>=k && matrixRow[j].score == matrixLastKRow[j].score+GAP_START+GAP_EXTEND*k) {
                    spaceForOtherRow[j] += k;
                    i = i - k;
                    break;
                } else if(j>=k && matrixRow[j].score == matrixRow[j-k].score+GAP_START+GAP_EXTEND*k) {
                    spaceRow[i] += k;
                    j = j - k;
                    break;
                } else {
                    k++;
                }
            }
        }
    }
}


__global__
void kernel(char *centerSeq, char *seqs, int centerSeqLength, int *seqsSize, cudaPitchedPtr matrix3DPtr, short *space, short *spaceForOther, int maxLength, int workload) {

    int tid = get_tid;
    if(tid >= workload) return;

    // 得到当前线程要计算的串
    int width = maxLength + 1;
    char *seq = seqs + width * tid;

    int m = centerSeqLength;
    int n = seqsSize[tid];

    // 当前匹配的字符串所需要填的空格数组的位置
    short *spaceRow = space + tid * (m+1);
    short *spaceForOtherRow = spaceForOther + tid * width;

    // 计算使用的DP矩阵
    cuda_nw_3d(m, n, centerSeq, seq, matrix3DPtr);
    cuda_backtrack_3d(m, n, centerSeq, seq, matrix3DPtr, spaceRow, spaceForOtherRow);

    //printMatrix(spaceForOtherRow, 1, n+1);
}




// totalWorkload:int    当前GPU设备需要计算的所有串的个数
void cuda_msa(int offset, GPUData data, string centerSeq, int maxLength, short *space, short *spaceForOther) {

    int sWidth = centerSeq.size() + 1;      // d_space的宽度
    int soWidth = maxLength + 1;            // d_spaceForOther的宽度

    // 根据参数，每个kernel计算SEQUENCES_PER_KERNEL条串
    // workload 为本次kernel需要计算的量
    int SEQUENCES_PER_KERNEL = BLOCKS * THREADS;
    int workload = data.totalWorkload < SEQUENCES_PER_KERNEL ? data.totalWorkload : SEQUENCES_PER_KERNEL;

    // 给存储空格信息申请空间
    // d_space, d_spaceForOther 是循环利用的
    cudaMalloc((void**)&data.d_space, workload*sWidth*sizeof(short));
    cudaMalloc((void**)&data.d_spaceForOther, workload*soWidth*sizeof(short));

    // 分配一个3D的DP Matrix
    size_t freeMem, totalMem;
    cudaExtent matrixSize = make_cudaExtent(sizeof(DPCell) * soWidth, sWidth, SEQUENCES_PER_KERNEL);
    cudaMalloc3D(&data.matrix3DPtr, matrixSize);
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("freeMem :%luMB, totalMem: %luMB\n", freeMem/1024/1024, totalMem/1024/1024);

    for(int i = 0; i <= data.totalWorkload / SEQUENCES_PER_KERNEL; i++) {
        if(i==data.totalWorkload/SEQUENCES_PER_KERNEL)           // 最后一次kernel计算量
            workload = data.totalWorkload % SEQUENCES_PER_KERNEL;

        // 此次kernel计算的起始串的位置
        int startIdx = i * SEQUENCES_PER_KERNEL + offset;
        printf("%d. startIdx: %d, workload: %d\n", i, startIdx, workload);

        cudaMemsetAsync(data.d_space, 0, workload*sWidth*sizeof(short), data.stream);
        cudaMemsetAsync(data.d_spaceForOther, 0, workload*soWidth*sizeof(short), data.stream);

        // 1. 把这次kernel要计算的串和串的长度信息传送给GPU
        cudaMemcpyAsync(data.d_seqs, data.h_seqs+(maxLength+1)*(i*SEQUENCES_PER_KERNEL), (maxLength+1)*workload*sizeof(char), cudaMemcpyHostToDevice, data.stream);
        cudaMemcpyAsync(data.d_seqsSize, data.h_seqsSize+(i*SEQUENCES_PER_KERNEL), sizeof(int)*workload, cudaMemcpyHostToDevice, data.stream);

        // 2. Kernel计算
        kernel<<<BLOCKS, THREADS, 0, data.stream>>>(data.d_centerSeq, data.d_seqs, centerSeq.size(), data.d_seqsSize, data.matrix3DPtr, data.d_space, data.d_spaceForOther, maxLength, workload);
        cudaError_t err  = cudaGetLastError();
        if ( cudaSuccess != err )
            printf("Error: %d, %s\n", err, cudaGetErrorString(err));

        // 3. 将计算得到的将空格信息传回给CPU
        // TODO：使用Pipeline可以重叠数据传输和kernel计算
        cudaMemcpyAsync(space+startIdx*sWidth, data.d_space, workload*sWidth*sizeof(short), cudaMemcpyDeviceToHost, data.stream);
        cudaMemcpyAsync(spaceForOther+startIdx*soWidth, data.d_spaceForOther, workload*soWidth*sizeof(short), cudaMemcpyDeviceToHost, data.stream);
    }

}


/**
  * 支持多个GPU
  * gpuWorkload:int     需要由GPU执行的工作量，平均分给各个GPU
  * centerSeq:string    中心串
  * seqs:vector<string> 除中心串外的所有串
  * maxLength:int       所有串的最长长度
  */
void multi_gpu_msa(int gpuWorkload, string centerSeq, vector<string> seqs, int maxLength, short *space, short *spaceForOther) {
    if(gpuWorkload <= 0) return;

    if(GPU_NUM==0)
        cudaGetDeviceCount(&GPU_NUM);       // 如果用户没设置GPU数量则由程序自动读取
    if(GPU_NUM==0) {
        printf("No CUDA capable devices available.");
        return;
    }


    int workload = gpuWorkload / GPU_NUM;

    // 为每个GPU设置数据，分配空间
    GPUData gpuData[GPU_NUM];
    int sWidth = centerSeq.size() + 1;      // d_space的宽度
    //int soWidth = maxLength + 1;          // d_spaceForOther的宽度
    for(int i = 0; i < GPU_NUM; i++) {
        cudaSetDevice(i);

        int offset = i * workload;

        // 计算任务量，最后一块GPU还要计算多余的余数
        gpuData[i].totalWorkload =  workload + ((i==GPU_NUM-1) ? (gpuWorkload%GPU_NUM) : 0);

        // 创建stream
        cudaStreamCreate(&(gpuData[i].stream));

        // 1. 中心串
        cudaMalloc((void**)(&(gpuData[i].d_centerSeq)), sWidth * sizeof(char));
        cudaMemcpy(gpuData[i].d_centerSeq, centerSeq.c_str(), sWidth *sizeof(char), cudaMemcpyHostToDevice);

        // 2. 将需要匹配串拼接成一个长串传到GPU
        //    为实现数据传输和计算重叠，需要Pinned Memory
        int w = maxLength + 1;
        cudaMallocHost((void**)(&(gpuData[i].h_seqs)), w*gpuData[i].totalWorkload*sizeof(char));
        for(int k = 0; k < gpuData[i].totalWorkload; k++) {
            char *p = &(gpuData[i].h_seqs[k * w]);
            strcpy(p, seqs[k+offset].c_str());
        }
        cudaMalloc((void**)(&(gpuData[i].d_seqs)), w*gpuData[i].totalWorkload*sizeof(char));

        // 3. 将要匹配的串的长度也计算好传给GPU，因为在GPU上计算长度比较慢
        //    为实现数据传输和计算重叠，需要Pinned Memory
        cudaMallocHost((void**)(&(gpuData[i].h_seqsSize)), sizeof(int)*gpuData[i].totalWorkload);
        for(int k = 0; k < gpuData[i].totalWorkload; k++)
            gpuData[i].h_seqsSize[k] = seqs[k+offset].size();
        cudaMalloc((void**)(&(gpuData[i].d_seqsSize)), sizeof(int)*gpuData[i].totalWorkload);
    }

    // 每个GPU并行计算任务
    for(int i = 0; i < GPU_NUM; i++) {
        cudaSetDevice(i);
        cuda_msa(i*workload, gpuData[i], centerSeq, maxLength, space, spaceForOther);
    }


    // 等待所有GPU设备完成后释放资源
    for(int i = 0; i < GPU_NUM; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(gpuData[i].stream);
        cudaFreeHost(gpuData[i].h_seqs);
        cudaFreeHost(gpuData[i].h_seqsSize);
        /*
        cudaFree(gpuData[i].d_centerSeq);
        cudaFree(gpuData[i].d_seqs);
        cudaFree(gpuData[i].d_seqsSize);
        cudaFree(gpuData[i].d_space);
        cudaFree(gpuData[i].d_spaceForOther);
        cudaFree(gpuData[i].matrix3DPtr.ptr);
        */
        cudaStreamDestroy(gpuData[i].stream);
    }
}
