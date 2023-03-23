#include <iostream>  // std::cout
#include "func.h"
//--------函数--------//
// initData 初始化数据
void initData(int readsCount, Data &data) {
  // 查询设备属性
  cudaDeviceProp deviceProp;  // 设备属性
  cudaGetDeviceProperties(&deviceProp, 0);
  data.alignSize = deviceProp.textureAlignment;  // 数据对齐长度
  unsigned long memSize, dataSize;
  memSize = deviceProp.totalGlobalMem;  // 内存大小
  memSize *= 0.85;  // 避免溢出
  // 分配统一内存
  cudaMallocManaged(&data.flag, sizeof(unsigned int));  // flag
  // 分配主机内存
  cudaMallocHost(&data.read, sizeof(unsigned int)*2048);  // 2048=65536/32
  cudaMallocHost(&data.length, sizeof(unsigned int));  // 一条序列长度
  cudaMallocHost(&data.offset, sizeof(unsigned long));  // 一条序列偏移
  // 分配设备内存
  dataSize = sizeof(unsigned int)*readsCount;  // lengths大小
  cudaMalloc(&data.lengths, dataSize);
  memSize -= dataSize;  // 去除lengths部分
  dataSize = sizeof(unsigned long)*readsCount;  // offsets大小
  cudaMalloc(&data.offsets, dataSize);
  memSize -= dataSize;  // 去除offsets部分
  cudaMalloc(&data.reads, memSize);  // 剩下的都是reads
}
// packing打包数据
void packing(Data &data, std::string &line) {
  data.length[0] = line.size();  // 序列的长度
  unsigned int length = data.length[0];  // 缓存常用变量
  for (int i=0; i<length/32*32; i+=32) {  // 打包长度为32倍数的部分
    unsigned int high = 0;
    unsigned int low = 0;
    for (int j=0; j<32; j++) {
      unsigned int temp = line[i+j];
      high <<= 1;
      high += temp>>1;  // 打包高位
      low <<= 1;
      low += temp&1;  // 打包低位
    }
    data.read[i/32*2+0] = high;
    data.read[i/32*2+1] = low;
  }
  if (length%32 > 0) {  // 如果有剩余则继续打包
    unsigned int high = 0;
    unsigned int low = 0;
    for (int i=length/32*32; i<length; i++) {
      unsigned int temp = line[i];
      high <<= 1;
      high += temp>>1;  // 打包高位
      low <<= 1;
      low += temp&1;  // 打包低位
    }
    high <<= 32-length%32;
    low <<= 32-length%32;
    data.read[length/32*2+0] = high;
    data.read[length/32*2+1] = low;
  }
}
//  动态规划核函数
__global__ void kernel_dynamic(unsigned int *reads, unsigned int *lengths,
unsigned long *offsets, unsigned int *flag, float threshold,
unsigned long represent) {
  // 准备数据
  __shared__ unsigned int text[4096];  // 4096 = 65536/32*2
  unsigned int textLength = lengths[represent];
  unsigned int text32Length = textLength%32>0?(textLength/32+1)*32:textLength;  // 补齐长度
  unsigned long textOffset = offsets[represent];
  for (int i=threadIdx.x; i<4096; i+= blockDim.x) {  // 拷贝query序列
    text[i] = reads[textOffset+i];
  }
  unsigned int line[2048];  // 暂存每行的结果
  memset(line, 0xFFFFFFFF, sizeof(unsigned int)*2048);  // 初始化
  // 计算部分
  unsigned int indext = blockIdx.x*blockDim.x+threadIdx.x;  // 当前线程编号
  unsigned int wholeThread = gridDim.x*blockDim.x;  // 线程总数
  for (unsigned index=indext; index<represent; index+=wholeThread) {  // 内循环
    int queryLength = lengths[index];
    int query32Length = queryLength%32>0?(queryLength/32+1)*32:queryLength;  // 补齐长度
    unsigned int queryOffset = offsets[index];
    unsigned int *query = reads+queryOffset;
    if (flag[0] != 0) return;  // 聚类完成
    int shift = ceil((float)textLength-(float)queryLength*threshold);
    shift = ceil((float)shift/32.0f);  // 相对对角线，左右偏移方块个数
    // 处理长度为32整数倍的部分
    for (int i=0; i<queryLength/32; i++) {  // 遍历列
      unsigned int column[32] = {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};
      unsigned int queryHight = query[i*2+0];  // 高位
      unsigned int queryLow = query[i*2+1];  // 低位
        int jstart = i-shift;
        jstart = max(jstart, 0);
        int jend = i+shift;
        jend = min(jend, textLength/32);
      for (int j=jstart; j<=jend; j++) {  // 遍历行
        unsigned int texth = text[j*2+0];
        unsigned int textl = text[j*2+1];
        unsigned int row = line[j];
        for (int k=0; k<32; k++) {  // 遍历query的32列
          unsigned int queryh = queryHight>>k&1 ? 0xFFFFFFFF : 0x00000000;
          unsigned int queryl = queryLow  >>k&1 ? 0xFFFFFFFF : 0x00000000;
          unsigned int temp1 = textl ^ queryl;
          unsigned int temp2 = texth ^ queryh;
          unsigned int match = (~temp1)&(~temp2);
          unsigned int unmatch = ~match;
          unsigned int temp3 = row & match;
          unsigned int temp4 = row & unmatch;
          unsigned int carry = column[k];
          unsigned int temp5 = row + carry;
          unsigned int carry1 = temp5 < row;
          temp5 += temp3;
          unsigned int carry2 = temp5 < temp3;
          carry = carry1 | carry2;
          row = temp5 | temp4;
          column[k] = carry;
        }
        line[j] = row;
      }
    }
    // 处理剩下的部分
    if (queryLength%32 > 0) {  // 如果query有剩余
      unsigned int column[32] = {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};
      unsigned int queryLow = query[(query32Length/32)*2+0];
      unsigned int queryHight = query[(query32Length/32)*2+1];
    int jstart = queryLength/32-shift;
    jstart = max(jstart, 0);
    int jend = queryLength/32+shift;
    jend = min(jend, textLength/32);
      for (int j=jstart; j<=jend; j++) {  // 遍历行
        unsigned int textl = text[j*2+0];
        unsigned int texth = text[j*2+1];
        unsigned int row = line[j];
        for (int k=0; k<queryLength%32; k++) {  // 遍历query的剩余列
          unsigned int queryh = queryHight>>k&1 ? 0xFFFFFFFF : 0x00000000;
          unsigned int queryl = queryLow  >>k&1 ? 0xFFFFFFFF : 0x00000000;
          unsigned int temp1 = textl ^ queryl;
          unsigned int temp2 = texth ^ queryh;
          unsigned int match = (~temp1)&(~temp2);
          unsigned int unmatch = ~match;
          unsigned int temp3 = row & match;
          unsigned int temp4 = row & unmatch;
          unsigned int carry = column[k];
          unsigned int temp5 = row + carry;
          unsigned int carry1 = temp5 < row;
          temp5 += temp3;
          unsigned int carry2 = temp5 < temp3;
          carry = carry1 | carry2;
          row = temp5 | temp4;
          column[k] = carry;
        }
        line[j] = row;
      }
    }
    // 统计结果，判断是否相似
    unsigned int sum = 0;
    unsigned int result;
    for (int i=0; i<textLength/32; i++) {
      result = line[i];
      for (int j=0; j<32; j++) {
        sum += result>>j&1^1;
      }
    }
    result = line[textLength/32];
    for (int i=0; i<textLength%32; i++) {
      sum += result>>i&1^1;
    }
    unsigned int cutoff = ceil(textLength*threshold);
    if (sum >= cutoff) {  // 相似
      flag[0] = 1;
    }
    // printf("similarity: %d\n", textOffset);
    // if (flag[0] != 0) return;  // 聚类完成
  }
}


// kernel_dynamic2 动态规划
__global__ void kernel_dynamic2(unsigned int *reads, unsigned int *lengths,
unsigned long *offsets, unsigned int *flag, float threshold,
unsigned long represent) {
// __global__ void kernel_dynamic2(float threshold, int *lengths, long *offsets,
// int *gaps, unsigned int *compressed, int *baseCutoff, int *cluster,
// int *jobList, int jobCount, int representative) {
    //----准备数据----//
    __shared__ unsigned int bases[4096];  // 65536/32*2
    unsigned int text = represent;
    unsigned long textStart = offsets[represent];
    unsigned int textLength = lengths[text];
    for (int i=threadIdx.x; i<textLength/32+1; i+=blockDim.x) {  // 拷贝数据
        bases[i*2+0] = reads[textStart+i*2+0];
        bases[i*2+1] = reads[textStart+i*2+1];
    }
    //----开始计算----//
    // 批量计算阶段
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= represent) return;  // 超出范围
    unsigned int line[2048] = {0xFFFFFFFF};  // 保存结果
    for (int i=0; i<2048; i++) {
        line[i] = 0xFFFFFFFF;
    }
    long queryStart = offsets[index];
    int queryLength = lengths[index];
    int shift = ceil((float)textLength-(float)queryLength*threshold);
    shift = ceil((float)shift/32.0f);  // 相对对角线，左右偏移方块个数
    for (int i=0; i<queryLength/32; i++) {  // 遍历query
        unsigned int column[32] = {0};
        unsigned int queryLow = reads[queryStart+i*2+0];
        unsigned int queryHight = reads[queryStart+i*2+1];
        int jstart = i-shift;
        jstart = max(jstart, 0);
        int jend = i+shift;
        jend = min(jend, textLength/32);
        for (int j=jstart; j<=jend; j++) {  // 遍历text
            unsigned int textl = bases[j*2+0];
            unsigned int texth = bases[j*2+1];
            unsigned int row = line[j];
            for (int k=0; k<32; k++) {  // 32*32的核心
                unsigned int queryl = 0x00000000;
                if (queryLow>>k&1) queryl = 0xFFFFFFFF;
                unsigned int queryh = 0x00000000;
                if (queryHight>>k&1) queryh = 0xFFFFFFFF;
                unsigned int temp1 = textl ^ queryl;
                unsigned int temp2 = texth ^ queryh;
                unsigned int match = (~temp1)&(~temp2);
                unsigned int unmatch = ~match;
                unsigned int temp3 = row & match;
                unsigned int temp4 = row & unmatch;
                unsigned int carry = column[k];
                unsigned int temp5 = row + carry;
                unsigned int carry1 = temp5 < row;
                temp5 += temp3;
                unsigned int carry2 = temp5 < temp3;
                carry = carry1 | carry2;
                row = temp5 | temp4;
                column[k] = carry;  //
            }
            line[j] = row;
        }
    }
    // query补齐阶段
    unsigned int column[32] = {0};
    unsigned int queryLow = reads[queryStart+(queryLength/32)*2+0];
    unsigned int queryHight = reads[queryStart+(queryLength/32)*2+1];
    int jstart = queryLength/32-shift;
    jstart = max(jstart, 0);
    int jend = queryLength/32+shift;
    jend = min(jend, textLength/32);
    for (int j=jstart; j<=jend; j++) {  // 遍历text
        unsigned int textl = bases[j*2+0];
        unsigned int texth = bases[j*2+1];
        unsigned int row = line[j];
        for (int k=0; k<queryLength%32; k++) {  // 32*32的核心
            unsigned int queryl = 0x00000000;
            if (queryLow>>k&1) queryl = 0xFFFFFFFF;
            unsigned int queryh = 0x00000000;
            if (queryHight>>k&1) queryh = 0xFFFFFFFF;
            unsigned int temp1 = textl ^ queryl;
            unsigned int temp2 = texth ^ queryh;
            unsigned int match = (~temp1)&(~temp2);
            unsigned int unmatch = ~match;
            unsigned int temp3 = row & match;
            unsigned int temp4 = row & unmatch;
            unsigned int carry = column[k];
            unsigned int temp5 = row + carry;
            unsigned int carry1 = temp5 < row;
            temp5 += temp3;
            unsigned int carry2 = temp5 < temp3;
            carry = carry1 | carry2;
            row = temp5 | temp4;
            column[k] = carry;
        }
        line[j] = row;
    }
    //----统计结果----//
    int sum = 0;
    unsigned int result;
    for (int i=0; i<textLength/32; i++) {
        result = line[i];
        for (int j=0; j<32; j++) {
            sum += result>>j&1^1;
        }
    }
    result = line[textLength/32];
    for (int i=0; i<textLength%32; i++) {
        sum += result>>i&1^1;
    }
    int cutoff = textLength*threshold;
    if (sum > cutoff) flag[0] = 1;
    if (flag[0] == 1) return;
}











// dynamic 动态规划
void dynamic(Data &data, Option &option, unsigned long represent) {
  kernel_dynamic<<<64, 128>>>(data.reads, data.lengths, data.offsets, data.flag,
  option.threshold, represent);
}
// checkErr 检查错误
void checkErr() {
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err);
  }
}