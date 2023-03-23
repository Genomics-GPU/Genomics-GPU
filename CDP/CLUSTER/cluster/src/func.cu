#include <iostream>  // cout
#include <vector>  // vector
#include <fstream>  // ifstream
#include <mpi.h>  // mpi
#include "cmdline.h"  // cmdline
// #include "timer.h"  // timer
#include "func.h"

// initialization 初始化
void initialization(int argc, char **argv, Option &option) {  // 初始化
  // 解析参数
  cmdline::parser parser;  // 解析器
  parser.add<float>("similarity", 's', "similarity 0.8-0.99",
    false, 0.95, cmdline::range(0.8, 0.99));  // 相似度
  parser.parse_check(argc, argv);
  option.similarity = parser.get<float>("similarity");  // 相似度
  // MPI初始化
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &option.size);
  MPI_Comm_rank(MPI_COMM_WORLD, &option.rank);
  // 选择设备
  int deviceCount;  // 节点上的设备数
  cudaGetDeviceCount(&deviceCount);
  int device = option.rank%deviceCount;
  cudaSetDevice(device);  // 根据rank选择设备
}
// readOffset 读offset文件
void readOffset(std::vector<long> &dataOffset) {
  std::ifstream file("index.txt");  // 打开文件
  std::string line;
  getline(file, line);  // 读序列个数
  long readsCount = stringToNum<long>(line);  // 序列个数
  // 跳过nameOffset
  for (long i=0; i<readsCount; i++) {
    getline(file, line);
  }
  // 读dataOffset
  dataOffset.resize(readsCount);  // 申请空间
  for (long i=0; i<readsCount; i++) {
    getline(file, line);
    dataOffset[i] = stringToNum<long>(line);
  }
  file.close();
}
// kernel_baseToNumber 碱基转数字并去除gap
__global__ void kernel_baseToNumber(char **reads, unsigned short *lengths,
unsigned short *netLengths, long readsCount) {
  int index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  int loop = gridDim.x*blockDim.x;  // 一共有多少线程
  for (long i=index; i<readsCount; i+=loop) {  // 遍历序列
    unsigned short length = lengths[i];  // 原始长度
    unsigned short netLength = 0;  // 净长度
    for (unsigned short j=0; j<length; j++) {
      switch (reads[i][j]) {  // 实际是寄存器计算，比用数组更快
        case 'A':
          reads[i][netLength] = 0;
          netLength += 1;
          break;
        case 'C':
          reads[i][netLength] = 1;
          netLength += 1;
          break;
        case 'G':
          reads[i][netLength] = 2;
          netLength += 1;
          break;
        case 'T':
          reads[i][netLength] = 3;
          netLength += 1;
          break;
        case 'U':
          reads[i][netLength] = 3;
          netLength += 1;
          break;
        case 'a':
          reads[i][netLength] = 0;
          netLength += 1;
          break;
        case 'c':
          reads[i][netLength] = 1;
          netLength += 1;
          break;
        case 'g':
          reads[i][netLength] = 2;
          netLength += 1;
          break;
        case 't':
          reads[i][netLength] = 3;
          netLength += 1;
          break;
        case 'u':
          reads[i][netLength] = 3;
          netLength += 1;
          break;
        default:  // 跳过gap
          break;
      }
    }
    netLengths[i] = netLength;
    __syncthreads();  // 保证每个轮回步调一致
  }
}
// kernel_packData打包数据
__global__ void kernel_packData(char **reads, unsigned int **packedReads,
unsigned short *netLengths, long readsCount) {
  int index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  int loop = gridDim.x*blockDim.x;  // 一共有多少线程
  for (long i=index; i<readsCount; i+=loop) {  // 遍历序列
    unsigned short length = netLengths[i];  // 总长度
    // 高低位反向，为了后面动态规划进位
    for (int j=0; j<length; j+=32) {  // 每32个碱基生成一对int
      unsigned int low = 0;  // 低位
      unsigned int high = 0;  // 高位
      for (int k=31; k>-1; k--) {
        char base = reads[i][j+k];
        low = low<<1;
        high <<= 1;
        low += base&1;  // 打包低位
        high += base>>1;  // 打包高位
      }
      packedReads[i][j/32*2+0]=low;
      packedReads[i][j/32*2+1]=high;
    }
    __syncthreads();
  }
}
// kernel_makeWords 生成短词
__global__ void kernel_makeWords(char **reads, unsigned short **words,
unsigned short *netLengths, long readsCount) {
  int index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  int loop = gridDim.x*blockDim.x;  // 一共有多少线程
  for (long idx=index; idx<readsCount; idx+=loop) {  // 遍历序列
    unsigned int length = netLengths[idx];  // 总长度
    unsigned char temp = reads[idx][0]<<4+reads[idx][1]<<2+reads[idx][2];
    for (int i=3; i<length; i++) {
      temp <<= 2;
      temp += reads[idx][i];
      words[idx][temp] += 1;
    }
    __syncthreads();
  }
}
// readData 读序列数据
void readData(Option &option, Data &data) {
  // 声明变量
  std::vector<long> dataOffset;  // 序列偏移
  std::string line;  // 一行数据
  std::string whole_h;  // 全部数据
  long readsCount;  // 共有多少序列
  // 读序列数据和长度
  readOffset(dataOffset);  // 读序列偏移
  option.readsWhole = dataOffset.size();  // 总序列数据
  readsCount = option.readsWhole/option.size;
  readsCount += option.rank < dataOffset.size()%option.size;
  option.readsCount = readsCount;  // 得到序列个数
  cudaMalloc(&data.lengths, readsCount*sizeof(unsigned short));  // 序列长度
  cudaMallocHost(&data.lengths_h, readsCount*sizeof(unsigned short));
  std::ifstream file("data.txt");
  for (long i=option.rank, j=0; i<dataOffset.size(); i+=option.size, j++) {
    file.seekg(dataOffset[i], std::ios::beg);
    getline(file, line);
    whole_h += line;
    data.lengths_h[j] = line.size();
  }
  file.close();
  cudaMemcpy(data.lengths, data.lengths_h,
    readsCount*sizeof(unsigned short), cudaMemcpyHostToDevice);  // 拷贝长度
  // 拷贝数据
  char *whole;
  cudaMalloc(&whole, (whole_h.size()+32)*sizeof(char));  // 显存中的序列数据
  cudaMemcpy(whole, whole_h.data(),
    whole_h.size()*sizeof(char), cudaMemcpyHostToDevice);
  char **reads, **reads_h;  // 数据指针
  cudaMalloc(&reads, readsCount*sizeof(char*));
  cudaMallocHost(&reads_h, readsCount*sizeof(char*));
  char *wholeTemp = whole;
  for (long i=0; i<readsCount; i++) {  // 生成二维数组
    reads_h[i] = wholeTemp;
    wholeTemp += data.lengths_h[i];
  }
  cudaMemcpy(reads, reads_h, readsCount*sizeof(char*), cudaMemcpyHostToDevice);
  // 生成压缩数据和净长度
  cudaMalloc(&data.netLengths, readsCount*sizeof(unsigned short));  // 净长度
  cudaMallocHost(&data.netLengths_h, readsCount*sizeof(unsigned short));
  cudaMalloc(&data.packedReads, readsCount*sizeof(unsigned int*));  // 压缩数据
  cudaMallocHost(&data.packedReads_h, readsCount*sizeof(unsigned int*));
  for (long i=0; i<readsCount; i++) {  // 申请压缩后数据的显存
    unsigned short length = (data.lengths_h[i]+31)/32*2;  // 压缩后数据长度
    unsigned int *read;
    cudaMalloc(&read, length*sizeof(unsigned int));  // 分配一行数据空间
    data.packedReads_h[i] = read;
  }
  cudaMemcpy(data.packedReads, data.packedReads_h,
    readsCount*sizeof(unsigned int*), cudaMemcpyHostToDevice);
  kernel_baseToNumber<<<128, 128>>>(reads, data.lengths,
    data.netLengths, readsCount);  // 碱基转数字并去除gap
  kernel_packData<<<128, 128>>>(reads, data.packedReads,
    data.netLengths, readsCount);  // 压缩数据
  for (long i=0; i<readsCount; i++) {  // 申请压缩后数据的内存
    unsigned short length = (data.lengths_h[i]+31)/32*2;  // 压缩后数据长度
    unsigned int *read_h;
    cudaMallocHost(&read_h, length*sizeof(unsigned int));  // 分配一行数据
    cudaMemcpyAsync(read_h, data.packedReads_h[i],
      length*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    data.packedReads_h[i] = read_h;
  }
  cudaMemcpy(data.netLengths_h, data.netLengths,
    readsCount*sizeof(unsigned short), cudaMemcpyDeviceToHost);  // 拷回净长度
  // 申请短词空间
  cudaMalloc(&data.words, readsCount*sizeof(unsigned short*));  // 短词
  cudaMallocHost(&data.words_h, readsCount*sizeof(unsigned short*));
  for (long i=0; i<readsCount; i++) {
    unsigned short *word;
    cudaMalloc(&word, 256*sizeof(unsigned short));  // 分配一行数据
    cudaMemset(word, 0x00, 256*sizeof(unsigned short));
    data.words_h[i] = word;
  }
  cudaMemcpy(data.words, data.words_h,
    readsCount*sizeof(unsigned short*), cudaMemcpyHostToDevice);
  kernel_makeWords<<<128, 128>>>(reads, data.words,
    data.netLengths, readsCount);  // 生成短词
  for (long i=0; i<readsCount; i++) {  // 申请短词的内存空间
    unsigned short *word_h;
    cudaMallocHost(&word_h, 256*sizeof(unsigned short));  // 一行
    cudaMemcpyAsync(word_h, data.words_h[i],
      256*sizeof(unsigned short), cudaMemcpyDeviceToHost);
    data.words_h[i] = word_h;
  }
  // 释放空间
  cudaFree(whole); // 释放原始数据
  whole_h.clear();
  cudaFree(reads);  // 释放原始数据指针
  cudaFreeHost(reads_h);
  cudaDeviceSynchronize();
}
// defineMpiType 定义MPI数据结构
void defineMpiType (MPI_Datatype &MPI_REPRESENT) {
  int lengths[5] = {1, 1, 1, 256, 4096};  // 每个成员的长度
  MPI_Datatype types[5] = {MPI_LONG, MPI_UNSIGNED_SHORT,
    MPI_UNSIGNED_SHORT, MPI_UNSIGNED_SHORT, MPI_UNSIGNED};  //每个成员的类型
  Represent DummyRepresent;  // 用来描述要声明的类型
  MPI_Aint begin;
  MPI_Aint offsets[5];  // MPI_Aint 类似unsigned long
  MPI_Get_address(&DummyRepresent, &begin);  // 变量的起始位置
  MPI_Get_address(&DummyRepresent.order, &offsets[0]);
  MPI_Get_address(&DummyRepresent.length, &offsets[1]);
  MPI_Get_address(&DummyRepresent.netLength, &offsets[2]);
  MPI_Get_address(&DummyRepresent.word, &offsets[3]);
  MPI_Get_address(&DummyRepresent.packedRead, &offsets[4]);
  offsets[0] = MPI_Aint_diff(offsets[0], begin);  // 每个成员相对位置
  offsets[1] = MPI_Aint_diff(offsets[1], begin);
  offsets[2] = MPI_Aint_diff(offsets[2], begin);
  offsets[3] = MPI_Aint_diff(offsets[3], begin);
  offsets[4] = MPI_Aint_diff(offsets[4], begin);
  MPI_Type_create_struct(5, lengths, offsets, types, &MPI_REPRESENT);
  MPI_Type_commit(&MPI_REPRESENT);  // 提交新类型
}
// initBench 初始化bench
void initBench(Option &option, Bench &bench, long readsCount) {
  cudaMallocHost(&bench.ships, option.size*sizeof(long));  // 节点的旗舰序列
  memset(bench.ships, 0xFF, option.size*sizeof(long));  // 赋初值-1
  cudaMallocManaged(&bench.cluster, readsCount*sizeof(long));  // 聚类结果
  memset(bench.cluster, 0xFF, readsCount*sizeof(long));  // 赋初值-1
  cudaMallocManaged(&bench.remainList, readsCount*sizeof(long));  // 未聚类序列
  for (long i=0; i<readsCount; i++) {  // 未聚类列表赋初值
    bench.remainList[i] = i;
  }
  bench.remainCount = readsCount;  // 未聚类序列个数
  cudaMallocManaged(&bench.jobList, readsCount*sizeof(long));  // 任务列表
  memcpy(bench.jobList, bench.remainList, readsCount*sizeof(long));  // 赋初值
  bench.jobCount = readsCount;  // 需要计算序列个数
  cudaMallocManaged(&bench.represent, sizeof(Represent));  // 代表序列
  cudaDeviceSynchronize();  // 同步数据
}
// updateRepresent 更新代表序列
void updateRepresent(Option &option, Bench &bench, Data &data,
  MPI_Datatype &MPI_REPRESENT) {
  // 更新未聚类列表
  long count = 0;
  for (long i=0; i<bench.remainCount; i++) {
    long index = bench.remainList[i];
    if (bench.cluster[index] == -1) {  // 如果这个序列还没聚类
      bench.remainList[count] = index;
      count += 1;
    }
  }
  bench.remainCount = count;
  // 更新任务列表
  memcpy(bench.jobList, bench.remainList, bench.remainCount*sizeof(long));
  bench.jobCount = bench.remainCount;
  // 代表序列
  if (bench.remainCount > 0) {  // 如果聚类还没完成
    long index = bench.remainList[0];  // 代表序列的本地编号
    bench.represent[0].order = index*option.size+option.rank;  // 全局编号
    bench.represent[0].length = data.lengths_h[index];  // 序列长度
    bench.represent[0].netLength = data.netLengths_h[index];  // 序列净长度
    memcpy(bench.represent[0].word, data.words_h[index],
      256*sizeof(unsigned short));  // 短词
    memcpy(bench.represent[0].packedRead, data.packedReads_h[index],
      (data.lengths_h[index]+31)/32*2*sizeof(unsigned int));  // 压缩后数据
    bench.ship = bench.represent[0].order;  // 全局编号
  } else {
    bench.ship = option.readsWhole;  // 表示已经聚完了
  }
  // 收集编号
  MPI_Gather(&bench.ship, 1, MPI_LONG, bench.ships,
    1, MPI_LONG, 0, MPI_COMM_WORLD);
  // 找最小
  bench.ship = option.readsWhole;
  for (int i=0; i<option.size; i++) {
    if (bench.ships[i] < bench.ship) {  // 找最小旗舰
      bench.ship = bench.ships[i];
    }
  }
  // 广播代表序列
  MPI_Bcast(&bench.ship, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(bench.represent, 1, MPI_REPRESENT,
    bench.ship%option.size, MPI_COMM_WORLD);
}
// kernel_filter 过滤
__global__ void kernel_filter(unsigned short netLength, unsigned short *word,
unsigned short *netLengths, unsigned short *lengths, unsigned short **words,
long *jobList, long jobCount, float similarity) {
  __shared__ unsigned short bases[256];
  for (int i=threadIdx.x; i<256; i+=blockDim.x) {
    bases[i] = word[i];
  }
  int textLength = netLength;
  int index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  int loop = gridDim.x*blockDim.x;  // 一共有多少线程
  for (long idx=index; idx<jobCount; idx+=loop) {  // 遍历序列
    long order = jobList[idx];
    int sum = 0;  // 短词数总和
    for (int i=0; i<256; i++) {
      sum += min(bases[i], words[order][i]);
    }
    int queryLength = netLengths[order];
    int length = min(textLength, queryLength);
    int threshold = length-ceil(length*(1-similarity))*4;
    threshold = max(threshold, 1);  // 最小是1
    if (sum < threshold) jobList[idx] = -1;  // 序列不相似
    __syncthreads();
  }
}
// updateJobs 更新任务列表
void updateJobs(Bench &bench) {
  long count = 0;
  for (long i=0; i<bench.jobCount; i++) {
    long target = bench.jobList[i];  // 任务目标
    if (target != -1) {  // 任务没被排除
      bench.jobList[count] = target;
      count += 1;
    }
  }
  bench.jobCount = count;
}
// kernel_dynamic 动态规划
__global__ void kernel_dynamic(
unsigned short netLength, unsigned short length, unsigned int *packedRead,
unsigned short *netLengths, unsigned short *lengths, unsigned int **packedReads,
long *jobList, long *cluster, long jobCount, float similarity, long order,
int rank, long *remainList) {
  //----准备数据----//
  unsigned short textLength = netLength;
  __shared__ unsigned int bases[4096];  // 65536/32*2
  for (int i=threadIdx.x; i<(textLength+31)/32*2; i+=blockDim.x) {  // 拷贝数据
    bases[i] = packedRead[i];
  }
  //----开始计算----//
  int index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  int loop = gridDim.x*blockDim.x;  // 一共有多少线程
  for (long idx=index; idx<jobCount; idx+=loop) {  // 遍历序列
    unsigned int line[2048];  // 保存结果 2048=65536/32
    memset(line, 0xFF, 2048*sizeof(unsigned int));  // 赋值-1 0表示比对上了
    long query = jobList[idx];
    unsigned short queryLength = netLengths[query];
    int shift = ceil((float)textLength-(float)queryLength*similarity);
    shift = ceil((float)shift/32.0f);  // 相对对角线，左右偏移方块个数
    // 遍历query的32整除部分
    for (int i=0; i<queryLength/32; i++) {
      unsigned int column[32] = {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
                                 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};
      unsigned int queryLow = packedReads[query][i*2+0];  // 低32位
      unsigned int queryHigh = packedReads[query][i*2+1];  // 高32位
      int jstart = i-shift;
      jstart = max(jstart, 0);
      int jend = i+shift;
      jend = min(jend, (textLength+31)/32);
      for (int j=jstart; j<=jend; j++) {  // 遍历text
        unsigned int textl = bases[j*2+0];  // 参与运算用小写字母
        unsigned int texth = bases[j*2+1];
        unsigned int row = line[j];
        for (int k=0; k<32; k++) {  // 32*32的核心
          unsigned int queryl = 0x00000000;  // 1位扩展成32位
          if (queryLow>>k&1) queryl = 0xFFFFFFFF;
          unsigned int queryh = 0x00000000;
          if (queryHigh>>k&1) queryh = 0xFFFFFFFF;
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
    // query补齐阶段
    unsigned int column[32] = {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
                               0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0};
    unsigned int queryLow = packedReads[query][queryLength/32*2+0];
    unsigned int queryHigh = packedReads[query][queryLength/32*2+1];
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
        if (queryHigh>>k&1) queryh = 0xFFFFFFFF;
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
    int sum1 = 0;
    for (int i=0; i<textLength%32; i++) {
      sum += result>>i&1^1;
      sum1 += result>>i&1^1;
    }
    int cutoff = ceil(min(queryLength, textLength)*similarity);
    if (sum >= cutoff) {  // 相似就聚类成功
      cluster[query] = order;  // 记录聚类结果
      // remainList[idx] = -1;  // 已经聚类就移除待聚类序列
    }
    __syncthreads();
  }
}
// clustering 聚类
void clustering(Option &option, Data &data, Bench &bench) {
  long readsCount = option.readsCount;
  MPI_Datatype MPI_REPRESENT;  // 定义的MPI结构体
  defineMpiType(MPI_REPRESENT);  // 声明MPI结构体
  initBench(option, bench, readsCount);  // 初始化bench
  while (true) {  // 聚类
    //----准备工作----//
    // 更新代表序列
    updateRepresent(option, bench, data, MPI_REPRESENT);
    if (bench.ship == option.readsWhole) break;  // 判断聚类完成
    // 打印进度
    if (option.rank == 0) {
      std::cout << "\r" << bench.ship+1 << "/" << option.readsWhole;
      std::cout << " " << bench.jobCount << std::flush;
    }
    //----过滤工作----//
    // long count1 = bench.jobCount;
    kernel_filter<<<128, 128>>>(bench.represent[0].netLength,
      bench.represent[0].word, data.netLengths, data.lengths, data.words,
      bench.jobList, bench.jobCount, option.similarity); 
    cudaDeviceSynchronize();  // 同步数据
    updateJobs(bench);  // 更新任务
    // long count2 = bench.jobCount;
    // std::cout << " " << count1 << " " << count2;
    //----比对工作----//
    if (bench.jobCount > 0) {  // 动态规划
      kernel_dynamic<<<128, 128>>>(bench.represent[0].netLength,
      bench.represent[0].length, bench.represent[0].packedRead,
        data.netLengths, data.lengths, data.packedReads,
        bench.jobList, bench.cluster, bench.jobCount, 
        option.similarity, bench.represent[0].order, option.rank,
        bench.remainList); 
    }
    cudaDeviceSynchronize();  // 同步数据
    // break;
  }
  if (option.rank == 0) {
    std::cout << "\r" << option.readsWhole << "/";
    std::cout << option.readsWhole << std::endl;
  }
  // 生成结果
  {
    long *cluster;
    std::vector<long> result;  // 代表序列的结果
    cudaMallocHost(&cluster, readsCount*option.size*sizeof(long));
    MPI_Gather(bench.cluster, readsCount, MPI_LONG, cluster,
      readsCount, MPI_LONG, 0, MPI_COMM_WORLD);
    if (option.rank == 0) {  // 统计结果
      for (long i=0; i<readsCount*option.size; i++) {
        long a = cluster[i];
        long b = i%readsCount*option.size+i/readsCount;
        if (a==b) result.push_back(a);
      }
      std::cout << "cluster: " << result.size() << std::endl;
    }
    cudaFreeHost(cluster);
    // 把结果写入文件
    if (option.rank == 0) {
      std::ifstream fileIn1("name.txt");  // 打开输入文件1
      std::ifstream fileIn2("data.txt");  // 打开输入文件2
      std::ofstream fileOut("result.txt");  // 打开输出文件
      std::string line1;
      std::string line2;
      long point = 0;  // result结果的指针
      for (long i=0; i<option.readsWhole; i++) {
        getline(fileIn1, line1);  // 读一行名
        getline(fileIn2, line2);  // 读一行数据
        if (i==result[point]) {
          // if (point<result.size()-1) 
          point += 1;
          fileOut << line1 << "\n";
          fileOut << line2 << "\n";
        }
        std::cout << "\r" << i;
        if (point==result.size()-1) break;  // 已经找到了全部代表序列
      }
      std::cout << std::endl;
      fileIn1.close();
      fileIn2.close();
      fileOut.close();
    }
  }
  // 生成结果
}
// finish 收尾函数
void finish(Option &option, Data &data, Bench &bench) {
  // 释放空间
  cudaFree(data.lengths);  // 序列长度
  cudaFreeHost(data.lengths_h);
  cudaFree(data.netLengths);  // 序列净长度
  cudaFreeHost(data.netLengths_h);
  cudaFree(data.packedReads);  // 压缩后的数据
  cudaFreeHost(data.packedReads_h);
  cudaFree(data.words);  // 短词
  cudaFreeHost(data.words_h);
  // 检查显卡错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << std::endl;
  }
  // 结束MPI
  MPI_Finalize();
}
