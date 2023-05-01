#ifndef __FUNCH__
#define __FUNCH__

#include <iostream>  // string
#include <sstream>  // istringstream
#include <mpi.h>  // MPI_Datatype

//--------数据--------//
struct Option {  // 设定的选项等信息
  int rank;  // mpi进程序号
  int size;  // mpi进行个数
  float similarity;  // 相似度阈值
  long readsCount;  // 节点序列数量
  long readsWhole;  // 总序列数量
};
struct Data {  // 显存和内存中的数据
  unsigned short *lengths;  // 序列长度
  unsigned short *lengths_h;  // 序列长度
  unsigned short *netLengths;  // 序列净长度
  unsigned short *netLengths_h;  // 序列净长度
  unsigned int **packedReads;  // 压缩后的数据
  unsigned int **packedReads_h;  // 压缩后的数据
  unsigned short **words;  // 生成的短词
  unsigned short **words_h;  // 生成的短词
};
struct Represent {  // 代表序列
  long order;  // 是第几条数据
  unsigned short length;  // 序列长度
  unsigned short netLength;  // 序列净长度
  unsigned short word[256];  // 短词
  unsigned int packedRead[4096];  // 压缩后的数据
};
struct Bench {  // 工作台
  long *ships;  // 每个节点的代表序列
  long ship;  // 最终确定的最小
  long *cluster;  // 存储聚类的结果
  long *remainList;  // 未聚类列表
  long remainCount;  // 未聚类数
  long *jobList;  // 任务列表
  long jobCount;  // 任务数
  Represent *represent;
};
//--------功能--------//
// string转数字
template <class Type> Type stringToNum(const std::string &str) {
  std::istringstream iss(str);
  Type num;
  iss >> num;
  return num;
}
//--------函数--------//
void initialization(int argc, char **argv, Option &option);  // 初始化
void readData(Option &option, Data &data);  // 读数据
void clustering(Option &option, Data&data, Bench &bench);  // 聚类
void finish(Option &option, Data &data, Bench &bench);  // 检查错误

#endif
