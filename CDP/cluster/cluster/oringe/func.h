#ifndef __FUNCH__
#define __FUNCH__

#include <iostream>  // string
#include <vector>  // vector

// 数据定义
struct Option {  // 设定
  std::string inputFile;  // 输入文件
  int wordLength;  // 短词长度
  float threshold;  // 相似度阈值
};
// n个序列：0 -- n-1 为text序列 n为query序列
struct Data {  // 只可以包含基础的数据结构
  // 统一内存
  unsigned int *flag;  // 是否被聚类 0失败 1成功
  // host部分
  unsigned int alignSize;  // 对齐长度
  unsigned int *read;  // 读入的一条序列
  unsigned int *length;  // 一条序列长度
  unsigned long *offset;  // 一条序列偏移
  // device部分
  unsigned int *reads;  // 序列数据
  unsigned int *lengths;  // 序列长度
  unsigned long *offsets;  // 序列起始位置
};
//函数定义
void initData(int readsCount, Data &data);  // 初始化数据
void packing(Data &data, std::string &line);  // 打包数据
void dynamic(Data &data, Option &option, unsigned long represent);  // 动态规划
void checkErr();  // 检擦错误
#endif
