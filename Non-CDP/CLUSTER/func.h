// 模仿cdhit写的聚类应用，有聚类步骤
#pragma once
#include <iostream>  // cout
#include <fstream>  // ifstream
#include <ctime>  // clock
#include <vector>  // vector
#include <cstring>  // memcpy
#include <algorithm>  // sort
//--------------------数据--------------------//
struct Option {  // 配置选项
    std::string inputFile;  // 输入文件名
    std::string outputFile;  // 输出文件名
    float threshold;  // 阈值
    int wordLength;  // 词长度
    int drop;  // 是否跳过过滤
    int pigeon;  // 是否启动鸽笼过滤
};
struct Read {  // 读长
    std::string name;
    std::string data;
};
struct Data {  // 显存和内存中的数据
    // // 原始数据
    int readsCount;  // 读长的数量
    int *lengths;  // 存储读长的长度
    long *offsets;  // 存储读长的开端
    char *reads;  // 存储读长
    // 生成的数据
    int *prefix;  // 前置过滤数据
    unsigned short *words;  // 存储生成的短词
    int *wordCounts;  // 短词的个数
    unsigned short *orders;  // 存储短词的个数
    int *gaps;  // 序列中的gap数量
    unsigned int *compressed;  // 压缩后的数据
    // 阈值
    int *wordCutoff;  // word的阈值
    int *baseCutoff;  // 比对的阈值
};
struct Bench {  // 工作台
    unsigned short *table;  // 存储table
    int *cluster;  // 存储聚类的结果
    int *remainList;  // 未聚类列表
    int remainCount;  // 未聚类数
    int *jobList;  // 任务列表
    int jobCount;  // 任务数
    int representative;  // 代表序列
};
//--------------------普通函数--------------------//
// checkOption 检查输入
void checkOption(int argc, char **argv, Option &option);
// readFile 读文件
void readFile(std::vector<Read> &reads, Option &option);
// copyData 拷贝数据
void copyData(std::vector<Read> &reads, Data &data);
// baseToNumber 碱基转换为数字
void baseToNumber(Data &data);
// createPrefix 生成前置过滤
void createPrefix(Data &data);
// createWords 生成短词
void createWords(Data &data, Option &option);
// sortWords 排序短词 (gpu希尔比std::sort快3倍)
void sortWords(Data &data, Option &option);
// mergeWords 合并相同短词
void mergeWords(Data &data);
// createCutoff 生成阈值
void createCutoff(Data &data, Option &option);
// deleteGap 去除gap
void deleteGap(Data &data);
// compressData 压缩数据
void compressData(Data &data);
//--------------------聚类过程--------------------//
// 聚类
void clustering(Option &option, Data &data, Bench &bench);
//--------------------收尾函数--------------------//
// saveFile 保存结果
void saveFile(Option &option, std::vector<Read> &reads, Bench &bench);
// 检查显卡错误
void checkError();
