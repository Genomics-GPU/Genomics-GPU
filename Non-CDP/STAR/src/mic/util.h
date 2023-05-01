#ifndef _UTIL_H_
#define _UTIL_H_
#include <string>
#include <vector>
#include <iostream>

typedef struct FastaSeqs_t {
    std::vector<std::string> titles;
    std::vector<std::string> seqs;
} FastaSeqs;

/**
 * 读入Fasta格式的文件
 */
FastaSeqs readFastaFile(const char *path);

/**
 * 将MSA的结果写入到Fasta格式文件
 */
void writeFastaFile(const char *path, std::vector<std::string> titles, std::vector<std::string> alignedSeqs);

/**
 * 解析main函数的参数，设置全局变量
 * 返回argv中不是选项的下标，即input和ouput的路径
 */
int parseOptions(int argc, char* argv[]);

/**
 * 显示帮助信息
 */
void displayUsage();


bool configureKernel(int centerSeqLength, int maxLength, unsigned long sumLength);

void writeArrayToFile(short *array, int size, char *path);
void readArrayFromFile(short *array, char *path);

#endif
