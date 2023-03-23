#ifndef FUNCH
#define FUNCH

#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>
#include <cstring>
#include <algorithm>
#include <CL/sycl.hpp>
struct Option {
    std::string inputFile;
    std::string outputFile;
    float threshold;
    int wordLength;
};
struct Read {
    std::string name;
    std::string data;
};
struct Data {
  
    int readsCount;
    long readsLength;
    int *lengths_dev;
    long *offsets_dev;
    char *reads_dev;
  
    int *prefix_dev;
    unsigned short *words_dev;
    int *wordCounts_dev;
    unsigned short *orders_dev;
    int *gaps_dev;
    unsigned int *compressed_dev;
  
    int *wordCutoff_dev;
    int *baseCutoff_dev;
};
struct Bench {
    unsigned short *table_dev;
    int *cluster;
    int *remainList;
    int remainCount;
    int *jobList;
    int jobCount;
    int representative;
    int *result;
};
void checkOption(int argc, char **argv, Option &option);
void selectDevice(Option &option);
void readFile(std::vector<Read> &reads, Option &option);
void copyData(std::vector<Read> &reads, Data &data, Option &option);
void baseToNumber(Data &data);
void createPrefix(Data &data);
void createWords(Data &data, Option &option);
void sortWords(Data &data, Option &option);
void mergeWords(Data &data);
void createCutoff(Data &data, Option &option);
void deleteGap(Data &data);
void compressData(Data &data);
void clustering(Option &option, Data &data, Bench &bench);
void saveFile(Option &option, std::vector<Read> &reads, Bench &bench);
#endif
