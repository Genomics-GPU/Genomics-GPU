#include "func.h"
int main(int argc, char **argv) {
    clock_t start = clock();
  
    Option option;
    checkOption(argc, argv, option);
    selectDevice(option);
    std::vector<Read> reads;
    readFile(reads, option);
    Data data;
    copyData(reads, data, option);
    baseToNumber(data);
  
    createPrefix(data);
    createWords(data, option);
    sortWords(data, option);
    mergeWords(data);
    createCutoff(data, option);
    deleteGap(data);
    compressData(data);
  
    Bench bench;
    clustering(option, data, bench);
  
    saveFile(option, reads, bench);
    std::cout << "time consuming：" << (clock() - start) / 1000 << "ms" << std::endl;
}
