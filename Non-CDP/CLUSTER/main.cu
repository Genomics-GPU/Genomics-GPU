#include "func.h"
//--------------------主体--------------------//
int main(int argc, char **argv) {
    clock_t start = clock();
    //----准备数据----//
    Option option;
    checkOption(argc, argv, option);  // 检查配置 ok
    std::vector<Read> reads;
    readFile(reads, option);  // 读文件 ok
    Data data;
    copyData(reads, data);  // 拷贝数据 ok
    baseToNumber(data);  // 碱基转数字 ok
    //----预处理----//
    createPrefix(data);  // 生成前置过滤 ok
    createWords(data, option);  // 生成短词 ok
    sortWords(data, option);  // 排序短词 ok
    mergeWords(data);  // 合并相同短词 ok
    createCutoff(data, option);  // 生成阈值 ok
    deleteGap(data);  // 去除gap ok
    compressData(data);  // 压缩数据 ok
    //----去冗余----//
    Bench bench;
    clustering(option, data, bench);  // 聚类 ok
    //----收尾----//
    saveFile(option, reads, bench);  // 保存结果 ok
    checkError();
    std::cout << "聚类耗时：" << (clock()-start)/1000 << "ms" << std::endl;
}
