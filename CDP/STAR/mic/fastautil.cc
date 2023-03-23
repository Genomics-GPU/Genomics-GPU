#include <stdio.h>
#include <assert.h>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include "util.h"
using namespace std;

/**
 * 处理FASTA格式文件的工具
 * 功能：
 * 1. 统计文件中共有多少条串
 * 2. 计算所有串的平均长度
 * 3. 截取前n条串输出到指定文件
 *
 * 用法
 * ./fastautil input_path [output_path] [n]
 */

int main(int argc, char* argv[]) {
    assert(argc >= 2);

    FastaSeqs fs = readFastaFile(argv[1]);
    vector<string> seqs = fs.seqs;

    int maxLength = seqs[0].size();
    int minLength = seqs[0].size();
    long long sumLength = 0;

    for(int i = 0; i < seqs.size(); i++) {
        sumLength += seqs[i].size();
        if(seqs[i].size() > maxLength)
            maxLength = seqs[i].size();
        if(seqs[i].size() < minLength)
            minLength = seqs[i].size();
    }

    int avgLength = sumLength / seqs.size();

    printf("\n\n=========================================\n");
    printf("Sequences Size: %lu\n", seqs.size());
    printf("Max: %d, Min: %d, Avg: %d\n", maxLength, minLength, avgLength);
    printf("=========================================\n\n");

    /**
     * 将前n行输出到指定文件
     */
    if( argc == 4) {
        char *outputPath = argv[2];
        int n = atoi(argv[3]);

        printf("Write %d Sequences to %s ...\n", n, outputPath);
        seqs.erase(seqs.begin()+n, seqs.end());
        writeFastaFile(outputPath, fs.titles, seqs);
        printf("Done.\n");
    }


    return 0;

}
