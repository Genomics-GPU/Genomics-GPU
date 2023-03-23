#include <stdio.h>
#include <assert.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include "util.h"
using namespace std;

bool compareLength(const string &str1, const string &str2) {
    return str1.size() < str2.size();
}

/**
 * 处理FASTA格式文件的工具
 * 功能：读入文件然后按照串的长度排序后输出
 *
 * 用法
 * ./sort input_path output_path
 */

int main(int argc, char* argv[]) {
    assert(argc >= 2);

    FastaSeqs fs = readFastaFile(argv[1]);
    vector<string> seqs = fs.seqs;

    vector<int> seqsSize;

    printf("Sorting ...\n");
    sort(seqs.begin(), seqs.end(), compareLength);
    printf("Done. Max: %lu, Min: %lu\n", seqs[0].size(), seqs[seqs.size()-1].size());

    // 输出文件
    char *outputPath = argv[2];
    printf("Write %lu Sequences to %s ...\n", seqs.size(), outputPath);
    writeFastaFile(outputPath, fs.titles, seqs);
    printf("Done.\n");

    return 0;

}
