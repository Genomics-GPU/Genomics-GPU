#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <cuda.h>
#include "util.h"
#include "global.h"
using namespace std;

FastaSeqs readFastaFile(const char *path) {
    vector<string> titles;
    vector<string> seqs;
    string buff, line, title;

    ifstream file;
    file.open(path);
    assert(file);

    while(getline(file, buff)) {
        if(buff.empty() || buff[0] == '>') {
            if(!line.empty()) {
                seqs.push_back(line);
                titles.push_back(title);
            }
            if(buff[0] == '>')
                title = buff;
            line = "";
            continue;
        } else {
            line += buff;
        }
    }

    if(!line.empty() && !title.empty()) {
        seqs.push_back(line);
        titles.push_back(title);
    }

    file.close();
    FastaSeqs fastaSeqs = {titles, seqs};
    return fastaSeqs;
}


void writeFastaFile(const char* path, vector<string> titles, vector<string> alignedSeqs) {
    ofstream file(path);
    if(file.is_open()) {
        for(int i=0;i<alignedSeqs.size();i++) {
            file<<titles[i]<<endl;
            int lines = alignedSeqs[i].size() / 60;        // 60个字符一行
            lines = alignedSeqs[i].size() % 60 == 0 ? lines : lines+1;
            for(int k = 0; k < lines; k++)
                file<<alignedSeqs[i].substr(k*60, 60)<<endl;
        }
    }

    file.close();
}

void displayUsage() {
    printf("Usage :\n");
    printf("./msa.out [options] input_path output_path\n");
    printf("Options:\n");
    printf("\t-d\t: DNA/RNA alignment (default)\n");
    printf("\t-p\t: PROTEIN alignment\n");
    printf("\t-m\t: specify the score matrix of PROTEIN alignment (default use BLOSUM62)\n");
    printf("\t-g\t: use GPU only (default use both GPU and CPU)\n");
    printf("\t-c\t: use CPU only (default use both GPU and CPU)\n");
    printf("\t-w <int>\t: specify the workload ratio of CPU / CPU\n");
    printf("\t-b <int>\t: specify the number of blocks per grid\n");
    printf("\t-t <int>\t: specify the number of threads per block\n");
    printf("\t-n <int>\t: specify the number of GPU devices should be used\n");
}


int parseOptions(int argc, char* argv[]) {
    if (argc < 3) {
        displayUsage();
        return -1;                          // 不执行程序
    }

    int oc;
    while ((oc = getopt(argc, argv, "gcdpm:w:b:t:n:")) != -1) {
        switch (oc) {
        case 'g':                       // 只使用GPU
            MODE = GPU_ONLY;
            break;
        case 'c':                       // 只使用CPU (OpenMP)
            MODE = CPU_ONLY;
            break;
        case 'd':                       // 对DNA或RNA进行比对
            TYPE = DNA;
            break;
        case 'p':                       // 对蛋白质进行比对
            TYPE = PROTEIN;
            break;
        case 'm':                       // 对蛋白质进行比对
            MATRIX = optarg;
            break;
        case 'w':                       // 设置任务比例
            WORKLOAD_RATIO = atof(optarg);
            break;
        case 'b':                       // 设置Blocks数量
            BLOCKS = atoi(optarg);
            break;
        case 't':                       // 设置Threads数量
            THREADS = atoi(optarg);
            break;
        case 'n':
            GPU_NUM = atoi(optarg);     // 设置使用的GPU数量
            break;
        case '?':                       // 输入错误选项，不执行程序
            displayUsage();
            return -1;
        }
    }

    return optind;
}


bool configureKernel(int centerSeqLength, int maxLength, unsigned long sumLength) {

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    // 每次两两匹配的DP矩阵所需要的空间
    size_t matrixSize = sizeof(short) * (centerSeqLength+1) * (maxLength+1);

    // 得到每个Kernel可以执行的串数（即可并发的总线程数BLOCKS*THREADS）
    // 不应该使用所有的空闲内存，在此留出一部分20%
    freeMem = (freeMem - sizeof(char) * sumLength) / 10 * 8;        // *0.8中间会转成double有可能截短
    int seqs = freeMem / matrixSize;

    printf("freeMem: %luMB, sumLengthSize: %luMB, matrix :%luKB, seqs: %d\n", freeMem/1024/1024, sumLength/1024/1024, matrixSize/1024, seqs);

    // 先判断用户设置的<BLOCKS, THREADS>是否满足内存限制
    // 如果不满足，则自动设置一个<BLOCKS, THREADS>
    if(seqs >= BLOCKS*THREADS)
        return true;

    // 在满足内存限制的前提下，
    // 满足BLOCKS >= 3 且 THREADS >= 32则可以在GPU执行
    int b, t;
    for(t = THREADS; t >= 32; t -= 32) {
        b = seqs / t;
        if( b >= 3 && t >= 32) {
            BLOCKS = b;
            THREADS = t;
            return true;
        }
    }

    return false;
}
