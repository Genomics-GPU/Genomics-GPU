#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
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
    printf("\t-g\t: use GPU only (default use both GPU and CPU)\n");
    printf("\t-c\t: use CPU only (default use both GPU and CPU)\n");
    printf("\t-w <int>\t: specify the workload ratio of CPU / CPU\n");
    printf("\t-b <int>\t: specify the number of blocks per grid\n");
    printf("\t-t <int>\t: specify the number of threads per block\n");
    printf("\t-n <int>\t: specify the number of GPU devices should be used\n");
}


int parseOptions(int argc, char* argv[]) {
    if(argc < 3) {
        displayUsage();
        return -1;                          // 不执行程序
    }

    int oc;
    while((oc = getopt(argc, argv, "gcw:b:t:n:")) != -1) {
        switch(oc) {
            case 'g':                       // 只使用GPU
                MODE = GPU_ONLY;
                break;
            case 'c':                       // 只使用CPU (OpenMP)
                MODE = CPU_ONLY;
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


void writeArrayToFile(short *array, int size, char *path) {
    FILE *fout = fopen(path, "w");
    for(int i = 0; i < size; i++)
        fprintf(fout, "%d ", array[i]);
    fclose(fout);
}
void readArrayFromFile(short *array, char *path) {
    ifstream fin(path);
    int i = 0;
    while(fin >> array[i])
        ++i;
    fin.close();
}
