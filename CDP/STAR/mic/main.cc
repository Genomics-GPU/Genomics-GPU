#include <stdio.h>
#include "util.h"
#include "sp.h"
#include "center-star.h"
#include "nw.h"
#include "omp.h"
#include "global.h"
using namespace std;


/**
 * 定义全局变量
 * centerSeq 存储中心串
 * seqs 存储所有其他串
 */
string centerSeq;
vector<string> titles;
vector<string> seqs;    // 所有串
int maxLength;          // 最长的串的长度
int centerSeqIdx;


void pre_compute();

/**
 * 从path读如fasta格式文件，
 * 完成初始化工作并输出相关信息
 */
void init(const char *path) {
    // 读入所有字符串
    // centerSeq, 图中的纵向，决定了行数m
    // seqs[idx], 图中的横向，决定了列数n
    double start = omp_get_wtime();
    FastaSeqs fastaSeqs = readFastaFile(path);
    titles = fastaSeqs.titles;
    seqs = fastaSeqs.seqs;
    double end = omp_get_wtime();
    printf("Read Sequences, use time: %f\n", end-start);

    // 找出中心串
    start = omp_get_wtime();
    centerSeqIdx = findCenterSequence(seqs);
    end = omp_get_wtime();
    printf("Find the Center Sequence, use time: %f\n", end-start);

    centerSeq = seqs[centerSeqIdx];
    seqs.erase(seqs.begin() + centerSeqIdx);

    unsigned long sumLength = 0;
    maxLength = centerSeq.size();
    int minLength = centerSeq.size();
    for(int i=0;i<seqs.size();i++) {
        sumLength += seqs[i].size();
        if( maxLength < seqs[i].size())
            maxLength = seqs[i].size();
        if( minLength > seqs[i].size())
            minLength = seqs[i].size();
    }
    int avgLength = sumLength / seqs.size();

    // 输出相关信息
    printf("\n\n=========================================\n");
    printf("Sequences Size: %lu\n", seqs.size()+1);
    printf("Max: %d, Min: %d, Avg: %d\n", maxLength, minLength, avgLength);
    printf("Center Sequence Index: %d\n", centerSeqIdx);
    printf("Workload Ratio of GPU/CPU: %.2f:%d\n", (MODE==GPU_ONLY)?1:WORKLOAD_RATIO, (MODE==GPU_ONLY)?0:1);
    printf("Block Size: %d, Thread Size: %d\n", BLOCKS, THREADS);
    printf("=========================================\n\n");
}

/**
  * 将MSA结果输出到path文件中
  * 共有n条串，平均长度m
  * 构造带空格的中心串复杂度为:O(nm)
  * 构造带空格的其他条串复杂度为:O(nm)
  */
void output(short *space, short *spaceForOther, const char* path) {
    double start = omp_get_wtime();
    vector<string> allAlignedStrs;

    int sWidth = centerSeq.size() + 1;      // space[] 的每条串宽度
    int soWidth = maxLength + 1;            // spaceForOther[] 的每条串宽度

    // 将所有串添加的空格汇总到一个数组中
    // 然后给中心串插入空格
    string alignedCenter(centerSeq);
    vector<int> spaceForCenter(centerSeq.size()+1, 0);
    for(int pos = centerSeq.size(); pos >= 0; pos--) {
        int count = 0;
        for(int idx = 0; idx < seqs.size(); idx++)
            count = (space[idx*sWidth+pos] > count) ? space[idx*sWidth+pos] : count;
        spaceForCenter[pos] = count;
        if(spaceForCenter[pos] > 0)
            //printf("pos:%d, space:%d\n", pos, spaceForCenter[pos]);
            alignedCenter.insert(pos, spaceForCenter[pos], '-');
    }

    //printf("\n\n%s\n", alignedCenter.c_str());
    //allAlignedStrs.push_back(alignedCenter);

    for(int idx = 0; idx < seqs.size(); idx++) {
        int shift = 0;
        string alignedStr(seqs[idx]);
        // 先插入自己比对时的空格
        for(int pos = seqs[idx].size(); pos >= 0; pos--) {
            if(spaceForOther[idx*soWidth+pos] > 0)
                alignedStr.insert(pos, spaceForOther[idx*soWidth+pos], '-');
        }
        // 再插入其他串比对时引入的空格
        for(int pos = 0; pos < spaceForCenter.size(); pos++) {
            int num = spaceForCenter[pos] - space[idx*sWidth+pos];
            if(num > 0) {
                alignedStr.insert(pos+shift, num, '-');
            }
            shift += spaceForCenter[pos];
        }
        //printf("%s\n", alignedStr.c_str());
        allAlignedStrs.push_back(alignedStr);
    }
    allAlignedStrs.insert(allAlignedStrs.begin()+centerSeqIdx, alignedCenter);

    // 将结果写入文件
    //writeFastaFile(path, titles, allAlignedStrs);
    double end = omp_get_wtime();
    printf("write %lu sequences to the output file: %s, use time: %f\n", allAlignedStrs.size(), path, end-start);

}

/**
  * 使用GPU和CPU计算MSA
  * 返回CPU/GPU的运行时间比，用于pre_compute计算WORKLOAD_RATIO
  * space: out
  * spaceForOther: out
  */
void msa(short *space, short *spaceForOther, vector<string> seqs, int gpuWorkCount) {
    double cpu_time;

    // CPU, 做剩下的部分(seqs.size() - gpuWorkCount)
    double start = omp_get_wtime();
    cpu_msa(centerSeq, seqs, gpuWorkCount, space, spaceForOther, maxLength);
    double end = omp_get_wtime();
    cpu_time = end - start;
    printf("CPU DP calulation, use time: %f\n", cpu_time);
}


int main(int argc, char *argv[]) {

    double start = omp_get_wtime();

    // 解析用户参数
    //int argvIdx = parseOptions(argc, argv);
    // 输入错误选项或选项不够时不执行程序
    //if(argvIdx < 0) return 0;

    const char *inputPath = argv[1];
    const char *outputPath = argv[2];
    if(argc>3)
        OMP_THREADS = atoi(argv[3]);

    // 读入所有串，找出中心串
    init( inputPath );

    // Host端的纪录空格的数组
    // 为是计算和数据传输可以重叠，需要使用Pinned Memory
    short *space = new short[seqs.size() * (centerSeq.size() + 1)]();
    short *spaceForOther = new short[seqs.size() * (maxLength + 1)]();

    // 只使用CPU
    int workCount = 0;

    // MSA计算
    msa(space, spaceForOther, seqs, workCount);

    //writeArrayToFile(space, seqs.size()*(centerSeq.size()+1), "./tmp1.txt");
    //writeArrayToFile(spaceForOther, seqs.size()*(maxLength+1), "./tmp2.txt");
    //readArrayFromFile(space, "./tmp1.txt");
    //readArrayFromFile(spaceForOther, "./tmp2.txt");

    // 输出结果
    output(space, spaceForOther, outputPath);

    delete[] space;
    delete[] spaceForOther;

    double end = omp_get_wtime();
    printf("total time: %f\n", end-start);
}


