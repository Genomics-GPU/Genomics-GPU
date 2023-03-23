#include <stdio.h>
#include <bitset>
#include <string.h>
#include "center-star.h"
using namespace std;


/**
 * 将8 char = 16 bits 转换成一个整数作为下标
 * 若遇到不识别的字母直接返回-1
 */
int charsToIndex(const char *str) {
    bitset<16> bits(0x0000);
    for(int i=0;i<8;i++) {
        switch(str[i]) {
            case 'A':       // 00
                break;
            case 'C':       // 01
                bits[i*2+1] = 1;
                break;
            case 'T':       // 10
            case 'U':
                bits[i*2] = 1;
                break;
            case 'G':       // 11
                bits[i*2] = 1;
                bits[i*2+1] = 1;
                break;
            default:        // 遇到不识别的字母，比如N,X等
                return -1;
        }
    }
    return (int) (bits.to_ulong());
}

/**
 * 一条串中的每个索引最多只能增加一次
 * 使用一个额外的bool[65536]来纪录是否已经加过一次
 */
void setOccVector(const char *str, int *vec) {
    bool flag[65536] = {false};

    int len = strlen(str);
    int n = len / 8;
    for(int i=0;i<n;i++) {
        int index = charsToIndex(str+i*8);
        if(index>=0 && !flag[index]) {
            vec[index]++;
            flag[index] = true;
        }
    }
}


/**
 * 查询一条串中的每一段在其他串中出现的次数
 * 出现最多的一条串作为中心串
 */
int countSequences(const char *str, int *vec) {
    int len = strlen(str);
    int n = len / 8;
    int count = 0;
    for(int i=0;i<n;i++) {
        int index = charsToIndex(str+i*8);
        if(index >= 0)
            count += vec[index];
    }

    return count;
}


/**
 * 将每条串分为p个小段，每个小段长度为8字节
 * 8 char = 16 bits，2^16最大为65535
 * 'A' = 00
 * 'C' = 01
 * 'T' = 10, 'U' = 10
 * 'G' = 11
 * 使用一个int[65536]数组来统计每一段的出现次数
 * 再此遍历所有串，找出在其他串中重复出现小段数最多的串作为中心串
 */
int findCenterSequence(vector<string> sequences) {
    int vec[65536] = {0};

    for(int i = 0; i < sequences.size(); i++) {
        setOccVector(sequences[i].c_str(), vec);
    }

    int maxIndex = 0, maxCount = 0;
    for(int i = 0; i < sequences.size(); i++) {
        int count = countSequences(sequences[i].c_str(), vec);
        if(count > maxCount) {
            maxIndex = i;
            maxCount = count;
        }
        //printf("seq: %d, count: %d,\n", i++, count);
    }

    //printf("maxIndex: %d, maxCount:%d\n", maxIndex, maxCount);

    return maxIndex;
}

