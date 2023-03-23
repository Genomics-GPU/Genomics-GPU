#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "sp.h"
using namespace std;

int min(int v1, int v2) {
    return v1 < v2 ? v1 : v2;
}

/**
 * 计算两个串的Sum of Pairs得分
 */
int sumOfPair(const char *str1, const char *str2) {
    int sp = 0;
    int len1 = strlen(str1);
    int len2 = strlen(str2);
    for(int i=0;i<min(len1, len2);i++) {
        if(str1[i] != str2[i])
            sp++;
    }
    sp += abs(len1 - len2);
    return sp;
}

/**
 * 计算整个MSA的Sum of Pairs得分
 */
int sumOfPairs(list<string> seqs) {
    long sp = 0;
    list<string>::iterator it1, it2;
    for(it1=seqs.begin();it1!=seqs.end();it1++) {
        const char *str1 = (*it1).c_str();
        for(it2=seqs.begin();it2!=seqs.end();it2++) {
            const char *str2 = (*it2).c_str();
            sp += sumOfPair(str1, str2);
        }
    }

    return sp;
}
