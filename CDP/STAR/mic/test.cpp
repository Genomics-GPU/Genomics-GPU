#include <vector>
#include <string>
#include <stdio.h>
#include <assert.h>
#include "util.h"
using namespace std;

/**
 * 测试用文件，与MSA程序无关
 */

int main(int argc, char *argv[]) {
    assert(argc>=3);

    int len = 100;

    vector<string> seqs = readFastaFile(argv[1]);
    vector<string> res;
    for(int i=0;i<seqs.size();i++) {
        string str = seqs[i].substr(0, len);
        printf("length: %d\n", str.size());
        res.push_back(str);
    }
    writeFastaFile(argv[2], res);
    return 0;
}
