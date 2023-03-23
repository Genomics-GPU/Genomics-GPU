#include <stdio.h>
#include "omp.h"
#include "nw.h"
#include "global.h"
using namespace std;



void printMatrix(short **matrix, int m, int n) {
    for(int i=0;i<m;i++) {
        for(int j=0;j<n;j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int cpu_max(int v1, int v2) {
    return v1 > v2 ? v1 : v2;
}
int cpu_max(int v1, int v2, int v3) {
    return max(max(v1, v2), v3);
}

short** nw(string str1, string str2) {

    // m行, n列
    int m = str1.size();
    int n = str2.size();

    // 直接定义二维数组，比使用vector<vector>的形式节省内存
    // 缺点是需要自己管理内存释放
    short **matrix = new short*[m+1];
    for(int i = 0; i <= m; i++)
        matrix[i] = new short[n+1];
    short **x_matrix = new short*[m+1];         // gap in str1
    for(int i = 0; i <= m; i++)
        x_matrix[i] = new short[n+1];
    short **y_matrix = new short*[m+1];         // gap in str2
    for(int i = 0; i <= m; i++)
        y_matrix[i] = new short[n+1];

    // 初始化矩阵
    for(int j = 0; j <= n; j++) {
        //matrix[0][j] = j * MISMATCH;
        matrix[0][j] = MIN_SCORE;
        x_matrix[0][j] = GAP_START + j * GAP_EXTEND;
        y_matrix[0][j] = MIN_SCORE;
    }
    for(int i = 0; i <= m; i++) {
        //matrix[i][0] = i * MISMATCH;
        matrix[i][0] = MIN_SCORE;
        x_matrix[i][0] = MIN_SCORE;
        y_matrix[i][0] = GAP_START + i * GAP_EXTEND;
    }
    matrix[0][0] = 0;


    for(int i = 1; i <= m; i++) {
        for(int j = 1; j <= n; j++) {
            /*
            int up = matrix[i-1][j] + GAP;
            int left = matrix[i][j-1] + GAP;
            int diag = matrix[i-1][j-1] + ((str1[i-1]==str2[j-1])?MATCH:MISMATCH);
            matrix[i][j] = cpu_max(up, left, diag);
            */

            x_matrix[i][j] = cpu_max(
                    GAP_START+GAP_EXTEND+matrix[i][j-1],
                    GAP_EXTEND+x_matrix[i][j-1]);
                    //GAP_START+GAP_EXTEND+y_matrix[i][j-1]);
            y_matrix[i][j] = cpu_max(
                    GAP_START+GAP_EXTEND+matrix[i-1][j],
                    //GAP_START+GAP_EXTEND+x_matrix[i-1][j],
                    GAP_EXTEND+y_matrix[i-1][j]);

            short score = (str1[i-1]==str2[j-1] ? MATCH : MISMATCH) + matrix[i-1][j-1];
            matrix[i][j] = cpu_max(score, x_matrix[i][j], y_matrix[i][j]);
        }
    }

    // printMatrix(matrix, m, n);
    // 释放x_matrix和y_matrix, matrix在backtrack中释放
    for(int i = 0; i <= m; i++) {
        delete[] x_matrix[i];
        delete[] y_matrix[i];
    }
    delete[] x_matrix;
    delete[] y_matrix;

    return matrix;
}

void backtrack(short **matrix, string centerSeq, string seq, int seqIdx, short *space, short *spaceForOther, int maxLength) {
    int m = centerSeq.size();
    int n = seq.size();

    int sWidth = m + 1;
    int soWidth = maxLength + 1;

    int i = m, j = n;
    while(i!=0 || j!=0) {
        int score = (centerSeq[i-1] == seq[j-1]) ? MATCH : MISMATCH;
        if(i>0 && j>0 && score+matrix[i-1][j-1]==matrix[i][j]) {
            i--;
            j--;
        } else {
            int k = 1;
            while(true) {
                if(i>=k && matrix[i][j]==matrix[i-k][j]+GAP_START+GAP_EXTEND*k) {
                    spaceForOther[seqIdx*soWidth+j] += k;
                    i = i - k;
                    break;
                } else if(j>=k && matrix[i][j]==matrix[i][j-k]+GAP_START+GAP_EXTEND*k) {
                    space[seqIdx*sWidth+i] += k;
                    j = j - k;
                    break;
                } else {
                    k++;
                }
            }
        }

    }

}


void cpu_msa(string centerSeq, vector<string> seqs, int startIdx, short *space, short *spaceForOther, int maxLength) {

    if(startIdx >= seqs.size()) return;

    double start, end;

    // 计算DP矩阵, 执行backtrack
    omp_set_num_threads(OMP_THREADS);
    printf("OMP THREADS: %d\n", OMP_THREADS);
    #pragma omp parallel for
    for(int idx = startIdx; idx < seqs.size(); idx++) {
        short **matrix = nw(centerSeq, seqs[idx]);
        backtrack(matrix, centerSeq, seqs[idx], idx, space, spaceForOther, maxLength);

    	// 释放matrix[(m+1, n+1]内存
    	for(int i = 0; i <= centerSeq.size(); i++)
            delete[] matrix[i];
    	delete[] matrix;

        // printf("%d/%lu, sequence length:%lu\n", idx+1, seqs.size(), seqs[idx].size());
    }

}
