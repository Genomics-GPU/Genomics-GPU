#include "global.h"

int BLOCKS = 12;
int THREADS = 256;

double WORKLOAD_RATIO = 1;  // 默认GPU/CPU任务比例1:1，即各自负责一半的串

int MODE = CPU_GPU;         // 默认同时使用GPU和CPU

int GPU_NUM = 0;            // 由程序自动读取GPU数量，用户也可以通过参数指定

int OMP_THREADS = 2;
