#include "func.h"  // 数据结构与函数
#include "timer.h"  // timer

int main(int argc, char **argv) {
  Timer timer;
  timer.start();
  // 声明数据
  Option option;  // 选项
  Data data;  // 数据
  Bench bench;  // 聚类的工作台
  // 计算
  initialization(argc, argv, option);  // 初始化
  readData(option, data); // 读序列数据
  clustering(option, data, bench);  // 聚类
  // long sum = 0;
  // for (long i=0; i<option.readsWhole; i++) {
  //   if (bench.cluster[i] == i) sum += 1;
  // }
  // std::cout << sum << std::endl;
  finish(option, data, bench);  // 收尾
  timer.getDuration();  // 计时
}
