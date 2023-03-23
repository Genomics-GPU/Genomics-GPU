#include <iostream>  // cout
#include <fstream>  // ifstream
#include <algorithm>  // min max
// #include <cstring>  // stoi
#include "cmdline.h"  // cmdline
#include "timer.h"  // timer
#include "func.h"  // 数据结构

int main(int argc, char **argv) {
  // 计时
  Timer timer;
  timer.start();
  // 解析参数
  cmdline::parser parser;  // 解析器
  parser.add<std::string>("input", 'i', "input name", true, "");  // 输入文件
  parser.add<int>("word", 'w', "word length", false, 7, cmdline::range(4, 8));  // 短词长度
  parser.add<float>("threshold", 't', "similarity threshold", false, 0.95, cmdline::range(0.8, 0.99));  // 相似度阈值
  parser.parse_check(argc, argv);  // 解析
  Option option;
  option.inputFile = parser.get<std::string>("input");
  option.wordLength = parser.get<int>("word");
  option.threshold = parser.get<float>("threshold");
  // 读取序列个数
  std::string line;
  std::ifstream file(option.inputFile);
  getline(file, line);  // 读取序列个数
  unsigned long readsCount = stoi(line);  // 序列总个数
  // 初始化数据
  Data data;
  initData(readsCount, data);
  // 处理数据
  unsigned long represent = 0;  // 有多少个代表序列
  for (unsigned long i=0; i<readsCount; i++) {  // 遍历序列
    // 主机端处理数据
    std::string line;
    getline(file, line);  // 读一条序列
    packing(data, line);  // 打包数据
    // 打包数据
    cudaMemcpy(data.reads+data.offset[0], data.read, sizeof(unsigned int)*data.length[0], cudaMemcpyHostToDevice);  // 拷贝序列数据
    cudaMemcpy(data.lengths+represent, data.length, sizeof(unsigned int), cudaMemcpyHostToDevice);  // 拷贝序列长度
    cudaMemcpy(data.offsets+represent, data.offset, sizeof(unsigned long), cudaMemcpyHostToDevice);  // 拷贝序列偏移
    data.flag[0] = 0;  // 标记复位
    // 开始计算
    if (represent > 0) {  // 第一条序列无需比对
      cudaDeviceSynchronize();  // 同步
      dynamic(data, option, represent);  // 动态规划比对
    }
    // 根据结果处理数据
    cudaDeviceSynchronize();  // 同步
    if (!data.flag[0] && represent<4500) {  // 是新的代表序列
      represent += 1;
      unsigned int temp = (line.size()+31)/32*2;
      temp = (temp+data.alignSize-1)/data.alignSize*data.alignSize;
      data.offset[0] += temp;  // 更新偏移
    }
    std::cout << "\r" << i+1 << "/" << readsCount << " " << represent;
    std::cout.flush();
  }
  std::cout << "\n";
  std::cout << "cluster: " << represent << "\n";
  // 收尾
  checkErr();
  file.close();
  timer.getDuration();  // 总耗时
  // std::cout << datah.clusters.size() << "\n";
}
