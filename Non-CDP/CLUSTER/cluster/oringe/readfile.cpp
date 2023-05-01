// 读文件，并排序
// 原始文件是fasta文件，新文件分为两个
//   一个文件存储序列的名字
//   一个存储序列的数据
// 这些文件都是按照顺序存储的，从长到短
// ./a.out 输入文件 序列名文件 序列数据文件
#include <iostream>  // std::cout
#include <fstream>  // fstream
#include <string>  // string
#include <vector>  // vector
#include <algorithm>  // sort
#include "timer.h"  // 计时器
#include "cmdline.h"  // 参数解析器
//--------定义数据--------//
struct Read {  // 记录一条序列
  std::string name;  // 序列的名字
  std::string data;  // 序列的数据
};
struct Data{  // 各种数据
  std::string inputFile;  // 输入文件
  std::string nameFile;  // 序列名文件
  std::string dataFile;  // 序列数据文件
  std::vector<Read> reads;  // 数据节点
} dt;
Timer timer;  // 计时器
//--------定义函数--------//
// 去除gap并转为数字
void deGap(std::string &line) {
  int index = 0;
  for (int i=0; i<line.size(); i++) {
    switch (line[i]) {
      case 'A':
        line[index] = 0;
        index += 1;
        break;
      case 'a':
        line[index] = 0;
        index += 1;
        break;
      case 'C':
        line[index] = 1;
        index += 1;
        break;
      case 'c':
        line[index] = 1;
        index += 1;
        break;
      case 'G':
        line[index] = 2;
        index += 1;
        break;
      case 'g':
        line[index] = 2;
        index += 1;
        break;
      case 'T':
        line[index] = 3;
        index += 1;
        break;
      case 't':
        line[index] = 3;
        index += 1;
        break;
      default:
        break;
    }
  }
  line.resize(index);
}
// readFile 读文件
void readFile() {
  timer.start();  // 开始计时
  std::ifstream file(dt.inputFile);
  std::string line;  // 临时变量
  long point=0, end=0;  // 文件指针
  file.seekg(0, std::ios::end);
  end = file.tellg();  // 获取文件长度
  file.seekg(0, std::ios::beg);
  Read read;  // 临时变量
  getline(file, line);  // 读第一行
  read.name = line;
  read.data = "";
  while(getline(file, line)) {
    if (line[0] == '>') {  // 读到序列名
      deGap(read.data);
      if (read.data.size() <= 65536) dt.reads.push_back(read);  // 入队
      read.name = line;
      read.data = "";
      if (dt.reads.size()%(50*1000) == 0) {  // 打印进度
        point = file.tellg();
        std::cout << "\rRead " << int(1.0f*point/end*100) << "%";
        std::cout.flush();  // 刷新 避免无输出
      }
    } else {
      read.data += line;
    }
  }
  deGap(read.data);
  dt.reads.push_back(read);
  std::cout << "\rRead 100%\n";  // 进度完成
  std::cout.flush();
  file.close();
  std::cout << "Read:\t\t\t" << dt.reads.size() << " sequences." << std::endl;
  std::cout << "Read:\t\t\t"; timer.getDuration();  // 统计耗时
}
// sortReads排序
void sortReads() {
  timer.start();  // 开始计时
  std::sort(dt.reads.begin(), dt.reads.end(), [](Read &a, Read &b) {
    return a.data.size() > b.data.size();
  });
  std::cout << "Sort:\t\t\t"; timer.getDuration();  // 统计耗时
}
// storeResult 结果写入文件
void storeResult() {
  timer.start();  // 开始计时
  std::ofstream nameFile(dt.nameFile);  // 序列名
  std::ofstream dataFile(dt.dataFile);  // 序列内容
  for (int i=0; i<dt.reads.size(); i++) {
    nameFile << dt.reads[i].name << "\n";  // \n比std::endl快狠多
    dataFile << dt.reads[i].data << "\n";
    if (i%(50*1000) == 0) {  // 打印进度
      std::cout << "\rStore " << int(1.0f*i/dt.reads.size()*100) << "%";
      std::cout.flush();  // 刷新 避免无输出
    }
  }
  std::cout << "\rStore 100%\n";  // 进度完成
  std::cout.flush();
  nameFile.close();
  dataFile.close();
  std::cout << "Store:\t\t\t"; timer.getDuration();  // 统计耗时
}
//--------主函数--------//
int main(int argc, char **argv) {
  // 解析参数
  cmdline::parser parser;
  parser.add<std::string>("input", 'i', "input file", true, "");
  parser.add<std::string>("name", 'n', "name file", true, "");
  parser.add<std::string>("data", 'd', "data file", true, "");
  parser.parse_check(argc, argv);
  dt.inputFile = parser.get<std::string>("input");
  dt.nameFile = parser.get<std::string>("name");
  dt.dataFile = parser.get<std::string>("data");
  // 计算
  std::cout << "Read data start:\t"; timer.getTimeNow();  // 开始时间戳
  readFile();  // 读文件
  sortReads();  // 排序
  storeResult();  // 写回文件
  std::cout << "Read data finish:\t"; timer.getTimeNow();  // 结束时间戳
}
