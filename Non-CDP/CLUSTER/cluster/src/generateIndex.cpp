// 读文件，并排序，然后生成name.txt data.txt index.txt
// name.txt 只保存序列名
// data.txt 只保存序列数据
// index.txt 序列个数 name.txt中序列名起始地址 data.txt中序列数据起始地址
// ./a.out xx.fasta
// 这部分不是重点，所以完成功能就行，不用管效率，反而代码越简单越好。
#include <iostream>  // cout
#include <fstream>  // fstream
#include <string>  // string
#include <vector>  // vector
#include <algorithm>  // sort
#include "timer.h"  // 计时器
#include "cmdline.h"  // 解析器
//--------数据--------//
struct Read {  // 记录一条序列的位置
  unsigned long nameOffset;  // 序列名的起始位置
  unsigned long dataOffset;  // 序列数据的起始位置
  unsigned short length;  // 序列数据长度 最长65536
};
//--------函数--------//
// readFile 读文件
void readFile(std::string &inputFile) {
  // 数据结构
  std::vector<Read> reads;  // 存储每条序列的长度和偏移
  std::string line;
  // 第一遍读文件索引
  std::ifstream file(inputFile);
  while(true) {  // 读到跳出为止
    // 读序列名
    unsigned long nameOffset = file.tellg();  // 序列名起始位置
    getline(file, line);  // 读序列名
    unsigned long dataOffset = file.tellg();  // 序列数据起始位置
    unsigned long length = 0;  // 序列长度清零
    // 读序列数据
    while (true) {
      getline(file, line);
      if (line[0] == '>') {  // 读到下一行了
        unsigned long point = file.tellg();
        point -= line.size();
        point -= 1;  // 还有换行符
        file.seekg(point, std::ios::beg);  // 退回上一行
        break;
      }
      length += line.size();
      if (file.peek() == EOF) break;  // 读到了结尾
    }
    // 写入节点
    if (length>10 && length < 65536) {  // 序列长度范围11-65535
      Read read;  // 加入新节点
      read.nameOffset = nameOffset;
      read.dataOffset = dataOffset;
      read.length = length;
      reads.push_back(read);
    }
    if (file.peek() == EOF)  break;  // 读到结尾跳出
  }
  file.close();
  // 排序
  std::sort(reads.begin(), reads.end(), [](Read &a, Read &b) {
    return a.length > b.length;
  });
  // 第二遍 结果写入文件
  file.open(inputFile);
  std::ofstream nameFile("name.txt");  //存储序列名
  std::ofstream dataFile("data.txt");  //存储序列数据
  std::vector<unsigned long> nameIndex;  // 序列名索引
  std::vector<unsigned long> dataIndex;  // 序列数据索引
  std::vector<unsigned short> dataLength;  // 序列长度索引
  unsigned long readsCount = 0;  // 序列条数
  std::string multiLine;  // 多行组成的序列数据
  for (int i=0; i<reads.size(); i++) {  // 遍历序列
    file.seekg(reads[i].dataOffset, std::ios::beg);  // 从头开始读数据
    multiLine.clear();  // 用前先清空
    while(true) {  // 把多行数据读成一行
      getline(file, line);
      if (line[0] == '>') break;  // 读到下一行了
      multiLine += line;
      if (file.peek() == EOF) break;  // 读到尾行了
    }
    dataIndex.push_back(dataFile.tellp());  // 序列数据索引
    dataLength.push_back(multiLine.size());  // 序列数据长度
    dataFile << multiLine << "\n";
    file.seekg(reads[i].nameOffset, std::ios::beg);  // 移动到序列名
    getline(file, line);
    nameIndex.push_back(nameFile.tellp());  // 序列名索引
    nameFile << line << "\n";
    readsCount += 1;  // 序列数加一
  }
  file.close();
  nameFile.close();
  dataFile.close();
  // 写入结果文件的索引
  std::ofstream indexFile("index.txt");  //存储索引
  indexFile << readsCount << std::endl;  // 序列个数
  for (int i=0; i<readsCount; i++) {  // 序列名索引
    indexFile << nameIndex[i] << '\n';
  }
  for (int i=0; i<readsCount; i++) {  // 序列数据索引
    indexFile << dataIndex[i] << '\n';
  }
  for (int i=0; i<readsCount; i++) {  // 序列长度索引
    indexFile << dataLength[i] << '\n';
  }
  indexFile.close();
}
//--------主函数--------//
int main(int argc, char **argv) {
  // 输入文件名
  cmdline::parser parser;  // 解析器
  parser.add<std::string>("input", 'i', "input file", true, "");
  parser.parse_check(argc, argv);
  std::string inputFile = parser.get<std::string>("input");
  // 计算
  Timer timer;
  std::cout << "Read data start:\t"; timer.getTimeNow();  // 开始时间戳
  timer.start();  // 开始计时
  readFile(inputFile);  // 读文件
  std::cout << "Read data finish:\t"; timer.getTimeNow();  // 结束时间戳
  std::cout << "read file total time:\t";
  timer.getDuration();  // 耗时
}
