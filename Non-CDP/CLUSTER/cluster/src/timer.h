#ifndef __TIMER__  // 防御声明 pragma once 通用性有问题
#define __TIMER__
// chrono库实现的计时器
#include <iostream>  // cout
#include <chrono>  // system_clock
#include <ctime>  // time_t
//--------类声明--------//
class Timer {  // 计时器
  public:
    void getTimeNow();
    void start();
    void pause();
    void resume();
    void getDuration();
  private:
    std::chrono::system_clock::time_point now;  // 时间戳
    std::time_t time_now;  // 时间戳的格式化版
    std::chrono::steady_clock::time_point t1;  // 计时器开始
    std::chrono::steady_clock::time_point t2;  // 计时器结束
    std::chrono::duration<double> duration;  // 耗时
    int p;  // 是否暂停了
};
//--------方法--------//
void Timer::getTimeNow() {  // 输出当前时间戳
  now = std::chrono::system_clock::now();
  time_now = std::chrono::system_clock::to_time_t(now);
  std::cout << ctime(&time_now);
}
void Timer::start() {  // 开始计时
  t1 = std::chrono::steady_clock::now();
  duration = std::chrono::duration<double>(0);
  p = 0;
}
void Timer::pause() {  // 暂停计时
  t2 = std::chrono::steady_clock::now();
  duration += std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  p = 1;
}
void Timer::resume() {  // 恢复计时
  t1 = std::chrono::steady_clock::now();
  p = 0;
}
void Timer::getDuration() {  // 输出耗时
  if (p == 0) pause();  // 没暂停需要先暂停
  std::cout << duration.count() << " seconds." << std::endl;
}
#endif
