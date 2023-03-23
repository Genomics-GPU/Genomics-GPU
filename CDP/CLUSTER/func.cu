#include "func.h"
//--------------------普通函数--------------------//
// printUsage 输出用法
void printUsage() {
    std::cout << "请校验输入参数"  << std::endl;
    std::cout << "a.out i inputFile t threshold" << std::endl;
    exit(0);  // 退出程序
}
// checkOption 检查输入
void checkOption(int argc, char **argv, Option &option) {
    if (argc%2 != 1) printUsage();  // 参数个数不对
    option.inputFile = "testData.fasta";  // 输入文件名
    option.outputFile = "result.fasta";  // 输出文件名
    option.threshold = 0.95;  // 默认阈值
    option.wordLength = 0;  // 默认词长
    option.drop = 3;  // 0不跳，1跳，3自动
    option.pigeon = 0;  // 默认不启动鸽笼过滤
    for (int i=1; i<argc; i+=2) {  // 遍历参数
        switch (argv[i][0]) {
        case 'i':
            option.inputFile = argv[i+1];
            break;
        case 'o':
            option.outputFile = argv[i+1];
            break;
        case 't':
            option.threshold = std::stof(argv[i+1]);
            break;
        case 'w':
            option.wordLength = std::stoi(argv[i+1]);
            break;
        case 'd':
            option.drop = std::stoi(argv[i+1]);
            break;
        case 'p':
            option.pigeon = std::stoi(argv[i+1]);
            break;
		case 'g': 
			option.grid_size = std::stoi(argv[i + 1]); 
			break; 
		case 'b':
			option.buffer_size = std::stoi(argv[i + 1]); 
			break; 
        default:
            printUsage();
            break;
        }
    }
    if (option.threshold < 0.8 || option.threshold >= 1) {  // 阈值
        std::cout << "阈值超出范围" << std::endl;
        std::cout << "0.8<=阈值<1" << std::endl;
        printUsage();
    }
    if (option.wordLength == 0) {  // 词长度
        if (option.threshold<0.88) {
            option.wordLength = 4;
        } else if (option.threshold<0.94) {
            option.wordLength = 5;
        } else if (option.threshold<0.97) {
            option.wordLength = 6;
        } else {
            option.wordLength = 7;
        }
    } else {
        if (option.wordLength<4 || option.wordLength>8) {
            std::cout << "词长超出范围" << std::endl;
            std::cout << "4<=词长<=8" << std::endl;
            printUsage();
        }
    }
    if (option.drop == 3) {  // 是否跳过过滤
        if (option.threshold <= 0.879) {
            option.drop = 1;
        } else {
            option.drop = 0;
        }
    } else {
        if (option.drop != 0 && option.drop != 1) {
            std::cout << "drop错误" << std::endl;
            std::cout << "drop=0/1" << std::endl;
            printUsage();
        }
    }
    if (option.pigeon != 1 && option.pigeon != 0) {  // 鸽笼过滤
        std::cout << "pigeon错误" << std::endl;
        std::cout << "pigeon=0/1" << std::endl;
        printUsage();
    }
    std::cout << "输入文件:\t" << option.inputFile << std::endl;
    std::cout << "输出文件:\t" << option.outputFile << std::endl;
    std::cout << "相似阈值:\t" << option.threshold << std::endl;
    std::cout << "word长度:\t" << option.wordLength << std::endl;
    std::cout << "不跑过滤器\t" << option.drop << std::endl;
	std::cout << "Grid size:\t" << option.grid_size << std::endl; 
	std::cout << "Buffer size:\t" << option.buffer_size << std::endl;
}
// readFile 读文件
void readFile(std::vector<Read> &reads, Option &option) {
    std::ifstream file(option.inputFile);
    Read read;
    std::string line;
    long end = 0;  // 文件结尾指针
    long point = 0;  // 当前指针
    file.seekg(0, std::ios::end);
    end = file.tellg();
    file.seekg(0, std::ios::beg);
    while(true) {
        getline(file, line);  // 读read名
        read.name = line;
        while (true) {  // 读read数据
            point = file.tellg();
            getline(file, line);
            if (line[0] == '>') {  // 读到下一个序列
                file.seekg(point, std::ios::beg);
                reads.push_back(read);
                read.name = "";  // 清空read
                read.data = "";
                break;
            } else {  // 读到数据
                read.data += line;
            }
            point = file.tellg();
            if (point == end){  // 读到结尾
                reads.push_back(read);
                read.data = "";
                read.name = "";
                break;
            }
        }
        if (point == end) break;  // 读到结尾
    }
    file.close();
    std::sort(reads.begin(), reads.end(), [](Read &a, Read &b) {
        return a.data.size() > b.data.size();  // 从大到小
    });  // 排序
    std::cout << "读文件完成" << std::endl;
    std::cout << "最长/最短：\t" << reads[0].data.size() << "/";
    std::cout << reads[reads.size()-1].data.size() << std::endl;
    std::cout << "序列数：\t" << reads.size() << std::endl;
}
// copyData 拷贝数据
void copyData(std::vector<Read> &reads, Data &data) {
    data.readsCount = reads.size();
    int readsCount = data.readsCount;
    cudaMallocManaged(&data.lengths, readsCount*sizeof(int));
    cudaMallocManaged(&data.offsets, (readsCount+1)*sizeof(long));
    data.offsets[0] = 0;
    for (int i=0; i<readsCount; i++) {  // 填充lengths和offsets
        int length = reads[i].data.size();
        data.lengths[i] = length;
        data.offsets[i+1] = data.offsets[i]+length/32*32+32;
    }
    cudaMallocManaged(&data.reads, data.offsets[readsCount]*sizeof(char));
    for (int i=0; i<readsCount; i++) {  // 填充reads
        long start = data.offsets[i];
        int length = data.lengths[i];
        memcpy(&data.reads[start], reads[i].data.c_str(), length*sizeof(char));
    }
    cudaDeviceSynchronize();  // 同步数据
    std::cout << "拷贝数据完成" << std::endl;
}
// kernel_baseToNumber 碱基转换为数字
__global__ void kernel_baseToNumber(char *reads, long length) {
    long index = threadIdx.x+blockDim.x*blockIdx.x;
    while (index < length) {
        switch (reads[index]) {  // 实际是寄存器计算，比用数组更快
        case 'A':
            reads[index] = 0;
            break;
        case 'a':
            reads[index] = 0;
            break;
        case 'C':
            reads[index] = 1;
            break;
        case 'c':
            reads[index] = 1;
            break;
        case 'G':
            reads[index] = 2;
            break;
        case 'g':
            reads[index] = 2;
            break;
        case 'T':
            reads[index] = 3;
            break;
        case 't':
            reads[index] = 3;
            break;
        case 'U':
            reads[index] = 3;
            break;
        case 'u':
            reads[index] = 3;
            break;
        default:
            reads[index] = 4;
            break;
        }
        index += 128*128;
    }
}
// baseToNumber 碱基转换为数字
void baseToNumber(Data &data) {
    int readsCount = data.readsCount;
    long length = data.offsets[readsCount];  // 总长度
    kernel_baseToNumber<<<128, 128>>>(data.reads, length);
    cudaDeviceSynchronize();  // 同步数据
    std::cout << "碱基转数字完成" << std::endl;
}
// kernel_createPrefix 生成前置过滤
__global__ void kernel_createPrefix(int *lengths,
long *offsets, char *reads, int *prefix, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    int base[5] = {0};  // 碱基的统计结果
    long start = offsets[index];  // 起始位置
    int length = lengths[index];
    for (int i=0; i<length; i++) {
        switch(reads[start+i]) {
            case 0:
                base[0] += 1;
                break;
            case 1:
                base[1] += 1;
                break;
            case 2:
                base[2] += 1;
                break;
            case 3:
                base[3] += 1;
                break;
            case 4:
                base[4] += 1;
                break;
        }
    }
    prefix[index*4+0] = base[0];
    prefix[index*4+1] = base[1];
    prefix[index*4+2] = base[2];
    prefix[index*4+3] = base[3];
}
// createPrefix 生成前置过滤
void createPrefix(Data &data) {
    int readsCount = data.readsCount;
    cudaMallocManaged(&data.prefix, readsCount*4*sizeof(int));  // 前置过滤
    kernel_createPrefix<<<(readsCount+127)/128, 128>>>
    (data.lengths, data.offsets, data.reads, data.prefix, readsCount);
    cudaDeviceSynchronize();
    std::cout << "生成前置过滤完成" << std::endl;
}
// kernel_createWords 生成短词
__global__ void kernel_createWords(int *lengths, long *offsets, char *reads,
unsigned short *words, int *wordCounts, int readsCount, int wordLength) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    long start = offsets[index];  // 起始位置
    int length = lengths[index];
    if (length < wordLength) {  // 序列过短
        wordCounts[index] = 0;
        return;
    }
    int count = 0;  // word的个数
    for (int i=wordLength-1; i<length; i++) {
        unsigned short word = 0;
        int flag = 0;  // 是否有gap
        for (int j=0; j<wordLength; j++) {
            unsigned char base = reads[start+i-j];
            word += base<<j*2;
            if (base == 4) flag = 1;
        }
        if (flag == 0) {  // 没有gap
            words[start+count] = word;
            count += 1;
        }
    }
    wordCounts[index] = count;
}
// createWords 生成短词
void createWords(Data &data, Option &option) {
    int readsCount = data.readsCount;
    int wordLength = option.wordLength;
    int length = data.offsets[readsCount];
    cudaMallocManaged(&data.words, length*sizeof(unsigned short));  // 短词
    cudaMallocManaged(&data.wordCounts, readsCount*sizeof(int));  // 短词
    kernel_createWords<<<(readsCount+127)/128, 128>>>
    (data.lengths, data.offsets, data.reads, data.words,
    data.wordCounts, readsCount, wordLength);  // 生成短词
    cudaDeviceSynchronize();  // 同步数据
    std::cout << "生成短词完成" << std::endl;
}
// kernel_sortWords 排序短词
__global__ void kernel_sortWords(long *offsets, unsigned short *words,
int *wordCounts, int wordLength, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    long start = offsets[index];
    int wordCount = wordCounts[index];
    // 希尔排序
    for (int gap=wordCount/2; gap>0; gap/=2){  // 每次增量
        for (int i=gap; i<wordCount; i++) {
            for (int j=i-gap; j>=0; j-=gap) {
                if (words[start+j] > words[start+j+gap]) {
                    unsigned int temp = words[start+j];
                    words[start+j] = words[start+j+gap];
                    words[start+j+gap] = temp;
                } else {
                    break;
                }
            }
        }
    }
}
// sortWords 排序短词 (gpu希尔比std::sort快3倍)
void sortWords(Data &data, Option &option) {
    int readsCount = data.readsCount;
    kernel_sortWords<<<(readsCount+127)/128, 128>>>
    (data.offsets, data.words, data.wordCounts, option.wordLength, readsCount);
    cudaDeviceSynchronize();  // 同步数据
    std::cout << "排序短词完成" << std::endl;
}
// kernel_mergeIndex 合并相同短词
__global__ void kernel_mergeWords(long *offsets, unsigned short *words,
int *wordCounts, unsigned short *orders, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    long start = offsets[index];
    int wordCount = wordCounts[index];
    unsigned int preWord = words[start];
    unsigned int current;
    unsigned short count = 0;
    for (int i=0; i<wordCount; i++) {  // 合并相同的index，orders为相同个数
        current = words[start+i];
        if (preWord == current) {
            count += 1;
            orders[start+i] = 0;
        } else {
            preWord = current;
            orders[start+i] = 0;
            orders[start+i-1] = count;
            count = 1;
        }
    }
    orders[start+wordCount-1] = count;
}
// mergeWords 合并相同短词
void mergeWords(Data &data) {
    int readsCount = data.readsCount;
    long length = data.offsets[readsCount];
    cudaMallocManaged(&data.orders, length*sizeof(unsigned short));
    kernel_mergeWords<<<(readsCount+127)/128, 128>>>
    (data.offsets, data.words, data.wordCounts, data.orders, readsCount);
    cudaDeviceSynchronize();  // 同步数据
    std::cout << "合并短词完成" << std::endl;
}
// kernel_createCutoff 生成阈值
__global__ void kernel_createCutoff(int *lengths, int *wordCutoff,
int *baseCutoff, float threshold, int wordLength, int readsCount) {
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= readsCount) return;  // 超出范围
    // 生成wordCutoff
    int length = lengths[index];
    int required = length - wordLength + 1;
    int cutoff = ceil((float)length*(1.0-threshold))*wordLength;
    required -= cutoff;
    required = max(required, 1);
    float offset = 0;  // 加快计算的系数
    if (threshold >= 0.9) {
        offset = 1.1-fabs(threshold-0.95)*2;
    } else {
        offset = 1;
    }
    // offset = 1;
    required = ceil((float)required*offset);
    wordCutoff[index] = required;
    // 生成baseCutoff
    required = ceil((float)length*threshold);
    baseCutoff[index] = required;

}
// createCutoff 生成阈值
void createCutoff(Data &data, Option &option) {
    int readsCount = data.readsCount;
    cudaMallocManaged(&data.wordCutoff, readsCount*sizeof(int));  // word阈值
    cudaMallocManaged(&data.baseCutoff, readsCount*sizeof(int));  // base阈值
    kernel_createCutoff<<<(readsCount+127)/128, 128>>>
    (data.lengths, data.wordCutoff, data.baseCutoff,
    option.threshold, option.wordLength, readsCount);
    cudaDeviceSynchronize();  // 同步数据
    std::cout << "生成阈值完成" << std::endl;
}
// kernel_deleteGap 去除gap
__global__ void kernel_deleteGap(int *lengths, long *offsets,
char *reads, int *gaps, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    long start = offsets[index];
    int length = lengths[index];
    int count = 0;
    int gap = 0;
    for (int i=0; i<length; i++) {
        char base = reads[start+i];
        if (base < 4) {  // 正常碱基就正常复制
            reads[start+count] = base;
            count += 1;
        } else {  // 遇到gap就跳过
            gap += 1;
        }
    }
    gaps[index] = gap;
}
// deleteGap 去除gap
void deleteGap(Data &data) {
    int readsCount = data.readsCount;
    cudaMallocManaged(&data.gaps, readsCount*sizeof(int));
    kernel_deleteGap<<<(readsCount+127)/128, 128>>>
    (data.lengths, data.offsets, data.reads, data.gaps, readsCount);
    cudaDeviceSynchronize();  // 同步数据
    std::cout << "去除gap完成" << std::endl;
}
//        A C G T
// low:   0 1 0 1
// hight: 0 0 1 1
// kernel_compresseData 压缩数据
__global__ void kernel_compressData(int *lengths, long *offsets,
char *reads, int *gaps, unsigned int *compressed, int readsCount) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= readsCount) return;  // 超出范围
    long readStart = offsets[index];
    long compressStart = readStart/16;
    int length = lengths[index] - gaps[index];
    length = length/32+1;
    for (int i=0; i<length; i++) {
        unsigned int low = 0;
        unsigned int hight = 0;
        for (int j=0; j<32; j++) {
            char base = reads[readStart+i*32+j];
            switch (base) {
                case 1:
                    low += 1<<j;
                    break;
                case 2:
                    hight += 1<<j;
                    break;
                case 3:
                    low += 1<<j;
                    hight += 1<<j;
                    break;
                default:
                    break;
            }
        }
        compressed[compressStart+i*2+0] = low;
        compressed[compressStart+i*2+1] = hight;
    }
}
// compressData 压缩数据
void compressData(Data &data) {
    int readsCount = data.readsCount;
    long length = data.offsets[readsCount];
    cudaMallocManaged(&data.compressed, length/16*sizeof(unsigned int));
    kernel_compressData<<<(readsCount+127)/128, 128>>>(data.lengths,
    data.offsets, data.reads, data.gaps, data.compressed, readsCount);
    cudaDeviceSynchronize();  // 同步数据
    std::cout << "压缩数据完成" << std::endl;
}
//--------------------聚类过程--------------------//
// initBench 初始化bench
__host__ void initBench(Bench &bench, int readsCount) {
    cudaMallocManaged(&bench.table, (1<<2*8)*sizeof(unsigned short));  // table
    memset(bench.table, 0, (1<<2*8)*sizeof(unsigned short));  // 清0
    cudaMallocManaged(&bench.cluster, readsCount*sizeof(int));  // 聚类结果
    for (int i=0; i<readsCount; i++) {  // 赋初值 -1：未聚类 其他：代表序列
        bench.cluster[i] = -1;
    }
    cudaMallocManaged(&bench.remainList, readsCount*sizeof(int));  // 未聚类列表
    for (int i=0; i<readsCount; i++) {  // 赋初值
        bench.remainList[i] = i;
    }
    bench.remainCount = readsCount;  // 未聚类序列个数
    cudaMallocManaged(&bench.jobList, readsCount*sizeof(int));  // 需要计算的序列
    for (int i=0; i<readsCount; i++) {  // 赋初值
        bench.jobList[i] = i;
    }
    bench.jobCount = readsCount;  // 需要计算序列个数
    bench.representative = -1;
    cudaDeviceSynchronize();  // 同步数据
}
// updateRepresentative 更新代表序列
__device__ void updateRepresentative(int *cluster, int &representative, int readsCount) {
    representative += 1;
    while (representative < readsCount) {
        if (cluster[representative] == -1) {  // 遇到代表序列了
            cluster[representative] = representative;
            break;
        } else {
            representative += 1;
        }
    }
}
__global__ void kernel_updateRepresentative(int *cluster,
int *representative, int readsCount) {
    *representative += 1;
    while (*representative < readsCount) {
        if (cluster[*representative] == -1) {  // 遇到代表序列了
            cluster[*representative] = *representative;
            break;
        } else {
            *representative += 1;
        }
    }
}
// undateRemain 更新未聚类列表
__device__ void updateRemain(int *cluster, int *remainList, int &remainCount) {
    int count = 0;
    for (int i=0; i<remainCount; i++) {
        int index = remainList[i];
        if (cluster[index] == -1) {
            remainList[count] = index;
            count += 1;
        }
    }
    remainCount = count;
}
// jobList -1被过滤掉了，其他表示需要计算的序列
// undateJobs 更新任务列表
__device__ void updatJobs(int *jobList, int &jobCount) {
    int count = 0;
    for (int i=0; i<jobCount; i++) {
        int value = jobList[i];
        if (value >= 0) {
            jobList[count] = value;
            count += 1;
        }
    }
    jobCount = count;
}
// kernel_makeTable 生成table
__global__ void kernel_makeTable(long *offsets,
unsigned short *words, int *wordCounts, unsigned short *orders,
unsigned short *table, int representative) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    long start = offsets[representative];
    int length = wordCounts[representative];
    for (int i=index; i<length; i+=128*128) {
        unsigned short word = words[start+i];
        unsigned short order = orders[start+i];
        if (order > 0) table[word] = order;  // 写标记
    }
}
// kernel_cleanTable 清零table
__global__ void kernel_cleanTable(long *offsets, unsigned short *words,
int* wordCounts, unsigned short *orders,
unsigned short *table, int representative) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    long start = offsets[representative];
    int length = wordCounts[representative];
    for (int i=index; i<length; i+=128*128) {
        unsigned short word = words[start+i];
        unsigned short order = orders[start+i];
        if (order > 0) table[word] = 0;  // 清零
    }
}
// kernel_preFilter 前置过滤
__global__ void kernel_preFilter(int *prefix, int *baseCutoff,
int *jobList, int jobCount, int representative) {
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= jobCount) return;  // 超出范围
    int text = representative;  // 实际的模式序列
    int query = jobList[index];  // 实际的查询序列
    int offsetOne = text*4;  // 模式序列起始位置
    int offsetTwo = query*4;  // 查询序列起始位置
    int sum = 0;
    sum += min(prefix[offsetOne+0], prefix[offsetTwo+0]);
    sum += min(prefix[offsetOne+1], prefix[offsetTwo+1]);
    sum += min(prefix[offsetOne+2], prefix[offsetTwo+2]);
    sum += min(prefix[offsetOne+3], prefix[offsetTwo+3]);
    int cutoff = baseCutoff[query];
    if (sum < cutoff) {  // 没通过过滤
        jobList[index] = -1;
    }
}
// kernel_filter 过滤
__global__ void kernel_filter(long *offsets, unsigned short *words,
int *wordCounts, unsigned short *orders, int *wordCutoff,
unsigned short *table, int *jobList, int jobCount) {
    if (blockIdx.x >= jobCount) return;  // 超出范围
    int query = jobList[blockIdx.x];  // 查询序列的编号
    __shared__ int result[128];  // 每个线程的结果
    result[threadIdx.x] = 0;  // 清零
    long start = offsets[query];
    int length = wordCounts[query];
    for (int i=threadIdx.x; i<length; i+=128) {
        unsigned short value = words[start+i];  // word的值
        result[threadIdx.x] += min(table[value], orders[start+i]);
    }
    __syncthreads();  // 同步一下
    for (int i=128/2; i>0; i/=2) {  // 规约
        if (threadIdx.x>=i) return;  // 超出范围
        result[threadIdx.x] += result[threadIdx.x+i];
        __syncthreads();
    }
    int cutoff = wordCutoff[query];
    if (result[0] < cutoff) {  // 没通过过滤
        jobList[blockIdx.x] = -1;
    }
}
// kernel_dynamic 动态规划
__global__ void kernel_dynamic(int *lengths, long *offsets, int *gaps,
unsigned int *compressed, int *baseCutoff, int *cluster, int *jobList,
int jobCount, int representative, int readsCount, int buffer_size) {
    //----准备数据----//
    __shared__ unsigned int bases[2048];  // 65536/32
    int text = representative;
    long textStart = offsets[text]/16;
    int textLength = lengths[text]-gaps[text];
    for (int i=threadIdx.x; i<textLength/32+1; i+=blockDim.x) {  // 拷贝数据
        bases[i*2+0] = compressed[textStart+i*2+0];
        bases[i*2+1] = compressed[textStart+i*2+1];
    }
    //----开始计算----//
    // 批量计算阶段
    int index = threadIdx.x+blockDim.x*blockIdx.x;
    if (index >= jobCount) return;  // 超出范围
    unsigned int line[2048] = {0xFFFFFFFF};  // 保存结果
    for (int i=0; i<2048; i++) {
        line[i] = 0xFFFFFFFF;
    }
    int query = jobList[index];
    long queryStart = offsets[query] / 16;
    int queryLength = lengths[query] - gaps[query];
    for (int i=0; i<queryLength/32; i++) {  // 遍历query
        unsigned int column[32] = {0};
        unsigned int queryLow = compressed[queryStart+i*2+0];
        unsigned int queryHight = compressed[queryStart+i*2+1];
        for (int j=0; j<textLength/32+1; j++) {  // 遍历text
            unsigned int textl = bases[j*2+0];
            unsigned int texth = bases[j*2+1];
            unsigned int row = line[j];
            for (int k=0; k<32; k++) {  // 32*32的核心
                unsigned int queryl = 0x00000000;
                if (queryLow>>k&1) queryl = 0xFFFFFFFF;
                unsigned int queryh = 0x00000000;
                if (queryHight>>k&1) queryh = 0xFFFFFFFF;
                unsigned int temp1 = textl ^ queryl;
                unsigned int temp2 = texth ^ queryh;
                unsigned int match = (~temp1)&(~temp2);
                unsigned int unmatch = ~match;
                unsigned int temp3 = row & match;
                unsigned int temp4 = row & unmatch;
                unsigned int carry = column[k];
                unsigned int temp5 = row + carry;
                unsigned int carry1 = temp5 < row;
                temp5 += temp3;
                unsigned int carry2 = temp5 < temp3;
                carry = carry1 | carry2;
                row = temp5 | temp4;
                column[k] = carry;  //
            }
            line[j] = row;
        }
    }
    // query补齐阶段
    unsigned int column[32] = {0};
    unsigned int queryLow = compressed[queryStart+(queryLength/32)*2+0];
    unsigned int queryHight = compressed[queryStart+(queryLength/32)*2+1];
    for (int j=0; j<textLength/32+1; j++) {  // 遍历text
        unsigned int textl = bases[j*2+0];
        unsigned int texth = bases[j*2+1];
        unsigned int row = line[j];
        for (int k=0; k<queryLength%32; k++) {  // 32*32的核心
            unsigned int queryl = 0x00000000;
            if (queryLow>>k&1) queryl = 0xFFFFFFFF;
            unsigned int queryh = 0x00000000;
            if (queryHight>>k&1) queryh = 0xFFFFFFFF;
            unsigned int temp1 = textl ^ queryl;
            unsigned int temp2 = texth ^ queryh;
            unsigned int match = (~temp1)&(~temp2);
            unsigned int unmatch = ~match;
            unsigned int temp3 = row & match;
            unsigned int temp4 = row & unmatch;
            unsigned int carry = column[k];
            unsigned int temp5 = row + carry;
            unsigned int carry1 = temp5 < row;
            temp5 += temp3;
            unsigned int carry2 = temp5 < temp3;
            carry = carry1 | carry2;
            row = temp5 | temp4;
            column[k] = carry;
        }
        line[j] = row;
    }
    //----统计结果----//
    int sum = 0;
    unsigned int result;
    for (int i=0; i<textLength/32; i++) {
        result = line[i];
        for (int j=0; j<32; j++) {
            sum += result>>j&1^1;
        }
    }
    result = line[textLength/32];
    for (int i=0; i<textLength%32; i++) {
        sum += result>>i&1^1;
    }
    int cutoff = baseCutoff[query];
    if (sum > cutoff) {
		// We want to place a number in this spot. The first value in cluster is the length, so we should increment it (we're adding 1 more item). 
		int *spot = &(cluster[query * buffer_size]);
		unsigned int length = atomicInc((unsigned int*)spot, buffer_size);
		if (length == buffer_size) {
			printf("Error: buffer reached max capacity. Increase buffer size (b) or decrease grid size (g).\n");
			__trap(); 
		}
		spot[length + 1] = text;
    } else {
        jobList[index] = -1;
    }
}

__global__ void kernel_iteration(Data data, Bench& bench, int readsCount, int index, const int grid_size, const int buffer_size, int* cluster, unsigned short *tables, int *job_lists) {
	index = index * grid_size + blockIdx.x;
	if (index >= readsCount) return; 
	
	// Create kernel-specific items.
	unsigned short *table = &(tables[blockIdx.x * (1 << 2 * 8)]);
	
	int *jobList = &(job_lists[blockIdx.x * readsCount]); 
	int jobCount = readsCount - index;
	for (int i = 0; i < jobCount; i++) 
		jobList[i] = i + index;
	
	//----Filtering work----//
	// Generate table ok
	kernel_makeTable << < 128, 128 >>> (data.offsets, data.words,
		data.wordCounts, data.orders, table, index);
	
	kernel_preFilter << < (jobCount + 127) / 128, 128 >>>
		(data.prefix, data.baseCutoff, jobList,
			jobCount, index);
	cudaDeviceSynchronize(); // synchronize data
	
	updatJobs(jobList, jobCount); // update job ok
	if (jobCount > 0) { // standard filter ok
		kernel_filter << < jobCount, 128 >>>
			(data.offsets, data.words, data.wordCounts, data.orders,
				data.wordCutoff, table, jobList, jobCount);
		cudaDeviceSynchronize(); // synchronize data
	}
	
	//----Comparison work----//
	updatJobs(jobList, jobCount); // update job ok
	if (jobCount > 0) { // dynamic programming ok
		kernel_dynamic << < (jobCount + 127) / 128, 128 >>> (data.lengths,
			data.offsets, data.gaps, data.compressed, data.baseCutoff,
			cluster, jobList, jobCount, index, readsCount, buffer_size);
	}
	
	//----Finishing work----//
	// clear table oka
	kernel_cleanTable << < 128, 128 >>> (data.offsets, data.words,
		data.wordCounts, data.orders, table, index);
}

// Returns the smallest valid value in the buffer at the index, or readsCount if it doesn't exist. 
__device__ int get_smallest(const int buffer_size, const int readsCount, int* valid, int* cluster, int index) {
	int *ptr = &(cluster[index * buffer_size]);
	int length = ptr[0];
	
	// Choose the smallest valid item from the list. 
	int smallest = readsCount;
	for (int i = 1; i <= length; i++)
		if (ptr[i] < smallest && valid[ptr[i]])
			smallest = ptr[i];
	
	return smallest; 
}

// Cluster is offset by the start point. 
__global__ void kernel_set_valid_parallel(const int limit, const int buffer_size, const int readsCount, int* valid, int* cluster) {
	int index = blockIdx.x * blockDim.x + threadIdx.x; 
	if (index >= limit) return; 
	
	int *ptr = &(cluster[index * buffer_size]); 
	int smallest = get_smallest(buffer_size, readsCount, valid, cluster, index); 
	if (smallest != readsCount) {
		ptr[0] = 1; // length = 1 
		ptr[1] = smallest; 
	}
	else ptr[0] = 0; // length = 0 
}

__global__ void kernel_set_valid_serial(const int readsCount, const int grid_size, const int buffer_size, const int max_thread_count, int* valid, int* cluster, int iteration) {
	int start = iteration * grid_size; 
	int end = min(readsCount, start + grid_size);

	for (int read = start; read < end; read++) {
		int smallest = get_smallest(buffer_size, readsCount, valid, cluster, read);
				
		int index; 
		if (smallest == readsCount) {
			index = read; 
			valid[index] = 1; 
		}
		else index = smallest;

		// Note: cluster is normally offset by the buffer size, 
		// but we ignore the offset here to save time later on. 
		cluster[read] = index;
	}
	
	if (end != readsCount) {
		int *offset_cluster = &(cluster[end * buffer_size]); 
		int limit = readsCount - end; 
		int threads = min(max_thread_count, limit); 
		int blocks = (int)ceilf((float)limit / threads); 
		kernel_set_valid_parallel<<<threads, blocks>>>(limit, buffer_size, readsCount, valid, offset_cluster); 
	}
}

// 聚类
void clustering(Option &option, Data &data, Bench &bench) {
	int readsCount = data.readsCount;
	initBench(bench, readsCount); // Initialize bench ok
	std::cout << "Representative sequence/total sequence count:" << std::endl;
	
	int buffer_size = option.buffer_size; 
	
	int *d_cluster;
	cudaMalloc(&d_cluster, sizeof(int) * readsCount * buffer_size);
	cudaMemset(d_cluster, 0, sizeof(int) * readsCount * buffer_size); 
	
	int grid_size = option.grid_size;
	
	unsigned short *d_tables; 
	cudaMalloc(&d_tables, sizeof(unsigned short) * grid_size * (1 << 2 * 8)); 
	
	int *d_job_lists; 
	cudaMalloc(&d_job_lists, sizeof(int) * readsCount * grid_size); 
	
	int *d_valid; 
	cudaMalloc(&d_valid, sizeof(int) * readsCount); 
	cudaMemset(d_valid, 0, sizeof(int) * readsCount);
	
	int iters = (int)ceil((double)readsCount / grid_size); 
	for (int i = 0; i < iters; i++) {
		kernel_iteration<<<grid_size,1,1>>>(data, bench, readsCount, i, grid_size, buffer_size, d_cluster, d_tables, d_job_lists);
		kernel_set_valid_serial<<<1,1>>>(readsCount, grid_size, buffer_size, 1024, d_valid, d_cluster, i); 
	}
	
	bench.cluster = (int*)malloc(sizeof(int) * readsCount); 
	cudaMemcpy(bench.cluster, d_cluster, sizeof(int) * readsCount, cudaMemcpyDeviceToHost);
	checkError(); 
}
//--------------------收尾函数--------------------//
// saveFile 保存结果
void saveFile(Option &option, std::vector<Read> &reads, Bench &bench) {
    std::ofstream file(option.outputFile);
    int sum = 0;
    for (int i=0; i<reads.size(); i++) {
        if (bench.cluster[i] == i) {
            file << reads[i].name << std::endl;
            file << reads[i].data << std::endl;
            sum++;
        }
    }
    file.close();
    std::cout << "聚类：" << sum << "个" << std::endl;
}
// 检查显卡错误
void checkError() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }
}
