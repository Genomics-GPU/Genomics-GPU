#include "func.h"
void printUsage() {
    std::cout << "use like this: cluster i inputFile t threshold" << std::endl;
    exit(0);
}
void checkOption(int argc, char **argv, Option &option) {
    if (argc%2 != 1) printUsage();
    option.inputFile = "testData.fasta";
    option.outputFile = "result.fasta";
    option.threshold = 0.95;
    option.wordLength = 0;
    for (int i=1; i<argc; i+=2) {
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
        default:
            printUsage();
            break;
        }
    }
    if (option.threshold < 0.8 || option.threshold >= 1) {
        std::cout << "similarity out of range" << std::endl;
        std::cout << "0.8<=similarity<1" << std::endl;
        printUsage();
    }
    if (option.wordLength == 0) {
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
            std::cout << "word length out of range" << std::endl;
            std::cout << "4<=word length<=8" << std::endl;
            printUsage();
        }
    }
    std::cout << "input file:\t" << option.inputFile << std::endl;
    std::cout << "output file:\t" << option.outputFile << std::endl;
    std::cout << "similarity:\t" << option.threshold << std::endl;
    std::cout << "word length:\t" << option.wordLength << std::endl;
}
sycl::device device;
sycl::queue queue;
void selectDevice(Option &option) {
    try {
        device = sycl::device(sycl::gpu_selector());
    } catch (sycl::exception const& error) {
        std::cout << "Cannot select a GPU: " << error.what() << std::endl;
    }
    std::cout << "Using: " << device.get_info<sycl::info::device::name>();
    std::cout << std::endl;
    queue = sycl::queue(device);
}
void readFile(std::vector<Read> &reads, Option &option) {
    std::ifstream file(option.inputFile);
    Read read;
    std::string line;
    long end = 0;
    long point = 0;
    file.seekg(0, std::ios::end);
    end = file.tellg();
    file.seekg(0, std::ios::beg);
    while(true) {
        getline(file, line);
        read.name = line;
        while (true) {
            point = file.tellg();
            getline(file, line);
            if (line[0] == '>') {
                file.seekg(point, std::ios::beg);
                reads.push_back(read);
                read.name = "";
                read.data = "";
                break;
            } else {
                read.data += line;
            }
            point = file.tellg();
            if (point == end){
                reads.push_back(read);
                read.data = "";
                read.name = "";
                break;
            }
        }
        if (point == end) break;
    }
    file.close();
    std::sort(reads.begin(), reads.end(), [](Read &a, Read &b) {
        return a.data.size() > b.data.size();
    });
    std::cout << "read file complete" << std::endl;
    std::cout << "longest/shortest：\t" << reads[0].data.size() << "/";
    std::cout << reads[reads.size()-1].data.size() << std::endl;
    std::cout << "reads count：\t" << reads.size() << std::endl;
}
void copyData(std::vector<Read> &reads, Data &data, Option &option) {
  
    data.readsCount = reads.size();
    int readsCount = data.readsCount;
    int *lengths = new int[readsCount];
    long *offsets = new long[readsCount+1];
    offsets[0] = 0;
    for (int i=0; i<readsCount; i++) {
        int length = reads[i].data.size();
        lengths[i] = length;
        offsets[i+1] = offsets[i]+length/32*32+32;
    }
    data.readsLength = offsets[readsCount];
    long readsLength = data.readsLength;
    char *reads_buf = new char[readsLength];
    for (int i=0; i<readsCount; i++) {
        long start = offsets[i];
        int length = lengths[i];
        memcpy(&reads_buf[start], reads[i].data.c_str(), length*sizeof(char));
    }
  
    data.reads_dev = sycl::malloc_device<char>(readsLength, queue);
    data.lengths_dev = sycl::malloc_device<int>(readsCount, queue);
    data.offsets_dev = sycl::malloc_device<long>(readsCount+1, queue);
    queue.memcpy(data.reads_dev, reads_buf, sizeof(char)*readsLength);
    queue.memcpy(data.lengths_dev, lengths, sizeof(int)*readsCount);
    queue.memcpy(data.offsets_dev, offsets, sizeof(long)*(readsCount+1));
    queue.wait_and_throw();
    delete[] lengths;
    delete[] offsets;
    delete[] reads_buf;
    std::cout << "copy data complete" << std::endl;
}
void baseToNumber(Data &data) {
    int readsCount = data.readsCount;
    long readsLength = data.readsLength;
    queue.submit([&](sycl::handler &handler) {
        sycl::stream out(16*128, 16, handler);
        handler.parallel_for(sycl::nd_range<1>(128*128, 128),
        [=](sycl::nd_item<1> item) {
            long index = item.get_global_linear_id();
            while (index < readsLength) {
                switch (data.reads_dev[index]) {
                case 'A':
                    data.reads_dev[index] = 0;
                    break;
                case 'a':
                    data.reads_dev[index] = 0;
                    break;
                case 'C':
                    data.reads_dev[index] = 1;
                    break;
                case 'c':
                    data.reads_dev[index] = 1;
                    break;
                case 'G':
                    data.reads_dev[index] = 2;
                    break;
                case 'g':
                    data.reads_dev[index] = 2;
                    break;
                case 'T':
                    data.reads_dev[index] = 3;
                    break;
                case 't':
                    data.reads_dev[index] = 3;
                    break;
                case 'U':
                    data.reads_dev[index] = 3;
                    break;
                case 'u':
                    data.reads_dev[index] = 3;
                    break;
                default:
                    data.reads_dev[index] = 4;
                    break;
                }
                index += 128*128;
            }
        });
    });
    queue.wait_and_throw();
    std::cout << "base to number complete" << std::endl;
}
void createPrefix(Data &data) {
    int readsCount = data.readsCount;
    
    data.prefix_dev = sycl::malloc_device<int>(readsCount*4, queue);
    queue.submit([&](sycl::handler &handler) {
        sycl::stream out(16*128, 16, handler);
        handler.parallel_for(sycl::nd_range<1>
        ((readsCount+127)/128*128, 128), [=](sycl::nd_item<1> item) {
            int index = item.get_global_linear_id();
            if (index >= readsCount) return;
            int base[5] = {0};
            int length = data.lengths_dev[index];
            long start = data.offsets_dev[index];
            for (int i=0; i<length; i++) {
                switch(data.reads_dev[start+i]) {
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
            data.prefix_dev[index*4+0] = base[0];
            data.prefix_dev[index*4+1] = base[1];
            data.prefix_dev[index*4+2] = base[2];
            data.prefix_dev[index*4+3] = base[3];
        });
    });
    queue.wait_and_throw();
    std::cout << "make prefilter complete" << std::endl;
}
void createWords(Data &data, Option &option) {
    int readsCount = data.readsCount;
    int wordLength = option.wordLength;
    long readsLength = data.readsLength;
    data.words_dev = sycl::malloc_device<unsigned short>(readsLength, queue);
    data.wordCounts_dev = sycl::malloc_device<int>(readsCount, queue);
    queue.submit([&](sycl::handler &handler) {
        sycl::stream out(16*128, 16, handler);
        handler.parallel_for(sycl::nd_range<1>
        ((readsCount+127)/128*128, 128), [=](sycl::nd_item<1> item) {
            int index = item.get_global_linear_id();
            if (index >= readsCount) return;
            int length = data.lengths_dev[index];
            long start = data.offsets_dev[index];
            if (length < wordLength) {
                data.wordCounts_dev[index] = 0;
                return;
            }
            int count = 0;
            for (int i=wordLength-1; i<length; i++) {
                unsigned short word = 0;
                int flag = 0;
                for (int j=0; j<wordLength; j++) {
                    unsigned char base = data.reads_dev[start+i-j];
                    word += base<<j*2;
                    if (base == 4) flag = 1;
                }
                if (flag == 0) {
                    data.words_dev[start+count] = word;
                    count += 1;
                }
            }
            data.wordCounts_dev[index] = count;
        });
    });
    queue.wait_and_throw();
    std::cout << "make word complete" << std::endl;
}
void sortWords(Data &data, Option &option) {
    int readsCount = data.readsCount;
    long readsLength = data.readsLength;
    
    queue.submit([&](sycl::handler &handler) {
        sycl::stream out(16*128, 16, handler);
        handler.parallel_for(sycl::nd_range<1>
        ((readsCount+127)/128*128, 128), [=](sycl::nd_item<1> item) {
            int index = item.get_global_linear_id();
            if (index >= readsCount) return;
            long start = data.offsets_dev[index];
            int wordCount = data.wordCounts_dev[index];
            for (int gap=wordCount/2; gap>0; gap/=2){
                for (int i=gap; i<wordCount; i++) {
                    for (int j=i-gap; j>=0; j-=gap) {
                        if(data.words_dev[start+j]>data.words_dev[start+j+gap]){
                            unsigned int temp = data.words_dev[start+j];
                            data.words_dev[start+j]=data.words_dev[start+j+gap];
                            data.words_dev[start+j+gap] = temp;
                        } else {
                            break;
                        }
                    }
                }
            }
        });
    });
    queue.wait_and_throw();
    std::cout << "sort word complete" << std::endl;
}
void mergeWords(Data &data) {
    int readsCount = data.readsCount;
    long readsLength = data.readsLength;
    data.orders_dev = sycl::malloc_device<unsigned short>(readsLength, queue);
    queue.submit([&](sycl::handler &handler) {
        sycl::stream out(16*128, 16, handler);
        handler.parallel_for(sycl::nd_range<1>
        ((readsCount+127)/128*128, 128), [=](sycl::nd_item<1> item) {
            int index = item.get_global_linear_id();
            if (index >= readsCount) return;
            long start = data.offsets_dev[index];
            int wordCount = data.wordCounts_dev[index];
            unsigned int preWord = data.words_dev[start];
            unsigned int current;
            unsigned short count = 0;
            for (int i=0; i<wordCount; i++) {
                current = data.words_dev[start+i];
                if (preWord == current) {
                    count += 1;
                    data.orders_dev[start+i] = 0;
                } else {
                    preWord = current;
                    data.orders_dev[start+i] = 0;
                    data.orders_dev[start+i-1] = count;
                    count = 1;
                }
            }
            data.orders_dev[start+wordCount-1] = count;
        });
    });
    queue.wait_and_throw();
    std::cout << "merge word complete" << std::endl;
}
void createCutoff(Data &data, Option &option) {
    int readsCount = data.readsCount;
    float threshold = option.threshold;
    int wordLength = option.wordLength;
    
    data.wordCutoff_dev = sycl::malloc_device<int>(readsCount, queue);
    data.baseCutoff_dev = sycl::malloc_device<int>(readsCount, queue);
    queue.submit([&](sycl::handler &handler) {
        sycl::stream out(16*128, 16, handler);
        handler.parallel_for(sycl::nd_range<1>
        ((readsCount+127)/128*128, 128), [=](sycl::nd_item<1> item) {
            int index = item.get_global_linear_id();
            if (index >= readsCount) return;
          
            int length = data.lengths_dev[index];
            int required = length - wordLength + 1;
            int cutoff = sycl::ceil((float)length*(1.0f-threshold))*wordLength;
            required -= cutoff;
            required = sycl::max(required, 1);
            float offset = 0;
            if (threshold >= 0.9) {
                offset = 1.1 - sycl::fabs(threshold - 0.95) * 2;
            } else {
                offset = 1;
            }
            offset = 1;
            required = sycl::ceil((float)required * offset);
            data.wordCutoff_dev[index] = required;
          
            required = sycl::ceil((float)length * threshold);
            data.baseCutoff_dev[index] = required;
        });
    });
    queue.wait_and_throw();
    std::cout << "make threshold complete" << std::endl;
}
void deleteGap(Data &data) {
    int readsCount = data.readsCount;
    
    data.gaps_dev = sycl::malloc_device<int>(readsCount, queue);
    queue.submit([&](sycl::handler &handler) {
        sycl::stream out(16*128, 16, handler);
        handler.parallel_for(sycl::nd_range<1>
        ((readsCount+127)/128*128, 128), [=](sycl::nd_item<1> item) {
            int index = item.get_global_linear_id();
            if (index >= readsCount) return;
            long start = data.offsets_dev[index];
            int length = data.lengths_dev[index];
            int count = 0;
            int gap = 0;
            for (int i=0; i<length; i++) {
                char base = data.reads_dev[start+i];
                if (base < 4) {
                    data.reads_dev[start+count] = base;
                    count += 1;
                } else {
                    gap += 1;
                }
            }
            data.gaps_dev[index] = gap;
        });
    });
    queue.wait_and_throw();
    std::cout << "delete gap complete" << std::endl;
}
void compressData(Data &data) {
    int readsCount = data.readsCount;
    long readsLength = data.readsLength/16;
    
    data.compressed_dev = sycl::malloc_device<unsigned int>(readsLength, queue);
    queue.submit([&](sycl::handler &handler) {
        sycl::stream out(16*128, 16, handler);
        handler.parallel_for(sycl::nd_range<1>
        ((readsCount+127)/128*128, 128), [=](sycl::nd_item<1> item) {
            int index = item.get_global_linear_id();
            if (index >= readsCount) return;
            long readStart = data.offsets_dev[index];
            long compressStart = readStart/16;
            int length = data.lengths_dev[index] - data.gaps_dev[index];
            length = length/32+1;
            for (int i=0; i<length; i++) {
                unsigned int low = 0;
                unsigned int hight = 0;
                for (int j=0; j<32; j++) {
                    char base = data.reads_dev[readStart+i*32+j];
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
                data.compressed_dev[compressStart+i*2+0] = low;
                data.compressed_dev[compressStart+i*2+1] = hight;
            }
        });
    });
    queue.wait_and_throw();
    std::cout << "compress data complete" << std::endl;
}
void initBench(Bench &bench, int readsCount, sycl::queue queue) {
  
    bench.table_dev = sycl::malloc_device<unsigned short>((1<<2*8), queue);
    queue.memset(bench.table_dev, 0, (1<<2*8)*sizeof(unsigned short));
    bench.cluster = sycl::malloc_shared<int>(readsCount, queue);
    for (int i=0; i<readsCount; i++) {
        bench.cluster[i] = -1;
    }
    bench.remainList = sycl::malloc_shared<int>(readsCount, queue);     for (int i=0; i<readsCount; i++) {
        bench.remainList[i] = i;
    }
    bench.remainCount = readsCount;
    bench.jobList = sycl::malloc_shared<int>(readsCount, queue);     for (int i=0; i<readsCount; i++) {
        bench.jobList[i] = i;
    }
    bench.jobCount = readsCount;
    bench.representative = -1;
    queue.wait_and_throw();
}
void updateRepresentative(int *cluster, int &representative, int readsCount) {
    representative += 1;
    while (representative < readsCount) {
        if (cluster[representative] == -1) {
            cluster[representative] = representative;
            break;
        } else {
            representative += 1;
        }
    }
}
void updateRemain(int *cluster, int *remainList, int &remainCount) {
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
void updatJobs(int *jobList, int &jobCount) {
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
void kernel_makeTable(long *offsets, unsigned short *words,
int *wordCounts, unsigned short *orders,unsigned short *table,
int representative, sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    long start = offsets[representative];
    int length = wordCounts[representative];
    for (int i=index; i<length; i+=128*128) {
        unsigned short word = words[start+i];
        unsigned short order = orders[start+i];
        if (order > 0) table[word] = order;
    }
}
void kernel_cleanTable(long *offsets, unsigned short *words,
int* wordCounts, unsigned short *orders, unsigned short *table,
int representative, sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    long start = offsets[representative];
    int length = wordCounts[representative];
    for (int i=index; i<length; i+=128*128) {
        unsigned short word = words[start+i];
        unsigned short order = orders[start+i];
        if (order > 0) table[word] = 0;
    }
}
void kernel_preFilter(int *prefix, int *baseCutoff, int *jobList,
int jobCount, int representative, sycl::nd_item<3> item) {
    int index;
    index=item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    if (index >= jobCount) return;
    int text = representative;
    int query = jobList[index];
    int offsetOne = text*4;
    int offsetTwo = query*4;
    int sum = 0;
    sum += sycl::min(prefix[offsetOne + 0], prefix[offsetTwo + 0]);
    sum += sycl::min(prefix[offsetOne + 1], prefix[offsetTwo + 1]);
    sum += sycl::min(prefix[offsetOne + 2], prefix[offsetTwo + 2]);
    sum += sycl::min(prefix[offsetOne + 3], prefix[offsetTwo + 3]);
    int cutoff = baseCutoff[query];
    if (sum < cutoff) {
        jobList[index] = -1;
    }
}
void kernel_filter(long *offsets, unsigned short *words, int *wordCounts,
unsigned short *orders, int *wordCutoff, unsigned short *table, int *jobList,
int jobCount, sycl::nd_item<3> item, int *result) {
    if (item.get_group(2) >= jobCount) return;
    int query = jobList[item.get_group(2)];
    result[item.get_local_id(2)] = 0;
    long start = offsets[query];
    int length = wordCounts[query];
    for (int i = item.get_local_id(2); i<length; i+=128) {
        unsigned short value = words[start+i];
        result[item.get_local_id(2)] +=
        sycl::min(table[value], orders[start + i]);
    }
    item.barrier(sycl::access::fence_space::local_space);
    for (int i=128/2; i>0; i/=2) {
        if (item.get_local_id(2) >= i) {
        } else {
            result[item.get_local_id(2)] += result[item.get_local_id(2)+i];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    if (item.get_local_id(2) == 0) {
        int cutoff = wordCutoff[query];
        if (result[0] < cutoff) {
            jobList[item.get_group(2)] = -1;
        }
    }
}
void kernel_dynamic(int *lengths, long *offsets, int *gaps,
unsigned int *compressed, int *baseCutoff, int *cluster, int *jobList,
int jobCount, int representative, sycl::nd_item<3> item,
unsigned int *bases) {
  
    int text = representative;
    long textStart = offsets[text]/16;
    int textLength = lengths[text]-gaps[text];
    for (int i = item.get_local_id(2); i < textLength/32+1;
    i += item.get_local_range().get(2)) {         bases[i*2+0] = compressed[textStart+i*2+0];
        bases[i*2+1] = compressed[textStart+i*2+1];
    }
  
  
    int index;
    index = item.get_local_id(2)+item.get_local_range().get(2)*item.get_group(2);
    if (index >= jobCount) return;
    unsigned int line[2048] = {0xFFFFFFFF};
    for (int i=0; i<2048; i++) {
        line[i] = 0xFFFFFFFF;
    }
    int query = jobList[index];
    long queryStart = offsets[query] / 16;
    int queryLength = lengths[query] - gaps[query];
    for (int i=0; i<queryLength/32; i++) {
        unsigned int column[32] = {0};
        unsigned int queryLow = compressed[queryStart+i*2+0];
        unsigned int queryHight = compressed[queryStart+i*2+1];
        for (int j=0; j<textLength/32+1; j++) {
            unsigned int textl = bases[j*2+0];
            unsigned int texth = bases[j*2+1];
            unsigned int row = line[j];
            for (int k=0; k<32; k++) {
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
    }
  
    unsigned int column[32] = {0};
    unsigned int queryLow = compressed[queryStart+(queryLength/32)*2+0];
    unsigned int queryHight = compressed[queryStart+(queryLength/32)*2+1];
    for (int j=0; j<textLength/32+1; j++) {
        unsigned int textl = bases[j*2+0];
        unsigned int texth = bases[j*2+1];
        unsigned int row = line[j];
        for (int k=0; k<queryLength%32; k++) {
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
        cluster[query] = text;
    } else {
        jobList[index] = -1;
    }
}
void clustering(Option &option, Data &data, Bench &bench) {
    int readsCount = data.readsCount;
    
    initBench(bench, readsCount, queue);
    std::cout << "now/whole:" << std::endl;
    while (true) {
      
      
        updateRepresentative(bench.cluster, bench.representative, readsCount);
        if (bench.representative >= readsCount) break;
      
        std::cout << "\r" << bench.representative+1 << "/" << readsCount;
        std::flush(std::cout);
      
        updateRemain(bench.cluster, bench.remainList, bench.remainCount);
      
        memcpy(bench.jobList, bench.remainList, bench.remainCount*sizeof(int));
        bench.jobCount = bench.remainCount;
        queue.wait_and_throw();
      
        try {
            queue.submit([&](sycl::handler &handler) {
                handler.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 128)*
                sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
                [=](sycl::nd_item<3> item) {
                    kernel_makeTable(data.offsets_dev, data.words_dev, data.wordCounts_dev,
                    data.orders_dev, bench.table_dev, bench.representative, item);
                });
            });
        } catch (sycl::exception const& error) {
            std::cout << "make table: " << error.what() << std::endl;
        }
        queue.wait_and_throw();
        updatJobs(bench.jobList, bench.jobCount);
        if (bench.jobCount > 0) {
            try {
                queue.submit([&](sycl::handler &handler) {
                    handler.parallel_for(sycl::nd_range<3>
                    (sycl::range<3>(1, 1, (bench.jobCount+127)/128)*
                    sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
                    [=](sycl::nd_item<3> item) {
                        kernel_preFilter(data.prefix_dev, data.baseCutoff_dev,
                        bench.jobList, bench.jobCount,
                        bench.representative, item);
                    });
                });
            } catch (sycl::exception const& error) {
                std::cout << "prefilter: " << error.what() << std::endl;
            }
        }
        queue.wait_and_throw();
        updatJobs(bench.jobList, bench.jobCount);
        if (bench.jobCount > 0) {
            try {
            queue.submit([&](sycl::handler &handler) {
                sycl::accessor<int, 1, sycl::access::mode::read_write,
                sycl::access::target::local>result(sycl::range<1>(128), handler);
                handler.parallel_for(sycl::nd_range<3>(sycl::range<3>
                (1, 1, bench.jobCount)*sycl::range<3>(1, 1, 128),sycl::range<3>
                (1, 1, 128)), [=](sycl::nd_item<3> item) {
                    kernel_filter(data.offsets_dev, data.words_dev, data.wordCounts_dev,
                    data.orders_dev, data.wordCutoff_dev, bench.table_dev, bench.jobList,
                    bench.jobCount, item, result.get_pointer());
                });
            });
            } catch (sycl::exception const& error) {
                std::cout << "word filter: " << error.what() << std::endl;
            }
        }
        queue.wait_and_throw();
      
        updatJobs(bench.jobList, bench.jobCount);
        if (bench.jobCount > 0) {
            try {
                queue.submit([&](sycl::handler &handler) {
                    sycl::accessor<unsigned int, 1,
                    sycl::access::mode::read_write, sycl::access::target::local>
                    bases_acc_ct1(sycl::range<1>(2048), handler);
                    handler.parallel_for(sycl::nd_range<3>(sycl::range<3>
                    (1, 1, (bench.jobCount+127)/128)*sycl::range<3>(1, 1, 128),
                    sycl::range<3>(1, 1, 128)), [=](sycl::nd_item<3> item) {
                        kernel_dynamic(data.lengths_dev, data.offsets_dev, data.gaps_dev,
                        data.compressed_dev, data.baseCutoff_dev, bench.cluster,
                        bench.jobList, bench.jobCount, bench.representative,
                        item, bases_acc_ct1.get_pointer());
                    });
                });
            } catch (sycl::exception const& error) {
                std::cout << "dynamic: " << error.what() << std::endl;
            }
        }
      
        try {
            queue.submit([&](sycl::handler &handler) {
                handler.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 128)*
                sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
                [=](sycl::nd_item<3> item) {
                    kernel_cleanTable(data.offsets_dev, data.words_dev, data.wordCounts_dev,
                    data.orders_dev, bench.table_dev, bench.representative, item);
                });
            });
        } catch (sycl::exception const& error) {
            std::cout << "clean table: " << error.what() << std::endl;
        }
        queue.wait_and_throw();
    }
    std::cout << std::endl;
}
void saveFile(Option &option, std::vector<Read> &reads, Bench &bench) {
    int readsCount = reads.size();
    std::ofstream file(option.outputFile);
    int sum = 0;
    for (int i=0; i<readsCount; i++) {
        int order = bench.cluster[i];
        if (order == i) {
            file << reads[i].name << std::endl;
            file << reads[i].data << std::endl;
            sum++;
        }
    }
    file.close();
    std::cout << "cluster：" << sum << std::endl;
}
