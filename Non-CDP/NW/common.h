
#ifndef _COMMON_H_
#define _COMMON_H_

#define MATCH       1
#define MISMATCH    (-1)
#define INSERTION   (-1)
#define DELETION    (-1)

void nw_gpu0(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N);
void nw_gpu1(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N);
void nw_gpu2(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N);
void nw_gpu3(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N);

#endif

