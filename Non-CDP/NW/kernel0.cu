#include "common.h"
#include "timer.h"

#define BLOCK_DIM 128

__global__ void nw_gpu0_kernel (unsigned char * reference_d, unsigned char* query_d, int* matrix_d, unsigned int N, unsigned int iteration, unsigned int round) {
	int position = blockDim.x*blockIdx.x + threadIdx.x;
	int r = 0;
	int q = 0;
	if(round == 1) {
		r  = iteration - 1 - position;
		q = position;
	}
	else if (round == 2) {
		r = N - position - 1;
		q = N - iteration + position;	
	}
	if( position < iteration ) {
		int top     = (q == 0)?((r + 1)*DELETION):(matrix_d[(q - 1)*N + r]);
	        int left    = (r == 0)?((q + 1)*INSERTION):(matrix_d[q*N + (r - 1)]);
	        int topleft = (q == 0)?(r*DELETION):((r == 0)?(q*INSERTION):(matrix_d[(q - 1)*N + (r - 1)]));
	        // Find scores based on neighbors
	        int insertion = top + INSERTION;
	        int deletion  = left + DELETION;
	       	int match     = topleft + ((query_d[q] == reference_d[r])?MATCH:MISMATCH);
	       	// Select best score
        	int max = (insertion > deletion)?insertion:deletion;
        	max = (match > max)?match:max;
	        matrix_d[q*N + r] = max;
	}		
		
} 

void nw_gpu0(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) {
	for (int i = 1; i < N+1; i++) {
		int numThreadsPerBlock = BLOCK_DIM;
		int numBlocks = (i+numThreadsPerBlock-1)/(numThreadsPerBlock);
		nw_gpu0_kernel <<< numBlocks, numThreadsPerBlock >>> (reference_d, query_d, matrix_d, N, i, 1);
		cudaDeviceSynchronize();
	}
	for (int i = N-1; i>0; i--){
		int numThreadsPerBlock = BLOCK_DIM;
		int numBlocks = (i + numThreadsPerBlock -1)/numThreadsPerBlock;
		nw_gpu0_kernel <<< numBlocks, numThreadsPerBlock >>> (reference_d, query_d, matrix_d, N, i, 2);
		cudaDeviceSynchronize();
	}


}
