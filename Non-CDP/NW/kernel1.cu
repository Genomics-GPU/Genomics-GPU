
#include "common.h"
#include "timer.h"

#define BLOCK_DIM 32

__global__ void nw_gpu1_kernel (unsigned char * reference_d, unsigned char* query_d, int* matrix_d, unsigned int N, unsigned int round) {
	
	__shared__ unsigned int q_offset;
	__shared__ unsigned int r_offset;
	__shared__ unsigned int loop_limit;

	if(threadIdx.x == 0){
		//Check if it is round 1 or 2 in overall matrix of blocks
		if (round == 1){
			q_offset = BLOCK_DIM*blockIdx.x;
			r_offset = BLOCK_DIM*(gridDim.x - 1 - blockIdx.x);
		}
		else if (round == 2){
			q_offset = BLOCK_DIM*((N + BLOCK_DIM - 1)/BLOCK_DIM - gridDim.x + blockIdx.x );
			r_offset = BLOCK_DIM*((N + BLOCK_DIM - 1)/BLOCK_DIM - blockIdx.x - 1);	
		}
		//Loop limit is used as a boundary check
		//If the block is not complete and some elements are out of bounds, we can loop fewer times
		loop_limit = (((N-q_offset) > BLOCK_DIM && (N-r_offset) > BLOCK_DIM) || N%BLOCK_DIM == 0)? 2*BLOCK_DIM : ((N-q_offset) < BLOCK_DIM && (N-r_offset) < BLOCK_DIM)? 2*(N % BLOCK_DIM)  : BLOCK_DIM + N % BLOCK_DIM;
	}
	__syncthreads();

	for (int i = 1; i < loop_limit; i++){
		//Check if it is round 1 or 2 within the block
		int idx = (i < BLOCK_DIM + 1)? i : 2*BLOCK_DIM - i;
		int q = 0;
		int r = 0;
		if (i < BLOCK_DIM + 1) {
			//This is round 1;
			q = threadIdx.x + q_offset;
			r = idx - threadIdx.x - 1 + r_offset;
		}
		else {
			//This is round 2
			q = BLOCK_DIM - idx + threadIdx.x + q_offset;
			r = BLOCK_DIM - threadIdx.x - 1 + r_offset;
		}

		if(threadIdx.x < idx && q < N && r < N) {
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
		__syncthreads();
	}
	
}


void nw_gpu1(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) {

	//Each tile is of dimension BLOCK_DIM*BLOCK_DIM
	//Max number of threads simultaneoulsy active in a tile is BLOCK_DIM
	//So number of threads per block is BLOCK_DIM	
	int numThreadsPerBlock = BLOCK_DIM;
	
	for (unsigned int i = 1; i < (N + BLOCK_DIM - 1)/BLOCK_DIM + 1; i++) {
		//Number of blocks (i.e. of tiles)  is equal to the iteration number
                int numBlocks = i;
                nw_gpu1_kernel <<< numBlocks, numThreadsPerBlock >>> (reference_d, query_d, matrix_d, N, 1);
                cudaDeviceSynchronize();
        }
        for (int i = (N + BLOCK_DIM - 1)/BLOCK_DIM -1; i>0; i--){
                int numBlocks = i;
                nw_gpu1_kernel <<< numBlocks, numThreadsPerBlock >>> (reference_d, query_d, matrix_d, N, 2);
                cudaDeviceSynchronize();
        }
}
