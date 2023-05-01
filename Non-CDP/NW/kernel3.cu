
#include "common.h"
#include "timer.h"

#define BLOCK_DIM 64

__global__ void nw_gpu3_kernel (unsigned char * reference_d, unsigned char* query_d, int* matrix_d, unsigned int N, unsigned int round) {
	
	__shared__ unsigned int q_offset;
	__shared__ unsigned int r_offset;
	__shared__ unsigned int loop_limit;
	__shared__ int matrix_s[BLOCK_DIM*BLOCK_DIM];

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
		int q_t = 0;
		int r_t = 0;
		if (i < BLOCK_DIM + 1) {
			//This is round 1;
			q_t = threadIdx.x;
			r_t = idx - threadIdx.x - 1;
		}
		else {
			//This is round 2
			q_t = BLOCK_DIM - idx + threadIdx.x;
			r_t = BLOCK_DIM - threadIdx.x - 1;
		}
		int q = q_t + q_offset;
		int r = r_t + r_offset;
		int max = 0;
		if(threadIdx.x < idx && q < N && r < N) {
			int top     = (q == 0)?((r + 1)*DELETION):(q_t == 0)?(matrix_d[(q - 1)*N + r]):matrix_s[(q_t -1)*BLOCK_DIM + r_t];
                	int left    = (r == 0)?((q + 1)*INSERTION):(r_t == 0)?(matrix_d[q*N + (r - 1)]):matrix_s[q_t*BLOCK_DIM + (r_t - 1)];
                	int topleft = (q == 0)?(r*DELETION):(r == 0)?(q*INSERTION):(q_t == 0 || r_t == 0)?(matrix_d[(q - 1)*N + (r - 1)]):matrix_s[(q_t - 1)*BLOCK_DIM + (r_t - 1)];
                	// Find scores based on neighbors

			int insertion = top + INSERTION;
                	int deletion  = left + DELETION;
                	int match     = topleft + ((query_d[q] == reference_d[r])?MATCH:MISMATCH);
                	// Select best score
                	max = (insertion > deletion)?insertion:deletion;
                	max = (match > max)?match:max;
			
			matrix_s[q_t*BLOCK_DIM + r_t] = max;
        	}
		__syncthreads();
	}
	for(int it = 0; it < BLOCK_DIM && q_offset + it < N; it++){
		if(r_offset + threadIdx.x < N){
			matrix_d[(q_offset + it)*N + r_offset +threadIdx.x] = matrix_s[it*BLOCK_DIM + threadIdx.x];
		}
	}
	
}


void nw_gpu3(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) {

	//Each tile is of dimension BLOCK_DIM*BLOCK_DIM
	//Max number of threads simultaneoulsy active in a tile is BLOCK_DIM
	//So number of threads per block is BLOCK_DIM	
	int numThreadsPerBlock = BLOCK_DIM;
	
	for (unsigned int i = 1; i < (N + BLOCK_DIM - 1)/BLOCK_DIM + 1; i++) {
		//Number of blocks (i.e. of tiles)  is equal to the iteration number
                int numBlocks = i;
                nw_gpu3_kernel <<< numBlocks, numThreadsPerBlock >>> (reference_d, query_d, matrix_d, N, 1);
                cudaDeviceSynchronize();
        }
        for (int i = (N + BLOCK_DIM - 1)/BLOCK_DIM -1; i>0; i--){
                int numBlocks = i;
                nw_gpu3_kernel <<< numBlocks, numThreadsPerBlock >>> (reference_d, query_d, matrix_d, N, 2);
                cudaDeviceSynchronize();
        }
}
