
#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include "common.h"
#include "timer.h"
#define BLOCK_DIM 128

void nw_cpu(unsigned char* reference, unsigned char* query, int* matrix, unsigned int N) {
    for(int q = 0; q < N; ++q) {
        for (int r = 0; r < N; ++r) {
            // Get neighbors
            int top     = (q == 0)?((r + 1)*DELETION):(matrix[(q - 1)*N + r]);
            int left    = (r == 0)?((q + 1)*INSERTION):(matrix[q*N + (r - 1)]);
            int topleft = (q == 0)?(r*DELETION):((r == 0)?(q*INSERTION):(matrix[(q - 1)*N + (r - 1)]));
            // Find scores based on neighbors
            int insertion = top + INSERTION;
            int deletion  = left + DELETION;
            int match     = topleft + ((query[q] == reference[r])?MATCH:MISMATCH);
            // Select best score
            int max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;
            matrix[q*N + r] = max;
        }
    }
}

void verify(int* matrix_cpu, int* matrix_gpu, unsigned int N) {
    for (unsigned int q = 0; q < N; ++q) {
        for (unsigned int r = 0; r < N; ++r) {
            if(matrix_cpu[q*N + r] != matrix_gpu[q*N + r]) {
                printf("\033[1;31mMismatch at q = %u, r = %u (CPU result = %d, GPU result = %d)\033[0m\n", q, r, matrix_cpu[q*N + r], matrix_gpu[q*N + r]);
                return;
            }
        }
    }
    printf("Verification succeeded\n");
}

void generateQuery(unsigned char* reference, unsigned char* query, unsigned int N) {
    const float PROB_MATCH = 0.80f;
    const float PROB_INS   = 0.10f;
    const float PROB_DEL   = 1.00f - PROB_MATCH - PROB_INS;
    assert(PROB_MATCH >= 0.00f && PROB_MATCH <= 1.00f);
    assert(PROB_INS   >= 0.00f && PROB_INS   <= 1.00f);
    assert(PROB_DEL   >= 0.00f && PROB_DEL   <= 1.00f);
    unsigned int r = 0, q = 0;
    while(r < N && q < N) {
        float prob = rand()*1.0f/RAND_MAX;
        if(prob < PROB_MATCH) {
            query[q++] = reference[r++]; // Match
        } else if(prob < PROB_MATCH + PROB_INS) {
            query[q++] = rand()%256; // Insertion
        } else {
            ++r; // Deletion
        }
    }
    while(q < N) {
        query[q++] = rand()%256; // Tail insertions
    }
}


/*__global__ void nw_gpu0_kernel (unsigned char * reference_d, unsigned char* query_d, int* matrix_d, unsigned int N, unsigned int iteration, unsigned int round) {
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

__global__ void nw_gpu0_cdp(unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) {
     // if (iSize == 1) return;
    // reduce block size to half
 // int nthreads = iSize>>1;
   // if(threadIdx.x == 0 && nthreads > 0) {
        for (int i = 1; i < N+1; i++) {
                int numThreadsPerBlock = BLOCK_DIM;
                int numBlocks = (i+numThreadsPerBlock-1)/(numThreadsPerBlock);
                nw_gpu0_kernel <<< numBlocks, numThreadsPerBlock >>> (reference_d, query_d, matrix_d, N, i, 1);
               // cudaDeviceSynchronize();
        }
        for (int i = N-1; i>0; i--){
                int numThreadsPerBlock = BLOCK_DIM;
                int numBlocks = (i + numThreadsPerBlock -1)/numThreadsPerBlock;
                nw_gpu0_kernel <<< numBlocks, numThreadsPerBlock >>> (reference_d, query_d, matrix_d, N, i, 2);
               // cudaDeviceSynchronize();
        }

   // }
}
*/

/*
__global__ void cdprun(int const iSize,int iDepth, unsigned char* reference_d, unsigned char* query_d, int* matrix_d, unsigned int N) {

	unsigned char* r_d_cdp = reference_d;
	 unsigned char* q_d_cdp = query_d;
	 int* m_d_cdp = matrix_d;
	 unsigned int N_cdp = N;


//   int tid = threadIdx.x；

       // char * c_x = x;
       // char * c_y = y;
//       kData* host_data = (kData*) malloc(sizeof(kData) * nWindows);
       // kData* device_data = data;
       //  result * device_results = results;
       // int ncdpb = nrun;
// int numThreadsPerBlock = BLOCK_DIM;
               // int numBlocks = (i+numThreadsPerBlock-1)/(numThreadsPerBlock);

   // printf("Recursion=%d: Hello World from thread %d block %d\n",iDepth,threadIdx.x,blockIdx.x);
    // condition to stop recursive execution
    if (iSize == 1) return;
    // reduce block size to half
    int nthreads = iSize>>1;
    // thread 0 launches child grid recursively
    if(threadIdx.x == 0 && nthreads > 0) {
        //nestedHelloWorld<<<1, nthreads>>>(nthreads,++iDepth);
       // alignKernel<<<ncdpb, 1>>>(c_x, c_y, device_data, device_results);
	     nw_gpu0_cdp <<< 2, 1>> >(r_d_cdp, q_d_cdp, m_d_cdp, N_cdp);
       // printf("-------> nested execution depth: %d\n",iDepth);
    }
}
*/



int main(int argc, char**argv) {

   // cudaDeviceSynchronize();

    // Parse arguments
    unsigned int N = 32000;
    unsigned int runGPUVersion0 = 0;
    unsigned int runGPUVersion1 = 0;
    unsigned int runGPUVersion2 = 0;
    unsigned int runGPUVersion3 = 0;
    int opt;
    while((opt = getopt(argc, argv, "N:0123")) >= 0) {
        switch(opt) {
            case 'N': N = atoi(optarg);     break;
            case '0': runGPUVersion0 = 1;   break;
            case '1': runGPUVersion1 = 1;   break;
            case '2': runGPUVersion2 = 1;   break;
            case '3': runGPUVersion3 = 1;   break;
            default:  fprintf(stderr, "\nUnrecognized option!\n");
                      exit(0);
        }
    }

    // Allocate memory and initialize data
    Timer timer;
    unsigned char* reference = (unsigned char*) malloc(N*sizeof(unsigned char));
    unsigned char* query = (unsigned char*) malloc(N*sizeof(unsigned char));
    int* matrix_cpu = (int*) malloc(N*N*sizeof(int));
    int* matrix_gpu = (int*) malloc(N*N*sizeof(int));
    for(unsigned int r = 0; r < N; ++r) {
        reference[r] = rand()%256;
    }
    generateQuery(reference, query, N);
    
    //Open file for writing in appending mode
    FILE *fp_seq;
    fp_seq = fopen("runtimes_seq.txt", "a");

    // Compute on CPU
    startTime(&timer);
    nw_cpu(reference, query, matrix_cpu, N);
    stopTime(&timer);
    printElapsedTimeToFile(timer, fp_seq);
    printElapsedTime(timer, "CPU time", CYAN);

   // int blocksize = 8;
     //       int size = 8;

       // dim3 block (blocksize, 1);
       // dim3 grid  ((size + block.x - 1) / block.x, 1);

    if(runGPUVersion0 || runGPUVersion1 || runGPUVersion2 || runGPUVersion3) {

        // Allocate GPU memory
        startTime(&timer);
        unsigned char *reference_d;
        unsigned char *query_d;
        int *matrix_d;
        cudaMalloc((void**) &reference_d, N*sizeof(unsigned char));
        cudaMalloc((void**) &query_d, N*sizeof(unsigned char));
        cudaMalloc((void**) &matrix_d, N*N*sizeof(int));
       // cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "Allocation time");

        // Copy data to GPU
        startTime(&timer);
        cudaMemcpy(reference_d, reference, N*sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaMemcpy(query_d, query, N*sizeof(unsigned char), cudaMemcpyHostToDevice);
       // cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "Copy to GPU time");

        if(runGPUVersion0) {

            // Reset
            cudaMemset(matrix_d, 0, N*N*sizeof(int));
            cudaDeviceSynchronize();
            
            // Open File
            FILE *fp_gpu0;
    		fp_gpu0 = fopen("runtimes_gpu0.txt", "a");

            // Compute on GPU with version 0
            startTime(&timer);
	    //statt cdp here...

            nw_gpu0 <<<128,68>>>(reference_d, query_d, matrix_d, N);

            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTimeToFile(timer, fp_gpu0);
            printElapsedTime(timer, "GPU kernel time (version 0)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            cudaMemcpy(matrix_gpu, matrix_d, N*N*sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "Copy from GPU time");

            // Verify
            verify(matrix_cpu, matrix_gpu, N);
            
            // Close file
            fclose(fp_gpu0);

        }

	      /* if(runGPUVersion3) {

            // Reset
            cudaMemset(matrix_d, 0, N*N*sizeof(int));
            cudaDeviceSynchronize();

            // Open File
            FILE *fp_gpu3;
                fp_gpu3 = fopen("runtimes_gpu3.txt", "a");

            // Compute on GPU with version 3
            startTime(&timer);
            nw_gpu3_cdp<<<1,1>>>(reference_d, query_d, matrix_d, N);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTimeToFile(timer, fp_gpu3);
            printElapsedTime(timer, "GPU kernel time (version 3)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            cudaMemcpy(matrix_gpu, matrix_d, N*N*sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "Copy from GPU time");

            // Verify
            verify(matrix_cpu, matrix_gpu, N);

            // Close file
            fclose(fp_gpu3);

        }*/

        if(runGPUVersion1) {

            // Reset
            cudaMemset(matrix_d, 0, N*N*sizeof(int));
            cudaDeviceSynchronize();
            
            // Open File
            FILE *fp_gpu1;
    		fp_gpu1 = fopen("runtimes_gpu1.txt", "a");

            // Compute on GPU with version 1
            startTime(&timer);
            nw_gpu1<<<128,68>>>(reference_d, query_d, matrix_d, N);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTimeToFile(timer, fp_gpu1);
            printElapsedTime(timer, "GPU kernel time (version 1)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            cudaMemcpy(matrix_gpu, matrix_d, N*N*sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "Copy from GPU time");

            // Verify
            verify(matrix_cpu, matrix_gpu, N);
            
            // Close file
            fclose(fp_gpu1);

         }
        
        if(runGPUVersion2) {

            // Reset
            cudaMemset(matrix_d, 0, N*N*sizeof(int));
            cudaDeviceSynchronize();
            
            // Open File
            FILE *fp_gpu2;
    		fp_gpu2 = fopen("runtimes_gpu2.txt", "a");

            // Compute on GPU with version 2
            startTime(&timer);
            nw_gpu2<<<128,68>>>(reference_d, query_d, matrix_d, N);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTimeToFile(timer, fp_gpu2);
            printElapsedTime(timer, "GPU kernel time (version 2)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            cudaMemcpy(matrix_gpu, matrix_d, N*N*sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "Copy from GPU time");

            // Verify
            verify(matrix_cpu, matrix_gpu, N);
            
            // Close file
            fclose(fp_gpu2);

        }

        if(runGPUVersion3) {

            // Reset
            cudaMemset(matrix_d, 0, N*N*sizeof(int));
            cudaDeviceSynchronize();
            
            // Open File
            FILE *fp_gpu3;
    		fp_gpu3 = fopen("runtimes_gpu3.txt", "a");

            // Compute on GPU with version 3
            startTime(&timer);
            nw_gpu3<<<128,68>>>(reference_d, query_d, matrix_d, N);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTimeToFile(timer, fp_gpu3);
            printElapsedTime(timer, "GPU kernel time (version 3)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            cudaMemcpy(matrix_gpu, matrix_d, N*N*sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "Copy from GPU time");

            // Verify
            verify(matrix_cpu, matrix_gpu, N);
            
            // Close file
            fclose(fp_gpu3);

        }

        // Free GPU memory
        startTime(&timer);
        cudaFree(reference_d);
        cudaFree(query_d);
        cudaFree(matrix_d);
        cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "Deallocation time");

    }
    
    //Close File
    fclose(fp_seq);

    // Free memory
    free(reference);
    free(query);
    free(matrix_cpu);
    free(matrix_gpu);

    return 0;

}

