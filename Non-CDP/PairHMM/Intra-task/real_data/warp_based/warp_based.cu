
	#include <iostream>
	#include <stdlib.h>
	#include<limits>
	#include <stdio.h>
	#include <string.h>
	#include <time.h>
	#include <cuda.h>
	#include <stdint.h>
	#include <math.h>
	#include <unistd.h>
	#include <omp.h>	
	using namespace std;

	struct NUM_ADD
	{
		short2 read_haplotype_number;
		int address_array;
	};

	double diff(timespec start, timespec end)
	{
	  double a=0;
	 if((end.tv_nsec-start.tv_nsec)<0)
	{
	a=end.tv_sec-start.tv_sec-1;
	a+=(1000000000+end.tv_nsec-start.tv_nsec)/1000000000.0;
	}
	else
	{
	a=end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;

	}
	return a;

	}

	__global__ void  pairHMM( int size, char * data,  NUM_ADD * num_add, float * result) // what is the maximum number of parameters?
	{
	 int offset=blockIdx.x;
	
	__shared__ short2 read_haplotype_number;
	__shared__ char * read_base_array;
	__shared__ float * parameter_array;
	__shared__ char4 * haplotype_base_array; 
	 while(offset<size)
	 {	
		 float result_block=0;
		__shared__ int round;
		__shared__ int skip;
		//as each time it will deal with 2 read&haplotype pairs
		// each block deal with one pairs of haplotype & read
		if( threadIdx.x==0)
		{
		read_haplotype_number=num_add[offset].read_haplotype_number;
		read_base_array=(char *) (data+num_add[offset].address_array);
		parameter_array=(float *) (read_base_array+(read_haplotype_number.x+127)/128*128);
		skip=(sizeof(float)*read_haplotype_number.x+128-1)/128*128/sizeof(float);
		haplotype_base_array=(char4 *) (parameter_array+skip*4);
	//	printf("%d %d %d \n", read_haplotype_number.x,read_haplotype_number.y, num_add[offset].address_array);
	//	printf("%p %p %p\n",read_base_array, parameter_array, haplotype_base_array);		
	//	printf("skip=%d\n", skip);
		round=(read_haplotype_number.x+blockDim.x-1)/blockDim.x;
		}
		__syncthreads();	

		__shared__ char haplotype_base_in_char[500];
		int hh=(read_haplotype_number.y+4-1)/4;
		int tt=(hh+blockDim.x-1)/blockDim.x;
		for(int ii=0;ii<tt;ii++)
		{	
			int aa=threadIdx.x+ii*blockDim.x;
			if(aa< hh)
			{
			char4 haplotype_base_in_thread;
			haplotype_base_in_thread=haplotype_base_array[aa]; //Is it right to get data from global memory
			haplotype_base_in_char[aa*4]=haplotype_base_in_thread.x;
			haplotype_base_in_char[aa*4+1]=haplotype_base_in_thread.y;
			haplotype_base_in_char[aa*4+2]=haplotype_base_in_thread.z;
			haplotype_base_in_char[aa*4+3]=haplotype_base_in_thread.w;
			//printf("%c %c %c %c\n", haplotype_base_in_thread.x,haplotype_base_in_thread.y,haplotype_base_in_thread.z, haplotype_base_in_thread.w);
			}
		}
	      __syncthreads();
		float MM,DD,II;
		__shared__ float MM_stored[500]; //left   all the 160 should be equal to the size of block, should I use dynamic share memory   size of MM, DD and II shold be the size of the block. 
		__shared__ float DD_stored[500]; //left 
		__shared__ float II_stored[500]; //left 
	 	char read_base;
		float D_0=1.329228e+36/(float)read_haplotype_number.y;
		int read_number=read_haplotype_number.x;
		int round_size;
		for(int i=0;i<round;i++)
		{
			round_size=(read_number>blockDim.x)?blockDim.x: read_number;
			read_number=(read_number>blockDim.x)?read_number-blockDim.x:0; // read_num is the remaining length at this round
			char read_base;
			float M=1.0f; //now 
			float Qm,Qm_1,alpha,beta,delta,epsion,xiksi;//thet;
			if(threadIdx.x<round_size ) // tid is from 0 ~ round_size-1
			{
				read_base=read_base_array[threadIdx.x+blockDim.x*i];
				delta=parameter_array[threadIdx.x+blockDim.x*i+skip];
				xiksi=parameter_array[threadIdx.x+blockDim.x*i+2*skip];
				alpha=parameter_array[threadIdx.x+blockDim.x*i+3*skip];
				epsion=0.1;					
				beta=0.9;
				Qm=parameter_array[threadIdx.x+blockDim.x*i];
				Qm_1=M-Qm;
				Qm=fdividef(Qm,3.0f);
				//printf("%d %e %e %e %e %e %e \n",threadIdx.x, Qm_1, Qm, alpha, beta, delta, xiksi);
			}
			//why not use else break;?  Because we use __syncthreads() we need to make sure that all threads could reach that point
			M=0;
			float I=0; //now
			float D=0; //now
			float MMM=0;//up left
			float DDD=0;//up left
			float III=0;//up left
		
			if(threadIdx.x==0&&i==0) DDD=D_0; // Just in the first round, it need to be D_0
			
			int current_haplotype_id=0;
			for(int j=0;j<round_size+read_haplotype_number.y-1;j++)
			{ 
				int aa=j-threadIdx.x;	
				if( aa>=0 && (current_haplotype_id<read_haplotype_number.y))
				{
					if(threadIdx.x==0)
					{	
					if(i>0)
					{
					MM=MM_stored[current_haplotype_id];
					II=II_stored[current_haplotype_id];
					DD=DD_stored[current_haplotype_id];
					}
					else
					{
					MM=0;
					II=0;
					DD=D_0;
					}
					}	
					float MID=__fadd_rn(III,DDD);
					DDD=DD;
					III=II;
					float DDM=__fmul_rn(M,xiksi);
					float IIMI=__fmul_rn(II,epsion);
					float MIIDD=__fmul_rn(beta,MID);
					char haplotype_base_each=haplotype_base_in_char[current_haplotype_id];
					float aa=(haplotype_base_each==read_base)? Qm_1:Qm;
					D=__fmaf_rn(D,epsion,DDM);
					//D=__fmaf_rn(D,thet,DDM);
					I=__fmaf_rn(MM,delta,IIMI);
					float MMID=__fmaf_rn(alpha,MMM,MIIDD);
					MMM=MM;
					current_haplotype_id++;
					M=__fmul_rn(aa,MMID);
					II=I;
					DD=D;
					MM=M;
				 }
				if(threadIdx.x==round_size-1 && i<round-1) // tid is the last thread but there are more round
                                {
                                        MM_stored[current_haplotype_id-1]=M;
                                        II_stored[current_haplotype_id-1]=I;
                                        DD_stored[current_haplotype_id-1]=D;
                                }

				if(threadIdx.x==round_size-1 && i==round-1)
					result_block=__fadd_rn(result_block,__fadd_rn(M,I));
				MM=__shfl_up(MM,1);
                                II=__shfl_up(II,1);
                                DD=__shfl_up(DD,1);
			}
		}
		if(threadIdx.x==round_size-1) 
		{
			result[offset]=result_block;
		}	
		offset+=gridDim.x;	
	 }
}

struct InputData
{
int read_size;
char read_base[260];
char base_quals[260];
char ins_quals[260];
char del_quals[260];
char gcp_quals[260];
int haplotype_size;
char haplotype_base[500];
};

	int main(int argc, char * argv[])
	{
		int INI=(log10f((std::numeric_limits<float>::max() / 16)));
		cudaFree(0);
		int size_each_for=4000000;
		//scanf("%d", &size_each_for);
		struct timespec start,finish;
		struct timespec start_total,finish_total;
		double total_time=0;
		double  computation_time=0,mem_cpy_time=0,read_time=0, data_prepare=0;
		FILE * file;
	//	file=fopen("pairHMM_input_store.txt","r");
	//	file=fopen("32_data.txt","r");
	//	file=fopen(argv[1], "r");
	//	file=fopen("/data/04068/sren/dir_chromosome-10/b.txt","r");
		file=fopen("a.txt","r");
		int size;
		fscanf(file,"%d",&size);
		clock_gettime(CLOCK_MONOTONIC_RAW,&start); 
		float ph2pr_h[128];
		for(int i=0;i<128;i++)
		{
			ph2pr_h[i]=powf(10.f, -((float)i) / 10.f);
		}
		
		clock_gettime(CLOCK_MONOTONIC_RAW,&finish);	
		data_prepare+=diff(start,finish);

		int total=0;
		float  read_read, haplotype_haplotype;
		
		while(!feof(file))
		{
			total+=size;
			char useless;
			useless=fgetc(file);
			
			clock_gettime(CLOCK_MONOTONIC_RAW,&start); 
			//scanf("%d", &fakesize);
			InputData *inputdata=(InputData* )malloc(size*(sizeof(InputData)));		
		//	int size_each_for=1000;
			for(int i=0;i<size;i++)
			{
				int read_size;
				fscanf(file,"%d\n",&inputdata[i].read_size);
				fscanf(file,"%s ",inputdata[i].read_base);
				read_size=inputdata[i].read_size;
			//	if(read_size>200) 
			//	printf("read size is bigger than 200: size is %d \n", read_size);
				read_read=read_size;
				for(int j=0;j<read_size;j++)
				{
				 int  aa;
				 fscanf(file,"%d ",&aa);
				 inputdata[i]. base_quals[j]=(char)aa;
				}

				for(int j=0;j<read_size;j++)
				{
				 int  aa;
				 fscanf(file,"%d ",&aa);
				 inputdata[i].ins_quals[j]=(char)aa;
				}
				for(int j=0;j<read_size;j++)
				{
				 int  aa;
				 fscanf(file,"%d ",&aa);
				 inputdata[i].del_quals[j]=(char)aa;
				}

				for(int j=0;j<read_size;j++)
				{
				 int  aa;
				if(j<read_size-1) fscanf(file,"%d ",&aa);
				else  fscanf(file,"%d \n",&aa);
				 inputdata[i].gcp_quals[j]=(char)aa;
				}

				fscanf(file,"%d\n",&inputdata[i].haplotype_size);
				fscanf(file, "%s\n",inputdata[i].haplotype_base);
				haplotype_haplotype=inputdata[i].haplotype_size;
			}
			clock_gettime(CLOCK_MONOTONIC_RAW,&finish);
			read_time+=diff(start,finish);
				
			float * result_h=(float *) malloc(sizeof(float)*size); //on cpu
                        clock_gettime(CLOCK_MONOTONIC_RAW,&start_total);

			
			char * data_h_total;
            		char * result_d_total;
			
			//printf("size=%d\n",size *700* sizeof (char)+size*200*4*sizeof(float)+size*sizeof(NUM_ADD) );
			int memory_malloc_size=(size*260+127)/128*128; //read_base
			memory_malloc_size+=(size*500+127)/128*128; // haplotyp_base;
			memory_malloc_size+=(size*260*4+127)/128*128;//parameter1;
			memory_malloc_size+=(size*260*4+127)/128*128;//parameter2;
			memory_malloc_size+=(size*260*4+127)/128*128;//parameter3;
			memory_malloc_size+=(size*260*4+127)/128*128;//parameter4;
			memory_malloc_size+=(size*4+127)/128*128;//result;
			memory_malloc_size+=(size*sizeof(NUM_ADD)+127)/128*128;//NUM_ADD;
			
			//printf("%d\n",memory_malloc_size);
			data_h_total=(char*)malloc(memory_malloc_size); //on cpu 
			cudaError err;
			err=cudaMalloc( (char **) &result_d_total, memory_malloc_size);
                        if(err!=cudaSuccess)
    			printf( "Error %d: %s!\n", err, cudaGetErrorString(err) );
			//printf("%p   %p  \n", result_d_total,result_d_total+memory_malloc_size);      
			//float * result_h=(float *) malloc(sizeof(float)*size); //on cpu
			char * data_d_total=result_d_total+(size*sizeof(float)+127)/128*128;  //on GPU

			//int num_streams=(size+size_each_for-1)/size_each_for;
			//cudaStream_t * streams=(cudaStream_t *) malloc(num_streams*sizeof(cudaStream_t));
		       //for(int aaa=0;aaa<num_streams;aaa++)
 	               //cudaStreamCreate(&streams[aaa]);
			
			//for(int aaa=0;aaa<num_streams;aaa++)
			//{
			//int size_in_each=size_each_for;
			//if(aaa==num_streams-1)
			//	size_in_each=size-aaa*size_each_for;
			
			//char * data_h=data_h_total+base*1500*sizeof(char)+base*sizeof(NUM_ADD);
			//char * data_h_begin=data_h; 
			char * data_h=data_h_total;  //cpu
			char * data_h_begin=data_h;  //cpu
			NUM_ADD *data_num_add=(NUM_ADD *) (data_h); //cpu
			
			data_h=data_h+(size*sizeof(NUM_ADD)+127)/128*128; // it is 64*x .thus we donot need to worry about alignment.
		
			int data_size=0;
			for(int i=0;i<size;i++)
			{
				int read_size=inputdata[i].read_size;
				int skip=(sizeof(float)*read_size+128-1)/128*128/sizeof(float);
				//float * parameter=(float *) malloc(skip*sizeof(float)*4);
				float parameter[1040];
				for(int j=0;j<read_size;j++)
				{
				    parameter[j]= ph2pr_h[inputdata[i].base_quals[j]&127 ];     //QM
				    parameter[j+skip]=ph2pr_h[inputdata[i].ins_quals[j]&127];      //Qi
				    parameter[j+skip*2]=ph2pr_h[inputdata[i].del_quals[j]&127];    //QD
				    parameter[j+skip*3]=1.0f-ph2pr_h[((int)(inputdata[i].ins_quals[j]&127)+(int)(inputdata[i].del_quals[j]&127))&127];  //alpha
			//	printf("%e %e %e %e\n", parameter[j],parameter[j+read_size], parameter[j+read_size*2],parameter[j+read_size*3]);
				}	
				
				char read_base_new[260];
				for(int j=0;j<read_size;j++)
				{	
				read_base_new[j]=inputdata[i].read_base[j];
				}	
		
				int haplotype_new_size=(inputdata[i].haplotype_size+4-1)/4;
				char4 haplotype_base_new[150];;
				for(int j=0;j<haplotype_new_size;j++)
				{
					haplotype_base_new[j].x=inputdata[i].haplotype_base[j*4];
					if(j*4+1<inputdata[i].haplotype_size)
					haplotype_base_new[j].y=inputdata[i].haplotype_base[j*4+1];
					if(j*4+2<inputdata[i].haplotype_size)
					haplotype_base_new[j].z=inputdata[i].haplotype_base[j*4+2];
					if(j*4+3<inputdata[i].haplotype_size)
					haplotype_base_new[j].w=inputdata[i].haplotype_base[j*4+3];			
				}

				data_num_add[i].read_haplotype_number.x=inputdata[i].read_size;
				data_num_add[i].read_haplotype_number.y=inputdata[i].haplotype_size;
				data_num_add[i].address_array=data_size;
					
				//read base
				memcpy(data_h,read_base_new,sizeof(char)*read_size);
				data_h+=(read_size+128-1)/128*128;
				data_size+=(read_size+128-1)/128*128;
				//printf("data_size=%d\n", data_size);
				//Parameter
				memcpy(data_h,parameter,sizeof(float) *skip*4);
				data_h+=sizeof(float) *skip*4;
				data_size+=sizeof(float) *skip*4;
				//printf("data_size=%d\n", data_size);
				
				//haplotype
				memcpy(data_h,haplotype_base_new,sizeof(char4)* haplotype_new_size);
				data_h+=(haplotype_new_size*sizeof(char4)+128-1)/128*128;
				data_size+=(haplotype_new_size*sizeof(char4)+128-1)/128*128;
				//printf("data_size=%d\n", data_size);
			}
			
			int data_size_to_copy=data_size+(size*sizeof(NUM_ADD)+127)/128*128;			
			char * data_d;
			float * result_d=(float *) (result_d_total);	
		//	cudaMemcpyAsync(data_d,data_h_begin,data_size_to_copy,cudaMemcpyHostToDevice,streams[aaa]);
			//printf("size_to_copy=%d\n", data_size_to_copy);
		//  call kernel
			int blocksize=32;
			int gridsize=size;
			//printf("before call\n");
			NUM_ADD * num_add_d=(NUM_ADD *) (data_d_total);
			data_d=data_d_total+(sizeof(NUM_ADD)*size+127)/128*128;
			err=cudaMemcpy(data_d_total,data_h_begin,data_size_to_copy,cudaMemcpyHostToDevice);
			if(err!=cudaSuccess)
    			printf( "Error %d: %s!\n", err, cudaGetErrorString(err) );
			clock_gettime(CLOCK_MONOTONIC_RAW,&start); 
			pairHMM<<<gridsize,blocksize>>> (size,data_d,num_add_d,result_d);
			//cudaDeviceSynchronize();
        		cudaMemcpy(result_h,result_d_total,size*sizeof(float),cudaMemcpyDeviceToHost);
 			clock_gettime(CLOCK_MONOTONIC_RAW,&finish);
			computation_time+=diff(start,finish);
			for(int i=0;i<size;i++)
			  float aa=(log10f((double)result_h[i]) - INI);		
	//printf("i=%d %e\n",i,result_h[i]);
		free(data_h_total);
		err=cudaFree(result_d_total);
		if(err!=cudaSuccess)
    		printf( "Error %d: %s!\n", err, cudaGetErrorString(err) );	
	
		clock_gettime(CLOCK_MONOTONIC_RAW,&finish_total);
                total_time+=diff(start_total,finish_total);

			
		 free(inputdata);
		 free(result_h);
		 fscanf(file,"%d",&size);
	//	if(total>10000)
	//		break;
		}//end of while
		
		clock_gettime(CLOCK_MONOTONIC_RAW,&start);
    
	 	cudaDeviceReset();
		clock_gettime(CLOCK_MONOTONIC_RAW,&finish);
		mem_cpy_time+=diff(start,finish);//(finish1.tv_nsec-start1.tv_nsec)/1000000000.0;

	//	printf("size %d\n",total);
		printf("read_time=%e  initial_time=%e  computation_time= %e total_time=%e\n",read_time, data_prepare,computation_time, computation_time+mem_cpy_time+data_prepare);
	//	printf("%d %d %d  %e\n", fakesize, read_read, haplotype_haplotype,computation_time);	
	  //  printf("GCUPS: %lf \n",  fakesize*read_read*haplotype_haplotype/computation_time/1000000000);
		printf("Total time=%e\n",total_time);
		return 0;
	}

