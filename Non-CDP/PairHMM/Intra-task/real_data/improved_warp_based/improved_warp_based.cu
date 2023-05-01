
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

	__global__ void  pairHMM( int size, char * data,  NUM_ADD * num_add, float * result,float * MG,float * DG, float * IG) // what is the maximum number of parameters?
	{
   	 int warp_index=(blockDim.x*blockIdx.x+threadIdx.x)/32;
    	 int warp_index_in_block=threadIdx.x/32;
    	 int offset=warp_index;
    //printf("threadIdx.x=%d \n", threadIdx.x);
         MG=MG+warp_index*500;
         IG=IG+warp_index*500;
         DG=DG+warp_index*500;

	 while(offset<size)
	 {	
		short2 read_haplotype_number;
		char * read_base_array;
		float * parameter_array;
		char4 * haplotype_base_array; 

		float result_block=0;
		int round;
	        int skip;
		//as each time it will deal with 2 read&haplotype pairs
		// each block deal with one pairs of haplotype & read
		read_haplotype_number=num_add[offset].read_haplotype_number;
		skip=(sizeof(float)*read_haplotype_number.x+127)/128*32;
		read_base_array=(char *) (data+num_add[offset].address_array);
		parameter_array=(float *) (read_base_array+(read_haplotype_number.x+127)/128*128);
		haplotype_base_array=(char4 *) (parameter_array+skip*4);
		int read_number=read_haplotype_number.x;
		int haplotype_number=read_haplotype_number.y;
		
		__shared__ char haplotype_base_in_char[500*128/32];
		
		int hh=(haplotype_number+4-1)/4;
		int thread_in_warp=threadIdx.x-warp_index_in_block*32;
		int tt=(hh+32-1)/32;
		for(int ii=0;ii<tt;ii++)
		{	
			int aa=thread_in_warp+ii*32;
			if(aa< hh)
			{
			char4 haplotype_base_in_thread;
			haplotype_base_in_thread=haplotype_base_array[aa]; //Is it right to get data from global memory
			haplotype_base_in_char[warp_index_in_block*500+aa*4]=haplotype_base_in_thread.x;
			haplotype_base_in_char[warp_index_in_block*500+aa*4+1]=haplotype_base_in_thread.y;
			haplotype_base_in_char[warp_index_in_block*500+aa*4+2]=haplotype_base_in_thread.z;
			haplotype_base_in_char[warp_index_in_block*500+aa*4+3]=haplotype_base_in_thread.w;
			//printf("%c %c %c %c\n", haplotype_base_in_thread.x,haplotype_base_in_thread.y,haplotype_base_in_thread.z, haplotype_base_in_thread.w);
			}
		}
	      //__syncthreads();
		
		float D_0=1.329228e+36/(float)haplotype_number;
		round=(read_number+32-1)/32;
		int round_size;
		for(int i=0;i<round;i++)
		{
			round_size=(read_number>32)?32: read_number;
			read_number=(read_number>32)?read_number-32:0; // read_num is the remaining length at this round
			char read_base;
			float M=1.0f; //now 
			float Qm,Qm_1,alpha,delta,xiksi;//thet;
			if(thread_in_warp<round_size ) // tid is from 0 ~ round_size-1
			{
				read_base=read_base_array[thread_in_warp+32*i];
				Qm=parameter_array[thread_in_warp+32*i];
				delta=parameter_array[thread_in_warp+32*i+skip];
				Qm_1=M-Qm;
				Qm=fdividef(Qm,3.0f);
				xiksi=parameter_array[thread_in_warp+32*i+2*skip];
				alpha=parameter_array[thread_in_warp+32*i+3*skip];
				//epsion=0.1;					
				//beta=0.9;
								//printf("%d %e %e %e %e %e %e \n",threadIdx.x, Qm_1, Qm, alpha, beta, delta, xiksi);
			}
			//why not use else break;?  Because we use __syncthreads() we need to make sure that all threads could reach that point
			M=0;
			float I=0; //now
			float D=0; //now
			float MMID=0;
			
			if(thread_in_warp==0&&i==0) MMID=__fmul_rn(0.9,D_0); // Just in the first round, it need to be D_0
			
			int current_haplotype_id=0;
			for(int j=0;j<round_size+haplotype_number-1;j++)
			{ 
				int aa=j-thread_in_warp;	
				float MM,DD,II;
				if( aa>=0 && (current_haplotype_id<haplotype_number))
				{
					
					if(thread_in_warp==0)
					{	
					if(i>0)
					{
					MM=MG[current_haplotype_id];
					II=IG[current_haplotype_id];
					DD=DG[current_haplotype_id];
					}
					else
					{
					MM=0;
					II=0;
					DD=D_0;
					}
					}	
					char haplotype_base_each=haplotype_base_in_char[warp_index_in_block*500+current_haplotype_id];
					float aa=(haplotype_base_each==read_base)? Qm_1:Qm;
				
					float MID=__fadd_rn(II,DD);
					float DDM=__fmul_rn(M,xiksi);
					float IIMI=__fmul_rn(II,0.1);
					M=__fmul_rn(aa,MMID);
					
					float MIIDD=__fmul_rn(0.9,MID);
					D=__fmaf_rn(D,0.1,DDM);
					I=__fmaf_rn(MM,delta,IIMI);
					current_haplotype_id++;
					II=I;
					DD=D;
					MMID=__fmaf_rn(alpha,MM,MIIDD);
					MM=M;
				 }
				if(thread_in_warp==round_size-1 && i<round-1) // tid is the last thread but there are more round
                                {
                                        MG[current_haplotype_id-1]=M;
                                        IG[current_haplotype_id-1]=I;
                                        DG[current_haplotype_id-1]=D;
                                }

				if(thread_in_warp==round_size-1 && i==round-1)
					result_block=__fadd_rn(result_block,__fadd_rn(M,I));
				MM=__shfl_up(MM,1);
                                II=__shfl_up(II,1);
                                DD=__shfl_up(DD,1);
			}
		}
		if(thread_in_warp==round_size-1) 
		{
			result[offset]=result_block;
		}	
		offset+=blockDim.x*gridDim.x/32;	
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
		float * MG;
            	float * DG;
		float * IG;
		cudaMalloc( (float **)& MG,sizeof(float) *4*150*500*3);
	        DG=MG+150*4*500;// ????
            	IG=DG+150*4*500;  //?????
			
		int size_each_for=4000000;
		//scanf("%d", &size_each_for);
		struct timespec start,finish,start_all,finish_all;
		double  computation_time=0,mem_cpy_time=0,read_time=0, data_prepare=0;
		FILE * file;
	//	file=fopen("a.txt","r");
	//	file=fopen("pairHMM_input_store.txt","r");
	//	file=fopen("32_data.txt","r");
		file=fopen("/data/04068/sren/dir_chromosome-10/b.txt", "r");
	//	file=fopen(argv[1],"r");
		//printf("OK\n");
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
		double total_time=0;
		while(!feof(file))
		{
			total+=size;
			char useless;
			useless=fgetc(file);
			
			clock_gettime(CLOCK_MONOTONIC_RAW,&start); 
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
		

	
			clock_gettime(CLOCK_MONOTONIC_RAW,&start_all);
			
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
			
			
			data_h_total=(char*)malloc(memory_malloc_size); //on cpu 
			cudaError err;
			err=cudaMalloc( (char **) &result_d_total, memory_malloc_size);
                        if(err!=cudaSuccess)
    			printf( "Error %d: %s!\n", err, cudaGetErrorString(err) );
			//printf("%p   %p  \n", result_d_total,result_d_total+memory_malloc_size);      
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
			//printf("before call\n");
		
				
			int data_size_to_copy=data_size+(size*sizeof(NUM_ADD)+127)/128*128;			
			char * data_d;
			float * result_d=(float *) (result_d_total);	
		
			NUM_ADD * num_add_d=(NUM_ADD *) (data_d_total);
			data_d=data_d_total+(sizeof(NUM_ADD)*size+127)/128*128;
                        clock_gettime(CLOCK_MONOTONIC_RAW,&start);
			err=cudaMemcpy(data_d_total,data_h_begin,data_size_to_copy,cudaMemcpyHostToDevice);
			if(err!=cudaSuccess)
    			printf( "Error %d: %s!\n", err, cudaGetErrorString(err) );

 			clock_gettime(CLOCK_MONOTONIC_RAW,&finish);
			mem_cpy_time+=diff(start,finish);	
			//  call kernel
			int blocksize=128;
		//	int gridsize=1+(size/(blocksize/32));
			int gridsize=150;	
		
			clock_gettime(CLOCK_MONOTONIC_RAW,&start);
			pairHMM<<<gridsize,blocksize>>> (size,data_d,num_add_d,result_d,MG,DG,IG);
        		cudaMemcpy(result_h,result_d_total,size*sizeof(float),cudaMemcpyDeviceToHost);
 			
			clock_gettime(CLOCK_MONOTONIC_RAW,&finish);
			computation_time+=diff(start,finish);
		// }
        //for(int aaa=0;aaa<num_streams;aaa++)
            //cudaStreamDestroy(streams[aaa]);
		    for(int i=0;i<size;i++)
 		float aa=(log10f((double)result_h[i]) - INI);  
	//	 printf("  i=%d  %e\n",i, result_h[i]);
	//	 printf("result_d_total=%p\n", result_d_total);
		
		free(data_h_total);
		err=cudaFree(result_d_total);
		if(err!=cudaSuccess)
    		printf( "Error %d: %s!\n", err, cudaGetErrorString(err) );	
			
		clock_gettime(CLOCK_MONOTONIC_RAW,&finish_all);
		total_time+=diff(start_all, finish_all);		

		 free(inputdata);
		 free(result_h);
		 fscanf(file,"%d",&size);
		//printf("%d\n",size);
	//	if(total>10000000)
	//	break;
		}//end of while
		
		clock_gettime(CLOCK_MONOTONIC_RAW,&start);
    
		cudaFree(MG);
	 	cudaDeviceReset();
		clock_gettime(CLOCK_MONOTONIC_RAW,&finish);
		mem_cpy_time+=diff(start,finish);//(finish1.tv_nsec-start1.tv_nsec)/1000000000.0;

	//	printf("size %d\n",total);
		printf("read_time=%e  initial_time=%e  computation_time= %e total_time=%e\n",read_time, data_prepare,computation_time, computation_time+mem_cpy_time);
		printf("total time=%e\n",total_time); 
	
	//   printf("GCUPS: %lf \n",  size*read_read*haplotype_haplotype/computation_time/1000000000);

		return 0;
	}

