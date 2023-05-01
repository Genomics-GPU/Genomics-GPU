	#include <iostream>
	#include<limits>
	#include <stdlib.h>
	#include <stdio.h>
	#include <string.h>
	#include <time.h>
	#include <cuda.h>
	#include <stdint.h>
	#include <math.h>
	#include <unistd.h>
	#include <omp.h>	
	 #include <algorithm>


	using namespace std;

	// 8 byte.   how to be 128byte?
	struct NUM_ADD
	{
		int read_number;
		int haplotype_number;	
	//	short2 read_haplotype;
		int  Read_array;
		int  read_large_length;
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

	__constant__ float  constant[10];
	__constant__ int  constant_int[10];
	

	__global__ void  pairHMM( int size, char * data,  NUM_ADD * num_add, float * result,float * MG,float * DG, float * IG ) // what is the maximum number of parameters?
	{
	//MG, DG and IG are global memory to store indermediate result?
	//each thread finish one computation		
	int offset=blockIdx.x*blockDim.x+threadIdx.x;
	MG=MG+offset;
	IG=IG+offset;
	DG=DG+offset;
	//if(threadIdx.x==0)
	//printf("%d %d %d %d %d\n", constant_int[0],constant_int[1], constant_int[2],constant_int[3], constant_int[4]);
	while(offset<size)
	 {	
		__shared__ float parameter1[256];
		__shared__ float parameter2[256];
		__shared__ float parameter3[256];
		__shared__ float parameter4[256];
	
		//NUM_ADD number_address;
		//number_address=num_add[offset];//get from global memory
		int read_large_length=num_add[offset].read_large_length;
		char4 * read_base_array=(char4 *)(data+num_add[offset].Read_array); // to caculate the address of read_base_array. 
		int  read_number=num_add[offset].read_number;
		int  haplotype_number=num_add[offset].haplotype_number;
		float  *parameter1_array=(float *) (read_base_array+(read_large_length+3)/4*32);
		read_large_length=read_large_length*32;
		float  *parameter2_array=(float *) (parameter1_array+read_large_length);
		float  *parameter3_array=(float *) (parameter1_array+read_large_length*2);
		float  *parameter4_array=(float *) (parameter1_array+read_large_length*3);
		char4 * haplotype_base_array=(char4 * )(parameter1_array+read_large_length*4);    	
		//haplotype is 4 byte. Thus, in a warp it is 4*32=128 byte. //we need to change the struct of haplotype

		float result_block=constant[5];
		char4 read_base_4;
		int i;
		
		char4 read_haplotype_base;
		//if number_address.read_number is even
		for(i=0;i<read_number/2;i++)
		{	
		//got read_base from globle memory  (which is 32*4 (char4) = 128 bytes )
		if(i%2==0)
		read_base_4=read_base_array[i/2*constant_int[2]];
		if(i%2==0)
                {
                read_haplotype_base.x=read_base_4.x;
                read_haplotype_base.z=read_base_4.y;
                }
                if(i%2==1)
                {
                read_haplotype_base.x=read_base_4.z;
                read_haplotype_base.z=read_base_4.w;
                }

		int skip=i*2*constant_int[2];
		parameter1[threadIdx.x]=parameter1_array[skip];
		parameter2[threadIdx.x]=parameter2_array[skip];
		parameter3[threadIdx.x]=parameter3_array[skip];
		parameter4[threadIdx.x]=parameter4_array[skip];
		skip+=constant_int[2];
		parameter1[threadIdx.x+blockDim.x]=parameter1_array[skip];
		parameter2[threadIdx.x+blockDim.x]=parameter2_array[skip];
		parameter3[threadIdx.x+blockDim.x]=parameter3_array[skip];
		parameter4[threadIdx.x+blockDim.x]=parameter4_array[skip];
	
		float Ml=constant[5];// left M;
		float Dl=constant[5];// left D;
		float Il=constant[5];// left I
		float M2=constant[5]; //left M2
		float D2=constant[5]; //left D2

		float MMID=constant[5];
		float MMID2=constant[5];
		float MU=constant[5];// up M;
		float IU=constant[5];// up I;
		float DU=constant[5];// up D;
		
		int hh=(haplotype_number+3)/4;
		for(int j=0;j<hh;j++)
		{
		char4 haplotype_base;
		haplotype_base=haplotype_base_array[j*constant_int[2]]; 
		
		for(int kk=0;kk<4;kk++)
		{
		
			if(j*4+kk==haplotype_number)
				break;
			if(kk==0)
                             read_haplotype_base.y=haplotype_base.x;
                        if(kk==1)
                             read_haplotype_base.y=haplotype_base.y;
                        if(kk==2)
                             read_haplotype_base.y=haplotype_base.z;
                        if(kk==3)
                             read_haplotype_base.y=haplotype_base.w;
				
			int index=(j*4+kk)*blockDim.x*gridDim.x;
			if(i>0)
			{
			//here should not using offset. But using the 
					//get MU,IU,DU from global memory
				MU=MG[index];
				IU=IG[index];
				DU=DG[index];
			}
			else
			{
				DU= constant[0]  /(float)haplotype_number;
				MMID=__fmul_rn(constant[3],DU);
			}
		
			float Qm,Qm_1,alpha,delta,xiksi;
			Qm=parameter1[threadIdx.x];	
			Qm_1=constant[1]-Qm;
			delta=parameter2[threadIdx.x];
			xiksi=parameter3[threadIdx.x];
			alpha=parameter4[threadIdx.x];
			Qm=fdividef(Qm,constant[2]);
		
			float MID=__fadd_rn(IU,DU);
			float DDM=__fmul_rn(Ml,xiksi);
			float IIMI=__fmul_rn(IU,constant[4]);
		
			float aa=(read_haplotype_base.y==read_haplotype_base.x)? Qm_1:Qm;
			float MIIDD=__fmul_rn(constant[3],MID);
			Ml=__fmul_rn(aa,MMID);
			Dl=__fmaf_rn(Dl,constant[4],DDM);
			Il=__fmaf_rn(MU,delta,IIMI);
			MMID=__fmaf_rn(alpha,MU,MIIDD);
	
			skip=threadIdx.x+blockDim.x;	
			Qm=parameter1[skip];	
			Qm_1=constant[1]-Qm;
			delta=parameter2[skip];
			Qm=fdividef(Qm,constant[2]);
			xiksi=parameter3[skip];
			alpha=parameter4[skip];
		
			 MID=__fadd_rn(Il,Dl);
			 DDM=__fmul_rn(M2,xiksi);
			 IIMI=__fmul_rn(Il,constant[4]);
			 aa=(read_haplotype_base.y==read_haplotype_base.z)?Qm_1:Qm;
			 MIIDD=__fmul_rn(constant[3], MID);	

			 M2=__fmul_rn(aa,MMID2);
			 D2=__fmaf_rn(D2,constant[4],DDM);
			 Il=__fmaf_rn(Ml,delta,IIMI);
			 MMID2=__fmaf_rn(alpha,Ml,MIIDD);
				
			if(i==read_number/2-1 && read_number%2==0)
			{
			result_block=__fadd_rn(result_block,__fadd_rn(M2,Il));
			}
			else
			{
				MG[index]=M2;
				IG[index]=Il;
				DG[index]=D2;
			}
		}//4
		}//haplotype
		}//read /2

		//following is only  48 registers
		for(i=i*2;i<read_number;i++)
                {
                
		if(i%4==0)
                {
                        read_base_4=read_base_array[i/4*constant_int[2]];
                }
                
		if(i%4==0) read_haplotype_base.x=read_base_4.x;
                if(i%4==1) read_haplotype_base.x=read_base_4.y;
                if(i%4==2) read_haplotype_base.x=read_base_4.z;
                if(i%4==3) read_haplotype_base.x=read_base_4.w;

                float Qm,Qm_1,alpha,delta,xiksi;
        		
		Qm=parameter1_array[i*constant_int[2]];	
		delta=parameter2_array[i*constant_int[2]];
		Qm_1=constant[1]-Qm;
		xiksi=parameter3_array[i*constant_int[2]];
		alpha=parameter4_array[i*constant_int[2]];
		Qm=fdividef(Qm,constant[2]);

                float Ml=0;// left M;
                float Dl=0;// left D;
                float Il=0;
                float MU=0;// up M;
                float IU=0;// up I;
                float DU=0;// up D;
                float MMID=0;

		 if(i==0)
                {
                DU=constant[0]/(float) haplotype_number;
                MMID=__fmul_rn(constant[3],DU);
                }

                int hh=(haplotype_number+4-1)/4;
                for(int j=0;j<hh;j++)
                {
                char4 haplotype_base;
                haplotype_base=haplotype_base_array[j*constant_int[2]];

                for(int kk=0;kk<4;kk++)
                {
                                if(j*4+kk==haplotype_number)
                                        break;

                                int index=(j*4+kk)*blockDim.x*gridDim.x;
                                if(i>0)
                                {
                                        //here should not using offset. But using the
                                        //get MU,IU,DU from global memory
                                        MU=MG[index];
                                        IU=IG[index];
                                        DU=DG[index];
                                }

                                float MID=__fadd_rn(IU,DU);
                                float DDM=__fmul_rn(Ml,xiksi);
                                float IIMI=__fmul_rn(IU,constant[4]);
                                if(kk==0)
                                       read_haplotype_base.y=haplotype_base.x;
                                if(kk==1)
                                        read_haplotype_base.y=haplotype_base.y;

				 if(kk==2)
                                        read_haplotype_base.y=haplotype_base.z;
                                if(kk==3)
                                        read_haplotype_base.y=haplotype_base.w;

                                float aa=(read_haplotype_base.y==read_haplotype_base.x)? Qm_1:Qm;

                                float MIIDD=__fmul_rn(constant[3],MID);
                                Ml=__fmul_rn(aa,MMID);
                                Il=__fmaf_rn(MU,delta,IIMI);
                                Dl=__fmaf_rn(Dl,constant[4],DDM);

                                MMID=__fmaf_rn(alpha,MU,MIIDD);

                                if(i<read_number-1)
                                {
                                MG[index]=Ml;
                                IG[index]=Il;
                                DG[index]=Dl;
                                }
                                else
                                        result_block=__fadd_rn(result_block,__fadd_rn(Ml,Il));
                        }//4
                } //haplotype

                }//read

		result[offset]=result_block;
		offset+=gridDim.x*blockDim.x ;	
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

bool operator<(const InputData &a, const InputData &b)
{
 //   return x.point_value > y.point_value;
        if(a.read_size<b.read_size) return true;
        if(a.read_size==b.read_size) return a.haplotype_size<b.haplotype_size;
        else
        return false;

}




int main(int argc, char * argv[])
{
		int INI=(log10f((std::numeric_limits<float>::max() / 16)));
		//printf("input value of size_each_for \n");
		//scanf("%d", &size_each_for);
		struct timespec start,finish;
		double  computation_time=0,mem_cpy_time=0,read_time=0, data_prepare=0;
		double total_time=0;
		float * MG;
		float * DG;
		float * IG;
		cudaMalloc( (float **)& MG,sizeof(float) *128*105*500*3);
		DG=MG+128*105*500;// ????
		IG=DG+128*105*500;  //?????

		FILE * file;
	//	file=fopen("pairHMM_input_store.txt","r");
	//	file=fopen(argv[1],"r");
		//file=fopen("32_data.txt","r");
	//	file=fopen("less.txt","r");
	file=fopen("/data/04068/sren/dir_chromosome-10/b.txt","r");
	//	 file=fopen("../a.txt","r");
		int size;
		fscanf(file,"%d",&size);

		clock_gettime(CLOCK_MONOTONIC_RAW,&start); 
		float ph2pr_h[128];
		for(int i=0;i<128;i++)
		{
			ph2pr_h[i]=powf(10.f, -((float)i) / 10.f);
		}
		cudaError err;
		
		int  constants_h_int[10];
		float constants_h[10];
		constants_h[0]=1.329228e+36;
		constants_h[1]=1.0;
		constants_h[2]=3.0;
		constants_h[3]=0.9;
		constants_h[4]=0.1;
		constants_h[5]=0.0;
		constants_h_int[0]=0;
		constants_h_int[1]=128;
		constants_h_int[2]=32;
		constants_h_int[3]=4;
		constants_h_int[4]=3;

		cudaMemcpyToSymbol(constant,constants_h,sizeof(float)*10 );
		cudaMemcpyToSymbol(constant_int,constants_h_int,sizeof(int)*10 );
			
	
		clock_gettime(CLOCK_MONOTONIC_RAW,&finish);	
		data_prepare+=diff(start,finish);
		
		int total=0;
		char * result_d_total;
		float read_read, haplotype_haplotype;
		while(!feof(file))
		{
			total+=size;
			char useless;
			useless=fgetc(file);
			
			clock_gettime(CLOCK_MONOTONIC_RAW,&start); 
			
			InputData *inputdata=(InputData* )malloc(size*(sizeof(InputData)));		
			for(int i=0;i<size;i++)
			{
				int read_size;
				fscanf(file,"%d\n",&inputdata[i].read_size);
				fscanf(file,"%s ",inputdata[i].read_base);
				read_size=inputdata[i].read_size;
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
			
   			float * result_h=(float *) malloc(sizeof(float)*size);
		
			struct timespec start_total,finish_total;
		
			clock_gettime(CLOCK_MONOTONIC_RAW,&start_total); 
			char * data_h_total;

			   std::sort(inputdata, inputdata+size);	
			//32 one chunck.
			int malloc_size_for_each_chunk=(65*4*32+260*4*32*4+125*4*32) ;
			int total_size=(size+31)/32*malloc_size_for_each_chunk+(size*sizeof(NUM_ADD)+127)/128*128;
			data_h_total=(char*)malloc(total_size);
		
			err=cudaMalloc( (char **) &result_d_total,total_size+size*sizeof(float));
        		if(err!=cudaSuccess)
                        printf("Error %d:%s !\n", err, cudaGetErrorString(err));
			char * data_d_total=result_d_total;
	             	float * result_d=(float *)(result_d_total+total_size);//last part is to store the result.     

			char * data_h=data_h_total;
			char * data_h_begin=data_h; 
			NUM_ADD *data_num_add=(NUM_ADD *) (data_h);
			
			data_h=data_h+(size*sizeof(NUM_ADD)+127)/128*128; // it is 64*x .thus we donot need to worry about alignment.
			int data_size=0;
		
			//for each chunk
			int total_in_each=(size+31)/32;
			for(int i=0;i<total_in_each;i++)
			{
			//each is 32 
			//printf("total_in_each %d\n",total_in_each);
			//read_base
			int long_read_size=0;
			//to find the longest read_size
			for(int j=0;j<32;j++)
			{
			if(i*32+j>=size)
				break;
			if(long_read_size<inputdata[i*32+j].read_size)
				long_read_size=inputdata[i*32+j].read_size;
			}

			int change_length=(long_read_size+3)/4;//because tile=4; each time deal with 4 read
			char4 read_base_data[32*65];
			for(int kk=0;kk<change_length;kk++)
			{
				for(int dd=0;dd<32;dd++) //
				{
					if(i*32+dd>=size)
						break;

					if(inputdata[i*32+dd].read_size<=kk*4)
						continue;
					else
					read_base_data[kk*32+dd].x=inputdata[i*32+dd].read_base[kk*4];
				
					if(inputdata[i*32+dd].read_size<=kk*4+1)
						continue;
					else
					read_base_data[kk*32+dd].y=inputdata[i*32+dd].read_base[kk*4+1];
					
					if(inputdata[i*32+dd].read_size<=kk*4+2)
						continue;
					else
					read_base_data[kk*32+dd].z=inputdata[i*32+dd].read_base[kk*4+2];
				
					if(inputdata[i*32+dd].read_size<=kk*4+3)
						continue;
					else
					read_base_data[kk*32+dd].w=inputdata[i*32+dd].read_base[kk*4+3];
				}
			}	
			//finish read_base

			float parameter1[260*32];//Qm//128 do not change to 128
			float parameter2[260*32];//QI//128 do not change to 128
			float parameter3[260*32];//QD/128 do not change to 128
			float parameter4[260*32];//alpha//128 do not change to 128
			for(int kk=0;kk<long_read_size;kk++)
			{
				for(int dd=0;dd<32;dd++)
				{
					if(i*32+dd>=size)
						break;
					
					if(inputdata[i*32+dd].read_size<=kk)
						continue;
					else
					{
					parameter1[kk*32+dd]= ph2pr_h[inputdata[i*32+dd].base_quals[kk]&127];   
					parameter2[kk*32+dd]= ph2pr_h[inputdata[i*32+dd].ins_quals[kk]&127]  ;
					parameter3[kk*32+dd]= ph2pr_h[inputdata[i*32+dd].del_quals[kk]&127] ;
					parameter4[kk*32+dd]= 1.0f-ph2pr_h[((int)(inputdata[i*32+dd].ins_quals[kk]&127)+(int)( inputdata[i*32+dd].del_quals[kk]&127))&127];
		//			printf("kk=%d  x=%d  y=%d z=%d w=%d \n ",kk,parameter1[kk*32+dd],parameter2[kk*32+dd],parameter3[kk*32+dd],parameter4[kk*32+dd] );
					}		
				}
			}
			
			//to haplotype into 32 char4
			int long_haplotype_size=0;
			//to find the longest hapltoype_size
			for(int j=0;j<32;j++)
			{
			if(i*32+j>=size)
				break;
			if(long_haplotype_size<inputdata[i*32+j].haplotype_size)
				long_haplotype_size=inputdata[i*32+j].haplotype_size;
			}

			int haplotype_change_length=(long_haplotype_size+3)/4;
			char4 haplotype_base_data[125*32];
			for(int kk=0;kk<haplotype_change_length;kk++)
			{
				for(int dd=0;dd<32;dd++)
				{
					if(i*32+dd>=size)
						break;
					if(inputdata[i*32+dd].haplotype_size<=kk*4)
						continue;
					else
					haplotype_base_data[kk*32+dd].x=inputdata[i*32+dd].haplotype_base[kk*4];
				
					if(inputdata[i*32+dd].haplotype_size<=kk*4+1)
						continue;
					else
					haplotype_base_data[kk*32+dd].y=inputdata[i*32+dd].haplotype_base[kk*4+1];
					
					if(inputdata[i*32+dd].haplotype_size<=kk*4+2)
						continue;
					else
					haplotype_base_data[kk*32+dd].z=inputdata[i*32+dd].haplotype_base[kk*4+2];
				
					if(inputdata[i*32+dd].haplotype_size<=kk*4+3)
						continue;
					else
					haplotype_base_data[kk*32+dd].w=inputdata[i*32+dd].haplotype_base[kk*4+3];
				}
			}

			//put data address to each pair of read and haplotype.
			// read address
			memcpy(data_h,read_base_data,sizeof(char4)*32*change_length);//128
			for(int kk=0;kk<32;kk++)
			{
				if(i*32+kk>=size) break;
				data_num_add[i*32+kk].read_number=inputdata[i*32+kk].read_size;
				data_num_add[i*32+kk].haplotype_number=inputdata[i*32+kk].haplotype_size;
				data_num_add[i*32+kk].Read_array=data_size+sizeof(char4)*kk;
		//		printf("set read size %d %d \n", data_num_add[i*32+kk].read_number,data_num_add[i*32+kk].haplotype_number);
			}

			data_h+=sizeof(char4)*32*change_length;
			data_size+=sizeof(char4)*32*change_length;
			
			//parameter address
			memcpy(data_h,parameter1,sizeof(float)*32*long_read_size);
			for(int kk=0;kk<32;kk++)
			{
				if(i*32+kk>=size) break;
				data_num_add[i*32+kk].read_large_length=long_read_size;
			}
			data_h+=sizeof(float)*32*long_read_size;
			data_size+=sizeof(float)*32*long_read_size;
			
			memcpy(data_h,parameter2,sizeof(float)*32*long_read_size);
			data_h+=sizeof(float)*32*long_read_size;
			data_size+=sizeof(float)*32*long_read_size;
		
			memcpy(data_h,parameter3,sizeof(float)*32*long_read_size);
			data_h+=sizeof(float)*32*long_read_size;
			data_size+=sizeof(float)*32*long_read_size;
		
			memcpy(data_h,parameter4,sizeof(float)*32*long_read_size);
			data_h+=sizeof(float)*32*long_read_size;
			data_size+=sizeof(float)*32*long_read_size;
		

			//haplotype address
			memcpy(data_h,haplotype_base_data,sizeof(char4)*32*haplotype_change_length);
			data_h+=sizeof(char4)*32*haplotype_change_length;
			data_size+=sizeof(char4)*32*haplotype_change_length;
			}
				
			int data_size_to_copy=data_size+(size*sizeof(NUM_ADD)+127)/128*128;			
			char * data_d;
			NUM_ADD * num_add_d=(NUM_ADD *) (data_d_total);
			data_d=data_d_total+(sizeof(NUM_ADD)*size+127)/128*128;
			//printf("data_d_total  %p   num_add_d  %p     data_d %p \n",data_d_total,  num_add_d,data_d);		
			int blocksize=128;
			int gridsize=105;
			dim3 block(blocksize);
			dim3 grid(gridsize);
			// global memory to be used by GPU kernels.
	 	
			clock_gettime(CLOCK_MONOTONIC_RAW,&start);
			err=cudaMemcpy(data_d_total,data_h_begin,data_size_to_copy,cudaMemcpyHostToDevice);
			if(err!=cudaSuccess)
			printf("Error %d: %s !\n", err, cudaGetErrorString(err));
			pairHMM<<<grid,block>>> (size,data_d,num_add_d, result_d,MG,DG,IG);
                        cudaMemcpy(result_h,result_d,size*sizeof(float),cudaMemcpyDeviceToHost);
                       	clock_gettime(CLOCK_MONOTONIC_RAW,&finish);
    			computation_time+=diff(start,finish);
		    	for(int i=0;i<size;i++)
float aa=(log10f((double)result_h[i]) - INI);
		//	   	printf("  i=%d  %e\n",i, result_h[i]);
		
			free(result_h);
			free(data_h_total);
         		cudaFree(result_d_total);
	//		
			
                       	clock_gettime(CLOCK_MONOTONIC_RAW,&finish_total);
			total_time+=diff(start_total,finish_total);		
			free(inputdata);
			fscanf(file,"%d",&size);
	//	if(total>10000)
	//		break;
		}
		
		clock_gettime(CLOCK_MONOTONIC_RAW,&start);
 		cudaFree(MG);   
	 	cudaDeviceReset();
		clock_gettime(CLOCK_MONOTONIC_RAW,&finish);
		mem_cpy_time+=diff(start,finish);//(finish1.tv_nsec-start1.tv_nsec)/1000000000.0;

		printf("read_time=%e  initial_time=%e  computation_time= %e total_time=%e\n",read_time, data_prepare,computation_time, total_time);
		//printf("GCUPS: %lf \n",  fakesize*read_read*haplotype_haplotype/computation_time/1000000000);
		printf("Total time=%e\n",total_time);
		return 0;
	}


