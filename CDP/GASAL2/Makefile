GPU_SM_ARCH=
MAX_QUERY_LEN=
N_CODE=
N_PENALTY=

GPU_COMPUTE_ARCH=$(subst sm,compute,$(GPU_SM_ARCH))
NVCC=/opt/apps/cuda/11.4/bin/nvcc
CC=g++
SRC_DIR=./src/
OBJ_DIR=./obj/
OBJO_DIR=./objo/
LIB_DIR=./lib/
INCLUDE_DIR=./include/

SOURCES=  args_parser.cpp host_batch.cpp ctors.cpp interfaces.cpp res.cpp gasal_align.cu 
LOBJS=$(patsubst %,%o,$(SOURCES))

LOBJS_PATH=$(addprefix $(OBJ_DIR),$(LOBJS))
VPATH=src:obj:lib
YELLOW=\033[1;33m
NC=\033[0m # No Color

ifeq ($(GPU_SM_ARCH),)
error1:
	@echo "Must specify GPU architecture as sm_xx"
endif
ifeq ($(MAX_QUERY_LEN),)
error2:
	@echo "Must specify maximum sequence length"
endif

ifeq ($(N_CODE),)
error3:
	@echo "Must specify the code for 'N'"
endif
#ifneq ($(GPU_SM_ARCH),clean)




## If your computer ships gcc-5.3.1 (at least for CUDA 8.0), this is the regular line. You might need to add: --compiler-options -fPIC 
## With Debian and clang, use: $(NVCC) -ccbin clang-3.8 --compiler-options -fpie

ifeq ($(N_PENALTY),)
%.cuo: %.cu
	$(NVCC) -c -g -O3 -std=c++11 -lcudart -rdc=true -lcudadevrt -Xcompiler -Wall,-DMAX_QUERY_LEN=$(MAX_QUERY_LEN),-DN_CODE=$(N_CODE) -Xptxas -Werror  --gpu-architecture=$(GPU_COMPUTE_ARCH) --gpu-code=$(GPU_SM_ARCH) -lineinfo --ptxas-options=-v --default-stream per-thread src/gasal_align.cu -o $(OBJ_DIR)$@
	$(NVCC) -c -g -O3 -std=c++11 -lcudart -rdc=true -lcudadevrt -Xcompiler -Wall,-DMAX_QUERY_LEN=$(MAX_QUERY_LEN),-DN_CODE=$(N_CODE)  --gpu-architecture=$(GPU_COMPUTE_ARCH) --gpu-code=$(GPU_SM_ARCH) src/gasal_align.cu -o gasal_align.o
	$(NVCC) -arch=sm_80 -lcudadevrt -dlink gasal_align.o -o file_link.o 

#	$(NVCC) -arch=sm_80 -dlink  -o file_link.o /obj/gasal_align.cuo  -lcudadevrt -lcudart
#%.o: %.cu
#	$(NVCC) -c -g -O3 -std=c++11 -lcudart -rdc=true -lcudadevrt -Xcompiler -Wall,-DMAX_QUERY_LEN=$(MAX_QUERY_LEN),-DN_CODE=$(N_CODE) -Xptxas -Werror  --gpu-architecture=$(GPU_COMPUTE_ARCH) --gpu-code=$(GPU_SM_ARCH) -lineinfo --ptxas-options=-v --default-stream per-thread $< -o file.o
#	$(NVCC) -dlink --gpu-architecture=$(GPU_COMPUTE_ARCH) -o bs_link.o file.o -lcudadevrt -lcudart -rdc=true
	
else
%.cuo: %.cu
	$(NVCC) -c -g -O3 -std=c++11 -lcudart -rdc=true -lcudadevrt -Xcompiler -Wall,-DMAX_QUERY_LEN=$(MAX_QUERY_LEN),-DN_CODE=$(N_CODE),-DN_PENALTY=$(N_PENALTY) -Xptxas -Werror  --gpu-architecture=$(GPU_COMPUTE_ARCH) --gpu-code=$(GPU_SM_ARCH) -lineinfo --ptxas-options=-v --default-stream per-thread $< -o $(OBJ_DIR)$@
#%.o: %.cu
#	$(NVCC) -c -g -O3 -std=c++11 -lcudart -rdc=true -lcudadevrt -Xcompiler -Wall,-DMAX_QUERY_LEN=$(MAX_QUERY_LEN),-DN_CODE=$(N_CODE) -Xptxas -Werror  --gpu-architecture=$(GPU_COMPUTE_ARCH) --gpu-code=$(GPU_SM_ARCH) -lineinfo --ptxas-options=-v --default-stream per-thread $< -o file.o
#	$(NVCC) -dlink --gpu-architecture=$(GPU_COMPUTE_ARCH) -o bs_link.o file.o -lcudadevrt -lcudart -rdc=true	
endif



## If your computer ships gcc-5.3.1 (at least for CUDA 8.0), this is the regular line. You might need to add: -fPIC 
ifeq ($(N_PENALTY),)
%.cppo: %.cpp
	$(CC) -c -g -O3 -std=c++11 -Wall -DMAX_QUERY_LEN=$(MAX_QUERY_LEN) -DN_CODE=$(N_CODE) -Werror $< -o $(OBJ_DIR)$@
	
else
%.cppo: %.cpp
	$(CC) -c -g -O3 -std=c++11 -Wall -DMAX_QUERY_LEN=$(MAX_QUERY_LEN) -DN_CODE=$(N_CODE) -DN_PENALTY=$(N_PENALTY) -Werror $< -o $(OBJ_DIR)$@
	
endif


all: clean makedir libgasal.a

makedir:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(OBJO_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(INCLUDE_DIR)
	@cp $(SRC_DIR)/*.h $(INCLUDE_DIR)
	@sed  -i "s/MAX_QUERY_LEN=[0-9]\{1,9\}/MAX_QUERY_LEN=$(MAX_QUERY_LEN)/" ./test_prog/Makefile
	 
ifeq ($(N_PENALTY),)
libgasal.a: $(LOBJS)
	ar -csru $(LIB_DIR)$@ $(LOBJS_PATH)
	@echo ""
	@echo -e "${YELLOW}WARNING:${NC}\"N_PENALTY\" is not defined"
else
libgasal.a: $(LOBJS)
	ar -csru $(LIB_DIR)$@ $(LOBJS_PATH)
endif
	
clean:
	rm -f -r $(OBJ_DIR) $(LIB_DIR) $(INCLUDE_DIR)  *~ *.exe *.cppo *.cuo *.txt *~

gasal_align.cuo: gasal.h gasal_kernels.h

gasal_align.o:  gasal.h gasal_kernels.h gasal_align.cu

