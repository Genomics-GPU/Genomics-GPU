CUDA_LD_LIBRARY=/opt/apps/cuda/11.4/targets/x86_64-linux/lib
ANALYSIS_FILENAME=analysis
# prefix1 can be optirun in case you need to run it from an optimus-enabled laptop.
PREFIX1=
#i prefix2 can be nvprof. use preferably the following : nvprof --profile-api-trace none -s -f -o /tmp/.nvprof/$(ANALYSIS_FILENAME).nvprof
PREFIX2=valgrind
#suffix1 and 2 can be an output file.
SUFFIX1=> golden.log
SUFFIX2=> out.log

PRGM=test_prog.out
 
OPTARGS1=-p -y local
OPTARGS2=-p -y local


FILES_HUMAN600=reads_600_human_10M.fasta ref_600_human_10M.fasta
FILES_HUMAN300=reads_300_human_10M.fasta ref_300_human_10M.fasta
FILES_HUMAN150=reads_150_human_10M.fasta ref_150_human_10M.fasta
FILES_20K=query_batch.fasta target_batch.fasta
FILES_262K=reads_150.fasta ref_150.fasta
FILES_SHORT=short_query_batch.fasta short_target_batch.fasta

.cpp.o:
	g++ -std=c++11 -g -c -O3 -Wall -Werror -fopenmp -lcudadevrt -I ../include  -o test_prog.o test_prog.cpp

all: clean test_prog.out

test_prog.out: test_prog.o
	g++ ../file_link.o ../gasal_align.o -std=c++11 -O3 -o test_prog.out -L$(CUDA_LD_LIBRARY) -L../lib test_prog.o -fopenmp -lcudart -lgasal -lcudadevrt 

clean:
	rm -f -r *~ *.exe *.o *.out
	
test_prog.o: Timer.h
 

human150: all
	$(PREFIX1) $(PREFIX2) ./$(PRGM) $(OPTARGS1) $(FILES_HUMAN150) $(SUFFIX1)

human150-2: all
	$(PREFIX1) $(PREFIX2) ./$(PRGM) $(OPTARGS2) $(FILES_HUMAN150) $(SUFFIX2)

human300: all
	$(PREFIX1) $(PREFIX2) ./$(PRGM) $(OPTARGS1) $(FILES_HUMAN300) $(SUFFIX1)

human300-2: all
	$(PREFIX1) $(PREFIX2) ./$(PRGM) $(OPTARGS2) $(FILES_HUMAN300) $(SUFFIX2)

human600: all
	$(PREFIX1) $(PREFIX2) ./$(PRGM) $(OPTARGS1) $(FILES_HUMAN600) $(SUFFIX1)

human600-2: all
	$(PREFIX1) $(PREFIX2) ./$(PRGM) $(OPTARGS2) $(FILES_HUMAN600) $(SUFFIX2)


run: all
	$(PREFIX1) $(PREFIX2) ./$(PRGM) $(OPTARGS1) $(FILES_SHORT) $(SUFFIX1)

run2: all
	$(PREFIX1) $(PREFIX2) ./$(PRGM) $(OPTARGS2) $(FILES_SHORT) $(SUFFIX2)


fullrun: all
	$(PREFIX1) $(PREFIX2) ./$(PRGM) $(OPTARGS1) $(FILES_20K) $(SUFFIX1)

fullrun2: all
	$(PREFIX1) $(PREFIX2) ./$(PRGM) $(OPTARGS2) $(FILES_20K) $(SUFFIX2)


262k: all
	$(PREFIX1) $(PREFIX2) ./$(PRGM) $(OPTARGS1) $(FILES_262K) $(SUFFIX1)

262k2: all
	$(PREFIX1) $(PREFIX2) ./$(PRGM) $(OPTARGS2) $(FILES_262K) $(SUFFIX2)





cuda-memcheck: all
	cuda-memcheck ./$(PRGM) $(OPTARGS1) $(FILES_20K) $(SUFFIX1)

cuda-gdb: all
	cuda-gdb --args ./test_prog.out -p -y local query_batch.fasta target_batch.fasta

valgrind: all
	valgrind ./test_prog.out -p -y local short_query_batch.fasta short_target_batch.fasta

gdb: all
	gdb --args ./test_prog.out -p -y local short_query_batch.fasta short_target_batch.fasta
