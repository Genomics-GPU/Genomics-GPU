nvbio_module(nvBowtie)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -rdc=true -Xptxas="-v")

addsources(
nvBowtie.cpp
)

nvbio_add_module_directory(bowtie2)
nvbio_add_module_directory(bowtie2/cuda)

message(STATUS "Current directory: ${CMAKE_CURRENT_BINARY_DIR}") 
foreach(src ${nvBowtie_srcs})
	if(src MATCHES ".cu$")
		string(REGEX MATCH "^.*\\/" relative_src_directory ${src})
		string(REGEX REPLACE "^.*\\/" "" src_name ${src})
		message(STATUS ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nvBowtie.dir/${relative_src_directory}nvBowtie_generated_${src_name}.o) 
		list(APPEND src_cu ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nvBowtie.dir/${relative_src_directory}nvBowtie_generated_${src_name}.o) 
	endif() 
endforeach() 
string(REPLACE ";" " " src_cu "${src_cu}")
separate_arguments(src_cu UNIX_COMMAND "${src_cu}") 

cuda_add_executable(nvBowtie ${nvBowtie_srcs})

get_target_property(nvBowtie_includes nvBowtie INCLUDE_DIRECTORIES)
foreach (dir ${nvBowtie_includes})
	string(APPEND INCLUDE_COMPILER_STRING "-I${dir} ")
endforeach()
separate_arguments(INCLUDE_COMPILER_STRING UNIX_COMMAND "${INCLUDE_COMPILER_STRING}") 

target_link_libraries(nvBowtie nvbio zlibstatic lz4 crcstatic ${SYSTEM_LINK_LIBRARIES})

get_target_property(all_libraries nvBowtie LINK_LIBRARIES) 
message(STATUS "Libraries: ${all_libraries}") 
foreach(lib ${all_libraries})
	string(APPEND LINK_LIBRARIES "-l${lib} ")
endforeach()
separate_arguments(LINK_LIBRARIES UNIX_COMMAND "${LINK_LIBRARIES}") 

# Generates the link.o file. This should be run after everything is compiled, but before everything is linked. 
add_custom_command(
	TARGET nvBowtie PRE_LINK
	COMMAND nvcc -dlink ${CUDA_NVCC_FLAGS} ${src_cu} -o ${CMAKE_CURRENT_BINARY_DIR}/link.o
	VERBATIM
)

target_link_options(nvBowtie PRIVATE link.o -lcudadevrt -L/usr/local/cuda/lib64)
