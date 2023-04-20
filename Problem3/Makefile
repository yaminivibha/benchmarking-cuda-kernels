SDK_INSTALL_PATH := /usr/local/cuda
NVCC= /usr/local/cuda/bin/nvcc
LIB := -L$(SDK_INSTALL_PATH)/lib64 
CC = g++
CFLAGS = -std=c++11 -Wall -Wextra -pedantic -O2
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := yva2002_partC.tar
EXECS :=  Convolution Convolution_tiled
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt

#######################################################################

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################
convKernel00.o : convKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

convKernel01.o : convKernel01.cu
	${NVCC} $< -c -o $@ $(OPTIONS)
	
Convolution: Convolution.cu convKernel.h convKernel00.o timer.o
	${NVCC} $< convKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

Convolution_tiled : conv_tiled.cu convKernel.h convKernel01.o timer.o
	${NVCC} $< convKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)

Convolution_cudnn : Convolution.cu convKernel.h convKernel00.o timer.o
	${NVCC} $< convKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)