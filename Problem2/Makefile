SDK_INSTALL_PATH := /usr/local/cuda
NVCC= /usr/local/cuda/bin/nvcc
LIB := -L$(SDK_INSTALL_PATH)/lib64 
CC = g++
CFLAGS = -std=c++11 -Wall -Wextra -pedantic -O2
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := yva2002_partB.tar
EXECS :=  KMillionAddition
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

KMillionAddition: KMillionAddition.cc 
	$(CC) $(CFLAGS) $< -o $@

KMillionAdditioncu: KMillionAddition.cu timer.o
	$(NVCC) $< -o $@ $(OPTIONS)