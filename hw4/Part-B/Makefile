###
### For COMS E6998 Spring 2023
### Instructor: Parajit Dube and Kaoutar El Maghraoui
### Makefile for CUDA1 assignment
### By Wim Bohm, Waruna Ranasinghe, and Louis Rabiet
### Created: 2011-01-27 DVN
### Last Modified: Nov 2014 WB, WR, LR
###

SDK_INSTALL_PATH :=  /cm/shared/apps/cuda11.7/toolkit/11.7.1
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS :=  q1 q2 q3
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

vecaddKernel01.o : vecaddKernel01.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

q2 : q2.cu vecaddKernel.h vecaddKernel01.o timer.o
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)

q3 : q3.cu vecaddKernel.h vecaddKernel01.o timer.o
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################

q1 : q1.cpp
	g++ $< -o $@ 






