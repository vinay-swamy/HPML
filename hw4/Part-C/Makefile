

SDK_INSTALL_PATH :=  /cm/shared/apps/cuda11.7/toolkit/11.7.1
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS :=  c1 c2
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

c1 : c1.cu timer.o
	${NVCC} $<  -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
c2 : c2.cu timer.o
	${NVCC} $<  -o $@ $(LIB) timer.o $(OPTIONS)




