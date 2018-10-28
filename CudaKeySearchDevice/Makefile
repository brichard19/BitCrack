NAME=CudaKeySearchDevice
CPPSRC:=$(wildcard *.cpp)
CUSRC:=$(wildcard *.cu)

all:	cuda

cuda:
	for file in ${CPPSRC} ; do\
		${CXX} -c $$file ${INCLUDE} -I${CUDA_INCLUDE} ${CXXFLAGS};\
	done

	for file in ${CUSRC} ; do\
		${NVCC} -c $$file -o $$file".o" ${NVCCFLAGS} -rdc=true ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH};\
	done

	${NVCC} -dlink -o cuda_libs.o *.cu.o -lcudadevrt -lcudart

	ar rvs ${LIBDIR}/lib$(NAME).a *.o

clean:
	rm -f *.o *.cu.o
	rm -f *.a