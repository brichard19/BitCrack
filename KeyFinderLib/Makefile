NAME=keyfinder
CPPSRC:=$(wildcard *.cpp)
CUSRC:=$(wildcard *.cu)

all:	cuda

cuda:
	for file in ${CPPSRC} ; do\
		${CXX} -c $$file ${INCLUDE} -I${CUDA_INCLUDE} ${CXXFLAGS};\
	done

	for file in ${CUSRC} ; do\
		${NVCC} -c $$file -o $$file".cuo" ${NVCCFLAGS} ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH};\
	done

	ar rvs ${LIBDIR}/lib$(NAME).a *.o *.cuo

clean:
	rm -f *.o *.cuo
	rm -f *.a