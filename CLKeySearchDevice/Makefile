NAME=CLKeySearchDevice
CPPSRC:=$(wildcard *.cpp)

all:
	cat sha256.cl secp256k1.cl ripemd160.cl keysearch.cl > bitcrack.cl
	${BINDIR}/embedcl bitcrack.cl bitcrack_cl.cpp _bitcrack_cl

	for file in ${CPPSRC} bitcrack_cl.cpp; do\
		${CXX} -c $$file ${INCLUDE} -I${OPENCL_INCLUDE} ${CXXFLAGS};\
	done

	ar rvs ${LIBDIR}/lib$(NAME).a *.o

clean:
	rm -rf *.a *.o bitcrack.cl bitcrack_cl.cpp
