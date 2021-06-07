NAME=CLKeySearchDevice
CPPSRC:=$(wildcard *.cpp)

all:
	cat ../clMath/sha256.cl ../clMath/secp256k1.cl ../clMath/ripemd160.cl bloomfilter.cl bitcoin.cl keysearch.cl > bitcrack.cl
	${BINDIR}/embedcl bitcrack.cl bitcrack_cl.cpp _bitcrack_cl

	for file in ${CPPSRC} bitcrack_cl.cpp; do\
		${CXX} -c $$file ${INCLUDE} -I${OPENCL_INCLUDE} ${CXXFLAGS};\
	done

	ar rvs ${LIBDIR}/lib$(NAME).a *.o

clean:
	rm -rf *.a *.o bitcrack.cl bitcrack_cl.cpp
