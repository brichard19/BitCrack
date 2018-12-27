NAME=CLUnitTests
CPPSRC:=$(wildcard *.cpp)

all:
	cat ../clMath/secp256k1.cl secp256k1test.cl > cltest.cl
	${BINDIR}/embedcl cltest.cl cltest.cpp _secp256k1_test_cl

	${CXX} -o clunittest.bin ${CPPSRC} ${INCLUDE} -I${OPENCL_INCLUDE} ${CXXFLAGS} ${LIBS} -L${OPENCL_LIB} -lclutil -lOpenCL
	mkdir -p $(BINDIR)

	cp clunittest.bin $(BINDIR)/clunittest



clean:
	rm -rf *.a *.o clunittest.bin cltest.cl cltest.cpp