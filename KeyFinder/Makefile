CPPSRC:=$(wildcard *.cpp)

all:
	${CXX} -o KeyFinder.bin ${CPPSRC} ${INCLUDE} -I${CUDA_INCLUDE} ${CXXFLAGS} ${LIBS} -L${CUDA_LIB} -lkeyfinder -laddressutil -lsecp256k1 -lcryptoutil -lsecp256k1 -lcudadevicecontext -lcudautil -llogger -lutil -lcudadevrt -lcudart -lcmdparse
	mkdir -p $(BINDIR)
	cp KeyFinder.bin $(BINDIR)/BitCrack

clean:
	rm -rf KeyFinder.bin
