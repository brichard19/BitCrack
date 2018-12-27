CPPSRC:=$(wildcard *.cpp)

all:
	${CXX} -o addrgen.bin ${CPPSRC} ${INCLUDE} -I${CUDA_INCLUDE} ${CXXFLAGS} ${LIBS} -laddressutil -lsecp256k1 -lcryptoutil -lsecp256k1 -lutil -lcmdparse
	mkdir -p $(BINDIR)
	cp addrgen.bin $(BINDIR)/addrgen


clean:
	rm -rf addrgen.bin
