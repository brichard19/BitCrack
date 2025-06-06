CPPSRC:=$(wildcard *.cpp)

all:
	${CXX} -o cudainfo.bin ${CPPSRC} ${INCLUDE} -I${CUDA_INCLUDE} ${CXXFLAGS} ${LIBS} -L${CUDA_LIB} -lcudautil -lcudart 
	mkdir -p $(BINDIR)
	cp cudainfo.bin $(BINDIR)/cudainfo

clean:
	rm -rf cudainfo.bin