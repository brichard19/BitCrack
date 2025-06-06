CPPSRC:=$(wildcard *.cpp)
NAME=embedcl

all:
	${CXX} -o ${NAME} ${CPPSRC} ${INCLUDE} ${CXXFLAGS}
	mkdir -p $(BINDIR)
	cp ${NAME} $(BINDIR)/${NAME}

clean:
	rm -rf ${NAME}