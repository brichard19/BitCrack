NAME=cryptoutil
SRC=$(wildcard *.cpp)

all:    ${SRC}
	for file in ${SRC} ; do\
		${CXX} -c $$file ${INCLUDE} ${CXXFLAGS};\
	done
	mkdir -p ${LIBDIR}
	ar rvs ${LIBDIR}/lib$(NAME).a *.o

clean:
	rm -rf *.o
