
CUR_DIR=$(shell pwd)
DIRS=util AddressUtil CmdParse CryptoUtil KeyFinderLib CLKeySearchDevice CudaKeySearchDevice cudaMath clUtil cudaUtil secp256k1lib Logger embedcl

INCLUDE = $(foreach d, $(DIRS), -I$(CUR_DIR)/$d)

LIBDIR=$(CUR_DIR)/lib
BINDIR=$(CUR_DIR)/bin
LIBS+=-L$(LIBDIR)

# C++ options
CXX=g++
CXXFLAGS=-O2 -std=c++11

# CUDA variables
COMPUTE_CAP=70
NVCC=nvcc
NVCCFLAGS=-std=c++11 -gencode=arch=compute_${COMPUTE_CAP},code=\"sm_${COMPUTE_CAP}\" -Xptxas="-v" -Xcompiler "${CXXFLAGS}"
#CUDA_HOME=/usr/local/cuda-9.2
CUDA_HOME=/usr/local/cuda-10.2
CUDA_LIB=${CUDA_HOME}/lib64
CUDA_INCLUDE=${CUDA_HOME}/include
CUDA_MATH=$(CUR_DIR)/cudaMath

# OpenCL variables
OPENCL_LIB=${CUDA_LIB}
OPENCL_INCLUDE=${CUDA_INCLUDE}
OPENCL_VERSION=110

export INCLUDE
export LIBDIR
export BINDIR
export NVCC
export NVCCFLAGS
export LIBS
export CXX
export CXXFLAGS
export CUDA_LIB
export CUDA_INCLUDE
export CUDA_MATH
export OPENCL_LIB
export OPENCL_INCLUDE
export BUILD_OPENCL
export BUILD_CUDA

TARGETS=dir_addressutil dir_cmdparse dir_cryptoutil dir_keyfinderlib dir_keyfinder dir_secp256k1lib dir_util dir_logger dir_addrgen

ifeq ($(BUILD_CUDA),1)
	TARGETS:=${TARGETS} dir_cudaKeySearchDevice dir_cudautil
endif

ifeq ($(BUILD_OPENCL),1)
	TARGETS:=${TARGETS} dir_embedcl dir_clKeySearchDevice dir_clutil dir_clunittest
	CXXFLAGS:=${CXXFLAGS} -DCL_TARGET_OPENCL_VERSION=${OPENCL_VERSION}
endif

all:	${TARGETS}

dir_cudaKeySearchDevice: dir_keyfinderlib dir_cudautil dir_logger
	make --directory CudaKeySearchDevice

dir_clKeySearchDevice: dir_embedcl dir_keyfinderlib dir_clutil dir_logger
	make --directory CLKeySearchDevice

dir_embedcl:
	make --directory embedcl

dir_addressutil:	dir_util dir_secp256k1lib dir_cryptoutil
	make --directory AddressUtil

dir_cmdparse:
	make --directory CmdParse

dir_cryptoutil:
	make --directory CryptoUtil

dir_keyfinderlib:	dir_util dir_secp256k1lib dir_cryptoutil dir_addressutil dir_logger
	make --directory KeyFinderLib

KEYFINDER_DEPS=dir_keyfinderlib

ifeq ($(BUILD_CUDA), 1)
	KEYFINDER_DEPS:=$(KEYFINDER_DEPS) dir_cudaKeySearchDevice
endif

ifeq ($(BUILD_OPENCL),1)
	KEYFINDER_DEPS:=$(KEYFINDER_DEPS) dir_clKeySearchDevice
endif

dir_keyfinder:	$(KEYFINDER_DEPS)
	make --directory KeyFinder

dir_cudautil:
	make --directory cudaUtil

dir_clutil:
	make --directory clUtil

dir_secp256k1lib:	dir_cryptoutil
	make --directory secp256k1lib

dir_util:
	make --directory util

dir_cudainfo:
	make --directory cudaInfo

dir_logger:
	make --directory Logger

dir_addrgen:	dir_cmdparse dir_addressutil dir_secp256k1lib
	make --directory AddrGen
dir_clunittest:	dir_clutil
	make --directory CLUnitTests

clean:
	make --directory AddressUtil clean
	make --directory CmdParse clean
	make --directory CryptoUtil clean
	make --directory KeyFinderLib clean
	make --directory KeyFinder clean
	make --directory cudaUtil clean
	make --directory secp256k1lib clean
	make --directory util clean
	make --directory cudaInfo clean
	make --directory Logger clean
	make --directory clUtil clean
	make --directory CLKeySearchDevice clean
	make --directory CudaKeySearchDevice clean
	make --directory embedcl clean
	make --directory CLUnitTests clean
	rm -rf ${LIBDIR}
	rm -rf ${BINDIR}
