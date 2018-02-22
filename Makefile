
CUR_DIR=$(shell pwd)
DIRS=util AddressUtil CmdParse CryptoUtil KeyFinderLib cudaDeviceContext cudaMath cudaUtil secp256k1lib

INCLUDE = $(foreach d, $(DIRS), -I$(CUR_DIR)/$d)

LIBDIR=$(CUR_DIR)/lib
BINDIR=$(CUR_DIR)/bin
LIBS+=-L$(LIBDIR)

# C++ options
CXX=g++
CXXFLAGS=-O2

# CUDA variables
COMPUTE_CAP=30
NVCC=nvcc
NVCCFLAGS=-gencode=arch=compute_${COMPUTE_CAP},code=\"sm_${COMPUTE_CAP}\" -Xptxas="-v" -Xcompiler "${CXXFLAGS}"
CUDA_HOME=/usr/local/cuda-7.5
CUDA_LIB=${CUDA_HOME}/lib64
CUDA_INCLUDE=${CUDA_HOME}/include
CUDA_MATH=$(CUR_DIR)/cudaMath



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


all:	dir_addressutil dir_cmdparse dir_cryptoutil dir_keyfinderlib dir_keyfinder dir_cudadevicecontext dir_cudautil dir_secp256k1lib dir_util

dir_addressutil:	dir_util dir_secp256k1lib dir_cryptoutil
	make --directory AddressUtil

dir_cmdparse:
	make --directory CmdParse

dir_cryptoutil:
	make --directory CryptoUtil

dir_keyfinderlib:	dir_util dir_secp256k1lib dir_cryptoutil dir_addressutil dir_cudautil dir_cudadevicecontext dir_cudautil
	make --directory KeyFinderLib

dir_keyfinder:	dir_keyfinderlib
	make --directory KeyFinder

dir_cudadevicecontext:
	make --directory cudaDeviceContext

dir_cudautil:
	make --directory cudaUtil

dir_secp256k1lib:	dir_cryptoutil
	make --directory secp256k1lib

dir_util:
	make --directory util


clean:
	make --directory AddressUtil clean
	make --directory CmdParse clean
	make --directory CryptoUtil clean
	make --directory KeyFinderLib clean
	make --directory KeyFinder clean
	make --directory cudaDeviceContext clean
	make --directory cudaUtil clean
	make --directory secp256k1lib clean
	make --directory util clean

	rm -rf ${LIBDIR}
	rm -rf ${BINDIR}