// FileStringsToHash.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cuda_runtime.h>
#include <set>
#include "util.h"
#include <map>
#include "picosha2.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>

using namespace std;



std::vector<std::string> ReadFileLines(const std::string& fileName)
{
	std::vector<std::string> lines;
	util::readLinesFromStream(fileName, lines);
	return lines;
}

int main()
{
	cout << "Hello World!\n";

	//std::string path("C:/Users/avira/Documents/Passwords/example.txt");
	string path("C:/Users/avira/Documents/Passwords/10-million-password-list-top-100000.txt");
	vector<string> lines = ReadFileLines(path);



	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//goto Error;
	}



}

