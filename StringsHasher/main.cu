// cd /home/hork/cuda-workspace/CudaSHA256/Debug/files
// time ~/Dropbox/FIIT/APS/Projekt/CpuSHA256/a.out -f ../file-list
// time ../CudaSHA256 -f ../file-list


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sha256.cuh"
#include <ctype.h>
#include <device_launch_parameters.h>
#include "cudabridge.h"
#include "util.h"
#include <iostream>
#include <chrono>

char* trim(char* str) {
	size_t len = 0;
	char* frontp = str;
	char* endp = NULL;

	if (str == NULL) { return NULL; }
	if (str[0] == '\0') { return str; }

	len = strlen(str);
	endp = str + len;

	/* Move the front and back pointers to address the first non-whitespace
	 * characters from each end.
	 */
	while (isspace((unsigned char)*frontp)) { ++frontp; }
	if (endp != frontp)
	{
		while (isspace((unsigned char)*(--endp)) && endp != frontp) {}
	}

	if (str + len - 1 != endp)
		*(endp + 1) = '\0';
	else if (frontp != str && endp == frontp)
		*str = '\0';

	/* Shift the string so that it starts at str so that if it's dynamically
	 * allocated, we can still free it on the returned pointer.  Note the reuse
	 * of endp to mean the front of the string buffer now.
	 */
	endp = str;
	if (frontp != str)
	{
		while (*frontp) { *endp++ = *frontp++; }
		*endp = '\0';
	}


	return str;
}

__global__ void sha256_cuda(JOB** jobs, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	if (i < n) {
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
		sha256_final(&ctx, jobs[i]->digest);

		if (i == 0)
		{
			printf("jobs[i]->data = \n");
			/*
			char* string = (char*)malloc(70);
			int k, i;
			for (i = 0, k = 0; i < 32; i++, k += 2)
			{
				sprintf(string + k, "%.2x", buff[i]);
				//printf("%02x", buff[i]);
			}
			string[64] = 0;
			return string;
			*/
			
		}
		/*
		SHA256_CTX ctx2;
		sha256_init(&ctx2);
		sha256_update(&ctx2, jobs[i]->digest, 64);
		sha256_final(&ctx2, jobs[i]->digest2);


		SHA256_CTX ctx3;
		sha256_init(&ctx3);
		sha256_update(&ctx3, jobs[i]->digest, 32);
		sha256_final(&ctx3, jobs[i]->digest3);
		*/
	}
}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}


void runJobs(JOB** jobs, int n) {
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	sha256_cuda << < numBlocks, blockSize >> > (jobs, n);
}


JOB* JOB_init(BYTE* data, long size, char* fname) {
	JOB* j;
	checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));	//j = (JOB *)malloc(sizeof(JOB));
	checkCudaErrors(cudaMallocManaged(&(j->data), size));
	j->data = data;
	j->size = size;
	for (int i = 0; i < 64; i++)
	{
		j->digest[i] = 0xff;
		//j->digest2[i] = 0xff;
		//j->digest3[i] = 0xff;
	}
	strcpy(j->fname, fname);
	return j;
}


int main(int argc, char** argv) {
	int i = 0, n = 0;
	size_t len;
	unsigned long temp;
	char* a_file = 0, * line = 0;
	BYTE* buff = 0;
	char option, index;
	JOB** jobs;

	std::string path("C:/Users/avira/Documents/Passwords/example.txt");
	std::vector<std::string> lines = util::ReadFileLines(path);

	auto t0 = std::chrono::high_resolution_clock::now();
	
	n = lines.size();
	checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB*)));
	//fseek(f, 0, SEEK_SET);
	n = 0;
	std::cout << "Before Loop." << std::endl;
	//std::string combined; //Works perfectly fine so long as it is contiguously allocated
	//std::vector<int> indexes; //You *might* be able to use int instead of size_t to save space
	for (std::string const& line : lines) {
		//std::copy(line.begin(), line.end(), buff);
		BYTE* buffer = 0;
		size_t length = line.size() + 1;
		checkCudaErrors(cudaMallocManaged(&buffer, length * sizeof(char)));
		std::copy(line.begin(), line.end(), buffer);
		jobs[n++] = JOB_init(buffer, length - 1, "test");
	}
	std::cout << "After Loop." << std::endl;

	auto t1 = std::chrono::high_resolution_clock::now();
	pre_sha256();
	runJobs(jobs, n);

	cudaDeviceSynchronize();
	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration_total = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();
	auto duration_gpu_work = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	printf("\t duration_gpu_work = %d microseconds  \n", duration_gpu_work);
	printf("\t duration_total = %d microseconds  \n", duration_total);
	
	//print_jobs(jobs, n);
	cudaDeviceReset();
	return 0;
}
