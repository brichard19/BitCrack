
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <vector>
#include "util.h"
#include <ostream>
#include <iostream>
#include "picosha2.h"

cudaError_t addWithCuda(std::string* combined, std::vector<int>* indexes);


__global__ void printStringKernel(
    char* lines, //combined string data, 
    int* indexes, //indexes telling us the beginning and end of each string,
    int indexes_size //number of strings being analyzed
)
{
    printf("Starting 'printStringKernel' on device\n");
    printf("lines  value:  %s \n", lines);
    printf("indexes  value:  %d \n", indexes[0]);

    int i = threadIdx.x;

    size_t id = threadIdx.x;//"Which String are we examining?"

    if (id >= indexes_size) {//Bounds Checking
        printf("thread id:  %d EXIT.\n", id);
        return;
    }
    char* string; //Beginning of the string
    int string_length = 0; //Beginning of the string
    if (id == 0) {//First String
        string = lines;
        string_length = indexes[0];
    }
    else {
        string_length = indexes[id] - indexes[id - 1];
        string = (lines + indexes[id - 1]);
    }
    printf("string length value:  %d \n", string_length);
	for (int i = 0 ; i<indexes_size ; i++)
	{
        printf("indexes[%d] = %d.\n", i, indexes[i]);
	}

    //std::string hash_hex_str;
   // picosha2::hash256_hex_string("1", "1");
    //printf("hash256_hex_string = %s.\n", hash_hex_str);

    char* string_end = (lines + indexes[id]); //end of the string
    for (; string != string_end; string++) {
        if (*string == 'A') {
            return;
        }
    }
}



std::vector<std::string> ReadFileLines(const std::string& fileName)
{
    std::vector<std::string> lines;
    util::readLinesFromStream(fileName, lines);
    return lines;
}


int main2()
{


    std::string path("C:/Users/avira/Documents/Passwords/example.txt");
    //std::string path("C:/Users/avira/Documents/Passwords/10-million-password-list-top-10000.txt");
    std::vector<std::string> lines = ReadFileLines(path);



    std::string path2("SourceFiles/list.txt");
    std::vector<std::string> lines2 = ReadFileLines(path2);



    std::string path3("list.txt");
    std::vector<std::string> lines3 = ReadFileLines(path2);

    std::cout << "Before Loop." << std::endl;
    std::string combined; //Works perfectly fine so long as it is contiguously allocated
    std::vector<int> indexes; //You *might* be able to use int instead of size_t to save space
    for (std::string const& line : lines) {
        std::cout << "In Loop." << std::endl;
        combined += line;
        indexes.emplace_back(combined.size());
    }
    std::cout << "After Loop." << std::endl;

    /*
	for(int i = 0 ; i < 100; i++)
	{
        std::string hash_hex_str;
        picosha2::hash256_hex_string(lines[0], hash_hex_str);
	}
    */
	
    /* If 'lines' initially consisted of ["Dog", "Cat", "Tree", "Yard"], 'combined' is now
 * "DogCatTreeYard", and 'indexes' is now [3, 6, 10, 14].
 */

 // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(&combined, &indexes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(std::string* combined, std::vector<int>* indexes)
{
    char* dev_combined = 0;
    int* dev_indexes = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


    // Allocate GPU buffers for three vectors (two input, one output)    .

    cudaStatus = cudaMalloc((void**)&dev_combined, combined->size() * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_indexes, indexes->size() * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_combined, combined->data(), combined->size() * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_indexes, indexes->data(), indexes->size() * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



    // Launch a kernel on the GPU with one thread for each element.
    printStringKernel <<< 1, 8 >>> (dev_combined, dev_indexes, indexes->size());

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }



Error:
    cudaFree(dev_combined);
    cudaFree(dev_indexes);

    return cudaStatus;
}
