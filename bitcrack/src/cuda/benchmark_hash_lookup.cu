#include "CudaHashLookup.h"
#include "cuda_runtime.h"
#include "CudaHashLookup.cuh" // For checkHash __device__ function prototype

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip> // For std::fixed, std::setprecision
#include <algorithm>
#include <set>
#include <array>
#include <sstream> // Required for std::ostringstream

// Assuming Logger.h and util.h are in parent directories as described
// Adjust paths if they are different or if these are part of a library
#include "../Logger/Logger.h"
#include "../util/util.h"

// hash160 struct is defined in CudaHashLookup.h, which is included above.
// No need to redefine it here.

// Host-side helper functions to mirror the processing in CudaHashLookup.cu
static unsigned int swp_host(unsigned int x) {
    return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

static void undoRMD160FinalRound_host(const unsigned int hIn[5], unsigned int hOut[5]) {
    unsigned int iv[5] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
    };

    for (int i = 0; i < 5; i++) {
        hOut[i] = swp_host(hIn[i]) - iv[(i + 1) % 5];
    }
}

// Helper to format strings for logger
template<typename... Args>
std::string format_string(const char* fmt, Args... args) {
    // Determine buffer size (a bit of a hack for C-style variadic args)
    // This is not perfectly safe, but often works for typical log messages.
    // A more robust solution would use C++20 std::format or a dedicated formatting library.
    char buffer[1024];
    snprintf(buffer, sizeof(buffer), fmt, args...);
    return std::string(buffer);
}

// Function to get free GPU memory
static size_t getFreeDeviceMemory() {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA Error getting memory info: " << cudaGetErrorString(err);
        Logger::log(LogLevel::Error, oss.str());
        return 0;
    }
    return free_mem;
}

// Function to generate N unique hash160 targets
std::vector<hash160> generate_unique_targets(int N) {
    std::vector<hash160> targets;
    targets.reserve(N);
    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<unsigned int> dist;

    std::set<std::array<unsigned int, 5>> unique_hashes_set;

    while (static_cast<int>(targets.size()) < N) {
        unsigned int temp_h_for_constructor[5];
        std::array<unsigned int, 5> current_h_array;
        for (int i = 0; i < 5; ++i) {
            temp_h_for_constructor[i] = dist(rng);
            current_h_array[i] = temp_h_for_constructor[i];
        }
        if (unique_hashes_set.find(current_h_array) == unique_hashes_set.end()) {
            unique_hashes_set.insert(current_h_array);
            targets.push_back(hash160(temp_h_for_constructor)); // Construct using existing constructor
        }
    }
    return targets;
}

// Function to generate M test hash160 values
std::vector<hash160> generate_test_data(int M, const std::vector<hash160>& targets, std::vector<bool>& is_target_host_truth) {
    std::vector<hash160> test_hashes;
    test_hashes.reserve(M);
    is_target_host_truth.assign(M, false);

    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<unsigned int> dist_val;

    std::uniform_int_distribution<size_t> target_idx_dist;
    if (!targets.empty()) {
        target_idx_dist = std::uniform_int_distribution<size_t>(0, targets.size() - 1);
    }

    int targets_to_select = M / 2;
    if (targets.empty() && targets_to_select > 0) {
        targets_to_select = 0;
    }
    int new_hashes_to_generate = M - targets_to_select;

    std::vector<std::pair<hash160, bool>> temp_test_data;
    temp_test_data.reserve(M);

    std::set<std::array<unsigned int, 5>> targets_as_arrays_set;
    for(const auto& t : targets) {
        std::array<unsigned int, 5> arr;
        for(int i=0; i<5; ++i) arr[i] = t.h[i];
        targets_as_arrays_set.insert(arr);
    }

    std::set<size_t> selected_indices;
    if (!targets.empty()) {
        for (int i = 0; i < targets_to_select; ++i) {
            size_t idx;
            if (selected_indices.size() >= targets.size()) break;
            do {
                idx = target_idx_dist(rng);
            } while (selected_indices.count(idx));
            selected_indices.insert(idx);
            temp_test_data.push_back({targets[idx], true});
        }
    }

    for (int i = 0; i < new_hashes_to_generate; ++i) {
        unsigned int temp_h_for_constructor[5]; // Temporary array
        std::array<unsigned int, 5> h_new_arr;
        do {
            for (int j = 0; j < 5; ++j) {
                temp_h_for_constructor[j] = dist_val(rng); // Fill temp array
                h_new_arr[j] = temp_h_for_constructor[j];
            }
        } while (targets_as_arrays_set.count(h_new_arr));
        temp_test_data.push_back({hash160(temp_h_for_constructor), false}); // Construct using the temp array
    }

    std::shuffle(temp_test_data.begin(), temp_test_data.end(), rng);

    test_hashes.clear();
    is_target_host_truth.clear();
    for(const auto& item : temp_test_data) {
        test_hashes.push_back(item.first);
        is_target_host_truth.push_back(item.second);
    }

    return test_hashes;
}

// CUDA Kernel
__global__ void benchmarkCheckHashKernel(const unsigned int* test_hashes_flat, bool* results, int num_test_hashes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_test_hashes) {
        results[idx] = checkHash(&test_hashes_flat[idx * 5]);
    }
}


int main(int argc, char **argv) {
    // Logger::init is not available in the provided Logger.h
    // Logger::setLogFile(""); // Optionally direct log to a file

    const int N_TARGETS = 1000000;
    const int M_TEST_HASHES_REQUESTED = 100000;
    std::ostringstream oss;

    Logger::log(LogLevel::Info, "Starting CudaHashLookup Benchmark...");

    oss.str(""); oss.clear();
    oss << "Generating " << N_TARGETS << " unique target hash160 values...";
    Logger::log(LogLevel::Info, oss.str());
    auto targets = generate_unique_targets(N_TARGETS);
    oss.str(""); oss.clear();
    oss << "Generated " << targets.size() << " targets.";
    Logger::log(LogLevel::Info, oss.str());

    oss.str(""); oss.clear();
    oss << "Generating up to " << M_TEST_HASHES_REQUESTED << " test hash160 values (~50% from targets, ~50% new)...";
    Logger::log(LogLevel::Info, oss.str());
    std::vector<bool> is_actually_target_truth;
    auto test_hashes_original = generate_test_data(M_TEST_HASHES_REQUESTED, targets, is_actually_target_truth);
    const int ACTUAL_M_TEST_HASHES = test_hashes_original.size();
    oss.str(""); oss.clear();
    oss << "Generated " << ACTUAL_M_TEST_HASHES << " test hashes.";
    Logger::log(LogLevel::Info, oss.str());

    if (ACTUAL_M_TEST_HASHES == 0) {
        Logger::log(LogLevel::Warning, "No test hashes were generated. Benchmark cannot continue meaningfully.");
        return 0;
    }

    int actual_positives_in_test_set = 0;
    for(int i=0; i < ACTUAL_M_TEST_HASHES; ++i) {
        if(is_actually_target_truth[i]) actual_positives_in_test_set++;
    }
    oss.str(""); oss.clear();
    oss << "Test set contains " << actual_positives_in_test_set << " true positives and " << (ACTUAL_M_TEST_HASHES - actual_positives_in_test_set) << " true negatives.";
    Logger::log(LogLevel::Info, oss.str());

    CudaHashLookup lookup;

    oss.str(""); oss.clear();
    oss << "Benchmarking CudaHashLookup::setTargets() for " << N_TARGETS << " targets...";
    Logger::log(LogLevel::Info, oss.str());
    size_t mem_before = getFreeDeviceMemory();
    oss.str(""); oss.clear();
    oss << std::fixed << std::setprecision(2) << "GPU Memory Free (Before setTargets): " << mem_before / (1024.0 * 1024.0) << " MB";
    Logger::log(LogLevel::Info, oss.str());

    auto set_targets_start_time = std::chrono::high_resolution_clock::now();
    cudaError_t setResult = lookup.setTargets(targets);
    auto set_targets_end_time = std::chrono::high_resolution_clock::now();

    if (setResult != cudaSuccess) {
        oss.str(""); oss.clear();
        oss << "CudaHashLookup::setTargets() failed: " << cudaGetErrorString(setResult);
        Logger::log(LogLevel::Error, oss.str());
        return 1;
    }

    auto set_targets_duration = std::chrono::duration_cast<std::chrono::milliseconds>(set_targets_end_time - set_targets_start_time);
    oss.str(""); oss.clear();
    oss << "CudaHashLookup::setTargets() duration: " << set_targets_duration.count() << " ms";
    Logger::log(LogLevel::Info, oss.str());

    size_t mem_after = getFreeDeviceMemory();
    oss.str(""); oss.clear();
    oss << std::fixed << std::setprecision(2) << "GPU Memory Free (After setTargets): " << mem_after / (1024.0 * 1024.0) << " MB";
    Logger::log(LogLevel::Info, oss.str());
    if (mem_before >= mem_after) {
        oss.str(""); oss.clear();
        oss << std::fixed << std::setprecision(2) << "GPU Memory Used by Filter: " << (mem_before - mem_after) / (1024.0 * 1024.0) << " MB";
        Logger::log(LogLevel::Info, oss.str());
    } else {
        Logger::log(LogLevel::Info, "GPU Memory Free (After setTargets) was unexpectedly higher or memory info failed. Memory used calculation skipped.");
    }

    std::vector<unsigned int> test_hashes_flat(ACTUAL_M_TEST_HASHES * 5);
    oss.str(""); oss.clear();
    oss << "Preprocessing " << ACTUAL_M_TEST_HASHES << " test hashes with undoRMD160FinalRound_host...";
    Logger::log(LogLevel::Info, oss.str());
    for (int i = 0; i < ACTUAL_M_TEST_HASHES; ++i) {
        unsigned int processed_h[5];
        undoRMD160FinalRound_host(test_hashes_original[i].h, processed_h);
        for (int j = 0; j < 5; ++j) {
            test_hashes_flat[i * 5 + j] = processed_h[j];
        }
    }
    Logger::log(LogLevel::Info, "Preprocessing complete.");

    unsigned int* d_test_hashes_flat = nullptr;
    unsigned char* d_results_byte = nullptr;

    oss.str(""); oss.clear();
    oss << "Allocating GPU memory for " << ACTUAL_M_TEST_HASHES << " test hashes and results...";
    Logger::log(LogLevel::Info, oss.str());
    cudaError_t allocErr = cudaMalloc((void**)&d_test_hashes_flat, ACTUAL_M_TEST_HASHES * 5 * sizeof(unsigned int));
    if (allocErr != cudaSuccess) {
        oss.str(""); oss.clear();
        oss << "Failed to allocate memory for d_test_hashes_flat: " << cudaGetErrorString(allocErr);
        Logger::log(LogLevel::Error, oss.str());
        return 1;
    }
    allocErr = cudaMalloc((void**)&d_results_byte, ACTUAL_M_TEST_HASHES * sizeof(bool));
    if (allocErr != cudaSuccess) {
        oss.str(""); oss.clear();
        oss << "Failed to allocate memory for d_results_byte: " << cudaGetErrorString(allocErr);
        Logger::log(LogLevel::Error, oss.str());
        cudaFree(d_test_hashes_flat);
        return 1;
    }
    Logger::log(LogLevel::Info, "GPU memory allocation successful.");

    oss.str(""); oss.clear();
    oss << "Copying " << ACTUAL_M_TEST_HASHES << " test hashes (flattened and processed) to GPU...";
    Logger::log(LogLevel::Info, oss.str());
    cudaError_t copyErr = cudaMemcpy(d_test_hashes_flat, test_hashes_flat.data(), ACTUAL_M_TEST_HASHES * 5 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (copyErr != cudaSuccess) {
        oss.str(""); oss.clear();
        oss << "Failed to copy test_hashes_flat to GPU: " << cudaGetErrorString(copyErr);
        Logger::log(LogLevel::Error, oss.str());
        cudaFree(d_test_hashes_flat);
        cudaFree(d_results_byte);
        return 1;
    }
    Logger::log(LogLevel::Info, "Test hashes copied to GPU.");

    oss.str(""); oss.clear();
    oss << "Launching benchmarkCheckHashKernel for " << ACTUAL_M_TEST_HASHES << " hashes...";
    Logger::log(LogLevel::Info, oss.str());
    int threadsPerBlock = 256;
    int blocksPerGrid = (ACTUAL_M_TEST_HASHES + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t kernel_start_event, kernel_stop_event;
    cudaEventCreate(&kernel_start_event);
    cudaEventCreate(&kernel_stop_event);

    cudaEventRecord(kernel_start_event);
    benchmarkCheckHashKernel<<<blocksPerGrid, threadsPerBlock>>>(d_test_hashes_flat, (bool*)d_results_byte, ACTUAL_M_TEST_HASHES);
    cudaEventRecord(kernel_stop_event);

    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        oss.str(""); oss.clear();
        oss << "Kernel launch or execution failed: " << cudaGetErrorString(kernelErr);
        Logger::log(LogLevel::Error, oss.str());
    } else {
        cudaEventSynchronize(kernel_stop_event);
        float kernel_time_ms = 0;
        cudaEventElapsedTime(&kernel_time_ms, kernel_start_event, kernel_stop_event);
        oss.str(""); oss.clear();
        oss << std::fixed << std::setprecision(3) << "benchmarkCheckHashKernel execution time: " << kernel_time_ms << " ms";
        Logger::log(LogLevel::Info, oss.str());
        if (ACTUAL_M_TEST_HASHES > 0) {
            oss.str(""); oss.clear();
            oss << std::fixed << std::setprecision(3) << "Average lookup time per hash: " << (kernel_time_ms * 1000.0f) / ACTUAL_M_TEST_HASHES << " us";
            Logger::log(LogLevel::Info, oss.str());
        }
    }

    Logger::log(LogLevel::Info, "Copying results from GPU to host...");
    std::vector<unsigned char> host_results_char(ACTUAL_M_TEST_HASHES);
    copyErr = cudaMemcpy(host_results_char.data(), d_results_byte, ACTUAL_M_TEST_HASHES * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    if (copyErr != cudaSuccess) {
        oss.str(""); oss.clear();
        oss << "Failed to copy results from GPU: " << cudaGetErrorString(copyErr);
        Logger::log(LogLevel::Error, oss.str());
    } else {
        Logger::log(LogLevel::Info, "Results copied to host.");

        std::vector<bool> host_kernel_results(ACTUAL_M_TEST_HASHES);
        for(int i=0; i < ACTUAL_M_TEST_HASHES; ++i) {
            host_kernel_results[i] = (host_results_char[i] != 0);
        }

        Logger::log(LogLevel::Info, "--- Verification Results ---");
        int false_positives = 0;
        int false_negatives = 0;
        int true_positives_found = 0;
        int true_negatives_correct = 0;
        int num_actual_negatives_in_test_set = ACTUAL_M_TEST_HASHES - actual_positives_in_test_set;

        for (int i = 0; i < ACTUAL_M_TEST_HASHES; ++i) {
            if (is_actually_target_truth[i]) {
                if (host_kernel_results[i]) {
                    true_positives_found++;
                } else {
                    false_negatives++;
                }
            } else {
                if (host_kernel_results[i]) {
                    false_positives++;
                } else {
                    true_negatives_correct++;
                }
            }
        }
        oss.str(""); oss.clear(); oss << "Total test hashes processed: " << ACTUAL_M_TEST_HASHES;
        Logger::log(LogLevel::Info, oss.str());
        oss.str(""); oss.clear(); oss << "Actual positives in test set (expected true): " << actual_positives_in_test_set;
        Logger::log(LogLevel::Info, oss.str());
        oss.str(""); oss.clear(); oss << "Actual negatives in test set (expected false): " << num_actual_negatives_in_test_set;
        Logger::log(LogLevel::Info, oss.str());
        Logger::log(LogLevel::Info, " ");
        oss.str(""); oss.clear(); oss << "True positives correctly found by kernel: " << true_positives_found << " (out of " << actual_positives_in_test_set << ")";
        Logger::log(LogLevel::Info, oss.str());
        oss.str(""); oss.clear(); oss << "True negatives correctly identified by kernel: " << true_negatives_correct << " (out of " << num_actual_negatives_in_test_set << ")";
        Logger::log(LogLevel::Info, oss.str());
        Logger::log(LogLevel::Info, " ");
        oss.str(""); oss.clear(); oss << "False positives (kernel said true, but was negative): " << false_positives;
        Logger::log(LogLevel::Info, oss.str());
        oss.str(""); oss.clear(); oss << "False negatives (kernel said false, but was positive): " << false_negatives;
        Logger::log(LogLevel::Info, oss.str());

        double fp_rate = 0;
        if (num_actual_negatives_in_test_set > 0) {
            fp_rate = static_cast<double>(false_positives) / num_actual_negatives_in_test_set;
        }
        oss.str(""); oss.clear();
        oss << std::fixed << std::setprecision(6) << "False Positive Rate: " << fp_rate << " (" << false_positives << " / " << num_actual_negatives_in_test_set << ")";
        Logger::log(LogLevel::Info, oss.str());

        if (false_negatives > 0) {
            oss.str(""); oss.clear();
            oss << "WARNING: " << false_negatives << " false negatives encountered! This is unexpected for Bloom filters and indicates a potential issue.";
            Logger::log(LogLevel::Warning, oss.str());
        }
    }

    Logger::log(LogLevel::Info, "Cleaning up GPU resources...");
    cudaFree(d_test_hashes_flat);
    cudaFree(d_results_byte);
    cudaEventDestroy(kernel_start_event);
    cudaEventDestroy(kernel_stop_event);

    Logger::log(LogLevel::Info, "Benchmark finished.");
    return 0;
}
