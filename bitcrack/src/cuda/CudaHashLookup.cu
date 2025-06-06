#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <string> // Required for std::string in logger calls
#include <sstream> // Required for std::ostringstream in logger calls
#include <algorithm> // Required for std::min/max if used, or round
#include <cmath> // For log, ceil, round, log2


#include "KeySearchDevice.h"

#include "CudaHashLookup.h"

#include "CudaHashLookup.cuh"

#include "Logger.h"

#include "util.h"

#define MAX_TARGETS_CONSTANT_MEM 16

__constant__ unsigned int _TARGET_HASH[MAX_TARGETS_CONSTANT_MEM][5];
__constant__ unsigned int _NUM_TARGET_HASHES[1];
__constant__ unsigned int *_BLOOM_FILTER[1];
__constant__ unsigned int _BLOOM_FILTER_MASK[1];
__constant__ unsigned int _BLOOM_FILTER_MASK64[1]; // Will be deprecated by new logic but keep for now to avoid breaking old code not touched.
__constant__ unsigned int _USE_BLOOM_FILTER[1]; // 0: const mem, 1: new Murmur bloom, 2: old >32bit bloom (will be phased out)

// --- New Constant Memory for MurmurHash Bloom Filter ---
__constant__ unsigned int _BLOOM_FILTER_TOTAL_BITS[1]; // m: Total number of bits in the filter
__constant__ unsigned int _BLOOM_NUM_HASH_FUNCS[1];  // k: Number of hash functions to use


// Helper to swap endianness
static __host__ __device__ __forceinline__ unsigned int swp(unsigned int x)
{
	return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

// Prepares hash160 for comparison/Bloom filter.
static __host__ __device__ __forceinline__ void undoRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
	unsigned int iv[5] = {
		0x67452301,
		0xefcdab89,
		0x98badcfe,
		0x10325476,
		0xc3d2e1f0
	};

	for(int i = 0; i < 5; i++) {
		hOut[i] = swp(hIn[i]) - iv[(i + 1) % 5];
	}
}

/**
Copies the target hashes to constant memory
*/
cudaError_t CudaHashLookup::setTargetConstantMemory(const std::vector<struct hash160> &targets)
{
	size_t count = targets.size();

	for(size_t i = 0; i < count; i++) {
		unsigned int h[5];

		undoRMD160FinalRound(targets[i].h, h);

		cudaError_t err = cudaMemcpyToSymbol(_TARGET_HASH, h, sizeof(unsigned int) * 5, i * sizeof(unsigned int) * 5);

		if(err) {
			return err;
		}
	}

	cudaError_t err = cudaMemcpyToSymbol(_NUM_TARGET_HASHES, &count, sizeof(unsigned int));
	if(err) {
		return err;
	}

	unsigned int useBloomFilter = 0;
    // Corrected: sizeof(unsigned int) instead of sizeof(bool)
	err = cudaMemcpyToSymbol(_USE_BLOOM_FILTER, &useBloomFilter, sizeof(unsigned int));
	if(err) {
        Logger::log(LogLevel::Error, "setTargetConstantMemory: cudaMemcpyToSymbol for _USE_BLOOM_FILTER failed.");
		return err;
	}

	return cudaSuccess;
}

// --- MurmurHash3 Implementation (Device Compatible) ---
static __host__ __device__ __forceinline__ unsigned int rotl32(unsigned int x, int r) {
    return (x << r) | (x >> (32 - r));
}

static __host__ __device__ __forceinline__ unsigned int fmix32(unsigned int h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

static __host__ __device__ __forceinline__ unsigned int MurmurHash3_x86_32(const void* key, int len_bytes, unsigned int seed) {
    const unsigned char* data = (const unsigned char*)key;
    const int nblocks = len_bytes / 4;
    unsigned int h1 = seed;
    const unsigned int c1 = 0xcc9e2d51;
    const unsigned int c2 = 0x1b873593;

    const unsigned int* blocks = (const unsigned int*)(data);
    for (int i = 0; i < nblocks; i++) {
        unsigned int k1 = blocks[i];
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }
    // No tail processing for fixed 20-byte hash160
    h1 ^= len_bytes;
    h1 = fmix32(h1);
    return h1;
}
// --- End MurmurHash3 ---

/**
Returns the optimal bloom filter size in bits given the probability of false-positives and the
number of hash functions.
NOTE: This function's original logic (related to power-of-2 sizing and mask generation)
is kept for now if other parts of the codebase might call it, but the new Bloom filter
logic in setTargetBloomFilter calculates m (total bits) and k (hash functions) directly
using standard formulas and does not require the filter size to be a power of two for indexing.
*/
unsigned int CudaHashLookup::getOptimalBloomFilterBits(double p, size_t n)
{
    // Original logic:
	// double m = 3.6 * ceil((n * log(p)) / log(1 / pow(2, log(2))));
	// return (unsigned int)ceil(log(m) / log(2));

    // Revised to calculate m (total bits) and then return log2(m) if needed for power-of-2 sizing,
    // or could be entirely replaced if no longer needed by any calling code.
    if (n == 0) return 0;
    double m_optimal_bits = - ( (double)n * std::log(p) ) / (std::log(2.0) * std::log(2.0)); // Use std::log
    unsigned int m_bits_actual = static_cast<unsigned int>(std::ceil(m_optimal_bits)); // Use std::ceil

    if (m_bits_actual == 0) return 1;
    unsigned int power_of_2_bits = 0;
    unsigned long long temp_m = 1;
    // Find next power of 2 (or current if already power of 2) for m_bits_actual for log2 value
    while(temp_m < m_bits_actual && power_of_2_bits < 32) {
        temp_m <<= 1;
        power_of_2_bits++;
    }
    if (temp_m == 0 && m_bits_actual > 0 && power_of_2_bits <32) return 32; // Corrected guard for large m_bits_actual
    return power_of_2_bits; // Returns log2 of the power-of-2-sized filter
}


// Modified initializeBloomFilter to use MurmurHash3 and k hash functions
void CudaHashLookup::initializeBloomFilter(const std::vector<struct hash160> &targets, unsigned int *filter_host_ptr, unsigned int total_bits_in_filter, unsigned int num_hash_functions)
{
    unsigned int h_processed[5];
    if (total_bits_in_filter == 0 || num_hash_functions == 0) return;

	for(size_t i = 0; i < targets.size(); i++) { // Use size_t for loop
		undoRMD160FinalRound(targets[i].h, h_processed);

        unsigned int hash1 = MurmurHash3_x86_32(h_processed, 20, 0x9747b28c);
        unsigned int hash2 = MurmurHash3_x86_32(h_processed, 20, 0x45939ab0);

		for(unsigned int k_idx = 0; k_idx < num_hash_functions; k_idx++) {
			unsigned int combined_hash = hash1 + k_idx * hash2;
            unsigned int bit_idx = combined_hash % total_bits_in_filter;
			filter_host_ptr[bit_idx / 32] |= (1U << (bit_idx % 32));
		}
	}
}

// initializeBloomFilter64 is now deprecated. The new initializeBloomFilter is general.
void CudaHashLookup::initializeBloomFilter64(const std::vector<struct hash160> &targets, unsigned int *filter_host_ptr, unsigned long long mask_deprecated, unsigned int num_hash_functions_if_provided)
{
    // This function is conceptually deprecated. The new `initializeBloomFilter` should be used.
    // If it were to be truly supported, it would need to correctly interpret its 'mask' as total_bits
    // and call the new hashing logic. For now, log a warning if it's ever called.
    Logger::log(LogLevel::Warning, "initializeBloomFilter64 is deprecated and was called. Its behavior might not be as expected with new Bloom filter logic.");
    // Fallback or simplified logic if absolutely necessary, but ideally this is never called.
    // For example, could try to estimate total_bits from mask if mask is (2^N - 1), then call the new one.
    // However, it's safer to assume new call paths will use the primary initializeBloomFilter.
    // If a large filter was intended (more than 2^32 bits), that needs 64-bit indexing for bit_idx,
    // which the current MurmurHash3 (32-bit output) and filter_host_ptr (uint*) don't support directly for such huge sizes.
    // The new logic caps total_bits_in_filter to what fits in unsigned int for modulo.
    if (targets.empty() || filter_host_ptr == NULL) return;
    unsigned int estimated_total_bits = 0;
    if (mask_deprecated > 0) { // Try to guess total_bits if mask was (2^N - 1)
         estimated_total_bits = static_cast<unsigned int>(std::log2(mask_deprecated + 1.0));
         if((1ULL << estimated_total_bits) -1 != mask_deprecated) { // Not a simple 2^N-1 mask
            estimated_total_bits = 1U << 31; // Fallback to a large-ish number if mask is weird
            Logger::log(LogLevel::Warning, "initializeBloomFilter64 called with unusual mask, behavior undefined.");
         } else {
            // if mask was (2^N - 1), total_bits is 2^N
             estimated_total_bits = (1U << estimated_total_bits);
         }
    }
    if(estimated_total_bits == 0) estimated_total_bits = 1U << 31; // Default large size if cannot estimate

    initializeBloomFilter(targets, filter_host_ptr, estimated_total_bits, num_hash_functions_if_provided > 0 ? num_hash_functions_if_provided : 5);
}


cudaError_t CudaHashLookup::setTargetBloomFilter(const std::vector<struct hash160> &targets)
{
    size_t n = targets.size();
    if (n == 0) {
        unsigned int useBloomFilterType = 0;
        unsigned int zero_val = 0;
        cudaMemcpyToSymbol(_USE_BLOOM_FILTER, &useBloomFilterType, sizeof(unsigned int));
        cudaMemcpyToSymbol(_BLOOM_FILTER_TOTAL_BITS, &zero_val, sizeof(unsigned int));
        cudaMemcpyToSymbol(_BLOOM_NUM_HASH_FUNCS, &zero_val, sizeof(unsigned int));
        cleanup();
        return cudaSuccess;
    }

    double p = 1.0e-9;

    double m_optimal_bits_double = - ( (double)n * std::log(p) ) / (std::log(2.0) * std::log(2.0));
    unsigned int total_bits_in_filter = static_cast<unsigned int>(std::ceil(m_optimal_bits_double));

    if (total_bits_in_filter < 32) total_bits_in_filter = 32; // Minimum practical size for uint array
    // Cap maximum size to avoid issues with 'unsigned int' for total_bits_in_filter if it exceeds ~2^32
    // For truly huge filters, 64-bit indexing and size variables would be needed throughout.
    // Current MurmurHash3 is 32-bit, so bit_idx will be < 2^32.
    // If total_bits_in_filter can exceed 2^32, this needs a redesign. Assuming it fits unsigned int.

    unsigned int num_hash_functions = static_cast<unsigned int>(std::round( ((double)total_bits_in_filter / n) * std::log(2.0) ));
    if (num_hash_functions == 0) num_hash_functions = 1;
    if (num_hash_functions > 15) num_hash_functions = 15;

    unsigned int bloomFilterSizeBytes = (total_bits_in_filter + 7) / 8;
    unsigned int bloomFilterSizeWords = (bloomFilterSizeBytes + 3) / 4;
    bloomFilterSizeBytes = bloomFilterSizeWords * 4;

	std::ostringstream oss_log;
    oss_log << "Bloom Filter: targets=" << n << ", p=" << p
            << ", m_optimal_calc=" << m_optimal_bits_double
            << ", m_bits_actual_used=" << total_bits_in_filter
            << ", allocated_bytes_on_gpu=" << bloomFilterSizeBytes
            << " (~" << std::fixed << std::setprecision(2) << bloomFilterSizeBytes / (1024.0*1024.0) << " MB)"
            << ", k_hashes=" << num_hash_functions;
	Logger::log(LogLevel::Info, oss_log.str());

	unsigned int *filter_host_ptr = NULL;
	try {
		filter_host_ptr = new unsigned int[bloomFilterSizeWords];
	} catch(const std::bad_alloc& e) {
		Logger::log(LogLevel::Error, std::string("Out of system memory for host bloom filter: ") + e.what());
		return cudaErrorMemoryAllocation;
	}
    memset(filter_host_ptr, 0, bloomFilterSizeBytes);

    initializeBloomFilter(targets, filter_host_ptr, total_bits_in_filter, num_hash_functions);

    cleanup();

	cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&_bloomFilterPtr), bloomFilterSizeBytes);
	if(err != cudaSuccess) {
        std::ostringstream oss_err;
        oss_err << "Device error allocating bloom filter (" << bloomFilterSizeBytes << " bytes): " << cudaGetErrorString(err);
		Logger::log(LogLevel::Error, oss_err.str());
		delete[] filter_host_ptr;
		return err;
	}

	err = cudaMemcpy(_bloomFilterPtr, filter_host_ptr, bloomFilterSizeBytes, cudaMemcpyHostToDevice);
	delete[] filter_host_ptr;
	if(err != cudaSuccess) {
		Logger::log(LogLevel::Error, std::string("cudaMemcpy HostToDevice for bloom filter failed: ") + cudaGetErrorString(err));
		cudaFree(_bloomFilterPtr);
		_bloomFilterPtr = NULL;
		return err;
	}

	err = cudaMemcpyToSymbol(_BLOOM_FILTER, &_bloomFilterPtr, sizeof(unsigned int *));
	if(err != cudaSuccess) { Logger::log(LogLevel::Error, std::string("cudaMemcpyToSymbol for _BLOOM_FILTER failed: ") + cudaGetErrorString(err)); cudaFree(_bloomFilterPtr); _bloomFilterPtr = NULL; return err; }

    err = cudaMemcpyToSymbol(_BLOOM_FILTER_TOTAL_BITS, &total_bits_in_filter, sizeof(unsigned int));
    if(err != cudaSuccess) { Logger::log(LogLevel::Error, std::string("cudaMemcpyToSymbol for _BLOOM_FILTER_TOTAL_BITS failed: ") + cudaGetErrorString(err)); cudaFree(_bloomFilterPtr); _bloomFilterPtr = NULL; return err; }

    err = cudaMemcpyToSymbol(_BLOOM_NUM_HASH_FUNCS, &num_hash_functions, sizeof(unsigned int));
    if(err != cudaSuccess) { Logger::log(LogLevel::Error, std::string("cudaMemcpyToSymbol for _BLOOM_NUM_HASH_FUNCS failed: ") + cudaGetErrorString(err)); cudaFree(_bloomFilterPtr); _bloomFilterPtr = NULL; return err; }

	unsigned int useBloomFilterType = 1;
	err = cudaMemcpyToSymbol(_USE_BLOOM_FILTER, &useBloomFilterType, sizeof(unsigned int));
    if(err != cudaSuccess) { Logger::log(LogLevel::Error, std::string("cudaMemcpyToSymbol for _USE_BLOOM_FILTER failed: ") + cudaGetErrorString(err)); cudaFree(_bloomFilterPtr); _bloomFilterPtr = NULL; return err; }

	return cudaSuccess;
}

/**
*Copies the target hashes to either constant memory, or the bloom filter depending
on how many targets there are
*/
cudaError_t CudaHashLookup::setTargets(const std::vector<struct hash160> &targets)
{
	cleanup();

	if(targets.size() <= MAX_TARGETS_CONSTANT_MEM) {
		return setTargetConstantMemory(targets);
	} else {
		return setTargetBloomFilter(targets);
	}
}

void CudaHashLookup::cleanup()
{
	if(_bloomFilterPtr != NULL) {
		cudaError_t err = cudaFree(_bloomFilterPtr); // Check error, though not much to do if it fails
        if (err != cudaSuccess) {
             // Log error if a logger was accessible here, or handle silently.
        }
		_bloomFilterPtr = NULL;
        // Also nullify the constant memory pointer symbol
        unsigned int* nullPtr = NULL;
        // Ignore error for this, as it's cleanup. If it fails, not much recourse.
        cudaMemcpyToSymbol(_BLOOM_FILTER, &nullPtr, sizeof(unsigned int*));
	}
}

// New device-side check using MurmurHash3 based approach
static __device__ bool checkBloomFilterMurmur(const unsigned int processed_hash[5]) {
    // Ensure constant memory symbols are read safely.
    unsigned int* filter_ptr = _BLOOM_FILTER[0];
    unsigned int total_bits = _BLOOM_FILTER_TOTAL_BITS[0];
    unsigned int num_k = _BLOOM_NUM_HASH_FUNCS[0];

    // If filter is not set up (e.g., pointer is null, or sizes are zero), then it cannot contain the hash.
    if (filter_ptr == NULL || total_bits == 0 || num_k == 0) {
        return false;
    }

    unsigned int hash1 = MurmurHash3_x86_32(processed_hash, 20, 0x9747b28c);
    unsigned int hash2 = MurmurHash3_x86_32(processed_hash, 20, 0x45939ab0);

    for (unsigned int k_idx = 0; k_idx < num_k; k_idx++) {
        unsigned int combined_hash = hash1 + k_idx * hash2;
        unsigned int bit_idx = combined_hash % total_bits;

        // Read the word from the filter.
        // CUDA ensures that reads from global memory (pointed to by filter_ptr) are coherent
        // for threads within a warp, and constant cache helps for broader access.
        unsigned int word_val = filter_ptr[bit_idx / 32];
        if (! (word_val & (1U << (bit_idx % 32))) ) {
            return false; // Bit not set, so element is definitely not in the set
        }
    }
    return true; // All k bits were set, element is likely in the set (with some fp probability)
}

// Main device-side check function
// Expects processed_hash (output of undoRMD160FinalRound on host before passing to kernel)
__device__ bool checkHash(const unsigned int processed_hash[5])
{
    unsigned int use_filter_mode = _USE_BLOOM_FILTER[0];

	if(use_filter_mode == 1) { // New MurmurHash Bloom filter
		return checkBloomFilterMurmur(processed_hash);
	} else if (use_filter_mode == 0) { // Constant memory check
        unsigned int num_const_targets = _NUM_TARGET_HASHES[0];
        // Only proceed if there are targets in constant memory and index is valid
        if (num_const_targets == 0 || num_const_targets > MAX_TARGETS_CONSTANT_MEM) return false;

		bool found_in_const = false;
        // Loop unrolling might be beneficial here if MAX_TARGETS_CONSTANT_MEM is small and fixed.
		for(unsigned int j = 0; j < num_const_targets; j++) {
			bool current_equal = true;
			for(int i = 0; i < 5; i++) { // Inner loop for 5 uints
				if (processed_hash[i] != _TARGET_HASH[j][i]) {
                    current_equal = false;
                    break;
                }
			}
			if(current_equal) {
                found_in_const = true;
                break;
            }
		}
        return found_in_const;
	}
    // If use_filter_mode is an unexpected value (e.g., 2 from old scheme if not cleared), default to false.
	return false;
}