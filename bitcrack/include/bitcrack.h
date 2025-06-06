#ifndef BITCRACK_H
#define BITCRACK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h> // For uint64_t, etc.
#include <stdbool.h> // For bool type in C

// --- Basic Data Structures ---

// Representing a HASH160 (20 bytes)
typedef struct {
    unsigned char bytes[20];
} BitCrackHash160;

// Representing a found key result
typedef struct {
    BitCrackHash160 hash160;    // The target hash160 that was found
    char private_key_hex[65];   // Null-terminated hex string for private key (64 chars + null)
    char public_key_hex[131];   // Null-terminated hex string for public key (compressed: 66, uncompressed: 130)
    bool compressed_pubkey;     // Was the public key associated with the find compressed?
} BitCrackSearchResult;

// Opaque type for the key finder engine instance
typedef struct BitCrackFinder BitCrackFinder;

// --- Callbacks ---

// Callback function type for reporting a found key
// Userdata is a pointer passed by the user to init, and given back here.
typedef void (*BitCrackResultCallback)(const BitCrackSearchResult* result, void* userdata);

// Callback function type for reporting progress/status
// Userdata is a pointer passed by the user to init, and given back here.
typedef void (*BitCrackStatusCallback)(
    uint64_t current_key_count, // Keys processed in the current run/since last status
    uint64_t total_key_count,   // Total keys processed by this finder instance so far
    double keys_per_second,
    const char* device_name,    // Optional: name of the device working
    void* userdata
);


// --- Core Library Functions ---

/**
 * @brief Initializes the BitCrack key finder.
 *
 * @param start_key_hex Starting private key in hex string format.
 * @param end_key_hex Ending private key in hex string format.
 * @param device_id The ID of the CUDA device to use.
 * @param points_per_thread CUDA kernel parameter (0 for default).
 * @param blocks CUDA kernel parameter (0 for default).
 * @param threads_per_block CUDA kernel parameter (0 for default).
 * @param result_cb Callback function for when a key is found. Can be NULL.
 * @param status_cb Callback function for progress updates. Can be NULL.
 * @param status_interval_seconds Interval in seconds for status updates (approximate).
 * @param userdata Pointer to user data to be passed to callbacks.
 * @return A pointer to a BitCrackFinder instance, or NULL on error.
 *         The caller is responsible for calling bitcrack_finder_destroy() on this pointer.
 */
BitCrackFinder* bitcrack_finder_create(
    const char* start_key_hex,
    const char* end_key_hex,
    int device_id, // Or other device selection parameters
    // Potentially add more CUDA config params here or use a struct
    unsigned int points_per_thread,
    unsigned int blocks,
    unsigned int threads_per_block,
    BitCrackResultCallback result_cb,
    BitCrackStatusCallback status_cb,
    unsigned int status_interval_seconds,
    void* userdata
);

/**
 * @brief Destroys a BitCrackFinder instance and frees associated resources.
 *
 * @param finder The BitCrackFinder instance to destroy.
 */
void bitcrack_finder_destroy(BitCrackFinder* finder);

/**
 * @brief Adds a list of target hash160s to the finder.
 *        Targets are provided as hex strings.
 *
 * @param finder The BitCrackFinder instance.
 * @param targets_hex Array of null-terminated hex strings representing hash160s.
 * @param num_targets The number of targets in the array.
 * @return 0 on success, non-zero on error.
 */
int bitcrack_finder_add_targets_hex(BitCrackFinder* finder, const char* targets_hex[], unsigned int num_targets);

/**
 * @brief Adds target hash160s from a file.
 *        Each line in the file should be a hash160 hex string.
 *
 * @param finder The BitCrackFinder instance.
 * @param file_path Path to the file containing target hash160s.
 * @return 0 on success, non-zero on error.
 */
int bitcrack_finder_add_targets_from_file(BitCrackFinder* finder, const char* file_path);


/**
 * @brief Starts the key search operation. This function will block until
 *        the search is completed or stopped.
 *
 * @param finder The BitCrackFinder instance.
 * @return 0 on successful completion, non-zero if an error occurred or stopped early.
 */
int bitcrack_finder_run(BitCrackFinder* finder);

/**
 * @brief Stops an ongoing key search operation.
 *        This function is thread-safe and can be called from another thread.
 *
 * @param finder The BitCrackFinder instance.
 */
void bitcrack_finder_stop(BitCrackFinder* finder);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // BITCRACK_H
