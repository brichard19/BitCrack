cmake_minimum_required(VERSION 3.10)
project(BitCrackCoreMinimal LANGUAGES CXX CUDA) # Added CUDA

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For CUDA
set(CMAKE_CUDA_STANDARD 11) # Optional: Explicitly set CUDA standard if different from CXX
set(CMAKE_CUDA_STANDARD_REQUIRED ON) # Optional

# Set CUDA architectures (e.g., 75 for Turing, 86 for Ampere, 90 for Hopper)
set(CMAKE_CUDA_ARCHITECTURES "75;86;90")

# Enable relocatable device code (RDC) for linking .cu files together
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Renaming library to bitcrackcore and adding more sources
add_library(bitcrackcore STATIC # Renamed from bitcrackcore_minimal
    # Core C++ sources (already listed)
    src/core/logger/Logger.cpp
    src/core/util_common/util.cpp
    src/core/Base58.cpp
    src/core/Rng.cpp
    # Adding other core C++ sources identified earlier
    src/core/KeyFinder.cpp
    src/core/hash.cpp
    src/core/checksum.cpp
    src/core/ripemd160.cpp
    src/core/sha256.cpp
    src/core/secp256k1.cpp

    # CUDA related sources (.cu files and .cpp files that are part of cuda library component)
    src/cuda/CudaAtomicList.cu
    src/cuda/CudaDeviceKeys.cu
    src/cuda/CudaHashLookup.cu
    src/cuda/CudaKeySearchDevice.cu   # The .cu file for device logic
    src/cuda/CudaKeySearchDevice.cpp # Host API for CudaKeySearchDevice
    src/cuda/cudabridge.cu
    src/cuda/cudaUtil.cpp            # Host utilities for CUDA components
)

target_include_directories(bitcrackcore PUBLIC # Renamed from bitcrackcore_minimal
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/logger
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/util_common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda # Added cuda include path
)

# Ensure CUDA toolkit components are found if not automatic (already added CUDA to project)
# This find_package call can be moved towards the top if preferred, but works here too.
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA Toolkit found by CMake (for bitcrackcore build).")
else()
    message(WARNING "CUDA Toolkit not found by CMake for bitcrackcore build.")
endif()

message(STATUS "CMakeLists.txt with bitcrackcore (including CUDA sources) processed by CMake.")
# --- bitcrack_cli Executable ---
add_executable(bitcrack_cli
    src/cli/main.cpp
    src/cli/CmdParse.cpp
    src/cli/ConfigFile.cpp
    src/cli/DeviceManager.cpp
    src/cli/cudaInfo_util/main.cpp # This might need to be cuda_info_main.cpp if renamed
                                   # Or be a separate executable target. For now, part of bitcrack_cli.
)

# Include directories for bitcrack_cli
target_include_directories(bitcrack_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cli
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cli/cudaInfo_util
    # bitcrackcore headers are available via target_link_libraries
)

# Link bitcrack_cli against bitcrackcore
target_link_libraries(bitcrack_cli PRIVATE bitcrackcore)

message(STATUS "CMakeLists.txt with bitcrackcore and bitcrack_cli processed by CMake.")
