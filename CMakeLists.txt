cmake_minimum_required(VERSION 3.30)
project(CodeForTest3WithC)

set(CMAKE_CXX_STANDARD 20)

# Force Release configuration
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MD")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")

# Set vcpkg toolchain file
set(CMAKE_TOOLCHAIN_FILE "C:/Users/user/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Set paths
set(VCPKG_INSTALLED_DIR "C:/Users/user/vcpkg/installed/x64-windows")
set(INCLUDE_DIR "${VCPKG_INSTALLED_DIR}/include")
set(LIB_DIR "${VCPKG_INSTALLED_DIR}/lib")

add_executable(CodeForTest3WithC main.cpp)

# Include directories
target_include_directories(CodeForTest3WithC PRIVATE ${INCLUDE_DIR})

# Link directories
target_link_directories(CodeForTest3WithC PRIVATE ${LIB_DIR})

# Link all required libraries
target_link_libraries(CodeForTest3WithC PRIVATE
        "${LIB_DIR}/armadillo.lib"
        "${LIB_DIR}/lapack.lib"
        "${LIB_DIR}/openblas.lib"
        "${LIB_DIR}/zlib.lib"
)