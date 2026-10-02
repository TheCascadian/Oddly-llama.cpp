message(STATUS  "GGML_SYCL_TARGET=${GGML_SYCL_TARGET}")

if (NOT GGML_SYCL_TARGET MATCHES "^(INTEL)$")
    message(FATAL_ERROR "GGML_SYCL_TARGET: Invalid target, the supported options are [INTEL]")
endif()

check_cxx_compiler_flag("-fsycl" SUPPORTS_SYCL)

if (DEFINED ENV{ONEAPI_ROOT})
    message(STATUS "Using oneAPI Release SYCL compiler (icpx).")
elseif(SUPPORTS_SYCL)
    message(WARNING "Using open-source SYCL compiler (clang++). Didn't detect ENV {ONEAPI_ROOT}.
        If you expected the oneAPI Release compiler, please install oneAPI & source it, like:
        source /opt/intel/oneapi/setvars.sh")
else()
    message(FATAL_ERROR "C++ compiler lacks SYCL support.")
endif()
message(STATUS "SYCL found")
#todo: AOT

ggml_add_backend_library(ggml-sycl
                         ggml-sycl.cpp
                         ../../include/ggml-sycl.h
                        )

file(GLOB   GGML_HEADERS_SYCL "*.hpp")
file(GLOB   GGML_SOURCES_SYCL CONFIGURE_DEPENDS "*.cpp")
file(GLOB   SRCS "template-instances/fattn-tile*.cpp")
list(APPEND GGML_SOURCES_SYCL ${SRCS})
file(GLOB   SRCS "template-instances/fattn-vec*.cpp")
list(APPEND GGML_SOURCES_SYCL ${SRCS})

list(REMOVE_ITEM GGML_SOURCES_SYCL "${CMAKE_CURRENT_SOURCE_DIR}/ptq1-t2.cpp")

target_sources(ggml-sycl PRIVATE ${GGML_HEADERS_SYCL} ${GGML_SOURCES_SYCL})

if (WIN32)
    # To generate a Visual Studio solution, using Intel C++ Compiler for ggml-sycl is mandatory
    if( ${CMAKE_GENERATOR} MATCHES "Visual Studio" AND NOT (${CMAKE_GENERATOR_TOOLSET} MATCHES "Intel C"))
        set_target_properties(ggml-sycl PROPERTIES VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
        set(CMAKE_CXX_COMPILER "icx")
        set(CMAKE_CXX_COMPILER_ID "IntelLLVM")
    endif()
    # Level Zero SDK path for Windows (only when GGML_SYCL_SUPPORT_LEVEL_ZERO_API is enabled)
    if(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        if(DEFINED ENV{LEVEL_ZERO_V1_SDK_PATH})
            set(LEVEL_ZERO_V1_SDK_PATH $ENV{LEVEL_ZERO_V1_SDK_PATH})
            if(EXISTS "${LEVEL_ZERO_V1_SDK_PATH}")
                target_include_directories(ggml-sycl PRIVATE "${LEVEL_ZERO_V1_SDK_PATH}/include")
                set(LEVEL_ZERO_V1_SDK_LIB_PATH "${LEVEL_ZERO_V1_SDK_PATH}/lib")
            else()
                message(WARNING "LEVEL_ZERO_V1_SDK_PATH set but folder not found: ${LEVEL_ZERO_V1_SDK_PATH}")
            endif()
        endif()
    endif()
endif()

macro(detect_and_find_package package_name)
    set(test_source "
    cmake_minimum_required(VERSION ${CMAKE_VERSION})
    project(check_package LANGUAGES CXX)
    find_package(${package_name} QUIET)
    ")

    set(test_dir "${CMAKE_CURRENT_BINARY_DIR}/check_package_${package_name}")
    file(WRITE "${test_dir}/CMakeLists.txt" "${test_source}")

    set(cmake_args "")
    if(CMAKE_GENERATOR)
        list(APPEND cmake_args "-G" "${CMAKE_GENERATOR}")
    endif()
    if(CMAKE_GENERATOR_PLATFORM)
        list(APPEND cmake_args "-A" "${CMAKE_GENERATOR_PLATFORM}")
    endif()
    if(CMAKE_GENERATOR_TOOLSET)
        list(APPEND cmake_args "-T" "${CMAKE_GENERATOR_TOOLSET}")
    endif()
    if(CMAKE_CXX_COMPILER)
        list(APPEND cmake_args "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} ${cmake_args} .
        WORKING_DIRECTORY "${test_dir}"
        RESULT_VARIABLE result
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(result EQUAL 0)
        find_package(${package_name} ${ARGN})
    else()
        message(WARNING "Detection of ${package_name} failed. The package might be broken or incompatible.")
        set(${package_name}_FOUND FALSE)
    endif()
endmacro()

detect_and_find_package(IntelSYCL)
if (IntelSYCL_FOUND)
    # Use oneAPI CMake when possible
    target_link_libraries(ggml-sycl PRIVATE IntelSYCL::SYCL_CXX)
else()
    # Fallback to the simplest way of enabling SYCL when using intel/llvm nightly for instance
    target_compile_options(ggml-sycl PRIVATE "-fsycl")
    target_link_options(ggml-sycl PRIVATE "-fsycl")
endif()

target_compile_options(ggml-sycl PRIVATE "-Wno-narrowing")

message(STATUS "GGML_SYCL_SUPPORT_LEVEL_ZERO_API ${GGML_SYCL_SUPPORT_LEVEL_ZERO_API}")
if (GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
    # Link against Level Zero loader for direct device memory allocation.
    # Avoids sycl::malloc_device triggering DMA-buf/TTM system RAM staging
    # in the xe kernel driver during multi-GPU inference.
    find_path(LEVEL_ZERO_INCLUDE_DIR level_zero/ze_api.h HINTS ${ONEAPI_ROOT}/include ${LEVEL_ZERO_V1_SDK_PATH}/include)
    find_library(ZE_LOADER_LIB ze_loader HINTS ${ONEAPI_ROOT}/lib ${LEVEL_ZERO_V1_SDK_LIB_PATH} ENV LD_LIBRARY_PATH)
    if(ZE_LOADER_LIB AND LEVEL_ZERO_INCLUDE_DIR)
        target_include_directories(ggml-sycl PRIVATE ${LEVEL_ZERO_INCLUDE_DIR})
        target_link_libraries(ggml-sycl PRIVATE ${ZE_LOADER_LIB})
        target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        message(STATUS "Level Zero loader found: ${ZE_LOADER_LIB}")
        message(STATUS "Level Zero headers found: ${LEVEL_ZERO_INCLUDE_DIR}")
    else()
        message(WARNING "Level Zero loader or headers not found, Level Zero support disabled")
    endif()
endif()

# Link against oneDNN
set(GGML_SYCL_DNNL 0)
if(GGML_SYCL_DNN)
    find_package(DNNL)
    if(DNNL_FOUND)
        if (NOT DEFINED DNNL_GPU_VENDOR)
            # default to intel target
            set(DNNL_GPU_VENDOR "INTEL")
            if(NOT "${GGML_SYCL_TARGET}" STREQUAL "INTEL")
                message(WARNING "oneDNN builds bundled with oneapi release only support INTEL target")
            endif()
        endif()

        # Verify oneDNN was compiled for the same target as llama
        if("${GGML_SYCL_TARGET}" STREQUAL "${DNNL_GPU_VENDOR}")
            target_link_libraries(ggml-sycl PRIVATE DNNL::dnnl)
            set(GGML_SYCL_DNNL 1)
            get_target_property(CONFIGS DNNL::dnnl IMPORTED_CONFIGURATIONS)
            foreach(CONFIG ${CONFIGS})
                get_target_property(DNNL_LIB DNNL::dnnl IMPORTED_LOCATION_${CONFIG})
                message(STATUS "Found oneDNN: ${DNNL_LIB}")
            endforeach()
        else()
            message(WARNING
                "oneDNN must be compiled for the same target as llama.cpp.
                 llama.cpp: ${GGML_SYCL_TARGET}, oneDNN: ${DNNL_GPU_VENDOR}.
                 Disabling oneDNN support.")
        endif()
    else()
        message(STATUS "oneDNN not found, disabling oneDNN support")
    endif()
else()
    message(STATUS "oneDNN support disabled by the user")
endif()
target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_DNNL=${GGML_SYCL_DNNL})

if (GGML_SYCL_F16)
    add_compile_definitions(GGML_SYCL_F16)
endif()

if (GGML_SYCL_TARGET STREQUAL "INTEL")
    add_compile_definitions(GGML_SYCL_WARP_SIZE=16)
    if (NOT GGML_SYCL_DEVICE_ARCH)
        target_link_options(ggml-sycl PRIVATE -Xs -ze-intel-greater-than-4GB-buffer-required)
    else()
        message(STATUS "Skipping -ze-intel-greater-than-4GB-buffer-required for spir64_gen AOT")
    endif()

    # Link against Intel oneMKL
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(SYCL_COMPILER ON)
    endif()
    find_package(MKL REQUIRED)

    # Intel's oneAPI compiler and oneMKL ship as versioned components. Mixing
    # releases can produce a successful SYCL compile followed by an unresolved
    # SYCL runtime symbol at link time. CMake caches these package paths, so
    # fail early with an actionable message instead of leaving the user with a
    # confusing linker error.
    set(_ggml_sycl_compiler_release "")
    set(_ggml_sycl_mkl_release "")
    if (CMAKE_CXX_COMPILER MATCHES "/compiler/([0-9]+\\.[0-9]+)/")
        set(_ggml_sycl_compiler_release "${CMAKE_MATCH_1}")
    elseif (CMAKE_CXX_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
        set(_ggml_sycl_compiler_release "${CMAKE_MATCH_1}")
    endif()
    if (MKL_DIR MATCHES "/mkl/([0-9]+\\.[0-9]+)/")
        set(_ggml_sycl_mkl_release "${CMAKE_MATCH_1}")
    endif()
    if (_ggml_sycl_compiler_release AND _ggml_sycl_mkl_release AND
        NOT _ggml_sycl_compiler_release STREQUAL _ggml_sycl_mkl_release)
        message(FATAL_ERROR
            "Intel SYCL toolchain mismatch: DPC++ ${_ggml_sycl_compiler_release} "
            "is paired with oneMKL ${_ggml_sycl_mkl_release} (from ${MKL_DIR}). "
            "Use matching oneAPI compiler and oneMKL releases, then configure a fresh build directory.")
    endif()
    target_link_libraries(ggml-sycl PRIVATE MKL::MKL_SYCL::BLAS)
else()
    # default for other target
    message(FATAL_ERROR "GGML_SYCL_TARGET is not supported")
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
endif()

if (GGML_SYCL_GRAPH)
    message(STATUS "find GGML_SYCL_GRAPH")
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_GRAPH)
endif()

if (GGML_SYCL_HOST_MEM_FALLBACK)
    message(STATUS "find GGML_SYCL_HOST_MEM_FALLBACK")
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_HOST_MEM_FALLBACK)
endif()

if (GGML_SYCL_DEVICE_ARCH)
    message(STATUS "GGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH} (AOT via spir64_gen)")
    target_compile_options(
        ggml-sycl PRIVATE
        -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${GGML_SYCL_DEVICE_ARCH}\""
    )

    # Pass through parallel job (process) count for parallelising the
    # `llvm-foreach -- ocloc` invocation for compiling AOT device images.
    include(ProcessorCount)
    ProcessorCount(_ggml_sycl_nproc)
    if (_ggml_sycl_nproc LESS 1)
        set(_ggml_sycl_nproc 1)
    endif()
    set(GGML_SYCL_MAX_PARALLEL_LINK_JOBS ${_ggml_sycl_nproc} CACHE STRING
        "Parallel ocloc jobs for spir64_gen AOT device-image lowering")
    target_link_options(
        ggml-sycl PRIVATE
        -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${GGML_SYCL_DEVICE_ARCH}\""
        -fsycl-max-parallel-link-jobs=${GGML_SYCL_MAX_PARALLEL_LINK_JOBS}
    )
endif()

# oneMKL enables external device-image imports. IGC builtins must be linked separately.
option(GGML_SYCL_T2_PORTABLE_LOADS "Use standard SYCL loads in the PTQ1 T2 kernels" OFF)
add_library(prism-sycl-t2 SHARED ptq1-t2.cpp)
if (GGML_SYCL_T2_PORTABLE_LOADS)
    target_compile_definitions(prism-sycl-t2 PRIVATE GGML_SYCL_T2_PORTABLE_LOADS)
endif()
target_compile_features(prism-sycl-t2 PRIVATE cxx_std_17)
target_compile_options(prism-sycl-t2 PRIVATE -fsycl)
target_link_options(prism-sycl-t2 PRIVATE -fsycl -fno-sycl-allow-device-image-dependencies)
set_target_properties(prism-sycl-t2 PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_link_libraries(ggml-sycl PRIVATE prism-sycl-t2)
if (GGML_SYCL_DEVICE_ARCH)
    target_compile_options(prism-sycl-t2 PRIVATE -fsycl-targets=spir64_gen)
    target_link_options(prism-sycl-t2 PRIVATE -fsycl-targets=spir64_gen
        -Xsycl-target-backend=spir64_gen "-device ${GGML_SYCL_DEVICE_ARCH}")
endif()
if (GGML_BACKEND_DL)
    if (GGML_BACKEND_DIR)
        set(_t2_install_dir ${GGML_BACKEND_DIR})
    else()
        set(_t2_install_dir ${CMAKE_INSTALL_BINDIR})
    endif()
else()
    set(_t2_install_dir ${CMAKE_INSTALL_LIBDIR})
endif()
set_property(TARGET ggml-sycl APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
install(TARGETS prism-sycl-t2 LIBRARY DESTINATION ${_t2_install_dir}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
