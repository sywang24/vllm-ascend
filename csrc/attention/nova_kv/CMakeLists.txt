# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
# This is an opt-in standalone launcher, outside the ACLNN operator registry.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    return()
endif()
project(nova_kv LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(SOC_VERSION Ascend910B3 CACHE STRING "Ascend SoC used for the NOVA kernel")
set(RUN_MODE npu)
set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest
    CACHE PATH "CANN toolkit installation")
include(${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

ascendc_library(nova_attention_shared SHARED op_kernel/nova_attention.cpp)
set_target_properties(nova_attention_shared PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
configure_file(nova_kernel_abi.json.in lib/nova_kernel_abi.json COPYONLY)
