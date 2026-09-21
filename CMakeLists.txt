cmake_minimum_required(VERSION 3.20)
project(rigid_kernels VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
option(RIGID_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)

add_library(rigid_kernels SHARED cpp/src/tanimoto.cpp)
target_compile_features(rigid_kernels PUBLIC cxx_std_17)
target_include_directories(rigid_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
    $<INSTALL_INTERFACE:include>)
target_compile_definitions(rigid_kernels PRIVATE RIGID_BUILD_SHARED)
target_compile_definitions(rigid_kernels INTERFACE RIGID_USE_SHARED)
set_target_properties(rigid_kernels PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(rigid_kernels PRIVATE /W4 /permissive-)
else()
    target_compile_options(rigid_kernels PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
    if(RIGID_ENABLE_SANITIZERS)
        target_compile_options(rigid_kernels PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(rigid_kernels PUBLIC -fsanitize=address,undefined)
    endif()
endif()

if(BUILD_TESTING)
    add_executable(test_tanimoto cpp/tests/test_tanimoto.cpp)
    target_link_libraries(test_tanimoto PRIVATE rigid_kernels)
    add_test(NAME packed_tanimoto COMMAND test_tanimoto)
endif()

install(TARGETS rigid_kernels LIBRARY DESTINATION . RUNTIME DESTINATION . ARCHIVE DESTINATION lib)
install(FILES cpp/include/rigid/tanimoto.h DESTINATION include/rigid)
