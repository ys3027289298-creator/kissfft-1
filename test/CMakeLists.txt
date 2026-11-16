function(add_kissfft_test_executable NAME)
    add_kissfft_executable(${NAME} ${ARGN})
    target_include_directories(${NAME} PRIVATE ..)

    if(HAVE_LASX)
        target_compile_options(${NAME} PRIVATE -mlasx)
    endif()

    if(HAVE_LSX)
        target_compile_options(${NAME} PRIVATE -mlsx)
    endif()

    add_test(NAME ${NAME} COMMAND ${NAME})
    set_tests_properties(${NAME} PROPERTIES TIMEOUT 3600)
endfunction()

set(KISSFFT_TEST_NUMFFTS 10000)

#
# Add tools-independent fastfilt_* (../tools/fft_*) executable without adding a test
#

add_kissfft_executable(fastfilt ../tools/fftutil.c)
target_include_directories(fastfilt PRIVATE ..)

#
# Add test executables and define tests
#

add_kissfft_test_executable(bm_kiss benchkiss.c pstats.c)
# add_test(NAME benchmar COMMAND ${NAME})
# set_tests_properties(${NAME} PROPERTIES TIMEOUT 3600)

include(FindPkgConfig)
if(KISSFFT_FLOAT)
    set(fftw3_pkg fftw3f)
else()
    set(fftw3_pkg fftw3)
endif()
# The fftw benchmark is optional: without the fftw development package the
# rest of the suite still configures and runs.
pkg_check_modules(fftw3 QUIET IMPORTED_TARGET ${fftw3_pkg})
if(fftw3_FOUND)
    add_kissfft_test_executable(bm_fftw benchfftw.c pstats.c)
    target_link_libraries(bm_fftw PRIVATE PkgConfig::fftw3)
else()
    message(WARNING "fftw3 (${fftw3_pkg}) not found: skipping the bm_fftw benchmark")
endif()

add_kissfft_test_executable(st twotonetest.c)

add_kissfft_test_executable(tkfc twotonetest.c)
target_compile_definitions(tkfc PRIVATE KFC_TEST)

add_kissfft_test_executable(ffr twotonetest.c)
add_kissfft_test_executable(tr test_real.c)
add_kissfft_test_executable(tsizeval test_size_validation.c)

if(KISSFFT_HAS_F16)
    # binary16 i/o layer: built with heap accounting so the test can assert the
    # exact footprint and that a transform allocates nothing.
    add_kissfft_test_executable(test_f16 test_f16.c ../kiss_fft_f16.c ../kiss_fft.c)
    target_compile_definitions(test_f16 PRIVATE TEST_F16_WRAP_MALLOC)
    # The test inspects NaN/infinity bit patterns, which -ffast-math (added
    # globally by the top-level CMakeLists) lets the compiler assume away.
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(test_f16 PRIVATE -fno-fast-math)
        target_link_options(test_f16 PRIVATE -Wl,--wrap=malloc -Wl,--wrap=free)
    endif()
endif()

add_kissfft_test_executable(next_fast_size next_fast_size.c)

add_kissfft_test_executable(testcpp testcpp.cc)

if(KISSFFT_DATATYPE MATCHES "^simd$")
    add_kissfft_test_executable(tsimd test_simd.c)
    target_compile_definitions(tsimd PRIVATE USE_SIMD)
    if(HAVE_LASX)
        target_compile_options(kissfft PRIVATE -mlasx)
    elseif(HAVE_LSX)
        target_compile_options(kissfft PRIVATE -mlsx)
    elseif(NOT MSVC)
        target_compile_options(kissfft PRIVATE -msse)
    else()
        target_compile_options(kissfft PRIVATE "/arch:SSE")
    endif()
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME testkiss.py COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/testkiss.py")
list(APPEND TESTKISS_PY_ENV "KISSFFT_DATATYPE=${KISSFFT_DATATYPE}")
list(APPEND TESTKISS_PY_ENV "KISSFFT_OPENMP=${KISSFFT_OPENMP}")
set_tests_properties(testkiss.py PROPERTIES
    TIMEOUT 3600
    ENVIRONMENT "${TESTKISS_PY_ENV}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
