#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "kissfft::kissfft-int32_t" for configuration "Release"
set_property(TARGET kissfft::kissfft-int32_t APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kissfft::kissfft-int32_t PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkissfft-int32_t.so.131.1.0"
  IMPORTED_SONAME_RELEASE "libkissfft-int32_t.so.131"
  )

list(APPEND _cmake_import_check_targets kissfft::kissfft-int32_t )
list(APPEND _cmake_import_check_files_for_kissfft::kissfft-int32_t "${_IMPORT_PREFIX}/lib/libkissfft-int32_t.so.131.1.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
