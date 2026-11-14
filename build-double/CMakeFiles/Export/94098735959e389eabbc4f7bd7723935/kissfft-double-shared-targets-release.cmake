#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "kissfft::kissfft-double" for configuration "Release"
set_property(TARGET kissfft::kissfft-double APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(kissfft::kissfft-double PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkissfft-double.so.131.1.0"
  IMPORTED_SONAME_RELEASE "libkissfft-double.so.131"
  )

list(APPEND _cmake_import_check_targets kissfft::kissfft-double )
list(APPEND _cmake_import_check_files_for_kissfft::kissfft-double "${_IMPORT_PREFIX}/lib/libkissfft-double.so.131.1.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
