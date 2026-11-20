# Construct the TVM source tree in the build directory. The nested FFI source
# comes from TileLang's release pin rather than TVM's older gitlink.
function(tilelang_prepare_tvm upstream output)
  find_package(Git REQUIRED)
  set(ffi_source "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm-ffi")
  if(NOT EXISTS "${upstream}/CMakeLists.txt" OR NOT EXISTS "${ffi_source}/CMakeLists.txt")
    message(FATAL_ERROR "Missing TVM or TVM FFI sources. Run git submodule update --init --recursive.")
  endif()
  set(patch "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/tvm-ffi-0.1.14.post0.patch")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${patch}")
  file(STRINGS "${patch}" patched REGEX "^\\+\\+\\+ b/")
  list(TRANSFORM patched REPLACE "^\\+\\+\\+ b/" "")
  set(scratch "${CMAKE_CURRENT_BINARY_DIR}/tvm-patch-input")
  foreach(path IN LISTS patched)
    configure_file("${upstream}/${path}" "${scratch}/${path}" COPYONLY)
  endforeach()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}"
            "${GIT_EXECUTABLE}" apply --whitespace=nowarn "${patch}"
    WORKING_DIRECTORY "${scratch}" COMMAND_ERROR_IS_FATAL ANY)

  # Mirror the compiler, runtime Python and headers so their installed copies
  # match the compiled objects. configure_file preserves unchanged outputs.
  foreach(dir IN ITEMS src include python cmake)
    file(GLOB_RECURSE files CONFIGURE_DEPENDS RELATIVE "${upstream}" "${upstream}/${dir}/*")
    list(FILTER files EXCLUDE REGEX "(__pycache__/|\\.pyc$)")
    foreach(path IN LISTS files)
      if(path IN_LIST patched)
        get_filename_component(parent "${output}/${path}" DIRECTORY)
        file(MAKE_DIRECTORY "${parent}")
        file(COPY_FILE "${scratch}/${path}" "${output}/${path}" ONLY_IF_DIFFERENT)
      else()
        configure_file("${upstream}/${path}" "${output}/${path}" COPYONLY)
      endif()
    endforeach()
    file(GLOB_RECURSE staged RELATIVE "${output}" "${output}/${dir}/*")
    foreach(path IN LISTS staged)
      if(NOT EXISTS "${upstream}/${path}")
        file(REMOVE "${output}/${path}")
      endif()
    endforeach()
  endforeach()

  file(GLOB entries RELATIVE "${upstream}" "${upstream}/*")
  list(REMOVE_ITEM entries .git src include python cmake 3rdparty build)
  list(FILTER entries EXCLUDE REGEX "^cmake-build-")
  foreach(entry IN LISTS entries)
    file(COPY "${upstream}/${entry}" DESTINATION "${output}"
         PATTERN ".git" EXCLUDE PATTERN "__pycache__" EXCLUDE)
  endforeach()
  file(COPY "${upstream}/3rdparty/" DESTINATION "${output}/3rdparty"
       PATTERN ".git" EXCLUDE PATTERN "tvm-ffi" EXCLUDE)
  file(COPY "${ffi_source}/" DESTINATION "${output}/3rdparty/tvm-ffi"
       PATTERN ".git" EXCLUDE PATTERN "__pycache__" EXCLUDE PATTERN "build" EXCLUDE)
endfunction()
