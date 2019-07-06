cmake_minimum_required(VERSION 3.1)

set (CMAKE_C_STANDARD 11)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_STATIC_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
   
   set (FLAGS "-O3 -funsigned-char -Wall  -Wl,--no-undefined")  
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
   set (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")
  
   set (EXTRA_LIBRARIES 
		"gomp"
		"m"
		)
endif()

## Build the regularisers package as a library
message("Creating Regularisers as a shared library")

message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message("CMAKE_C_FLAGS ${CMAKE_C_FLAGS}")
message("CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS}")
message("CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS}")
message("CMAKE_STATIC_LINKER_FLAGS ${CMAKE_STATIC_LINKER_FLAGS}")



add_library(fdiff SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/finite_diff.c)
add_executable(fdiffexe ${CMAKE_CURRENT_SOURCE_DIR}/src/finite_diff.c)

target_link_libraries(fdiff ${EXTRA_LIBRARIES} )
include_directories(fdiff PUBLIC 
				  ${CMAKE_CURRENT_SOURCE_DIR}/include
		          ${CMAKE_CURRENT_SOURCE_DIR})

## Install
include(GNUInstallDirs)
install(TARGETS fdiff
	RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
	CONFIGURATIONS ${CMAKE_BUILD_TYPE} 
	)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}) 


find_package(PythonInterp REQUIRED)
if (PYTHONINTERP_FOUND)
  message ("Current Python " ${PYTHON_VERSION_STRING} " found " ${PYTHON_EXECUTABLE})
endif()

set(PYTHON_DEST_DIR "" CACHE PATH "Directory of the Python wrappers")
if (PYTHON_DEST_DIR)
 set(PYTHON_DEST "${PYTHON_DEST_DIR}")
else() 
 set(PYTHON_DEST "${CMAKE_INSTALL_PREFIX}/python")
endif()
message(STATUS "Python wrappers will be installed in " ${PYTHON_DEST})


message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

set(CMAKE_BUILD_TYPE "Release")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/setup.py.in" "${CMAKE_CURRENT_BINARY_DIR}/setup.py")


find_package(PythonInterp)
find_package(PythonLibs)
if (PYTHONINTERP_FOUND)
  message(STATUS "Found PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
  message(STATUS "Python version ${PYTHON_VERSION_STRING}")
endif()
if (PYTHONLIBS_FOUND)
  message(STATUS "Found PYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIRS}")
  message(STATUS "Found PYTHON_LIBRARIES=${PYTHON_LIBRARIES}")
endif()

if (PYTHONINTERP_FOUND)
    message("Python found " ${PYTHON_EXECUTABLE})
    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/src/setup.py.in")
    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    #set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/module/__init__.py")
    set (DEPS       "${CMAKE_BINARY_DIR}/")
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    message("Core binary dir " ${CMAKE_BINARY_DIR}/Core/${CMAKE_BUILD_TYPE})
    
    if (CONDA_BUILD)
      add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/ccpi ${CMAKE_CURRENT_BINARY_DIR}/ccpi
                       COMMAND ${CMAKE_COMMAND} -E env CIL_VERSION=${CIL_VERSION}
                                                       PREFIX=${CMAKE_SOURCE_DIR}/src/Core
                                                       LIBRARY_INC=${CMAKE_SOURCE_DIR}/src/Core
                                                       LIBRARY_LIB=${CMAKE_BINARY_DIR}/src/Core
                                                       ${PYTHON_EXECUTABLE} ${SETUP_PY} install
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS cilreg)

    else()
      if (WIN32)
        add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/ccpi ${CMAKE_CURRENT_BINARY_DIR}/ccpi
                       COMMAND ${CMAKE_COMMAND} -E env CIL_VERSION=${CIL_VERSION}
                                                       PREFIX=${CMAKE_SOURCE_DIR}/src/Core
                                                       LIBRARY_INC=${CMAKE_SOURCE_DIR}/src/Core
                                                       LIBRARY_LIB=${CMAKE_BINARY_DIR}/src/Core/${CMAKE_BUILD_TYPE}
                                                       ${PYTHON_EXECUTABLE} ${SETUP_PY} build_ext --inplace
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS cilreg)
      else()
        add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/finite_diff ${CMAKE_CURRENT_BINARY_DIR}/ccpi
                       COMMAND ${CMAKE_COMMAND} -E env PREFIX=${CMAKE_SOURCE_DIR}/src/
                                                       LIBRARY_INC=${CMAKE_SOURCE_DIR}/src
                                                       LIBRARY_LIB=${CMAKE_BINARY_DIR}/
                                                       ${PYTHON_EXECUTABLE} ${SETUP_PY} build_ext --inplace
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS fdiff)
      endif()
      #set (PYTHON_DEST ${CMAKE_INSTALL_PREFIX}/python/)
      install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ccpi 
              DESTINATION ${PYTHON_DEST})
      
    endif()
    
    
    add_custom_target(PythonWrapper ALL DEPENDS ${OUTPUT})
    #install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
endif()
