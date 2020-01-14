cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set (LANGUAGES "C;CXX")
  
project(openacc_lesson LANGUAGES ${LANGUAGES})

#  cmake ../ -DOpenACC_ACCEL_TARGET=tesla:cc60 -DCMAKE_C_COMPILER=pgcc -DCMAKE_CXX_COMPILER=pgc++
find_package(OpenACC REQUIRED)




if (CMAKE_VERSION VERSION_LESS "3.16")
  
  foreach(lang IN LISTS LANGUAGES)
    
    
    string(REPLACE " " "\;" default_options ${OpenACC_${lang}_FLAGS})
    # message(WARNING "Creating OpenACC_${lang}_OPTIONS: ${default_options}")
    
    set(OpenACC_${lang}_OPTIONS ${default_options})
    message(STATUS "OpenACC ${lang} version " ${OpenACC_${lang}_VERSION})
    message(STATUS "OpenACC ${lang} flags " ${OpenACC_${lang}_FLAGS})
    
  endforeach()  

endif()




add_executable(ex7 example7.c)

add_definitions(-DROWS=3000 -DCOLS=3000)
target_compile_options(ex7 PRIVATE ${OpenACC_C_OPTIONS})
target_link_libraries(ex7 PRIVATE ${OpenACC_C_OPTIONS})

# run as ACC_DEVICE_TYPE=host ./ex5
# https://gcc.gnu.org/wiki/OpenACC#ACC_DEVICE_TYPE means it will run single threaded
