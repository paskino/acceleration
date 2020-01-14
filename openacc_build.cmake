find_package(OpenACC)

if (OpenACC_C_FOUND)

if (CMAKE_VERSION VERSION_LESS "3.16")
  
  foreach(lang IN LISTS LANGUAGES)
    
    
    string(REPLACE " " "\;" default_options ${OpenACC_${lang}_FLAGS})
    # message(WARNING "Creating OpenACC_${lang}_OPTIONS: ${default_options}")
    
    set(OpenACC_${lang}_OPTIONS ${default_options})
    message(STATUS "OpenACC ${lang} version " ${OpenACC_${lang}_VERSION})
    message(STATUS "OpenACC ${lang} flags " ${OpenACC_${lang}_FLAGS})
    
  endforeach()  

endif()

if (${CMAKE_C_COMPILER_ID} STREQUAL "PGI")

  message(WARNING "make sure you use target tesla:managed in OpenACC_ACCEL_TARGET")

  foreach(lang IN LISTS LANGUAGES)
    
    set (OpenACC_${lang}_OPTIONS "${OpenACC_${lang}_OPTIONS};-Minfo=all,intensity")
    message (STATUS "OpenACC_${lang}_OPTIONS ${OpenACC_${lang}_OPTIONS}")
    
  endforeach()  

endif()


add_library(fdiff_acc SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/src/axpby_openacc.c)

target_compile_options(fdiff_acc PRIVATE "${OpenACC_C_OPTIONS}")
target_link_libraries(fdiff_acc PRIVATE ${OpenACC_C_OPTIONS})

# unittest
add_executable(test_openacc ${CMAKE_CURRENT_SOURCE_DIR}/src/test_openacc.c)

target_compile_options(test_openacc PRIVATE ${OpenACC_C_OPTIONS})
target_link_libraries(test_openacc PRIVATE ${OpenACC_C_OPTIONS} fdiff_acc)

else()
  message(WARNING "No OpenACC capability found")
endif()