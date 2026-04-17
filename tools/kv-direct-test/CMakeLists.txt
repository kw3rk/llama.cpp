set(TARGET llama-kv-direct-test)
add_executable(${TARGET} kv-direct-test.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
