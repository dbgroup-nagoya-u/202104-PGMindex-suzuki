add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark pgmindexlib)
add_executable(2dbench 2dbench.cpp)
target_link_libraries(2dbench pgmindexlib)