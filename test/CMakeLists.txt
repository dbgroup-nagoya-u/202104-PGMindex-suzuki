add_executable(tests main.cpp tests.cpp)
target_link_libraries(tests pgmindexlib)
add_test(NAME test_all COMMAND tests)

add_executable(test_knn test_knn.cpp)
target_link_libraries(test_knn pgmindexlib)