# 定义宏简化重复操作
function(add_test_executable test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE ddla_lib)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# 调用宏添加测试
# add_test_executable(test_pzgemm)
add_test_executable(test_pdam)
add_test_executable(test_getrf_nopiv)