cmake_minimum_required(VERSION 3.20)

project(SansanE VERSION 0.1.0 LANGUAGES CXX)

add_library(sansane_tokenizer
    src/tokenizer.cpp
)

target_include_directories(sansane_tokenizer PUBLIC src)
target_compile_features(sansane_tokenizer PUBLIC cxx_std_23)
set_target_properties(sansane_tokenizer PROPERTIES
    CXX_EXTENSIONS OFF
)

add_executable(checkpoint_inspector
    src/main.cpp
)

target_link_libraries(checkpoint_inspector PRIVATE sansane_tokenizer)
target_compile_features(checkpoint_inspector PRIVATE cxx_std_23)
set_target_properties(checkpoint_inspector PROPERTIES
    CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(sansane_tokenizer PRIVATE /W4)
    target_compile_options(checkpoint_inspector PRIVATE /W4)
else()
    target_compile_options(sansane_tokenizer PRIVATE -Wall -Wextra)
    target_compile_options(checkpoint_inspector PRIVATE -Wall -Wextra)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(tokenizer_tests
        tests/tokenizer_tests.cpp
    )
    target_link_libraries(tokenizer_tests PRIVATE sansane_tokenizer)
    target_compile_features(tokenizer_tests PRIVATE cxx_std_23)
    set_target_properties(tokenizer_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(tokenizer_tests PRIVATE /W4)
    else()
        target_compile_options(tokenizer_tests PRIVATE -Wall -Wextra)
    endif()

    add_test(
        NAME tokenizer_tests
        COMMAND tokenizer_tests
            ${CMAKE_CURRENT_SOURCE_DIR}/../llm.c-reference/gpt2_tokenizer.bin
    )
    set_tests_properties(tokenizer_tests PROPERTIES SKIP_RETURN_CODE 77)
endif()
