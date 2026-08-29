cmake_minimum_required(VERSION 3.20)

project(SansanE VERSION 0.1.0 LANGUAGES CXX)

add_executable(checkpoint_inspector
    src/main.cpp
)

target_compile_features(checkpoint_inspector PRIVATE cxx_std_23)
set_target_properties(checkpoint_inspector PROPERTIES
    CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(checkpoint_inspector PRIVATE /W4 /permissive-)
else()
    target_compile_options(checkpoint_inspector PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
    )
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME checkpoint_inspector_help
        COMMAND checkpoint_inspector --help
    )
    add_test(NAME checkpoint_inspector_requires_path
        COMMAND checkpoint_inspector
    )
    set_tests_properties(checkpoint_inspector_requires_path PROPERTIES
        WILL_FAIL TRUE
    )
endif()
