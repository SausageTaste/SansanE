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
