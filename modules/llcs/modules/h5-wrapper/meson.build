project('h5-wrapper', 'cpp', default_options
        : ['default_library=static'])

    cmake = import('cmake')

#include directories
        h5_wrapper_inc = include_directories('include')

#sources
            h5_wrapper_src = ['src/h5-wrapper.cpp']

#submodules
    yaml_cpp_proj = cmake.subproject(
        'yaml-cpp') hdf5_proj = cmake.subproject('hdf5')

#external dependencies
                                    yaml_cpp_dep = yaml_cpp_proj.dependency(
        'yaml-cpp') hdf5_dep = hdf5_proj.dependency('hdf5-shared')
                                   nlohmann_json_dep = dependency(
        'nlohmann_json') jsoncpp_dep = dependency('jsoncpp')

#targets
        h5_wrapper = library(
            'h5-wrapper', h5_wrapper_src, include_directories
            : [h5_wrapper_inc], dependencies
            : [ jsoncpp_dep, nlohmann_json_dep, yaml_cpp_dep, hdf5_dep ])

#export dependencies
            h5_wrapper_dep = declare_dependency(
                include_directories
                : [h5_wrapper_inc], link_with
                : h5_wrapper, dependencies
                : [ jsoncpp_dep, nlohmann_json_dep, yaml_cpp_dep, hdf5_dep ])
