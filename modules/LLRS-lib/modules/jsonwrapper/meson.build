project('llrs-lib-jsonwrapper', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=static'])


# external dependencies
yaml_cpp_proj = cmake.subproject('yaml-cpp')
yaml_cpp_dep = yaml_cpp_proj.dependency('yaml-cpp')

# include directories
llrs_lib_jsonwrapper_inc = include_directories('include')

# sources
llrs_lib_jsonwrapper_src = [
    'src/JsonWrapper.cu',
  ]

# targets
llrs_lib_jsonwrapper = library('llrs-lib-jsonwrapper', 
    llrs_lib_jsonwrapper_src,
    include_directories: [llrs_lib_common_inc, llrs_lib_jsonwrapper_inc],
    dependencies: [jsoncpp_dep, nlohmann_json_dep, yaml_cpp_dep])

# dependencies
llrs_lib_jsonwrapper_dep = declare_dependency(
    include_directories: [llrs_lib_common_inc, llrs_lib_jsonwrapper_inc],
    link_with: llrs_lib_jsonwrapper,
    dependencies: [jsoncpp_dep, nlohmann_json_dep, yaml_cpp_dep])

