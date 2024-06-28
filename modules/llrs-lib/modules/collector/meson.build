project('llrs-lib-collector', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=static'])

# include directories
llrs_lib_collector_inc = include_directories('include')

# external dependencies
jsoncpp_dep = dependency('jsoncpp')

# sources
llrs_lib_collector_src = [
    'src/Collector.cu',
  ]

# targets
llrs_lib_collector = library('llrs-lib-collector', 
    llrs_lib_collector_src,
    include_directories: [llrs_lib_collector_inc],
    dependencies: [jsoncpp_dep])

llrs_lib_collector_activate = library('llrs-lib-collector-activate', 
    llrs_lib_collector_src,
    cuda_args: ['-D LOGGING_RUNTIME'],
    include_directories: [llrs_lib_collector_inc],
    dependencies: [jsoncpp_dep])


# dependencies
llrs_lib_collector_dep = declare_dependency(
    include_directories: [llrs_lib_collector_inc],
    link_with: llrs_lib_collector,
    dependencies: [jsoncpp_dep])

llrs_lib_collector_activate_dep = declare_dependency(
    include_directories: [llrs_lib_collector_inc],
    link_with: llrs_lib_collector_activate,
    dependencies: [jsoncpp_dep])
