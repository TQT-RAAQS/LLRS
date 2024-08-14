project(
    'llrs-lib-reconfiguration',
    'cuda',
    default_options: ['default_library=shared'],
    subproject_dir: 'algorithms',
)

# include directory
llrs_lib_reconfiguration_inc = include_directories('include')

# provided libraries
reconfiguration_provided_lib = meson.current_source_dir() / 'lib' / '*'

# export dependencies
llrs_lib_reconfiguration_dep = declare_dependency(
    include_directories: llrs_lib_reconfiguration_inc,
    link_args: [reconfiguration_provided_lib],
)
