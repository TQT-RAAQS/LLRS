project('experiment-hdf5-wrapper', 
        'cpp', 
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

# sources
h5_wrapper_src = [
    'src/shot_file.cpp',
    'src/globals_config.cpp'
  ]

# includes
experiment_hdf5_wrapper_inc = 'include'

# external dependencies
hdf5_dependency = dependency('hdf5', language: 'cpp')
boost_dependency = dependency('boost')

# targets
experiment_hdf5_wrapper_lib = library('experiment_hdf5_wrapper_lib', 
    h5_wrapper_src,
    include_directories: experiment_hdf5_wrapper_inc,
    dependencies: [hdf5_dependency, boost_dependency])

# dependencies
experiment_hdf5_wrapper_dep = declare_dependency(
    include_directories: experiment_hdf5_wrapper_inc,
    link_with: experiment_hdf5_wrapper_lib,
    dependencies: [hdf5_dependency, boost_dependency])
    
