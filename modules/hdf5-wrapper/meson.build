project('hdf5-wrapper', 
        'cpp', 
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

# sources
h5_wrapper_src = [
    'main.cpp'
  ]

h5_wrapper_inc = include_directories('include')

# targets
h5 = executable(
  'main',
  h5_wrapper_src,
  include_directories: h5_wrapper_inc
)
