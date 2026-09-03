project(
    'hdf5-wrapper',
    'cpp',
    default_options: ['cpp_std=c++14', 'default_library=shared'],
    subproject_dir: 'modules',
)

# Backward compatibility option for HDF5 type selection
hdf5_use_h5location = get_option('hdf5_use_h5location')

# sources
h5_wrapper_src = ['src/shot-file.cpp', 'src/globals-config.cpp']

# includes
hdf5_wrapper_inc = 'include'

# external dependencies
hdf5_dependency = dependency('hdf5', language: 'cpp')
boost_dependency = dependency('boost')

# Compile flags for backward compatibility
h5_compile_args = []
if hdf5_use_h5location
    h5_compile_args += ['-DHDF5_USE_H5LOCATION']
endif

# targets
hdf5_wrapper_lib = library(
    'hdf5_wrapper_lib',
    h5_wrapper_src,
    include_directories: hdf5_wrapper_inc,
    dependencies: [hdf5_dependency, boost_dependency],
    cpp_args: h5_compile_args,
)

# dependencies
hdf5_wrapper_dep = declare_dependency(
    include_directories: hdf5_wrapper_inc,
    link_with: hdf5_wrapper_lib,
    dependencies: [hdf5_dependency, boost_dependency],
)