project(
    'awg',
    'cuda',
    'cpp',
    default_options: ['cpp_std=c++14', 'default_library=shared'],
    subproject_dir: 'modules',
)

cmake = import('cmake')


# external dependencies
yaml_cpp_proj = cmake.subproject('yaml-cpp')
yaml_cpp_dep = yaml_cpp_proj.dependency('yaml-cpp')

# subdirectories
subdir('spcm')

# include directories
awg_inc = include_directories('include')

# sources
awg_src = ['src/awg.cpp']

# targets
awg = library(
    'awg',
    awg_src,
    include_directories: awg_inc,
    dependencies: [yaml_cpp_dep, spcm_dep],
)

# dependencies
awg_dep = declare_dependency(
    include_directories: awg_inc,
    link_with: awg,
    dependencies: [yaml_cpp_dep, spcm_dep],
)
