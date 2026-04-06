project(
    'qdac-client',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

cmake = import('cmake')
##########################

# submodules
cppzmq_proj = cmake.subproject('cppzmq')

# external dependencies
cppzmq_dep = cppzmq_proj.dependency('cppzmq')
libzmq_dep = dependency('libzmq', required: true)

yaml_cpp_proj = cmake.subproject('yaml-cpp')
yaml_cpp_dep = yaml_cpp_proj.dependency('yaml-cpp')

dependencies = [
    cppzmq_dep,
    libzmq_dep,
    yaml_cpp_dep
]

##########################

includes = ['include']
sources = [
    'src/qdac-client.cu',
]

##########################

qdac_client_lib = library(
    'qdac-client-lib',
    sources,
    include_directories: includes,
    dependencies: dependencies,
)

# dependency
qdac_client_dep = declare_dependency(
    include_directories: includes,
    link_with: qdac_client_lib,
    dependencies: dependencies,
)

##########################

test_executable = executable(
    'main',
    ['main.cu'] + sources,
    link_args: '-lstdc++fs',
    dependencies: dependencies,
    include_directories: includes
)