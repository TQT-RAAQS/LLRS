project(
    'shared-memory-handler',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

cmake = import('cmake')

# includes
shared_memory_handler_include_dir = 'include'

# sources
shared_memory_handler_main = ['main.cu']
shared_memory_handler_main2 = ['main2.cu']
shared_memory_handler_sources = [
    'src/shared-memory-handler.cu', 
    'src/master-shared-memory-handler.cu', 
    'src/shared-memory.cu',
    'src/master-shared-memory-handler-server.cu',
]

# submodules
zmq_server = subproject('zmq-server')
yaml_cpp = cmake.subproject('yaml-cpp')


# dependencies
yaml_cpp_dep = yaml_cpp.dependency('yaml-cpp')
boost_dep = dependency('boost', modules: ['filesystem', 'system'])
zmq_server_dep = zmq_server.get_variable('zmq_server_dep')

dependencies = [yaml_cpp_dep, boost_dep, zmq_server_dep]

# targets
shared_memory_handler = executable(
    'shared-memory-handler',
    shared_memory_handler_main + shared_memory_handler_sources,
    link_args: '-lstdc++fs',
    dependencies: dependencies,
    include_directories: shared_memory_handler_include_dir
)

# dependency variable
shared_memory_handler_lib = library(
    'shared_memory_handler_lib',
    shared_memory_handler_sources, # sources files
    include_directories: shared_memory_handler_include_dir,
    dependencies: dependencies,
)

shared_memory_handler_dep = declare_dependency(
    include_directories: shared_memory_handler_include_dir,
    link_with: shared_memory_handler_lib,
    dependencies: dependencies,
)