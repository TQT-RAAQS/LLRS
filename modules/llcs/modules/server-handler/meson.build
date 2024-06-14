project('llcs-server-handler', 
        'cpp', 
        default_options: ['cpp_std=c++14', 'default_library=static'],
        subproject_dir: 'modules')

# include directories
llcs_server_handler_inc = include_directories('include')

# sources
llcs_server_handler_src = [
    'src/handler.cpp',
  ]

# submodules
zmq_server_proj = subproject('zmq-server')

# dependencies
zmq_server_dep = zmq_server_proj.get_variable('zmq_server_dep')

# targets
llcs_server_handler = library('llcs-server_handler', 
    llcs_server_handler_src,
    include_directories: [llcs_server_handler_inc],
    dependencies: [zmq_server_dep])

# dependencies
llcs_server_handler_dep = declare_dependency(
    include_directories: [llcs_server_handler_inc],
    link_with: llcs_server_handler,
    dependencies: [zmq_server_dep])

