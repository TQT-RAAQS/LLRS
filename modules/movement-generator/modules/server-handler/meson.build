project('move-gen-server-handler', 
        'cpp', 
        default_options: ['cpp_std=c++14', 'default_library=static'],
        subproject_dir: 'modules')

# include directories
move_gen_server_handler_inc = include_directories('include')

# sources
move_gen_server_handler_src = [
    'src/handler.cpp',
  ]

# submodules
zmq_server_proj = subproject('zmq-server')

# dependencies
zmq_server_dep = zmq_server_proj.get_variable('zmq_server_dep')

# targets
move_gen_server_handler = library('move-gen-server_handler', 
    move_gen_server_handler_src,
    include_directories: [move_gen_server_handler_inc],
    dependencies: [zmq_server_dep])

# dependencies
move_gen_server_handler_dep = declare_dependency(
    include_directories: [move_gen_server_handler_inc],
    link_with: move_gen_server_handler,
    dependencies: [zmq_server_dep])

