project('zmq-server', 
        'cpp',
        default_options: ['default_library=static'])

cmake = import('cmake')

# include directories
zmq_server_inc =  include_directories('include')

# sources
zmq_server_src = [
    'src/server.cpp'
  ]

# submodules
cppzmq_proj = cmake.subproject('cppzmq')

# external dependencies
hdf5_dep = dependency('hdf5', language: 'cpp')
nlohmann_json_dep = dependency('nlohmann_json')
cppzmq_dep = cppzmq_proj.dependency('cppzmq')

# targets
zmq_server = library('zmq-server', 
    zmq_server_src,
    include_directories: [zmq_server_inc],
    dependencies: [nlohmann_json_dep, hdf5_dep, cppzmq_dep])

# export dependencies
zmq_server_dep = declare_dependency(
    include_directories: [zmq_server_inc],
    link_with: zmq_server,
    dependencies: [nlohmann_json_dep, hdf5_dep, cppzmq_dep])
