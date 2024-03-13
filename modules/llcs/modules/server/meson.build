project('llcs-server', 
        'cpp',
        default_options: ['default_library=static'])

cmake = import('cmake')

# include directories
llcs_server_inc =  include_directories('include')

# sources
llcs_server_src = [
    'src/server.cpp'
  ]

# submodules
hdf5_proj = cmake.subproject('hdf5')
cppzmq_proj = cmake.subproject('cppzmq')

# external dependencies
hdf5_dep = hdf5_proj.dependency('hdf5-shared')
nlohmann_json_dep = dependency('nlohmann_json')
cppzmq_dep = cppzmq_proj.dependency('cppzmq')

# targets
llcs_server = library('llcs-server', 
    llcs_server_src,
    include_directories: [llcs_server_inc],
    dependencies: [nlohmann_json_dep, hdf5_dep, cppzmq_dep])

# export dependencies
llcs_server_dep = declare_dependency(
    include_directories: [llcs_server_inc],
    link_with: llcs_server,
    dependencies: [nlohmann_json_dep, hdf5_dep, cppzmq_dep])
