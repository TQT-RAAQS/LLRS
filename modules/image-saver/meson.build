project(
    'image-saver',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

cmake = import('cmake')

# includes
image_saver_include_dir = 'include'

# sources
image_saver_main = ['main.cu']
image_saver_sources = ['src/image-saver-server.cu']

# submodules
fgc = subproject('fgc')
hdf5 = subproject('hdf5-wrapper')
zmq_server = subproject('zmq-server')
yaml_cpp = cmake.subproject('yaml-cpp')


# dependencies
fgc_dep = fgc.get_variable('fgc_dep')
hdf5_dep = hdf5.get_variable('hdf5_wrapper_dep')
zmq_server_dep = zmq_server.get_variable('zmq_server_dep')
yaml_cpp_dep = yaml_cpp.dependency('yaml-cpp')
boost_dep = dependency('boost', modules: ['filesystem', 'system'])
opencv_dep = dependency('opencv', required: true)

dependencies = [fgc_dep, hdf5_dep, zmq_server_dep, yaml_cpp_dep, boost_dep, opencv_dep, opencv_dep]

# targets
image_saver = executable(
    'image-saver',
    image_saver_main + image_saver_sources,
    link_args: '-lstdc++fs',
    dependencies: dependencies,
    include_directories: image_saver_include_dir
)

# dependency variable
image_saver_lib = library(
    'image_saver_lib',
    image_saver_sources, # sources files
    include_directories: image_saver_include_dir,
    dependencies: dependencies,
)
image_saver_dep = declare_dependency(
    include_directories: image_saver_include_dir,
    link_with: image_saver_lib,
    dependencies: dependencies,
)