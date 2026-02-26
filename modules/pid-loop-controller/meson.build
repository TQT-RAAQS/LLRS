project(
    'pid-loop-controller',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)
cmake = import('cmake')

pid_loop_controller_include = ['include']

yaml_cpp = cmake.subproject('yaml-cpp')
yaml_cpp_dep = yaml_cpp.dependency('yaml-cpp')

config_translator = subproject('configs-translator')
configs_translator_dep = config_translator.get_variable('configs_translator_dep')

dependencies = [yaml_cpp_dep, configs_translator_dep]

pid_loop_controller_sources = [
    'src/pid-loop-controller.cu',
    'src/pid-loop-phase-controller.cu',
    'src/linear-controller.cu',
]

pid_loop_controller_lib = library(
    'pid_loop_controller_lib',
    pid_loop_controller_sources, # sources files
    include_directories: pid_loop_controller_include,
    dependencies: dependencies,
)

# dependency
pid_loop_controller_dep = declare_dependency(
    include_directories: pid_loop_controller_include,
    link_with: pid_loop_controller_lib,
    dependencies: dependencies,
)