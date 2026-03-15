project(
    'feedback-controller',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)
cmake = import('cmake')

feedback_controller_include = ['include']

yaml_cpp = cmake.subproject('yaml-cpp')
yaml_cpp_dep = yaml_cpp.dependency('yaml-cpp')

config_translator = subproject('configs-translator')
configs_translator_dep = config_translator.get_variable('configs_translator_dep')

dependencies = [yaml_cpp_dep, configs_translator_dep]

feedback_controller_sources = [
    'src/pid-loop-controller.cu',
    'src/pid-loop-phase-controller.cu',
    'src/linear-controller.cu',
]

feedback_controller_lib = library(
    'feedback_controller_lib',
    feedback_controller_sources, # sources files
    include_directories: feedback_controller_include,
    dependencies: dependencies,
)

# dependency
feedback_controller_dep = declare_dependency(
    include_directories: feedback_controller_include,
    link_with: feedback_controller_lib,
    dependencies: dependencies,
)