project(
    'configs-translator',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

configs_translator_inc = ['include']

# sources
configs_translator_src = [
    'src/configs-translator.cu',
]

configs_translator_dep = []

# targets
configs_translator_lib = library(
    'configs-traslator',
    configs_translator_src,
    include_directories: configs_translator_inc,
    dependencies: configs_translator_dep,
)

# dependencies
configs_translator_dep = declare_dependency(
    include_directories: configs_translator_inc,
    link_with: configs_translator_lib,
    dependencies: configs_translator_dep,
)
