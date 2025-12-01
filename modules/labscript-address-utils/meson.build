project(
    'labscript-address-utils',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

# includes
labscript_address_utils_include = 'include'

# sources
labscript_address_utils_sources = ['src/labscript-address-utils.cu']

dependencies = []

# library
labscript_address_utils_lib = library(
    'labscript_address_utils_lib',
    labscript_address_utils_sources, # sources files
    include_directories: labscript_address_utils_include,
    dependencies: dependencies,
)

# dependency
labscript_address_utils_dep = declare_dependency(
    include_directories: labscript_address_utils_include,
    link_with: labscript_address_utils_lib,
    dependencies: dependencies,
)