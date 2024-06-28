project(
    'llrs-lib-utility',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
)

# include directories
llrs_lib_utility_inc = include_directories('include')

# sources
llrs_lib_utility_src = ['src/utility.cu']

# submodules
solver_proj = subproject('solver')

# dependencies
llrs_lib_solver_dep = solver_proj.get_variable('llrs_lib_solver_dep')

# targets
llrs_lib_utility = library(
    'llrs-lib-utility',
    llrs_lib_utility_src,
    include_directories: [llrs_lib_utility_inc],
    dependencies: [llrs_lib_solver_dep],
)

# dependencies
llrs_lib_utility_dep = declare_dependency(
    include_directories: [llrs_lib_utility_inc],
    link_with: llrs_lib_utility,
    dependencies: [llrs_lib_solver_dep],
)

