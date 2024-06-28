project(
    'LLRS-exe',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=shared'],
    subproject_dir: 'modules',
)

# sources
llrs_exe_src = ['main.cu']

# submodules
awg_proj = subproject('awg')
llrs_lib_proj = subproject('llrs-lib')

# external dependencies
awg_dep = awg_proj.get_variable('awg_dep')
llrs_lib_dep = llrs_lib_proj.get_variable('llrs_lib_dep')

# targets
llrs_exe = executable(
    'llrs-exe',
    llrs_exe_src,
    link_args: '-lstdc++fs',
    dependencies: [awg_dep, llrs_lib_dep],
    install: true,
)
