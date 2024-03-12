project('LLRS-exe', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=shared'])

# sources
llrs_exe_src = [
    'src/main.cu'
  ]

# submodules
llrs_lib_proj =                subproject('LLRS-lib')

# dependencies
llrs_lib_dep = fgc_proj.get_variable('llrs_lib_dep')

# targets
llrs_exe = executable('llrs-exe',
    llrs_lib_src,
    dependencies: [llrs_lib_dep],
    install: true)
