project('runtime-benchmarking', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=shared'])

# sources
rn_bench_src = [
    'src/main.cu'
  ]

# submodules
llrs_lib_proj =   subproject('llrs-lib')
awg_proj = subproject('awg')

# dependencies
llrs_lib_dep = llrs_lib_proj.get_variable('llrs_lib_dep')
awg_dep = awg_proj.get_variable('awg_dep')

# targets
rn_bench = executable('rn-bench',
    run_bench_src,
    cuda_args: '-D PRE_SOLVED',
    dependencies: [llrs_lib_dep, awg_dep])
