project('runtime-benchmarking', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=shared'])

# sources
run_bench_src = [
    'src/main.cu'
  ]

# submodules
llrs_lib_proj =   subproject('llrs-lib')
awg_proj = subproject('awg')

# dependencies
llrs_lib_dep = llrs_lib_proj.get_variable('llrs_lib_pre_solved_dep')
awg_dep = awg_proj.get_variable('awg_dep')

# targets
run_bench = executable('run-bench',
    run_bench_src,
    cuda_args: ['-D PRE_SOLVED', '-D LOGGING_RUNTIME'],
    dependencies: [llrs_lib_dep, awg_dep])
