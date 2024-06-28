project('runtime-benchmarking', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=shared'])

# sources
run_bench_src = [
    'src/main.cu'
  ]

# submodules
llrs_lib_proj =   subproject('llrs-lib')

# dependencies
llrs_lib_dep = llrs_lib_proj.get_variable('llrs_lib_runtime_dep')

# targets
run_bench = executable('run-bench',
    run_bench_src,
    cuda_args: ['-D LOGGING_RUNTIME'],
    dependencies: [llrs_lib_dep])
