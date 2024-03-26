project('operational-benchmarking', 
        'cuda', 
        default_options: ['cpp_std=c++14'])

# include directories
op_bench_inc = include_directories('include')

# sources
op_bench_atom_src = [
    'src/atom.cu'
  ]
op_bench_trap_array_src = [
    'src/trap-array.cu'
  ]
op_bench_src = [
    'src/main.cu'
  ]

# submodules
solver_proj = subproject('solver')
jsonwrapper_proj = subproject('jsonwrapper')
utility_proj = subproject('utility') 

# dependencies
solver_dep = solver_proj.get_variable('llrs_lib_solver_dep')
utility_dep = utility_proj.get_variable('llrs_lib_utility_dep')
jsonwrapper_dep = jsonwrapper_proj.get_variable('llrs_lib_jsonwrapper_dep')

# targets
op_bench_atom = library('op-bench-atom',
    op_bench_atom_src,
    include_directories: [op_bench_inc])
op_bench_trap_array = library('op-bench-trap-array',
    op_bench_trap_array_src,
    link_with: [op_bench_atom],
    include_directories: [op_bench_inc],
    dependencies: [solver_dep])
op_bench_exe = executable('op-bench',
    op_bench_src,
    link_with: [op_bench_trap_array],
    include_directories: [op_benchmarking_inc],
    dependencies: [solver_dep, utility_dep, jsonwrapper_dep])
