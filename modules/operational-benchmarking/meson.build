project('operational-benchmarking', 'cuda', default_options: ['cpp_std=c++14'])

# include directories
op_bench_inc = include_directories('include')

# sources
op_bench_atom_src = ['src/atom.cu']
op_bench_trap_array_src = ['src/trap-array.cu']
op_bench_src = ['src/main.cu']
op_bench_runtime_src = ['src/runtime.cu']
op_bench_operations_src = ['src/operations.cu']
op_bench_loss_src = ['src/loss.cu']

# submodules
solver_proj = subproject('solver')
utility_proj = subproject('utility')
collector_proj = subproject('collector')

# dependencies
solver_dep = solver_proj.get_variable('llrs_lib_solver_so_dep')
solver_runtime_dep = solver_proj.get_variable('llrs_lib_solver_runtime_dep')
utility_dep = utility_proj.get_variable('llrs_lib_utility_dep')
collector_dep = collector_proj.get_variable('llrs_lib_collector_activate_dep')

# targets
op_bench_atom = library(
    'op-bench-atom',
    op_bench_atom_src,
    include_directories: [op_bench_inc],
)
op_bench_trap_array = library(
    'op-bench-trap-array',
    op_bench_trap_array_src,
    link_with: [op_bench_atom],
    include_directories: [op_bench_inc],
    dependencies: [solver_dep],
)
op_bench_exe = executable(
    'op-bench',
    op_bench_src,
    link_with: [op_bench_trap_array],
    include_directories: [op_bench_inc],
    dependencies: [solver_dep, utility_dep],
)

op_bench_runtime_exe = executable(
    'op-bench-runtime',
    op_bench_runtime_src,
    include_directories: [op_bench_inc],
    dependencies: [solver_runtime_dep, utility_dep, collector_dep],
)

op_bench_operations_exe = executable(
    'op-bench-operations',
    op_bench_operations_src,
    include_directories: [op_bench_inc],
    link_with: [op_bench_trap_array],
    dependencies: [solver_runtime_dep, utility_dep],
)

op_bench_loss_exe = executable(
    'op-bench-loss',
    op_bench_loss_src,
    link_with: [op_bench_trap_array],
    include_directories: [op_bench_inc],
    dependencies: [solver_runtime_dep, utility_dep],
)

