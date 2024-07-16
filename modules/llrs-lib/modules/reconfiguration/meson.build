project(
    'llrs-lib-reconfiguration',
    'cuda',
    default_options: ['default_library=shared'],
    subproject_dir: 'algorithms',
)

add_project_arguments('-gencode arch=compute_75,code=sm_75' , language: 'cuda')

# include directory
llrs_lib_reconfiguration_inc = include_directories('include')

# subprojects
linear_exact_gpu_proj = subproject('gpu')
redrec_v2_proj = subproject('redrec-v2')
redrec_v2_dep = redrec_v2_proj.get_variable('redrec_v2_dep')
linear_exact_gpu_dep = linear_exact_gpu_proj.get_variable('linear_exact_gpu_dep')

# provided libraries
reconfiguration_provided_lib = meson.current_source_dir() / 'lib' / '*'

# export dependencies
llrs_lib_reconfiguration_dep = declare_dependency(
    include_directories: llrs_lib_reconfiguration_inc,
    link_args: [reconfiguration_provided_lib],
    dependencies: [linear_exact_gpu_dep, redrec_v2_dep]
)
