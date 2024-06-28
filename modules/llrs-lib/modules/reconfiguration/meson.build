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
linear_exact_v1_proj = subproject('linear-exact-v1')
linear_exact_v1 = linear_exact_v1_proj.get_variable('linear_exact_v1')
linear_exact_v2_proj = subproject('linear-exact-v2')
linear_exact_v2 = linear_exact_v2_proj.get_variable('linear_exact_v2')
redrec_v2_proj = subproject('redrec-v2')
redrec_v2 = redrec_v2_proj.get_variable('redrec_v2')
redrec_v3_proj = subproject('redrec-v3')
redrec_v3 = redrec_v3_proj.get_variable('redrec_v3')

# provided libraries
reconfiguration_provided_lib = meson.current_source_dir() / 'lib' / '*'

# export dependencies
llrs_lib_reconfiguration_dep = declare_dependency(
    include_directories: llrs_lib_reconfiguration_inc,
    link_args: reconfiguration_provided_lib,
)
