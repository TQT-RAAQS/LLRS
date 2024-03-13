project('LLRS', 
        'cpp', 'cuda', 'c',
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

add_global_arguments('-DPROJECT_BASE_DIR="' + meson.source_root() + '"' , language : ['c', 'cpp', 'cuda'])
add_global_arguments('-I' + meson.source_root() + '/modules/common-include' , language : ['c', 'cpp', 'cuda'])

# subprojects
fgc_proj = subproject('fgc')
awg_proj = subproject('awg')
llrs_lib_proj = subproject('llrs-lib')
llrs_exe_proj = subproject('llrs-exe')
llcs_proj = subproject('llcs')
