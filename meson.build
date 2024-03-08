project('LLRS', 
        'cpp', 'cuda', 'c',
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

add_global_arguments('-DPROJECT_BASE_DIR="' + meson.source_root() + '"' , language : ['c', 'cpp', 'cuda'])

# subprojects
fgc_proj = subproject('fgc')
awg_proj = subproject('awg')

