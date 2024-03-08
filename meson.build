project('LLRS', 
        'cpp', 'cuda', 'c',
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

cmake = import('cmake')


# subprojects

# external 

yaml_cpp_proj = cmake.subproject('yaml-cpp')

# internal

fgc_proj = subproject('fgc')

# subprojects dependencies

yaml_cpp_dep = yaml_cpp_proj.dependency('yaml-cpp')
