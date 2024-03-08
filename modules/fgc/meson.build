project('fgc', 
        'cpp',
        default_options: ['cpp_std=c++14', 'default_library=shared'])

# include directories
fgc_inc = include_directories('include', 'include/ActiveSDKv01.08.02')

# sources
fgc_src = [
    'src/activesilicon-1xcld.cpp',
    'src/buffer.cpp',
    'src/platform.cpp',
    'src/trace.cpp'
  ]

# targets
fgc = library('fgc', 
    fgc_src,
    include_directories: fgc_inc,
    dependencies: [])
