project('fgc', 
        'cuda',
        default_options: ['cpp_std=c++14', 'default_library=shared'])

# subdirectories
subdir('ActiveSDKv01.08.02')

# include directories
fgc_inc = include_directories('include')

# sources
fgc_src = [
    'src/activesilicon-1xcld.cu',
    'src/buffer.cu',
    'src/platform.cu',
    'src/trace.cu'
  ]

# targets
fgc = library('fgc', 
    fgc_src,
    include_directories: fgc_inc,
    dependencies: [activeSDK_dep])

# dependencies
fgc_dep = declare_dependency(
    include_directories: fgc_inc,
    link_with: fgc,
    dependencies: [activeSDK_dep])
    