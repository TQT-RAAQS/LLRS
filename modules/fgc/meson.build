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

# external dependencies
cuda_dep = dependency('cuda', version: '>=11',modules: ['cufft', 'cudart'])
boost_dep =  dependency('boost', modules: ['system'])
gl_dep = dependency('gl')

# targets
fgc = library('fgc', 
    fgc_src,
    include_directories: fgc_inc,
    link_args: ['-lpng', '-lXext'],
    dependencies: [activeSDK_dep, cuda_dep, boost_dep, gl_dep])

# dependencies
fgc_dep = declare_dependency(
    include_directories: fgc_inc,
    link_with: fgc,
    link_args: ['-lpng', '-lXext'],
    dependencies: [activeSDK_dep, cuda_dep, boost_dep, gl_dep])
    
