# spcm provided library
spcm_inc = include_directories('include')
spcm_provided_lib = meson.current_source_dir() / 'lib' / '*'

# spcm library

spcm_src = [
    'src/common/spcm_lib_card.cpp',
    'src/common/spcm_lib_data.cpp',
    'src/common/spcm_cuda_common.cu',
    'src/common/ostools/spcm_ostools.cpp'
  ] 

spcm_lib = library('spcm_lib',
    spcm_src,
    include_directories: spcm_inc
)

# cuda dep
cuda_dep = dependency('cuda', modules: ['cufft', 'cuda'])


spcm_dep = declare_dependency(link_with: spcm_lib, link_args: spcm_provided_lib, include_directories: spcm_inc, dependencies: cuda_dep) 
