project('ImageProcessingBenchmark', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

# sources
image_processing_timing_src = [
    'main.cu'
  ]

# submodules
image_processing_proj = subproject('image-processing')

# external dependencies
image_processing_dep = image_processing_proj.get_variable('llrs_lib_image_processing_dep') 


# targets
filtering_test = executable('filtering-test',
    image_processing_null_src,
    link_args: '-lstdc++fs',
    dependencies: [image_processing_dep])

