project('llrs-lib-image-processing', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=static'],
        subproject_dir: 'modules')


# include directories
llrs_lib_image_processing_inc = include_directories('include')

# sources
llrs_lib_image_processing_src = [
    'src/ImageProcessor.cu',
  ]

# targets
llrs_lib_image_processing = library('llrs-lib-image-processing', 
    llrs_lib_image_processing_src,
    include_directories: [llrs_lib_common_inc, llrs_lib_image_processing_inc],
    dependencies: [])

# dependencies
llrs_lib_image_processing_dep = declare_dependency(
    include_directories: [llrs_lib_common_inc, llrs_lib_image_processing_inc],
    link_with: llrs_lib_image_processing,
    dependencies: [])
