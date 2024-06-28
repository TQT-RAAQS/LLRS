project('llrs-lib-image-acquisition', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=static'])

# include directories
llrs_lib_image_acquisition_inc = include_directories('include')

# sources
llrs_lib_image_acquisition_src = [
    'src/acquisition.cu',
    'src/acquisition-stored.cu'
  ]


# submodules
collector_proj =          subproject('collector')
fgc_proj =                subproject('fgc')

# dependencies
llrs_lib_collector_dep = collector_proj.get_variable('llrs_lib_collector_dep')
llrs_lib_collector_activate_dep = collector_proj.get_variable('llrs_lib_collector_activate_dep')
fgc_dep = fgc_proj.get_variable('fgc_dep')

# targets
llrs_lib_image_acquisition = library('llrs-lib-image-acquisition', 
    llrs_lib_image_acquisition_src,
    include_directories: [llrs_lib_image_acquisition_inc],
    dependencies: [fgc_dep, llrs_lib_collector_dep])

llrs_lib_image_acquisition_runtime = library('llrs-lib-image-acquisition-runtime', 
    llrs_lib_image_acquisition_src,
    cuda_args: ['-D LOGGING_RUNTIME'],
    include_directories: [llrs_lib_image_acquisition_inc],
    dependencies: [fgc_dep, llrs_lib_collector_dep])


# dependencies
llrs_lib_image_acquisition_dep = declare_dependency(
    include_directories: [llrs_lib_image_acquisition_inc],
    link_with: llrs_lib_image_acquisition,
    dependencies: [fgc_dep, llrs_lib_collector_dep])

llrs_lib_image_acquisition_runtime_dep = declare_dependency(
    include_directories: [llrs_lib_image_acquisition_inc],
    link_with: llrs_lib_image_acquisition_runtime,
    dependencies: [fgc_dep, llrs_lib_collector_dep])

   