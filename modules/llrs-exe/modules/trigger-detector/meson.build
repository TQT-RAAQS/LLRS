project('llrs-exe-trigger-detector', 
        'cpp',
        default_options: ['default_library=static'])

# include directories
llrs_exe_trigger_detector_inc =  include_directories('include')

# sources
llrs_exe_trigger_detector_src = [
    'src/trigger-detector.cpp'
  ]

# submodules
awg_proj = subproject('awg')

# external dependencies
awg_dep = awg_proj.get_variable('awg_dep')

# targets
llrs_exe_trigger_detector = library('llrs-exe-trigger-detector', 
    llrs_exe_trigger_detector_src,
    include_directories: [llrs_exe_trigger_detector_inc],
    dependencies: [awg_dep])

# export dependencies
llrs_exe_trigger_detector_dep = declare_dependency(
    include_directories: [llrs_exe_trigger_detector_inc],
    link_with: llrs_exe_trigger_detector,
    dependencies: [awg_dep])

