project('llcs-trigger-detector', 
        'cpp'
        default_options: ['default_library=static'])

# include directories
llcs_trigger_detector_inc =  include_directories('include')

# sources
llcs_trigger_detector_src = [
    'src/trigger-detector.cpp'
  ]

# submodules
awg_proj = subproject('awg')

# external dependencies
awg_dep = awg_proj.get_variable('awg_dep')

# targets
llcs_trigger_detector = library('llcs-trigger-detector', 
    llcs_trigger_detector_src,
    include_directories: [llcs_trigger_detector_inc],
    dependencies: [awg_dep])

# export dependencies
llcs_trigger_detector_dep = declare_dependency(
    include_directories: [llcs_trigger_detector_inc],
    link_with: llcs_trigger_detector,
    dependencies: [awg_dep])

