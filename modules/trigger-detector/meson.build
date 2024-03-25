project('trigger-detector', 
        'cpp',
        default_options: ['default_library=static'])

# include directories
trigger_detector_inc =  include_directories('include')

# sources
trigger_detector_src = [
    'src/trigger-detector.cpp'
  ]

# submodules
awg_proj = subproject('awg')

# external dependencies
awg_dep = awg_proj.get_variable('awg_dep')

# targets
trigger_detector = library('llcs-trigger-detector', 
    trigger_detector_src,
    include_directories: [trigger_detector_inc],
    dependencies: [awg_dep])

# export dependencies
trigger_detector_dep = declare_dependency(
    include_directories: [trigger_detector_inc],
    link_with: trigger_detector,
    dependencies: [awg_dep])

