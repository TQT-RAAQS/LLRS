project('move-gen-synthesiser', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=static'],
        subproject_dir: 'modules')

# include directories
move_gen_synthesiser_inc = include_directories('include')

# sources
move_gen_synthesiser_src = [
    'src/synthesiser.cu',
  ]

# submodules
waveform_synthesis_proj = subproject('waveform-synthesis')
setup_proj = subproject('setup')
awg_proj = subproject('awg')
hdf_wrapper_proj = subproject('hdf5-wrapper')

# dependencies
llrs_lib_waveform_synthesis_dep = waveform_synthesis_proj.get_variable('llrs_lib_waveform_synthesis_dep')
awg_dep = awg_proj.get_variable('awg_dep')
llrs_lib_setup_dep = setup_proj.get_variable('llrs_lib_setup_dep')
hdf5_wrapper_dep = hdf_wrapper_proj.get_variable('hdf5_wrapper_dep')

# targets
move_gen_synthesiser = library('move-gen-synthesiser', 
    move_gen_synthesiser_src,
    include_directories: [move_gen_synthesiser_inc],
    dependencies: [llrs_lib_waveform_synthesis_dep, awg_dep, llrs_lib_setup_dep, hdf5_wrapper_dep])

# dependencies
move_gen_synthesiser_dep = declare_dependency(
    include_directories: [move_gen_synthesiser_inc],
    link_with: move_gen_synthesiser,
    dependencies: [llrs_lib_waveform_synthesis_dep, awg_dep, llrs_lib_setup_dep, hdf5_wrapper_dep])

