project('LLRS', 
        'cpp', 'cuda', 'c',
        default_options: ['cpp_std=c++14', 'default_library=shared'],
        subproject_dir: 'modules')

add_global_arguments('-DPROJECT_BASE_DIR="' + meson.source_root() + '"' , language : ['c', 'cpp', 'cuda'])
add_global_arguments('-I' + meson.source_root() + '/modules/common-include' , language : ['c', 'cpp', 'cuda'])
add_global_arguments('-D LOGGING_VERBOSE', language : ['c', 'cpp', 'cuda'])

# subprojects
fgc_proj = subproject('fgc')
awg_proj = subproject('awg')
llrs_lib_proj = subproject('llrs-lib')
llcs_proj = subproject('llcs')
llrs_exe_proj = subproject('llrs-exe')
operational_benchmarking_proj = subproject('operational-benchmarking')
runtime_benchmarking_proj = subproject('runtime-benchmarking')
image_acq_proj = subproject('image-acq-benchmarking')
wfm_streaming_bench_proj = subproject('wfm-streaming-benchmarking')
waveform_loader = subproject('waveform-loader')
hdf_wrapper = subproject('hdf5-wrapper')
