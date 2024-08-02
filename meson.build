project(
    'LLRS',
    'cpp',
    'cuda',
    'c',
    default_options: ['cpp_std=c++14', 'default_library=shared'],
    subproject_dir: 'modules',
)

add_global_arguments(
    '-DPROJECT_BASE_DIR="' + meson.source_root() + '"',
    '-I' + meson.source_root() + '/modules/common-include',
    '-D LOGGING_VERBOSE',
    language: ['c', 'cpp', 'cuda'],
)

add_global_arguments('-arch=sm_75', '-gencode=arch=compute_75,code=sm_75', '-Xcompiler=-fopenmp', language: 'cuda')

add_global_link_arguments('-lgomp', language: 'cuda')
# subprojects
fgc_proj = subproject('fgc')
awg_proj = subproject('awg')
llrs_lib_proj = subproject('llrs-lib')
hdf_wrapper = subproject('hdf5-wrapper')
llcs_proj = subproject('llcs')
llrs_exe_proj = subproject('llrs-exe')

operational_benchmarking_proj = subproject('operational-benchmarking')
runtime_benchmarking_proj = subproject('runtime-benchmarking')
image_acq_proj = subproject('image-acq-benchmarking')
image_processing_proj = subproject('image-processing-benchmarking')
wfm_streaming_bench_proj = subproject('wfm-streaming-benchmarking')
waveform_loader = subproject('waveform-loader')

movement_generator = subproject('movement-generator')
