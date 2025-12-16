project(
    'LLRS',
    'cpp',
    'cuda',
    'c',
    default_options: ['cpp_std=c++14', 'default_library=shared'],
    subproject_dir: 'modules',
)

fs = import('fs')
root = meson.project_source_root()

root_parts = root.split('/')
shared_root = '/' + join_paths(root_parts[1], root_parts[2], 'Z')

add_global_arguments(
    '-DPROJECT_BASE_DIR="' + root + '"',
    '-DSHARED_DRIVE_DIR="' + shared_root + '"',
    '-I' + meson.project_source_root() + '/modules/common-include',
    '-D LOGGING_VERBOSE',
    '-O3',
    language: ['c', 'cpp', 'cuda'],
)

add_global_arguments('-arch=sm_75', '-gencode=arch=compute_75,code=sm_75', '-Xcompiler=-fopenmp', language: 'cuda')

c_compiler = meson.get_compiler('c')
compiler_id = c_compiler.get_id()
compiler_version = c_compiler.version()
message('Using compiler: ', compiler_id, ' version ', compiler_version)

add_global_link_arguments('-lgomp', language: 'cuda')
# subprojects
labscript_address_utils_proj = subproject('labscript-address-utils')
pid_loop_controller_proj = subproject('pid-loop-controller')
timed_hash_queue_proj = subproject('timed-hash-queue')
microwave_handler_proj = subproject('microwave-handler')
configs_translator_proj = subproject('configs-translator')
fgc_proj = subproject('fgc')
awg_proj = subproject('awg')
llrs_lib_proj = subproject('llrs-lib')
hdf_wrapper = subproject('hdf5-wrapper')
llcs_proj = subproject('llcs')
llrs_exe_proj = subproject('llrs-exe')
trap_result_saver_proj = subproject('trap-result-saver')
ramsey_stabilizer_proj = subproject('ramsey-stabilizer')

operational_benchmarking_proj = subproject('operational-benchmarking')
runtime_benchmarking_proj = subproject('runtime-benchmarking')
image_acq_proj = subproject('image-acq-benchmarking')
image_processing_proj = subproject('image-processing-benchmarking')
wfm_streaming_bench_proj = subproject('wfm-streaming-benchmarking')
waveform_loader = subproject('waveform-loader')

movement_generator = subproject('movement-generator')
image_saver = subproject('image-saver')

shared_memory_handler = subproject('shared-memory-handler')
