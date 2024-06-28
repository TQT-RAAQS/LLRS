project(
    'ImageAcquisitionBenchmark',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=shared'],
    subproject_dir: 'modules',
)

# sources
image_acq_src = ['main.cu']

# submodules
awg_proj = subproject('awg')
fgc_proj = subproject('fgc')

# external dependencies
awg_dep = awg_proj.get_variable('awg_dep')
fgc_dep = fgc_proj.get_variable('fgc_dep')

# targets
image_acq = executable(
    'img-acq-benchmark',
    image_acq_src,
    link_args: '-lstdc++fs',
    dependencies: [awg_dep, fgc_dep],
)
