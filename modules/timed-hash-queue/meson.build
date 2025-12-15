project(
    'timed-hash-queue',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

includes = ['include']

dependencies = []

# sources
sources = [
    'src/timed-hash-queue.cu',
]

# dependencies

lib = library(
    'timed-hash-queue-lib',
    sources,
    include_directories: includes,
    dependencies: dependencies,
)

timed_hash_queue_dependency = declare_dependency(
    include_directories: [includes],
    link_with: lib,
    dependencies: dependencies,
)

