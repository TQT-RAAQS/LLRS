project('Linear-Exact-GPU', 
        'cuda')

# include directories
linear_exact_gpu_inc = include_directories('include')

# sources
linear_exact_gpu_src = [
    'src/gpufun.cu',
    'src/kernel.cu'
  ]

# targets
linear_exact_gpu = shared_library('linear-exact-gpu', 
    linear_exact_gpu_src,
    include_directories: [linear_exact_gpu_inc])

# export dependencies
linear_exact_gpu_dep = declare_dependency(
    include_directories: linear_exact_gpu_inc,
    link_with: linear_exact_gpu
)