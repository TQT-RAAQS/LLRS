project('linear-exact-v2', 
        'cuda')


# include directories
linear_exact_v2_inc = include_directories('include')

# sources
linear_exact_v2_src = [
    'src/lin_exact_1d_cpu_v2.cu',
  ]

# targets
linear_exact_v2 = library('linear-exact-v2', 
    linear_exact_v2_src,
    include_directories: [linear_exact_v2_inc],
    dependencies: [])

# dependencies
linear_exact_v2_dep = declare_dependency(
    include_directories: [linear_exact_v2_inc],
    link_with: linear_exact_v2)
