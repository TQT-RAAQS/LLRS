project('linear-exact-v1', 
        'cuda')


# include directories
linear_exact_v1_inc = include_directories('include')

# sources
linear_exact_v1_src = [
    'src/lin_exact_1d.cu',
  ]

# targets
linear_exact_v1 = library('linear-exact-v1', 
    linear_exact_v1_src,
    include_directories: [linear_exact_v1_inc],
    dependencies: [])
