project('redrec-v3', 
        'cuda')


# include directories
redrec_v3_inc = include_directories('include')

# sources
redrec_v3_src = [
    'src/redrec_cpu_v3.cu'
  ]

# subprojects
linear_exact_v2_proj = subproject('linear-exact-v2')

# dependencies
linear_exact_v2_dep =  linear_exact_v2_proj.get_variable('linear_exact_v2_dep')

# targets
redrec_v3 = library('redrec-v3', 
    redrec_v3_src,
    include_directories: [redrec_v3_inc],
    dependencies: [linear_exact_v2_dep])
