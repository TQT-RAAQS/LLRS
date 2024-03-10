project('redrec-v3', 
        'cuda')


# include directories
redrec_v3_inc = include_directories('include')

# sources
redrec_v3_src = [
    'src/rec_cpu_v3.cu'
  ]

# targets
redrec_v3 = library('redrec-v3', 
    redrec_v3_src,
    include_directories: [redrec_v3_inc],
    dependencies: [])
