project('redrec-v2', 
        'cuda')


# include directories
redrec_v2_inc = include_directories('include')

# sources
redrec_v2_src = [
    'src/redrec_v2_no_timer.cu'
  ]

# targets
redrec_v2 = library('redrec-v2', 
    redrec_v2_src,
    include_directories: [redrec_v2_inc],
    dependencies: [])
