project('RedRec-V2', 
        'cuda')

# include directories
redrec_v2_inc = include_directories('include')

# sources
redrec_v2_src = [
    'src/main.cu'
  ]

# targets
redrec_v2 = library('redrec-v2', 
    redrec_v2_src,
    include_directories: [redrec_v2_inc])

# export dependencies
redrec_v2_dep = declare_dependency(
    include_directories: redrec_v2_inc,
    link_with: redrec_v2 
)