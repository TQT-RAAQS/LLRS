project('move-generator', 
        'cuda', 
        default_options: ['cpp_std=c++14', 'default_library=static'],
        subproject_dir: 'modules')

# sources
move_gen_src = [
    'main.cu',
  ]

# submodules
awg_proj = subproject('awg')
synthesiser_proj = subproject('synthesiser')
server_proj = subproject('server')

# dependencies
awg_dep = awg_proj.get_variable('awg_dep')
synthesiser_dep = synthesiser_proj.get_variable('move_gen_synthesiser_dep')
server_dep = server_proj.get_variable('llcs_server_dep')


# targets
move_gen = executable('move-generator', 
    move_gen_src,
    dependencies: [awg_dep, synthesiser_dep, server_dep])

