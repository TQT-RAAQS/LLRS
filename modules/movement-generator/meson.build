project(
    'move-generator',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

# sources
move_gen_src = ['main.cu']

# submodules
awg_proj = subproject('awg')
synthesiser_proj = subproject('synthesiser')
move_gen_server_handler_proj = subproject('server-handler')

# dependencies
awg_dep = awg_proj.get_variable('awg_dep')
synthesiser_dep = synthesiser_proj.get_variable('move_gen_synthesiser_dep')
move_gen_server_handler_dep = move_gen_server_handler_proj.get_variable(
    'move_gen_server_handler_dep',
)


# targets
move_gen = executable(
    'move-generator',
    move_gen_src,
    link_args: '-lstdc++fs',
    dependencies: [awg_dep, synthesiser_dep, move_gen_server_handler_dep],
)

