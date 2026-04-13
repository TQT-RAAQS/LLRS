project(
    'string-math-parser',
    'cuda',
    default_options: ['cpp_std=c++14', 'default_library=static'],
    subproject_dir: 'modules',
)

ramsey_stabilizer_includes = ['include']

# submodules

# dependencies

string_math_parser = [
]

# sources
string_math_parser_sources = [
    'src/string-math-parser.cu',
]

# executable
string_math_parser_executable = executable(
    'string-math-parser',
    ['test.cu'] + string_math_parser_sources,
    link_args: '-lstdc++fs',
    dependencies: string_math_parser,
    include_directories: ramsey_stabilizer_includes
)