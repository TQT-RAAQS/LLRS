activeSDK_inc = include_directories('/usr/local/activesilicon/include')
activeSDK_lib = meson.current_source_dir() / 'lib' / '*'
activeSDK_dep = declare_dependency(
    link_args: [
        '-L/usr/local/activesilicon/lib64/',
        '-l:libphxapi-x86_64.so',
        '-l:libdvp.so',
    ],
    include_directories: activeSDK_inc,
)
