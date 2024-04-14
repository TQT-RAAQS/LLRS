import ctypes
from ctypes import *
from array import array
import pathlib

def execute_wrapper(algorithm, initial, target, Nt_x, Nt_y, solver_wrapper_so_file):
    init_arr = array('i', initial)
    init_ptr = cast(init_arr.buffer_info()[0], POINTER(c_int))

    targ_arr = array('i', target)
    targ_ptr = cast(targ_arr.buffer_info()[0], POINTER(c_int))
    dll = CDLL(solver_wrapper_so_file)  # Replace with the path to your SO file

    op_size = 4 # type, index, offset, block_size
    result = (c_int * (op_size * Nt_x * Nt_y * Nt_x * Nt_y))()
    sol_len = c_int(0)
    func = dll.solver_wrapper
    func.argtypes = [c_char_p, c_int, c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)]
    func(algorithm.encode(), Nt_x, Nt_y, init_ptr, targ_ptr, result, byref(sol_len))
    return [result[i] for i in range(sol_len.value*op_size)]


