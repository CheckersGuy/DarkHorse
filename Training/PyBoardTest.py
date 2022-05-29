import ctypes as ct
import pathlib
"""
Just will be the python implementation of the checker board
using the c++ implementation and ctypes
"""

libname = pathlib.Path().absolute().__str__() + "/libpyhelper.so"
c_lib = ct.CDLL(libname)

class Board:

    def __init__(self):


    def make_move(self, from_sq, to_sq, capture_list):
        pass
