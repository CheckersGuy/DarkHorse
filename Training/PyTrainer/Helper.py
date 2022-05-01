import ctypes
import pathlib
import numpy as np
import torch

libname = pathlib.Path().absolute().__str__() + "/libpyhelper.so"
c_lib = ctypes.CDLL(libname)


def print_fen_string(fen_string):
    c_lib.print_fen(ctypes.c_char_p(fen_string.encode("utf-8")))

def create_input(fen_string):
    output = np.zeros(128, dtype=np.float32)
    data_p = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_lib.get_input_from_fen(data_p, ctypes.c_char_p(fen_string.encode("utf-8")))
    out_hat = torch.Tensor(output)
    out = out_hat.view(4, 8, 4)
    return out
