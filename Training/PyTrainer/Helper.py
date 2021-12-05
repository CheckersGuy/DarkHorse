import ctypes
import pathlib
import numpy as np
import torch

libname = pathlib.Path().absolute() / "libpyhelper.so"
c_lib = ctypes.CDLL(libname)


def get_bucket_index(white_men, black_men, kings):
    return c_lib.get_bucket_index(ctypes.c_uint32(white_men), ctypes.c_uint32(black_men), ctypes.c_uint32(kings))


def patterns_op(white_men, black_men, kings, color):
    output = np.zeros(30, dtype=np.int64)
    data_p = output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    c_lib.patterns_op(ctypes.c_uint32(white_men), ctypes.c_uint32(black_men), ctypes.c_uint32(kings),
                      ctypes.c_int32(color),
                      data_p)
    return output


def create_input(white_men, black_men, kings, color):
    output = np.zeros(120, dtype=np.float32)
    data_p = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_lib.create_input(ctypes.c_uint32(white_men), ctypes.c_uint32(black_men), ctypes.c_uint32(kings),
                       ctypes.c_int32(color),
                       data_p)
    return output



def has_jumps(white_men, black_men, kings, color):
    return c_lib.has_jumps(ctypes.c_uint32(white_men), ctypes.c_uint32(black_men), ctypes.c_uint32(kings),
                           ctypes.c_int32(color))


def invert_pieces(pieces):
    return c_lib.invert_pieces(ctypes.c_uint32(pieces))


def num_pieces(white_men, black_men, kings):
    return c_lib.num_pieces(ctypes.c_uint32(white_men), ctypes.c_uint32(black_men), ctypes.c_uint32(kings))


def print_fen_string(color, white_men, black_men, kings):
    c_lib.print_fen(ctypes.c_int(color), ctypes.c_uint32(white_men), ctypes.c_uint32(black_men), ctypes.c_uint32(kings))


def print_position(white_men, black_men, kings):
    c_lib.print_position(ctypes.c_uint32(white_men), ctypes.c_uint32(black_men), ctypes.c_uint32(kings))


def num_black_kings(white_men, black_men, kings):
    return c_lib.count_black_kings(ctypes.c_uint32(white_men), ctypes.c_uint32(black_men), ctypes.c_uint32(kings))


def num_black_pawn(white_men, black_men, kings):
    return c_lib.count_black_pawn(ctypes.c_uint32(white_men), ctypes.c_uint32(black_men), ctypes.c_uint32(kings))


def num_white_pawn(white_men, black_men, kings):
    return c_lib.count_white_pawn(ctypes.c_uint32(white_men), ctypes.c_uint32(black_men), ctypes.c_uint32(kings))


def num_white_kings(white_men, black_men, kings):
    return c_lib.count_white_kings(ctypes.c_uint32(white_men), ctypes.c_uint32(black_men), ctypes.c_uint32(kings))
