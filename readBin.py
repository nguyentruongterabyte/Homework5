"""
    MSSV     : N20DCCN083
    Họ và tên: Nguyễn Thái Trưởng
"""
import numpy as np

def read_bin(fn, xsize):
    fid = open(fn, 'rb')
    if fid == -1:
        raise Exception(f'Could not open {fn}')
    x = np.fromfile(fid, dtype='uint8', count=xsize * xsize)
    x = x.reshape((xsize, xsize))
    fid.close()
    return x
