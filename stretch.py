"""
    MSSV     : N20DCCN083
    Họ và tên: Nguyễn Thái Trưởng
"""
import numpy as np

def stretch(x):
    xMax = np.max(x)
    xMin = np.min(x)
    scaleFactor = 255.0 / (xMax - xMin)
    y = np.round((x - xMin) * scaleFactor)
    return y.astype('uint8')
