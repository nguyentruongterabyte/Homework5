"""
    MSSV     : N20DCCN083
    Họ và tên: Nguyễn Thái Trưởng
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift

from readBin import read_bin
from stretch import stretch

X = read_bin('salesman.bin', 256)

# P1(a)
X2 = np.zeros((262, 262))
X2[3:259, 3:259] = X
Y2 = np.zeros((262, 262))
for row in range(3, 259):
    for col in range(3, 259):
        Y2[row, col] = np.sum(X2[row-3:row+4, col-3:col+4]) / 49
Y = stretch(Y2[3:258, 3:258])
Y1a = Y

# P1(b)
H = np.zeros((128, 128))
H[61:69, 61:69] = 1 / 49

Padsize = 256 + 128 - 1
ZPX = np.zeros((Padsize, Padsize))
ZPX[:256, :256] = X

plt.figure(figsize=(12, 6))
plt.subplot(2, 4, 1)
plt.imshow(X, cmap='gray')
plt.title('Original Image', fontsize=12)
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(ZPX, cmap='gray')
plt.title('Zero Padded Image', fontsize=12)
plt.axis('off')
plt.savefig('MZPX1b.png', bbox_inches='tight')

ZPH = np.zeros((Padsize, Padsize))
ZPH[:128, :128] = H
plt.subplot(2, 4, 3)
plt.imshow(stretch(ZPH), cmap='gray')
plt.title('Zero Padded Impulse Resp', fontsize=12)
plt.axis('off')
plt.savefig('MZPH1b.png', bbox_inches='tight')

ZPXtilde = fft2(ZPX)
ZPHtilde = fft2(ZPH)

ZPXtildeDisplay = stretch(np.log1p(np.abs(fftshift(ZPXtilde))))
plt.subplot(2, 4, 4)
plt.imshow(ZPXtildeDisplay, cmap='gray')
plt.title('Log−magnitude spectrum of zero padded image', fontsize=6)
plt.axis('off')
plt.savefig('MZPXtilde1b.png', bbox_inches='tight')

ZPHtildeDisplay = stretch(np.log1p(np.abs(fftshift(ZPHtilde))))
plt.subplot(2, 4, 5)
plt.imshow(ZPHtildeDisplay, cmap='gray')
plt.title('Log-magnitude spectrum of zero padded H image', fontsize=6)
plt.axis('off')
plt.savefig('MZPHtilde1b.png', bbox_inches='tight')

ZPYtilde = ZPXtilde * ZPHtilde
ZPY = ifft2(ZPYtilde)
ZPYtildeDisplay = stretch(np.log1p(np.abs(fftshift(ZPYtilde))))
plt.subplot(2, 4, 6)
plt.imshow(ZPYtildeDisplay, cmap='gray')
plt.title('Log-magnitude spectrum of zero padded result', fontsize=6)
plt.axis('off')
plt.savefig('MZPYtilde1b.png', bbox_inches='tight')

plt.subplot(2, 4, 7)
plt.imshow(stretch(ZPY), cmap='gray')
plt.title('Zero Padded Result', fontsize=12)
plt.axis('off')
plt.savefig('MZPY1b.png', bbox_inches='tight')

Y = stretch(ZPY[64: 319, 64:319])
plt.subplot(2, 4, 8)
plt.imshow(Y, cmap='gray')
plt.title('Final Filtered Image', fontsize=12)
plt.axis('off')
plt.savefig('MY1b.png', bbox_inches='tight')

# Compare this result image with the one from part (a)
max_difference = np.max(np.abs(Y - Y1a))
print(f"(b): max difference from part (a): {max_difference}")

plt.show()
