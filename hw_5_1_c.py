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
Y = stretch(Y2[3:259, 3:259])
Y1a = Y

# P1(c)
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(X, cmap='gray')
plt.title('Original Image', fontsize=12)
plt.axis('off')

H = np.zeros((256, 256))
H[125:133, 125:133] = 1/49

H2 = fftshift(H)
# print(H2)
plt.subplot(2, 2, 2)
plt.imshow(stretch(H2), cmap='gray')
plt.title('Zero Phase Impulse Resp', fontsize=12)
plt.axis('off')
plt.savefig('MH1c.png', bbox_inches='tight')

ZPX = np.zeros((512, 512))
ZPX[:256, :256] = X

ZPH2 = np.zeros((512, 512))
ZPH2[:128, :128] = H2[:128, :128]
ZPH2[:128, 384:512] = H2[:128, 128:256]
ZPH2[384:512, :128] = H2[128:256, :128]
ZPH2[384:512, 384:512] = H2[128:256, 128:256]

plt.subplot(2, 2, 3)
plt.imshow(stretch(ZPH2), cmap='gray')
plt.title('Zero Padded zero-phase H', fontsize=12)
plt.axis('off')
plt.savefig('MZPH1c.png', bbox_inches='tight')

Y = ifft2(fft2(ZPX) * fft2(ZPH2))
Y = stretch(Y[:256, :256])

plt.subplot(2, 2, 4)
plt.imshow(Y, cmap='gray')
plt.title('Final Filtered Image', fontsize=12)
plt.axis('off')
plt.savefig('MY1c.png', bbox_inches='tight')

max_difference = np.max(np.abs(Y - Y1a))
print(f"(c): max difference from part (a): {max_difference}")

plt.show()
