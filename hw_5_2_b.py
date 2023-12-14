"""
    MSSV     : N20DCCN083
    Họ và tên: Nguyễn Thái Trưởng
"""
import numpy as np

from readBin import read_bin
from stretch import stretch

import matplotlib.pyplot as plt

X = read_bin('girl2.bin', 256)

XNhi = read_bin('girl2Noise32Hi.bin', 256)
XN = read_bin('girl2Noise32.bin', 256)

xx_nhi = (XNhi - X) ** 2
MSE_Nhi = np.mean(xx_nhi)

xx_n = (XN - X) ** 2
MSE_N = np.mean(xx_n)

U_cutoff = 64
U, V = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
HLtildeCenter = (np.sqrt(U**2 + V**2) <= U_cutoff).astype(float)
HLtilde = np.fft.fftshift(HLtildeCenter)
Y1 = np.fft.ifft2(np.fft.fft2(X) * HLtilde).real
yy_y1 = (Y1 - X) ** 2
MSE_Y1 = np.mean(yy_y1)
print(f"MSE: ideal LPF on girl2: {MSE_Y1}")

Y1Nhi = np.fft.ifft2(np.fft.fft2(XNhi) * HLtilde).real
yy_y1nhi = (Y1Nhi - X) ** 2
MSE_Y1Nhi = np.mean(yy_y1nhi)
print(f"MSE: ideal LPF on Noise32Hi: {MSE_Y1Nhi}")
ISNR_Y1Nhi = 10 * np.log10(MSE_Nhi / MSE_Y1Nhi)
print(f"ISNR: ideal LPF on Noise32Hi: {ISNR_Y1Nhi} dB")

Y1N = np.fft.ifft2(np.fft.fft2(XN) * HLtilde).real
yy_y1n = (Y1N - X) ** 2
MSE_Y1N = np.mean(yy_y1n)
print(f"MSE: ideal LPF on Noise32: {MSE_Y1N}")
ISNR_Y1N = 10 * np.log10(MSE_N / MSE_Y1N)
print(f"ISNR: ideal LPF on Noise32: {ISNR_Y1N} dB")

# Display images
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(stretch(Y1), cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('LPF on girl2', fontsize=18)
plt.savefig('MY1.png', bbox_inches='tight')

plt.subplot(1, 3, 2)
plt.imshow(stretch(Y1Nhi), cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('LPF on Noise32Hi', fontsize=18)
plt.savefig('MY1Nhi.png', bbox_inches='tight')

plt.subplot(1, 3, 3)
plt.imshow(stretch(Y1N), cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('LPF on Noise32', fontsize=18)
plt.savefig('MY1N.png', bbox_inches='tight')

plt.show()
