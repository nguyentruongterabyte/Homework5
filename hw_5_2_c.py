"""
    MSSV     : N20DCCN083
    Họ và tên: Nguyễn Thái Trưởng
"""
import numpy as np
import matplotlib.pyplot as plt
from readBin import read_bin

X = read_bin('girl2.bin', 256)

XNhi = read_bin('girl2Noise32Hi.bin', 256)
XN = read_bin('girl2Noise32.bin', 256)

xx_nhi = (XNhi - X) ** 2
MSE_Nhi = np.mean(xx_nhi)

xx_n = (XN - X) ** 2
MSE_N = np.mean(xx_n)

U_cutoff_G = 64
SigmaG = 0.19 * 256 / U_cutoff_G

U, V = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))

GtildeCenter = np.exp((-2 * np.pi**2 * SigmaG**2) / (256**2) * (U**2 + V**2))
Gtilde = np.fft.fftshift(GtildeCenter)
G = np.fft.ifft2(Gtilde)
G2 = np.fft.fftshift(G)

ZPG2 = np.zeros((512, 512), dtype=np.complex128)
ZPG2[:256, :256] = G2


ZPX = np.zeros((512, 512), dtype=np.complex128)
ZPX[:256, :256] = X
yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPG2))
Y2 = yy[128:384, 128:384]
yy = np.abs(Y2 - X)**2
MSE_Y2 = np.mean(yy)

print(f'MSE: Gaussian LPF on girl2: {MSE_Y2}')

ZPX[:256, :256] = XNhi
yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPG2))
Y2Nhi = yy[128:384, 128:384]
yy = np.abs(Y2Nhi - X)**2
MSE_Y2Nhi = np.mean(yy)

print(f'MSE: Gaussian LPF on Noise32Hi: {MSE_Y2Nhi}')

ISNR_Y2Nhi = 10 * np.log10(MSE_Nhi / MSE_Y2Nhi)
print(f'ISNR: Gaussian LPF on Noise32Hi: {ISNR_Y2Nhi} dB')

ZPX[:256, :256] = XN
yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPG2))
Y2N = yy[128:384, 128:384]
yy = np.abs(Y2N - X)**2
MSE_Y2N = np.mean(yy)

print(f'MSE: Gaussian LPF on Noise32: {MSE_Y2N}')

ISNR_Y2N = 10 * np.log10(MSE_N / MSE_Y2N)
print(f'ISNR: Gaussian LPF on Noise32: {ISNR_Y2N} dB')

print(' ')

plt.figure(7)
plt.imshow(np.abs(Y2), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.title('Gauss1 on girl2', fontsize=18)
plt.savefig('MY2.png')

plt.figure(8)
plt.imshow(np.abs(Y2Nhi), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.title('Gauss1 on Noise32Hi', fontsize=18)
plt.savefig('MY2Nhi.png')

plt.figure(9)
plt.imshow(np.abs(Y2N), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.title('Gauss1 on Noise32', fontsize=18)
plt.savefig('MY2N.png')

plt.show()
