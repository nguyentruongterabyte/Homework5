"""
    MSSV     : N20DCCN083
    Họ và tên: Nguyễn Thái Trưởng
"""
import numpy as np
import matplotlib.pyplot as plt
from readBin import read_bin

X = read_bin('girl2.bin', 256)
XN = read_bin('girl2Noise32.bin', 256)
XNhi = read_bin('girl2Noise32Hi.bin', 256)

xx_nhi = (XNhi - X) ** 2
MSE_Nhi = np.mean(xx_nhi)

U_cutoff_G = 77.5
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
Y3 = yy[128:384, 128:384]
yy = np.abs(Y3 - X)**2
MSE_Y3 = np.mean(yy)

print(f'MSE: Gaussian2 LPF on girl2: {MSE_Y3}')

ZPX[:256, :256] = XNhi
yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPG2))
Y3Nhi = yy[128:384, 128:384]
yy = np.abs(Y3Nhi - X)**2
MSE_Y3Nhi = np.mean(yy)
xx_n = (XN - X) ** 2
MSE_N = np.mean(xx_n)

print(f'MSE: Gaussian2 LPF on Noise32Hi: {MSE_Y3Nhi}')

ISNR_Y3Nhi = 10 * np.log10(MSE_Nhi / MSE_Y3Nhi)
print(f'ISNR: Gaussian2 LPF on Noise32Hi: {ISNR_Y3Nhi} dB')

ZPX[:256, :256] = XN
yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPG2))
Y3N = yy[128:384, 128:384]
yy = np.abs(Y3N - X)**2
MSE_Y3N = np.mean(yy)

print(f'MSE: Gaussian2 LPF on Noise32: {MSE_Y3N}')

ISNR_Y3N = 10 * np.log10(MSE_N / MSE_Y3N)
print(f'ISNR: Gaussian2 LPF on Noise32: {ISNR_Y3N} dB')

plt.figure(10)
plt.imshow(np.abs(Y3), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.title('Gauss2 on girl2', fontsize=18)
plt.savefig('MY3.png')

plt.figure(11)
plt.imshow(np.abs(Y3Nhi), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.title('Gauss2 on Noise32Hi', fontsize=18)
plt.savefig('MY3Nhi.png')

plt.figure(12)
plt.imshow(np.abs(Y3N), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.title('Gauss2 on Noise32', fontsize=18)
plt.savefig('MY3N.png')

plt.show()
