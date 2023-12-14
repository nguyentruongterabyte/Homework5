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
print(f"MSE girl2Noise32Hi.bin: {MSE_Nhi}")

xx_n = (XN - X) ** 2
MSE_N = np.mean(xx_n)
print(f"MSE girl2Noise32.bin: {MSE_N}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(X, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Original Tiffany Image', fontsize=18)
plt.savefig('Mgirl2.png', bbox_inches='tight')

plt.subplot(1, 3, 2)
plt.imshow(XNhi, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('girl2Noise32Hi', fontsize=18)
plt.savefig('Mgirl2Noise32Hi.png', bbox_inches='tight')

plt.subplot(1, 3, 3)
plt.imshow(XN, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('girl2Noise32', fontsize=18)
plt.savefig('Mgirl2Noise32.png', bbox_inches='tight')

plt.show()
