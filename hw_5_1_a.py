"""
    MSSV     : N20DCCN083
    Họ và tên: Nguyễn Thái Trưởng
"""
import matplotlib.pyplot as plt
import numpy as np

from readBin import read_bin
from stretch import stretch

X = read_bin('salesman.bin', 256)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(X, cmap='gray')
plt.title('Origin Image', fontsize=18)
plt.axis('off')
plt.savefig('salesman.png', bbox_inches='tight')

X2 = np.zeros((262, 262))
X2[3:259, 3:259] = X
Y2 = np.zeros((262, 262))
for row in range(3, 259):
    for col in range(3, 259):
        Y2[row, col] = np.sum(X2[row - 3: row+4, col - 3:col+4]) / 49
Y = stretch(Y2[3:259, 3:259])

plt.subplot(1, 2, 2)
plt.imshow(Y, cmap='gray')
plt.title('Filtered Image', fontsize=18)
plt.axis('off')
plt.savefig('MY1a.png', bbox_inches='tight')
plt.show()
