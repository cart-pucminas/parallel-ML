import matplotlib.pyplot as plt
import numpy as np
import struct
import random

with open('./input/mnist/train-labels.idx1-ubyte', 'rb') as file:
    magic, size = struct.unpack(">II", file.read(8))
    labels = np.frombuffer(file.read(), dtype=np.uint8)

with open('./input/mnist/train-images.idx3-ubyte', 'rb') as file:
    magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
    images = np.frombuffer(file.read(), dtype=np.uint8).reshape(size, rows, cols)

index = random.randint(0, size - 1)

plt.figure(figsize=(5, 5))
plt.imshow(images[index], cmap='gray')
plt.title(f'Label: {labels[index]} (Index: {index})', fontsize=15)
plt.axis('off') # Cleaner look for your TCC
plt.show()
