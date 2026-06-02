import torch
import numpy as np

# Two uint8 values
a = torch.tensor(3, dtype=torch.uint8)
a_bits = np.unpackbits(a.numpy())
print("a =", a_bits)

b = torch.tensor(100, dtype=torch.uint8)
b_bits = np.unpackbits(b.numpy())
print("b =", b_bits)

# PyTorch automatically promotes the result
product = a * b
product_bits = np.unpackbits(product.numpy())
print("a x b =", product_bits)

c=int(a) * int(b)
print("a x b (binary) =", bin(c))