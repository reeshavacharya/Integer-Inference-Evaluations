import torch
import numpy as np
import warnings

print("--- PyTorch uint8 Division-by-Zero Test ---")
try:
    # Force uint8 tensors
    a_pt = torch.tensor([1], dtype=torch.uint8)
    b_pt = torch.tensor([0], dtype=torch.uint8)
    
    # Use floor division to keep it in the integer domain
    res_pt = a_pt // b_pt 
    print(f"PyTorch Success: {res_pt}")
except Exception as e:
    print(f"PyTorch Exception Caught: {type(e).__name__}: {e}")


print("\n--- NumPy uint8 Division-by-Zero Test ---")
# Catch warnings as errors to see NumPy's internal handling
warnings.filterwarnings("error") 
try:
    a_np = np.array([1], dtype=np.uint8)
    b_np = np.array([0], dtype=np.uint8)
    
    res_np = a_np // b_np
    print(f"NumPy Success: {res_np}")
except Exception as e:
    print(f"NumPy Exception Caught: {type(e).__name__}: {e}")