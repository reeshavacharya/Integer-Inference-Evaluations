import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Parameter Calculation (Scale and Zero-Point)
# ---------------------------------------------------------

def get_quantization_params(tensor, num_bits=32): 
    q_min, q_max = 0, (2**num_bits) - 1 
    
    min_val, max_val = tensor.min().item(), tensor.max().item()
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)
    
    scale = (max_val - min_val) / (q_max - q_min)
    if scale == 0:
        scale = 1e-8 
        
    zero_point = round(q_min - (min_val / scale))
    zero_point = max(q_min, min(q_max, zero_point))
    
    return scale, zero_point

def get_bias_quantization_params(scale_w, scale_x):
    scale_bias = scale_w * scale_x
    zero_point_bias = 0
    return scale_bias, zero_point_bias

def compute_integer_multiplier(scale_w, scale_x, scale_out):
    M = (scale_w * scale_x) / scale_out
    m0, exponent = math.frexp(M)
    shift = -exponent

    q_M0 = round(m0 * (1 << 31))
    if q_M0 == (1 << 31):
        q_M0 //= 2
        shift -= 1

    return int(q_M0), int(shift)

def multiply_by_quantized_multiplier(int64_accumulator, q_M0, shift):
    """
    Executes down-scaling using Python's native BigInt to safely bypass 
    PyTorch's lack of 128-bit tensors, guaranteeing zero overflow.
    """
    shape = int64_accumulator.shape
    # Flatten to list for native Python 128-bit math
    accum_list = int64_accumulator.flatten().tolist()
    
    total_shift = 31 + shift
    rounding_factor = 1 << (total_shift - 1)
    
    # Pure integer math: prevents overflow naturally in Python
    res_list = [((a * q_M0) + rounding_factor) >> total_shift for a in accum_list]
    
    # Cast back to int64 tensor
    return torch.tensor(res_list, dtype=torch.int64).view(shape)

def downscale_and_cast(int64_accumulator, q_M0, shift, z_out):
    scaled_accum = multiply_by_quantized_multiplier(int64_accumulator, q_M0, shift)
    q_out = scaled_accum + z_out

    # Saturating cast to uint32 limits
    q_out = torch.clamp(q_out, 0, 4294967295)
    return q_out.to(torch.uint32)

def compute_multiplier(scale_w, scale_x, scale_out):
    M = (scale_w * scale_x) / scale_out
    return M

# ---------------------------------------------------------
# 2. Quantization / Dequantization
# ---------------------------------------------------------

def quantize_tensor(tensor, scale, zero_point, dtype=torch.uint32, num_bits=32):
    q_tensor = torch.round(tensor / scale) + zero_point
    
    if dtype == torch.uint32:
        # Clamp strictly to the 32-bit limit to prevent any outlier from causing an overflow
        max_limit = (2**num_bits) - 1
        q_tensor = torch.clamp(q_tensor, 0, max_limit)
    elif dtype == torch.int64:
        # Biases are only added, never multiplied, so they get the full 64-bit limits
        q_tensor = torch.clamp(q_tensor, -9223372036854775808, 9223372036854775807)
        
    return q_tensor.to(dtype)

def dequantize_tensor(q_tensor, scale, zero_point):
    q_float = q_tensor.to(torch.float64) # Use float64 to prevent precision loss
    return scale * (q_float - zero_point)

# ---------------------------------------------------------
# 3. Integer Arithmetic Operations
# ---------------------------------------------------------

def integer_linear(q_x, q_w, z_x, z_w):
    # Upcast uint32 to int64 before subtracting zero-points to prevent underflow
    x_int = q_x.to(torch.int64) - z_x
    w_int = q_w.to(torch.int64) - z_w
    
    # F.linear handles the int64 accumulation
    return F.linear(x_int, w_int)

def integer_conv2d(q_x, q_w, z_x, z_w, stride=1, padding=0):
    x_int = q_x.to(torch.int64) - z_x
    w_int = q_w.to(torch.int64) - z_w
    
    return F.conv2d(x_int, w_int, stride=stride, padding=padding)

def add_bias(int64_accumulator, q_bias):
    bias_int64 = q_bias.to(torch.int64)
    if int64_accumulator.dim() == 4:
        bias_int64 = bias_int64.view(1, -1, 1, 1)
    return int64_accumulator + bias_int64

def quantized_relu(q_tensor, z_out):
    """
    Applies ReLU by clamping to the zero-point. 
    Includes a temporary int64 upcast to bypass PyTorch's missing UInt32 CPU kernels.
    """
    # 1. Cast the uint32 tensor to int64 to access PyTorch's supported CPU kernels
    q_int64 = q_tensor.to(torch.int64)
    
    # 2. Perform the mathematical clamp
    clamped = torch.clamp(q_int64, min=z_out)
    
    # 3. Safely cast the result natively back to uint32
    return clamped.to(torch.uint32)