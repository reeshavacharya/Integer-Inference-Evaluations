import math
import torch
import torch.nn.functional as F

# Static Fractional Bits for 32-bit Fixed Point (Q15.16)
F_BITS = 16
INT32_MIN = -2147483648
INT32_MAX = 2147483647

# ---------------------------------------------------------
# 1. Quantization / Dequantization
# ---------------------------------------------------------
def quantize_fixed_point(tensor, f_bits=F_BITS):
    """Converts float to static 32-bit fixed point (Q15.16)."""
    q_tensor = torch.round(tensor.to(torch.float64) * (2 ** f_bits))
    q_tensor = torch.clamp(q_tensor, INT32_MIN, INT32_MAX)
    return q_tensor.to(torch.int32)

def dequantize_fixed_point(q_tensor, f_bits=F_BITS):
    """Converts static 32-bit fixed point back to float."""
    return q_tensor.to(torch.float64) / (2 ** f_bits)

# ---------------------------------------------------------
# 2. Pure INT32 Arithmetic Operations (Pre-Truncation)
# ---------------------------------------------------------
def execute_and_shift_conv2d(q_x, q_w, stride=1, padding=0, f_bits=F_BITS):
    """32-bit integer conv2d using Pre-Truncation to preserve headroom."""
    shift_x = f_bits // 2
    shift_w = f_bits - shift_x
    
    # Apply truncation with rounding
    q_x_trunc = (q_x + (1 << (shift_x - 1))) >> shift_x
    q_w_trunc = (q_w + (1 << (shift_w - 1))) >> shift_w
    
    # Execute math through int64 pipeline to bypass CUDA uint32/int32 limitation
    accum = F.conv2d(
        q_x_trunc.to(torch.int64), 
        q_w_trunc.to(torch.int64), 
        stride=stride, 
        padding=padding
    )
    
    max_accum_val = accum.abs().max().item()
    max_bits_used = math.ceil(math.log2(max_accum_val + 1)) if max_accum_val > 0 else 0
    max_remainder = 0
    
    # Force the 32-bit Hardware Overflow Wall
    q_out = torch.clamp(accum, INT32_MIN, INT32_MAX)
    return q_out.to(torch.int32), max_bits_used, max_remainder


def execute_and_shift_linear(q_x, q_w, f_bits=F_BITS):
    """32-bit integer linear using Pre-Truncation."""
    shift_x = f_bits // 2
    shift_w = f_bits - shift_x
    
    q_x_trunc = (q_x + (1 << (shift_x - 1))) >> shift_x
    q_w_trunc = (q_w + (1 << (shift_w - 1))) >> shift_w
    
    accum = F.linear(q_x_trunc.to(torch.int64), q_w_trunc.to(torch.int64))
    
    max_accum_val = accum.abs().max().item()
    max_bits_used = math.ceil(math.log2(max_accum_val + 1)) if max_accum_val > 0 else 0
    max_remainder = 0
    
    # Force the 32-bit Hardware Overflow Wall
    q_out = torch.clamp(accum, INT32_MIN, INT32_MAX)
    return q_out.to(torch.int32), max_bits_used, max_remainder


def add_bias(int32_accumulator, q_bias):
    bias_int32 = q_bias.to(torch.int32)
    if int32_accumulator.dim() == 4:
        bias_int32 = bias_int32.view(1, -1, 1, 1)
        
    result = int32_accumulator.to(torch.int64) + bias_int32.to(torch.int64)
    return torch.clamp(result, INT32_MIN, INT32_MAX).to(torch.int32)

def fixed_point_relu(q_tensor):
    return torch.max(q_tensor, torch.zeros_like(q_tensor))

def fixed_point_max_pool2d(q_x, kernel_size=2, stride=2):
    # Safe float cast purely to utilize PyTorch's native C++ pooling operator
    q_x_float = q_x.to(torch.float32)
    q_out = F.max_pool2d(q_x_float, kernel_size=kernel_size, stride=stride)
    return q_out.to(torch.int32)

def fixed_point_adaptive_avg_pool2d(q_x, output_size=(7, 7)):
    q_x_float = q_x.to(torch.float64)
    # The mean of fixed-point values equals the fixed-point of the mean
    q_out = torch.round(F.adaptive_avg_pool2d(q_x_float, output_size))
    return torch.clamp(q_out, INT32_MIN, INT32_MAX).to(torch.int32)