import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Parameter Calculation (Dynamic Q-Format)
# ---------------------------------------------------------

def get_fractional_bits(tensor, num_bits=8):
    """
    Calculates the optimal number of fractional bits (f) for a float tensor.
    Formula: max_abs_val * 2^f <= max_int
    """
    max_int = (1 << (num_bits - 1)) - 1  # 127 for 8-bit signed
    abs_max = tensor.abs().max().item()
    
    if abs_max == 0:
        return 0  # Avoid log2(0) if tensor is all zeros
        
    # Calculate the max possible fractional bits without overflowing
    f_bits = math.floor(math.log2(max_int / abs_max))
    return int(f_bits)

# ---------------------------------------------------------
# 2. Quantization / Dequantization
# ---------------------------------------------------------

def quantize_fixed_point(tensor, f_bits, dtype=torch.int8):
    """
    Converts a real float tensor to a fixed-point integer tensor.
    q = round(r * 2^f)
    Uses PyTorch's native round() which defaults to round-to-nearest-even.
    """
    q_tensor = torch.round(tensor * (2 ** f_bits))
    
    # Clamp based on signed bounds
    if dtype == torch.int8:
        q_tensor = torch.clamp(q_tensor, -128, 127)
    elif dtype == torch.int32:
        q_tensor = torch.clamp(q_tensor, -2147483648, 2147483647)
        
    return q_tensor.to(dtype)

def dequantize_fixed_point(q_tensor, f_bits):
    """
    Converts a fixed-point integer tensor back to float.
    r = q * 2^(-f)
    """
    return q_tensor.to(torch.float32) * (2 ** -f_bits)

# ---------------------------------------------------------
# 3. Arithmetic Operations & Rounding
# ---------------------------------------------------------

def downscale_fixed_point(int32_accum, shift):
    """
    Downscales a 32-bit accumulator using right bit-shifts.
    Implements standard Round-to-Nearest, Ties-to-Even hardware logic.
    Finally saturates to int8 [-128, 127].
    """
    if shift <= 0:
        # If output requires MORE fractional bits than the accumulator, shift left
        result = int32_accum << (-shift)
    else:
        # Hardware logic for Ties-to-Even rounding via bitmasks
        
        # 1. Extract the primary right-shifted value
        shifted = int32_accum >> shift
        
        # 2. Extract the "Round Bit" (the bit just shifted out: shift - 1)
        round_bit = (int32_accum >> (shift - 1)) & 1
        
        # 3. Extract the "Sticky Bit" (are any bits below the round bit set to 1?)
        sticky_mask = (1 << (shift - 1)) - 1
        sticky_bit = (int32_accum & sticky_mask) != 0
        
        # 4. Extract the LSB of the shifted result
        lsb = shifted & 1
        
        # 5. Rounding Decision: Add 1 if Round bit is 1 AND (Sticky bit is 1 OR LSB is 1)
        round_up = round_bit & (sticky_bit | lsb)
        
        result = shifted + round_up

    # Saturating cast to signed int8
    return torch.clamp(result, -128, 127).to(torch.int8)

def fixed_point_relu(q_tensor):
    """
    Since Zero-Point is explicitly 0, ReLU is just a standard clamp at 0.
    """
    return torch.clamp(q_tensor, min=0)

def fixed_point_conv2d(q_x, q_w, stride=1, padding=0):
    """Pure integer convolution without zero points."""
    return F.conv2d(q_x.to(torch.int32), q_w.to(torch.int32), stride=stride, padding=padding)

def fixed_point_linear(q_x, q_w):
    """Pure integer linear matmul without zero points."""
    return F.linear(q_x.to(torch.int32), q_w.to(torch.int32))

def add_bias(int32_accumulator, q_bias):
    """Adds the 32-bit integer bias vector to the 32-bit accumulator."""
    bias_int32 = q_bias.to(torch.int32)
    if int32_accumulator.dim() == 4:
        bias_int32 = bias_int32.view(1, -1, 1, 1)
    return int32_accumulator + bias_int32