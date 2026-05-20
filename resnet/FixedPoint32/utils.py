import math
import torch
import torch.nn.functional as F

# Static Fractional Bits for Q15.16 format (32-bit)
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
# 2. Pure INT32 Arithmetic Operations (Post-Truncation via 64-bit MAC)
# ---------------------------------------------------------

def execute_and_shift_conv2d(q_x, q_w, stride=1, padding=0, f_bits=F_BITS):
    """32-bit fixed-point conv2d using an internal 64-bit accumulator."""
    # 1. Upcast inputs to int64 to simulate a hardware 64-bit MAC unit
    x_mac = q_x.to(torch.int64)
    w_mac = q_w.to(torch.int64)
    
    # 2. Multiply and Accumulate: The product natively lands at 32 fractional bits (16 + 16)
    accum = F.conv2d(x_mac, w_mac, stride=stride, padding=padding)
    
    # 3. Post-Truncation: Shift right by 16 to return to Q15.16 format
    q_out = (accum + (1 << (f_bits - 1))) >> f_bits
    
    return torch.clamp(q_out, INT32_MIN, INT32_MAX).to(torch.int32)


def execute_and_shift_linear(q_x, q_w, f_bits=F_BITS):
    """32-bit fixed-point linear matmul using an internal 64-bit accumulator."""
    x_mac = q_x.to(torch.int64)
    w_mac = q_w.to(torch.int64)
    
    accum = F.linear(x_mac, w_mac)
    
    # --- Capture Logging Stats ---
    max_accum_val = accum.abs().max().item()
    max_bits_used = math.ceil(math.log2(max_accum_val + 1)) if max_accum_val > 0 else 0
    
    # In Post-Truncation, the remainder is the exact fractional precision dropped
    mask = (1 << f_bits) - 1
    max_remainder = (accum.abs() & mask).max().item()
    # -----------------------------
    
    # Post-Truncation Shift
    q_out = (accum + (1 << (f_bits - 1))) >> f_bits
    q_out = torch.clamp(q_out, INT32_MIN, INT32_MAX).to(torch.int32)
    
    return q_out, max_bits_used, max_remainder


def add_bias(q_accum, q_bias):
    """Adds the Q15.16 bias to the Q15.16 accumulator."""
    bias_int32 = q_bias.to(torch.int32)
    if q_accum.dim() == 4:
        bias_int32 = bias_int32.view(1, -1, 1, 1)
    
    # Use int64 for the addition to prevent overflow, then clamp down
    sum_int64 = q_accum.to(torch.int64) + bias_int32.to(torch.int64)
    return torch.clamp(sum_int64, INT32_MIN, INT32_MAX).to(torch.int32)


def fixed_point_relu(q_tensor):
    return torch.clamp(q_tensor, min=0)


def fixed_point_global_avg_pool2d(q_in):
    """Sums spatial dimensions and divides by N using native integer division."""
    q_int64 = q_in.to(torch.int64)
    B, C, H, W = q_int64.shape
    N = H * W
    
    # Accumulate fully in int64 to avoid overflow
    accum = q_int64.sum(dim=(2, 3), keepdim=True)
    pooled = torch.div(accum + (N // 2), N, rounding_mode='floor')
    return torch.clamp(pooled, INT32_MIN, INT32_MAX).to(torch.int32)