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
    """Strict Q15.16 MAC: Configurable truncation and accumulation."""
    B, C_in, H, W = q_x.shape
    C_out, _, kH, kW = q_w.shape

    if padding > 0:
        q_x = torch.nn.functional.pad(q_x, (padding, padding, padding, padding))

    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1

    accum = torch.zeros((B, C_out, out_H, out_W), dtype=torch.int32, device=q_x.device)

    for kh in range(kH):
        h_start = kh
        h_end = h_start + out_H * stride
        for kw in range(kW):
            w_start = kw
            w_end = w_start + out_W * stride

            x_slice = q_x[:, :, h_start:h_end:stride, w_start:w_end:stride]
            w_slice = q_w[:, :, kh, kw].view(C_out, C_in, 1, 1)

            # 1. ALU Multiply (Q15.16 * Q15.16 = Q31.32 internal wire)
            prod_64 = x_slice.unsqueeze(1).to(torch.int64) * w_slice.to(torch.int64)

            # 2. Immediate Truncation
            prod_trunc = (prod_64 + (1 << (f_bits - 1))) >> f_bits
            prod_32 = prod_trunc.to(torch.int32)

            # 3. Modulo Math (Native 32-bit ALU)
            accum = accum + prod_32.sum(dim=2, dtype=torch.int32)

    return accum


def execute_and_shift_linear(q_x, q_w, f_bits=F_BITS):
    """Strict Q15.16 MAC: Configurable truncation and accumulation."""
    B, in_f = q_x.shape
    out_f = q_w.shape[0]
    
    accum = torch.zeros((B, out_f), dtype=torch.int32, device=q_x.device)

    for i in range(in_f):
        x_val = q_x[:, i:i+1]
        w_val = q_w[:, i:i+1].transpose(0, 1)

        # 1. ALU Multiply
        prod_64 = x_val.to(torch.int64) * w_val.to(torch.int64)

        # 2. Immediate Truncation
        prod_trunc = (prod_64 + (1 << (f_bits - 1))) >> f_bits
        prod_32 = prod_trunc.to(torch.int32)

        # 3. Modulo Math
        accum = accum + prod_32

    return accum, 0, 0


def add_bias(q_accum, q_bias):
    """Adds the Q15.16 bias to the Q15.16 accumulator."""
    bias_int32 = q_bias.to(torch.int32)
    if q_accum.dim() == 4:
        bias_int32 = bias_int32.view(1, -1, 1, 1)
    
    # Modulo Math
    return q_accum + bias_int32


def fixed_point_max_pool2d(q_in):
    """Executes a pure integer MaxPool."""
    return torch.amax(q_in.to(torch.int32), dim=(2, 3), keepdim=True)

def fixed_point_relu(q_tensor):
    """Executes a pure 32-bit integer ReLU."""
    # Since Q15.16 format has 0 float == 0 int, ReLU is a simple native clamp.
    return torch.clamp(q_tensor, min=0)

def fixed_point_gelu_lut(q_tensor, lut_dict):
    """Executes a pure 32-bit Q15.16 GELU using a precomputed Lookup Table."""
    lut = lut_dict["lut"].to(q_tensor.device)
    q_min = lut_dict["q_min"]
    q_max = lut_dict["q_max"]

    # 1. Clamp to safe LUT boundaries
    q_clamped = torch.clamp(q_tensor, q_min, q_max)
    
    # 2. Shift values down to act as 0-based array indices (using int64 safely for PyTorch indexing)
    indices = q_clamped.to(torch.int64) - q_min
    
    # 3. Vectorized Table Fetch
    return lut[indices]