import math
import torch
import torch.nn.functional as F

# Static Fractional Bits for ezDPS format (Q31.32)
F_BITS = 32
INT64_MIN = -9223372036854775808
INT64_MAX = 9223372036854775807

# ---------------------------------------------------------
# 1. Quantization / Dequantization
# ---------------------------------------------------------

def quantize_fixed_point(tensor, f_bits=F_BITS):
    """Converts float to static 64-bit fixed point (Q31.32)."""
    q_tensor = torch.round(tensor.to(torch.float64) * (2 ** f_bits))
    q_tensor = torch.clamp(q_tensor, INT64_MIN, INT64_MAX)
    return q_tensor.to(torch.int64)

def dequantize_fixed_point(q_tensor, f_bits=F_BITS):
    """Converts static 64-bit fixed point back to float."""
    return q_tensor.to(torch.float64) / (2 ** f_bits)

# ---------------------------------------------------------
# 2. Pure INT64 Arithmetic Operations (Pre-Truncation)
# ---------------------------------------------------------

def execute_and_shift_conv2d(q_x, q_w, stride=1, padding=0, f_bits=F_BITS):
    """Pure 64-bit integer conv2d using Pre-Truncation to prevent overflow."""
    shift_x = f_bits // 2
    shift_w = f_bits - shift_x
    
    q_x_trunc = (q_x + (1 << (shift_x - 1))) >> shift_x
    q_w_trunc = (q_w + (1 << (shift_w - 1))) >> shift_w
    
    accum = F.conv2d(q_x_trunc, q_w_trunc, stride=stride, padding=padding)
    
    # Capture ZK Stats
    max_accum_val = accum.abs().max().item()
    max_bits_used = math.ceil(math.log2(max_accum_val + 1)) if max_accum_val > 0 else 0
    max_remainder = 0 # 0 because precision dropped during pre-truncation

    q_out = torch.clamp(accum, INT64_MIN, INT64_MAX)
    return q_out, max_bits_used, max_remainder


def execute_and_shift_conv_transpose2d(q_x, q_w, stride=2, padding=0, output_padding=0, f_bits=F_BITS):
    """Pure 64-bit integer transposed conv2d using Pre-Truncation."""
    shift_x = f_bits // 2
    shift_w = f_bits - shift_x
    
    q_x_trunc = (q_x + (1 << (shift_x - 1))) >> shift_x
    q_w_trunc = (q_w + (1 << (shift_w - 1))) >> shift_w

    stride_h = stride[0] if isinstance(stride, tuple) else stride
    stride_w = stride[1] if isinstance(stride, tuple) else stride
    pad_h = padding[0] if isinstance(padding, tuple) else padding
    pad_w = padding[1] if isinstance(padding, tuple) else padding
    out_pad_h = output_padding[0] if isinstance(output_padding, tuple) else output_padding
    out_pad_w = output_padding[1] if isinstance(output_padding, tuple) else output_padding

    x_int64 = q_x_trunc.to(torch.int64)
    w_int64 = q_w_trunc.to(torch.int64)
    
    B, in_C, H, W = x_int64.shape
    in_C_w, out_C, kH, kW = w_int64.shape
    
    padded_H = (H - 1) * stride_h + kH
    padded_W = (W - 1) * stride_w + kW
    
    device = x_int64.device
    acc_padded = torch.zeros((B, out_C, padded_H, padded_W), dtype=torch.int64, device=device)
    x_flat = x_int64.view(B, in_C, H * W).permute(0, 2, 1) 
    
    for kh in range(kH):
        for kw in range(kW):
            w_slice = w_int64[:, :, kh, kw]
            dot_product = torch.matmul(x_flat, w_slice)
            dot_spatial = dot_product.permute(0, 2, 1).view(B, out_C, H, W)
            acc_padded[:, :, kh : kh + H*stride_h : stride_h, kw : kw + W*stride_w : stride_w] += dot_spatial
            
    end_h = padded_H - pad_h if pad_h > 0 else padded_H
    end_w = padded_W - pad_w if pad_w > 0 else padded_W
    accum = acc_padded[:, :, pad_h : end_h, pad_w : end_w]
    
    if out_pad_h > 0 or out_pad_w > 0:
        accum = F.pad(accum, (0, out_pad_w, 0, out_pad_h), value=0)
        
    # Capture ZK Stats
    max_accum_val = accum.abs().max().item()
    max_bits_used = math.ceil(math.log2(max_accum_val + 1)) if max_accum_val > 0 else 0
    max_remainder = 0
        
    q_out = torch.clamp(accum, INT64_MIN, INT64_MAX)
    return q_out, max_bits_used, max_remainder


def execute_and_shift_linear(q_x, q_w, f_bits=F_BITS):
    """Included for completeness (UNet uses 1x1 convolutions instead of linear layers)"""
    shift_x = f_bits // 2
    shift_w = f_bits - shift_x
    q_x_trunc = (q_x + (1 << (shift_x - 1))) >> shift_x
    q_w_trunc = (q_w + (1 << (shift_w - 1))) >> shift_w
    
    accum = F.linear(q_x_trunc, q_w_trunc)
    
    max_accum_val = accum.abs().max().item()
    max_bits_used = math.ceil(math.log2(max_accum_val + 1)) if max_accum_val > 0 else 0
    
    return torch.clamp(accum, INT64_MIN, INT64_MAX), max_bits_used, 0


def add_bias(q_accum, q_bias):
    bias_int64 = q_bias.to(torch.int64)
    if q_accum.dim() == 4:
        bias_int64 = bias_int64.view(1, -1, 1, 1)
    return q_accum + bias_int64


def fixed_point_relu(q_tensor):
    return torch.clamp(q_tensor, min=0)