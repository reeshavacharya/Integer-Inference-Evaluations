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
    
    # PyTorch struggles with native int32 Conv2d on GPUs.
    # We upcast to int64 purely to execute the math safely through CUDA, 
    # but strictly clamp the result to perfectly simulate a 32-bit physical accumulator.
    accum = F.conv2d(
        q_x_trunc.to(torch.int64), 
        q_w_trunc.to(torch.int64), 
        stride=stride, 
        padding=padding
    )
    
    # Capture ZK Stats
    max_accum_val = accum.abs().max().item()
    max_bits_used = math.ceil(math.log2(max_accum_val + 1)) if max_accum_val > 0 else 0
    max_remainder = 0
    
    # Force the 32-bit Overflow Wall
    q_out = torch.clamp(accum, INT32_MIN, INT32_MAX)
    return q_out.to(torch.int32), max_bits_used, max_remainder


def execute_and_shift_conv_transpose2d(q_x, q_w, stride=1, padding=0, output_padding=0, f_bits=F_BITS):
    shift_x = f_bits // 2
    shift_w = f_bits - shift_x
    
    q_x_trunc = (q_x + (1 << (shift_x - 1))) >> shift_x
    q_w_trunc = (q_w + (1 << (shift_w - 1))) >> shift_w
    
    B, C, H, W = q_x_trunc.shape
    out_C, in_C, kH, kW = q_w_trunc.shape 
    
    stride_h = stride[0] if isinstance(stride, tuple) else stride
    stride_w = stride[1] if isinstance(stride, tuple) else stride
    pad_h = padding[0] if isinstance(padding, tuple) else padding
    pad_w = padding[1] if isinstance(padding, tuple) else padding
    out_pad_h = output_padding[0] if isinstance(output_padding, tuple) else output_padding
    out_pad_w = output_padding[1] if isinstance(output_padding, tuple) else output_padding
    
    padded_H = (H - 1) * stride_h + kH
    padded_W = (W - 1) * stride_w + kW
    acc_padded = torch.zeros((B, out_C, padded_H, padded_W), dtype=torch.int64, device=q_x.device)
    
    x_int = q_x_trunc.to(torch.int64)
    w_int = q_w_trunc.to(torch.int64)
    
    for kh in range(kH):
        for kw in range(kW):
            w_spatial = w_int[:, :, kh, kw]
            x_reshaped = x_int.view(B, in_C, H * W)
            dot_spatial = torch.bmm(w_spatial.unsqueeze(0).expand(B, -1, -1), x_reshaped)
            dot_spatial = dot_spatial.view(B, out_C, H, W)
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
        
    # Force the 32-bit Overflow Wall
    q_out = torch.clamp(accum, INT32_MIN, INT32_MAX)
    return q_out.to(torch.int32), max_bits_used, max_remainder


def execute_and_shift_linear(q_x, q_w, f_bits=F_BITS):
    shift_x = f_bits // 2
    shift_w = f_bits - shift_x
    q_x_trunc = (q_x + (1 << (shift_x - 1))) >> shift_x
    q_w_trunc = (q_w + (1 << (shift_w - 1))) >> shift_w
    
    accum = F.linear(q_x_trunc.to(torch.int64), q_w_trunc.to(torch.int64))
    
    max_accum_val = accum.abs().max().item()
    max_bits_used = math.ceil(math.log2(max_accum_val + 1)) if max_accum_val > 0 else 0
    max_remainder = 0
    
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