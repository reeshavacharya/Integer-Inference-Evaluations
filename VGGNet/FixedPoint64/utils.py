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
    
    # 1. Split the target shift evenly across inputs
    shift_x = f_bits // 2          # 16
    shift_w = f_bits - shift_x     # 16
    
    # 2. Pre-Truncate (Arithmetic Right Shift with rounding factor)
    q_x_trunc = (q_x + (1 << (shift_x - 1))) >> shift_x
    q_w_trunc = (q_w + (1 << (shift_w - 1))) >> shift_w
    
    # 3. Execute Matrix Multiplication in pure int64
    q_out = F.conv2d(q_x_trunc, q_w_trunc, stride=stride, padding=padding)
    
    return torch.clamp(q_out, INT64_MIN, INT64_MAX)

def execute_and_shift_linear(q_x, q_w, f_bits=F_BITS):
    """Pure 64-bit integer linear matmul using Pre-Truncation."""
    
    shift_x = f_bits // 2
    shift_w = f_bits - shift_x
    
    q_x_trunc = (q_x + (1 << (shift_x - 1))) >> shift_x
    q_w_trunc = (q_w + (1 << (shift_w - 1))) >> shift_w
    
    # The product natively lands at 32 fractional bits. No post-shift required!
    accum = F.linear(q_x_trunc, q_w_trunc)
    
    # --- Capture ZK Stats ---
    max_accum_val = accum.abs().max().item()
    max_bits_used = math.ceil(math.log2(max_accum_val + 1)) if max_accum_val > 0 else 0
    
    # Remainder is 0 because the precision was dropped during the pre-truncation step.
    max_remainder = 0 
    # ------------------------
    
    q_out = torch.clamp(accum, INT64_MIN, INT64_MAX)
    return q_out, max_bits_used, max_remainder

def add_bias(q_accum, q_bias):
    bias_int64 = q_bias.to(torch.int64)
    if q_accum.dim() == 4:
        bias_int64 = bias_int64.view(1, -1, 1, 1)
    return q_accum + bias_int64

def fixed_point_relu(q_tensor):
    return torch.clamp(q_tensor, min=0)

def fixed_point_global_avg_pool2d(q_in):
    """Sums spatial dimensions and divides by N using native integer division."""
    q_int64 = q_in.to(torch.int64)
    B, C, H, W = q_int64.shape
    N = H * W
    
    accum = q_int64.sum(dim=(2, 3), keepdim=True)
    pooled = torch.div(accum + (N // 2), N, rounding_mode='floor')
    return torch.clamp(pooled, INT64_MIN, INT64_MAX).to(torch.int64)

def fixed_point_max_pool2d(q_tensor, kernel_size=2, stride=2):
    """Max pooling with 64-bit fixed point integers."""
    B, C, H, W = q_tensor.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1
    
    q_out = torch.zeros(B, C, out_h, out_w, dtype=torch.int64, device=q_tensor.device)
    
    for h in range(out_h):
        for w in range(out_w):
            h_start = h * stride
            w_start = w * stride
            h_end = h_start + kernel_size
            w_end = w_start + kernel_size
            
            patch = q_tensor[:, :, h_start:h_end, w_start:w_end]
            q_out[:, :, h, w] = patch.reshape(B, C, -1).max(dim=2)[0]
    
    return q_out

def fixed_point_adaptive_avg_pool2d(q_tensor, output_size=(7, 7)):
    """Adaptive average pooling with 64-bit fixed point integers."""
    B, C, H, W = q_tensor.shape
    out_h, out_w = output_size
    
    stride_h = H // out_h
    stride_w = W // out_w
    kernel_h = H - (out_h - 1) * stride_h
    kernel_w = W - (out_w - 1) * stride_w
    
    q_out = torch.zeros(B, C, out_h, out_w, dtype=torch.int64, device=q_tensor.device)
    
    for h in range(out_h):
        for w in range(out_w):
            h_start = h * stride_h
            w_start = w * stride_w
            h_end = h_start + kernel_h
            w_end = w_start + kernel_w
            
            patch = q_tensor[:, :, h_start:h_end, w_start:w_end]
            N = patch.shape[2] * patch.shape[3]
            accum = patch.sum(dim=(2, 3))
            q_out[:, :, h, w] = torch.div(accum + (N // 2), N, rounding_mode='floor')
    
    return torch.clamp(q_out, INT64_MIN, INT64_MAX).to(torch.int64)
