import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Parameter Calculation (Dynamic Q-Format)
# ---------------------------------------------------------

def get_fractional_bits(tensor, num_bits=8):
    """Calculates optimal fractional bits (f) for a float tensor."""
    max_int = (1 << (num_bits - 1)) - 1  # 127 for 8-bit signed
    abs_max = tensor.abs().max().item()
    
    if abs_max == 0:
        return 0 
        
    f_bits = math.floor(math.log2(max_int / abs_max))
    return int(f_bits)

# ---------------------------------------------------------
# 2. Quantization / Dequantization
# ---------------------------------------------------------

def quantize_fixed_point(tensor, f_bits, dtype=torch.int8):
    """Converts real float tensor to fixed-point integer tensor."""
    q_tensor = torch.round(tensor * (2 ** f_bits))
    if dtype == torch.int8:
        q_tensor = torch.clamp(q_tensor, -128, 127)
    elif dtype == torch.int32:
        q_tensor = torch.clamp(q_tensor, -2147483648, 2147483647)
    return q_tensor.to(dtype)

def dequantize_fixed_point(q_tensor, f_bits):
    """Converts fixed-point integer tensor back to float."""
    return q_tensor.to(torch.float32) * (2 ** -f_bits)

# ---------------------------------------------------------
# 3. Arithmetic Operations & Rounding
# ---------------------------------------------------------

def downscale_fixed_point(int32_accum, shift):
    """Downscales 32-bit accumulator using Round-to-Nearest, Ties-to-Even."""
    if shift <= 0:
        result = int32_accum << (-shift)
    else:
        shifted = int32_accum >> shift
        round_bit = (int32_accum >> (shift - 1)) & 1
        sticky_mask = (1 << (shift - 1)) - 1
        sticky_bit = (int32_accum & sticky_mask) != 0
        lsb = shifted & 1
        round_up = round_bit & (sticky_bit | lsb)
        result = shifted + round_up

    # We do NOT clamp here yet if we are using this for skip-connection alignment
    # We let the caller clamp to int8.
    return result

def fixed_point_relu(q_tensor):
    """ReLU in fixed-point is just clamping at 0."""
    return torch.clamp(q_tensor, min=0)

def fixed_point_conv2d(q_x, q_w, stride=1, padding=0):
    """Pure integer convolution."""
    return F.conv2d(q_x.to(torch.int32), q_w.to(torch.int32), stride=stride, padding=padding)

def fixed_point_linear(q_x, q_w):
    """Pure integer linear matmul."""
    return F.linear(q_x.to(torch.int32), q_w.to(torch.int32))

def add_bias(int32_accumulator, q_bias):
    bias_int32 = q_bias.to(torch.int32)
    if int32_accumulator.dim() == 4:
        bias_int32 = bias_int32.view(1, -1, 1, 1)
    return int32_accumulator + bias_int32

# ---------------------------------------------------------
# 4. ResNet Specific Operations (Alignment & Pooling)
# ---------------------------------------------------------

def fixed_point_add(q1, f1, q2, f2, f_out):
    """Aligns two branches to f_out via bit-shifts and adds them."""
    q1_32 = q1.to(torch.int32)
    q2_32 = q2.to(torch.int32)

    # Align q1 to f_out
    shift1 = f1 - f_out
    q1_aligned = downscale_fixed_point(q1_32, shift1) if shift1 > 0 else (q1_32 << -shift1)

    # Align q2 to f_out
    shift2 = f2 - f_out
    q2_aligned = downscale_fixed_point(q2_32, shift2) if shift2 > 0 else (q2_32 << -shift2)

    # Add and clamp back to int8
    q_add = q1_aligned + q2_aligned
    return torch.clamp(q_add, -128, 127).to(torch.int8)

def fixed_point_global_avg_pool2d(q_in):
    """Sums spatial dimensions and divides by N using native integer division."""
    q_int32 = q_in.to(torch.int32)
    B, C, H, W = q_int32.shape
    N = H * W
    
    # 1. Sum all pixels in the spatial window
    accum = q_int32.sum(dim=(2, 3), keepdim=True)
    
    # 2. Integer division by N with round-to-nearest logic
    # (Because this is post-ReLU, accum is always positive)
    pooled = torch.div(accum + (N // 2), N, rounding_mode='floor')
    
    # 3. Safely clamp back to int8
    return torch.clamp(pooled, -128, 127).to(torch.int8)