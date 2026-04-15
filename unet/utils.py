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

    # Clamp to signed int8 bounds
    return torch.clamp(result, -128, 127).to(torch.int8)

def align_tensor_for_concat(q_tensor, f_current, f_target):
    """Aligns an int8 tensor to a target f_bits format using bit-shifts."""
    q_int32 = q_tensor.to(torch.int32)
    shift = f_current - f_target
    return downscale_fixed_point(q_int32, shift)

def fixed_point_relu(q_tensor):
    """ReLU in fixed-point is just clamping at 0."""
    return torch.clamp(q_tensor, min=0)

def fixed_point_conv2d(q_x, q_w, stride=1, padding=0):
    """Pure integer convolution."""
    return F.conv2d(q_x.to(torch.int32), q_w.to(torch.int32), stride=stride, padding=padding)

def fixed_point_linear(q_x, q_w):
    """Pure integer linear matmul."""
    return F.linear(q_x.to(torch.int32), q_w.to(torch.int32))

def fixed_point_conv_transpose2d(q_x, q_w, stride=2, padding=0, output_padding=0):
    """Pure integer transposed convolution without zero-points."""
    stride_h = stride[0] if isinstance(stride, tuple) else stride
    stride_w = stride[1] if isinstance(stride, tuple) else stride
    pad_h = padding[0] if isinstance(padding, tuple) else padding
    pad_w = padding[1] if isinstance(padding, tuple) else padding
    out_pad_h = output_padding[0] if isinstance(output_padding, tuple) else output_padding
    out_pad_w = output_padding[1] if isinstance(output_padding, tuple) else output_padding

    x_int32 = q_x.to(torch.int32)
    w_int32 = q_w.to(torch.int32)
    
    B, in_C, H, W = x_int32.shape
    in_C_w, out_C, kH, kW = w_int32.shape
    
    padded_H = (H - 1) * stride_h + kH
    padded_W = (W - 1) * stride_w + kW
    
    device = x_int32.device
    acc_padded = torch.zeros((B, out_C, padded_H, padded_W), dtype=torch.int32, device=device)
    x_flat = x_int32.view(B, in_C, H * W).permute(0, 2, 1) 
    
    for kh in range(kH):
        for kw in range(kW):
            w_slice = w_int32[:, :, kh, kw]
            dot_product = torch.matmul(x_flat, w_slice)
            dot_spatial = dot_product.permute(0, 2, 1).view(B, out_C, H, W)
            acc_padded[:, :, kh : kh + H*stride_h : stride_h, kw : kw + W*stride_w : stride_w] += dot_spatial
            
    end_h = padded_H - pad_h if pad_h > 0 else padded_H
    end_w = padded_W - pad_w if pad_w > 0 else padded_W
    acc_cropped = acc_padded[:, :, pad_h : end_h, pad_w : end_w]
    
    if out_pad_h > 0 or out_pad_w > 0:
        acc_cropped = F.pad(acc_cropped, (0, out_pad_w, 0, out_pad_h), value=0)
        
    return acc_cropped

def add_bias(int32_accumulator, q_bias):
    bias_int32 = q_bias.to(torch.int32)
    if int32_accumulator.dim() == 4:
        bias_int32 = bias_int32.view(1, -1, 1, 1)
    return int32_accumulator + bias_int32