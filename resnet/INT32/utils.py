import math
import torch
import torch.nn.functional as F

INT32_MIN = -2147483648
INT32_MAX = 2147483647

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
    zero_point_bias = torch.tensor(0, dtype=torch.int32)
    return scale_bias, zero_point_bias


def compute_integer_multiplier(scale_w, scale_x, scale_out):
    M = (scale_w * scale_x) / scale_out
    m0, exponent = math.frexp(M)
    shift = -exponent
    q_M0 = round(m0 * (1 << 31))

    if q_M0 == (1 << 31):
        q_M0 //= 2
        shift -= 1

    return torch.tensor(q_M0, dtype=torch.int32), torch.tensor(shift, dtype=torch.int32)


# ---------------------------------------------------------
# The Custom 128-Bit Accumulator Simulation
# ---------------------------------------------------------


def multiply_by_quantized_multiplier(int32_accumulator, q_M0, shift):
    """
    Native 64-bit hardware scaling simulator.
    Acc (~28 bits) * M0 (31 bits) = ~59 bits. Fits safely inside int64 ALU.
    """
    # 1. Load into 64-bit ALU wire to prevent overflow during multiplication
    acc_64 = int32_accumulator.to(torch.int64)
    m0_64 = q_M0.to(torch.int64)

    # 2. Execute full hardware product (Max ~59 bits)
    prod_64 = acc_64 * m0_64

    # 3. Calculate target truncation shift
    total_shift = 31 + shift

    # 4. Inject hardware rounding bit (Standard DSP practice)
    if total_shift > 0:
        rounding_bit = 1 << (total_shift - 1)
        prod_64 = prod_64 + rounding_bit

    # 5. Immediate Truncation -> Drop the LSBs to scale down!
    result_64 = prod_64 >> total_shift

    # Note: result_64 is now safely scaled down and mathematically 
    # guaranteed to fit back inside an int32 register.
    return result_64.to(torch.int32)

# ---------------------------------------------------------
# 2. Quantization / Arithmetic Logic
# ---------------------------------------------------------


def downscale_and_cast(int32_accumulator, q_M0, shift, z_out, clamp=True):
    scaled_accum = multiply_by_quantized_multiplier(int32_accumulator, q_M0, shift)
    
    z_out_32 = torch.as_tensor(z_out, dtype=torch.int32, device=scaled_accum.device)
    
    if clamp:
        # Saturation Math (Clamping to INT32 range)
        q_out = scaled_accum + z_out_32.to(torch.int64)
        q_out = torch.clamp(q_out, INT32_MIN, INT32_MAX)
    else:
        # Modulo Math (Native 32-bit arithmetic)
        q_out = (scaled_accum.to(torch.int32) + z_out_32).to(torch.int64)
    
    return q_out.to(torch.int32)


def quantize_tensor(tensor, scale, zero_point, dtype=torch.uint32):
    zero_point_tensor = torch.as_tensor(zero_point, dtype=torch.int32, device=tensor.device)
    q_tensor = torch.round(tensor / scale) + zero_point_tensor.to(tensor.dtype)
    if dtype == torch.int32:
        q_tensor = torch.clamp(q_tensor, -2147483648, 2147483647)
    elif dtype == torch.int64:
        q_tensor = torch.clamp(q_tensor, -9223372036854775808, 9223372036854775807)
    return q_tensor.to(dtype)


def dequantize_tensor(q_tensor, scale, zero_point):
    q_float = q_tensor.to(torch.float64)
    return scale * (q_float - zero_point)


# def integer_linear(q_x, q_w, z_x, z_w):
#     x_int = q_x.to(torch.int64) - z_x
#     w_int = q_w.to(torch.int64) - z_w
#     return F.linear(x_int, w_int)


# def integer_conv2d(q_x, q_w, z_x, z_w, stride=1, padding=0):
#     x_int = q_x.to(torch.int64) - z_x
#     w_int = q_w.to(torch.int64) - z_w
#     return F.conv2d(x_int, w_int, stride=stride, padding=padding)


def add_bias(int64_accumulator, q_bias):
    bias_int64 = q_bias.to(torch.int64)
    if int64_accumulator.dim() == 4:
        bias_int64 = bias_int64.view(1, -1, 1, 1)
    return int64_accumulator + bias_int64


def add_bias_with_clamp(int32_accumulator, q_bias, clamp=True):
    bias_32 = q_bias.to(torch.int32)
    if int32_accumulator.dim() == 4:
        bias_32 = bias_32.view(1, -1, 1, 1)

    if clamp:
        summed = int32_accumulator.to(torch.int64) + bias_32.to(torch.int64)
        return torch.clamp(summed, INT32_MIN, INT32_MAX).to(torch.int32)

    return (int32_accumulator.to(torch.int32) + bias_32.to(torch.int32)).to(torch.int32)


def integer_add(q1, z1, scale1, q2, z2, scale2, z_out, scale_out, clamp=True):
    # Enforce pure 32-bit subtraction
    z1_32 = torch.as_tensor(z1, dtype=torch.int32, device=q1.device)
    z2_32 = torch.as_tensor(z2, dtype=torch.int32, device=q2.device)
    z_out_32 = torch.as_tensor(z_out, dtype=torch.int32, device=q1.device)

    x1 = q1.to(torch.int32) - z1_32
    x2 = q2.to(torch.int32) - z2_32

    M0_1, shift_1 = compute_integer_multiplier(scale1, 1.0, scale_out)
    M0_2, shift_2 = compute_integer_multiplier(scale2, 1.0, scale_out)

    val1 = multiply_by_quantized_multiplier(x1, M0_1, shift_1)
    val2 = multiply_by_quantized_multiplier(x2, M0_2, shift_2)

    if clamp:
        # Saturation Math
        q_out = val1 + val2 + z_out_32.to(torch.int64)
        return torch.clamp(q_out, INT32_MIN, INT32_MAX).to(torch.int32)
    else:
        # Modulo Math
        q_out = (val1.to(torch.int32) + val2.to(torch.int32) + z_out_32).to(torch.int64)
        return q_out.to(torch.int32)


def integer_global_avg_pool2d(q_in, z_in, scale_in, z_out, scale_out, clamp=True):
    N = q_in.size(2) * q_in.size(3)
    
    # Enforce pure 32-bit subtraction
    z_in_32 = torch.as_tensor(z_in, dtype=torch.int32, device=q_in.device)
    z_out_32 = torch.as_tensor(z_out, dtype=torch.int32, device=q_in.device)

    x = q_in.to(torch.int32) - z_in_32
    accum = x.sum(dim=(2, 3), keepdim=True)

    M0, shift = compute_integer_multiplier(scale_in, 1.0, scale_out * N)
    pooled = multiply_by_quantized_multiplier(accum, M0, shift)

    if clamp:
        # Saturation Math
        q_out = pooled + z_out_32.to(torch.int64)
        return torch.clamp(q_out, INT32_MIN, INT32_MAX).to(torch.int32)
    else:
        # Modulo Math
        q_out = (pooled.to(torch.int32) + z_out_32).to(torch.int64)
        return q_out.to(torch.int32)


def quantized_relu(q_tensor, z_out):
    z_out_32 = torch.as_tensor(z_out, dtype=torch.int32, device=q_tensor.device)
    q_int32 = q_tensor.to(torch.int32)
    clamped = torch.clamp(q_int32, min=z_out_32)
    return clamped.to(torch.int32)

def integer_gelu_lut(q_tensor, lut, q_min_bound, q_max_bound):
    """
    Executes a pure-integer GELU approximation using a precomputed Lookup Table.
    """
    # 1. Clamp the incoming integer tensor to our safe LUT bounds
    # (Use int64 temporarily just to safely handle PyTorch indexing syntax)
    q_min_32 = torch.as_tensor(q_min_bound, dtype=torch.int32, device=q_tensor.device)
    q_max_32 = torch.as_tensor(q_max_bound, dtype=torch.int32, device=q_tensor.device)
    q_clamped = torch.clamp(q_tensor.to(torch.int32), q_min_32, q_max_32)
    
    # 2. Shift the values down to act as 0-based array indices
    indices = q_clamped.to(torch.int64) - q_min_32.to(torch.int64)
    
    # 3. Vectorized Table Lookup (No math, just memory fetching!)
    q_out = lut[indices]
    
    return q_out.to(torch.int32)