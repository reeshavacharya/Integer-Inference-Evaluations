import math

import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Parameter Calculation (Scale and Zero-Point)
# ---------------------------------------------------------
def get_quantization_params(tensor, num_bits=32):
    q_min, q_max = 0, (2**num_bits) - 1
    min_val, max_val = tensor.min().item(), tensor.max().item()
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)
    
    scale = (max_val - min_val) / (q_max - q_min)
    if scale == 0: scale = 1e-8
        
    zero_point = round(q_min - (min_val / scale))
    zero_point = max(q_min, min(q_max, zero_point))
    return scale, zero_point

def get_bias_quantization_params(scale_w, scale_x):
    """
    Biases are quantized to 32-bit integers. 
    The zero-point is always 0, and the scale is the product of 
    weight scale and input activation scale[cite: 181, 184].
    """
    scale_bias = scale_w * scale_x
    zero_point_bias = 0
    return scale_bias, zero_point_bias

def integer_add(q1, z1, scale1, q2, z2, scale2, z_out, scale_out):
    """32-bit skip connection addition using int64 scaling to prevent overflow."""
    x1 = q1.to(torch.int64) - z1
    x2 = q2.to(torch.int64) - z2
    
    # Use compute_integer_multiplier for exact M0/shift logic
    M0_1, shift_1 = compute_integer_multiplier(scale1, 1.0, scale_out)
    M0_2, shift_2 = compute_integer_multiplier(scale2, 1.0, scale_out)
    
    val1 = multiply_by_quantized_multiplier(x1, M0_1, shift_1)
    val2 = multiply_by_quantized_multiplier(x2, M0_2, shift_2)
    
    q_out = val1 + val2 + z_out
    return torch.clamp(q_out, 0, 4294967295).to(torch.uint32)


def integer_global_avg_pool2d(q_in, z_in, scale_in, z_out, scale_out):
    """32-bit Global Average Pooling with int64 accumulation."""
    N = q_in.size(2) * q_in.size(3) 
    
    x = q_in.to(torch.int64) - z_in
    accum = x.sum(dim=(2, 3), keepdim=True) # Accumulate in int64
    
    # Incorporate the division by N directly into the hardware multiplier
    M0, shift = compute_integer_multiplier(scale_in, 1.0, scale_out * N)
    
    pooled = multiply_by_quantized_multiplier(accum, M0, shift)
    q_out = pooled + z_out
    return torch.clamp(q_out, 0, 4294967295).to(torch.uint32)

def compute_integer_multiplier(scale_w, scale_x, scale_out):
    """
    Computed OFFLINE during model conversion.
    Decomposes the floating-point multiplier M into an int32 multiplier (M0) 
    and a right bit-shift amount (n).
    """
    # Calculate the raw floating point M
    M = (scale_w * scale_x) / scale_out
    # math.frexp splits M into a mantissa m0 in [0.5, 1.0) and an exponent
    # such that M = m0 * 2^exponent
    m0, exponent = math.frexp(M)
    # The paper defines the multiplier as M0 * 2^(-n) [cite: 142-144].
    # Therefore, n is the negated exponent.
    shift = -exponent

    # Represent m0 as an int32 value. 
    # It is the int32 value nearest to (2^31 * m0)[cite: 146].
    q_M0 = round(m0 * (1 << 31))

    # Edge case handler: if m0 rounded exactly to the int32 limit
    if q_M0 == (1 << 31):
        q_M0 //= 2
        shift -= 1

    return int(q_M0), int(shift)

def multiply_by_quantized_multiplier(int64_accumulator, q_M0, shift):
    """Executes down-scaling using Python's native BigInt to bypass 128-bit overflow."""
    shape = int64_accumulator.shape
    accum_list = int64_accumulator.flatten().tolist()
    total_shift = 31 + shift
    rounding_factor = 1 << (total_shift - 1)
    
    res_list = [((a * q_M0) + rounding_factor) >> total_shift for a in accum_list]
    return torch.tensor(res_list, dtype=torch.int64).view(shape)

def downscale_and_cast(int64_accumulator, q_M0, shift, z_out):
    scaled_accum = multiply_by_quantized_multiplier(int64_accumulator, q_M0, shift)
    q_out = scaled_accum + z_out
    q_out = torch.clamp(q_out, 0, 4294967295)
    return q_out.to(torch.uint32)

def compute_multiplier(scale_w, scale_x, scale_out):
    """
    Computes the multiplier M used to downscale the 32-bit accumulator 
    back to the 8-bit output scale[cite: 138, 140].
    """
    # M = (S_1 * S_2) / S_3
    M = (scale_w * scale_x) / scale_out
    return M

def quantize_multiplier(real_multiplier):
    """
    Converts a floating-point multiplier into an integer M0 and a right-shift.
    Simulates fixed-point multiplier hardware.
    """
    if real_multiplier == 0.:
        return 0, 0
    
    # math.frexp returns m in [0.5, 1) and e such that real_multiplier = m * 2^e
    m, e = math.frexp(real_multiplier)
    
    # Scale m to fit into a signed 32-bit integer
    q_M0 = int(round(m * (1 << 31)))
    
    # Calculate the total right shift needed
    shift = 31 - e
    
    return q_M0, shift

# ---------------------------------------------------------
# 2. Quantization / Dequantization
# ---------------------------------------------------------

def quantize_tensor(tensor, scale, zero_point, dtype=torch.uint32):
    q_tensor = torch.round(tensor / scale) + zero_point
    if dtype == torch.uint32:
        q_tensor = torch.clamp(q_tensor, 0, 4294967295)
    elif dtype == torch.int64:
        q_tensor = torch.clamp(q_tensor, -9223372036854775808, 9223372036854775807)
    return q_tensor.to(dtype)

def dequantize_tensor(q_tensor, scale, zero_point):
    """
    Converts a quantized integer tensor (q) back to a real float tensor (r).
    r = S(q - Z)
    """
    # Cast to float32 before subtraction to avoid underflow in uint8
    q_float = q_tensor.to(torch.float32)
    return scale * (q_float - zero_point)

# ---------------------------------------------------------
# 3. Integer Arithmetic Operations
# ---------------------------------------------------------

def integer_linear(q_x, q_w, z_x, z_w):
    x_int = q_x.to(torch.int64) - z_x
    w_int = q_w.to(torch.int64) - z_w
    return F.linear(x_int, w_int)

def integer_conv2d(q_x, q_w, z_x, z_w, stride=1, padding=0):
    x_int = q_x.to(torch.int64) - z_x
    w_int = q_w.to(torch.int64) - z_w
    return F.conv2d(x_int, w_int, stride=stride, padding=padding)

def add_bias(int64_accumulator, q_bias):
    bias_int64 = q_bias.to(torch.int64)
    if int64_accumulator.dim() == 4:
        bias_int64 = bias_int64.view(1, -1, 1, 1)
    return int64_accumulator + bias_int64

def quantized_relu(q_tensor, z_out):
    """Upcasts to int64 for clamping to bypass missing PyTorch UInt32 CPU kernels."""
    q_int64 = q_tensor.to(torch.int64)
    clamped = torch.clamp(q_int64, min=z_out)
    return clamped.to(torch.uint32)