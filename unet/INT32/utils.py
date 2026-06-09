import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Parameter Calculation (Scale and Zero-Point)
# ---------------------------------------------------------
def get_quantization_params(tensor, num_bits=32):
    # Stretching the math to the absolute limits of the 32-bit container
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
    zero_point_bias = 0
    return scale_bias, zero_point_bias

def compute_integer_multiplier(scale_w, scale_x, scale_out):
    M = (scale_w * scale_x) / scale_out
    m0, exponent = math.frexp(M)
    shift = -exponent
    q_M0 = round(m0 * (1 << 31))
    return q_M0, shift

def compute_requantize_multiplier(scale_in, scale_out):
    M = scale_in / scale_out
    m0, exponent = math.frexp(M)
    shift = -exponent
    q_M0 = round(m0 * (1 << 31))
    return q_M0, shift

def quantize_tensor(tensor, scale, zero_point, dtype=torch.uint32, num_bits=32):
    q_tensor = torch.round(tensor / scale) + zero_point
    
    if dtype == torch.uint32:
        max_limit = (2**num_bits) - 1
        q_tensor = torch.clamp(q_tensor, 0, max_limit)
    elif dtype == torch.int32:
        q_tensor = torch.clamp(q_tensor, -2147483648, 2147483647)
    elif dtype == torch.int64:
        # Biases get full 64-bit bounds
        q_tensor = torch.clamp(q_tensor, -9223372036854775808, 9223372036854775807)
        
    return q_tensor.to(dtype)

# ---------------------------------------------------------
# 2. Integer Arithmetic Operations
# ---------------------------------------------------------
def add_bias(int32_accumulator, q_bias):
    bias_32 = q_bias.to(torch.int32)
    if int32_accumulator.dim() == 4:
        bias_32 = bias_32.view(1, -1, 1, 1)
    return (int32_accumulator.to(torch.int32) + bias_32.to(torch.int32)).to(torch.int32)

def multiply_by_quantized_multiplier(int32_accumulator, q_M0, shift):
    """
    Strict 32-bit hardware scaling simulator using 16-bit limb decomposition.
    No int64, no int16, no int8. Strictly torch.int32 arithmetic.
    """
    A = int32_accumulator.to(torch.int32)
    B = torch.as_tensor(q_M0, dtype=torch.int32, device=A.device)

    # 1. Unsigned Limb Decomposition (16-bit limbs)
    A_lo = A & 0xFFFF
    A_hi = (A >> 16) & 0xFFFF
    
    B_lo = B & 0xFFFF
    B_hi = (B >> 16) & 0xFFFF

    # 2. Cross Products (fits in 32-bit unsigned perfectly)
    P0 = A_lo * B_lo
    P1 = A_lo * B_hi
    P2 = A_hi * B_lo
    P3 = A_hi * B_hi

    # 3. Aligned Accumulation (simulating 64-bit sum)
    R0 = P0 & 0xFFFF
    carry1 = (P0 >> 16) & 0xFFFF

    S1 = carry1 + (P1 & 0xFFFF) + (P2 & 0xFFFF)
    R1 = S1 & 0xFFFF
    carry2 = S1 >> 16

    S2 = carry2 + ((P1 >> 16) & 0xFFFF) + ((P2 >> 16) & 0xFFFF) + (P3 & 0xFFFF)
    R2 = S2 & 0xFFFF
    carry3 = S2 >> 16

    S3 = carry3 + ((P3 >> 16) & 0xFFFF)
    R3 = S3 & 0xFFFF

    # Recombine into 32-bit registers (R_lo, R_hi)
    R_lo = R0 | (R1 << 16)
    R_hi = R2 | (R3 << 16)

    # 4. Apply Two's Complement Correction for Signed Multiplication
    is_A_neg = (A < 0).to(torch.int32)
    is_B_neg = (B < 0).to(torch.int32)
    
    R_hi = R_hi - (is_A_neg * B)
    R_hi = R_hi - (is_B_neg * A)

    # 5. Inject hardware rounding bit
    shift_tensor = torch.as_tensor(shift, dtype=torch.int32, device=A.device)
    total_shift = (31 + shift_tensor).to(torch.int32)
    
    one = torch.tensor(1, dtype=torch.int32, device=A.device)
    zero = torch.tensor(0, dtype=torch.int32, device=A.device)
    
    round_shift = torch.clamp(total_shift - 1, min=0)
    round_lo = torch.where(total_shift > 0, torch.where(round_shift < 32, one << round_shift, zero), zero)
    round_hi = torch.where(total_shift > 0, torch.where(round_shift >= 32, one << (round_shift - 32), zero), zero)
    
    # Safe 64-bit addition for rounding bit
    c_round = ((R_lo & 0xFFFF) + (round_lo & 0xFFFF)) >> 16
    carry_round = (((R_lo >> 16) & 0xFFFF) + ((round_lo >> 16) & 0xFFFF) + c_round) >> 16
    
    R_lo = R_lo + round_lo
    R_hi = R_hi + round_hi + carry_round

    # 6. Apply 64-bit arithmetic right shift
    shift_val = torch.clamp(total_shift, min=0)
    s_ge_32 = shift_val >= 32
    
    shift_hi = torch.clamp(shift_val - 32, min=0)
    shift_lo = torch.clamp(shift_val, max=31)
    
    thirty_two = torch.tensor(32, dtype=torch.int32, device=A.device)
    k = shift_lo
    mask = torch.where(
        k > 0,
        ~(torch.tensor(-1, dtype=torch.int32, device=A.device) << (thirty_two - k)),
        torch.tensor(-1, dtype=torch.int32, device=A.device)
    )
    
    R_lo_part = (R_lo >> k) & mask
    R_hi_part = torch.where(k > 0, R_hi << (thirty_two - k), zero)
    
    res_lt_32 = R_hi_part | R_lo_part
    res_ge_32 = R_hi >> shift_hi
    
    return torch.where(s_ge_32, res_ge_32, res_lt_32).to(torch.int32)

def downscale_and_cast(int32_accumulator, q_M0, shift, z_out):
    scaled_accum = multiply_by_quantized_multiplier(int32_accumulator, q_M0, shift)
    z_out_32 = torch.as_tensor(z_out, dtype=torch.int32, device=scaled_accum.device)
    
    # Modulo Math (Strict 32-bit arithmetic)
    q_out = (scaled_accum.to(torch.int32) + z_out_32).to(torch.int32)
    return q_out

def quantized_relu(q_tensor, z_out):
    z_out_32 = torch.as_tensor(z_out, dtype=torch.int32, device=q_tensor.device)
    q_int32 = q_tensor.to(torch.int32)
    clamped = torch.clamp(q_int32, min=z_out_32)
    return clamped.to(torch.int32)

def integer_gelu_lut(q_tensor, lut, z_out, q_min):
    q_tensor_int64 = q_tensor.to(torch.int64)
    z_out_tensor = torch.tensor(z_out, dtype=torch.int64, device=q_tensor.device)
    q_min_tensor = torch.tensor(q_min, dtype=torch.int64, device=q_tensor.device)
    
    # 1. Clamp to valid LUT bounds
    lut_size = len(lut)
    idx = q_tensor_int64 - q_min_tensor
    idx_clamped = torch.clamp(idx, 0, lut_size - 1)
    
    # 2. Gather outputs mapping
    q_out_int64 = torch.gather(lut.to(q_tensor.device).to(torch.int64), 0, idx_clamped.view(-1)).view(q_tensor.shape)
    
    # 3. Fallback for out-of-bounds
    # Positive out-of-bounds: GELU acts as identity (f(x) = x). Thus q_out = q_tensor
    q_out_int64 = torch.where(idx >= lut_size, q_tensor_int64, q_out_int64)
    
    # Negative out-of-bounds: GELU asymptotes to 0. In int space, 0 float = z_out
    q_out_int64 = torch.where(idx < 0, z_out_tensor, q_out_int64)

    return q_out_int64.to(torch.int32)

def requantize_tensor(q_tensor, z_old, z_new, q_M0, shift):
    z_old_32 = torch.as_tensor(z_old, dtype=torch.int32, device=q_tensor.device)
    z_new_32 = torch.as_tensor(z_new, dtype=torch.int32, device=q_tensor.device)
    
    x = (q_tensor.to(torch.int32) - z_old_32).to(torch.int32)
    
    scaled_x = multiply_by_quantized_multiplier(x, q_M0, shift)
    
    q_out = (scaled_x.to(torch.int32) + z_new_32).to(torch.int32)
    return q_out