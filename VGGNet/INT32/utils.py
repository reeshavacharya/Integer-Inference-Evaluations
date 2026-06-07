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
    return scale_bias, 0

def compute_integer_multiplier(scale_w, scale_x, scale_out):
    M = (scale_w * scale_x) / scale_out
    m0, exponent = math.frexp(M)
    shift = -exponent
    q_M0 = round(m0 * (1 << 31))

    if q_M0 == (1 << 31):
        q_M0 //= 2
        shift -= 1

    return torch.tensor(q_M0, dtype=torch.int32), torch.tensor(shift, dtype=torch.int32)

def quantize_tensor(tensor, scale, zero_point, dtype=torch.uint32, num_bits=32):
    q_tensor = torch.round(tensor / scale) + zero_point
    
    if dtype == torch.uint32:
        max_limit = (2**num_bits) - 1
        q_tensor = torch.clamp(q_tensor, 0, max_limit)
    elif dtype == torch.int32:
        q_tensor = torch.clamp(q_tensor, INT32_MIN, INT32_MAX)
        
    return q_tensor.to(dtype)


# ---------------------------------------------------------
# 2. Strict 32-Bit Hardware Scaling Simulator
# ---------------------------------------------------------

def multiply_by_quantized_multiplier(int32_accumulator, q_M0, shift):
    """
    Strict 32-bit hardware scaling simulator using 16-bit limb decomposition.
    No int64, no int16, no int8. Strictly torch.int32 arithmetic.
    """
    A = int32_accumulator.to(torch.int32)
    B = torch.tensor(q_M0, dtype=torch.int32, device=int32_accumulator.device)

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
    total_shift = torch.tensor(31 + shift, dtype=torch.int32, device=A.device) if isinstance(shift, int) else (31 + shift).to(torch.int32)
    
    one = torch.tensor(1, dtype=torch.int32, device=A.device)
    zero = torch.tensor(0, dtype=torch.int32, device=A.device)
    
    round_shift = torch.clamp(total_shift - 1, min=0)
    round_lo = torch.where(total_shift > 0, torch.where(round_shift < 32, one << round_shift, zero), zero)
    round_hi = torch.where(total_shift > 0, torch.where(round_shift >= 32, one << (round_shift - 32), zero), zero)
    
    # Safe 64-bit addition for rounding bit simulated in 32-bit
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


# ---------------------------------------------------------
# 3. Auxiliary Functions
# ---------------------------------------------------------

def add_bias(int32_accumulator, q_bias):
    bias_32 = q_bias.to(torch.int32)
    if int32_accumulator.dim() == 4:
        bias_32 = bias_32.view(1, -1, 1, 1)
    # Modulo Math
    return (int32_accumulator.to(torch.int32) + bias_32.to(torch.int32)).to(torch.int32)


def downscale_and_cast(int32_accumulator, q_M0, shift, z_out):
    scaled_accum = multiply_by_quantized_multiplier(int32_accumulator, q_M0, shift)
    
    z_out_32 = torch.as_tensor(z_out, dtype=torch.int32, device=scaled_accum.device)
    
    # Modulo Math (Strict 32-bit arithmetic)
    q_out = (scaled_accum.to(torch.int32) + z_out_32).to(torch.int32)
    return q_out.to(torch.uint32)


def quantized_relu(q_tensor, z_out):
    z_out_32 = torch.as_tensor(z_out, dtype=torch.int32, device=q_tensor.device)
    q_int32 = q_tensor.to(torch.int32)
    clamped = torch.clamp(q_int32, min=z_out_32)
    return clamped.to(torch.uint32)


def integer_gelu_lut(q_tensor, lut, z_out):
    # Map uint32 back to int32 range for array indexing
    # We use z_out (which is also z_in because GELU is layer-specific)
    z_out_32 = torch.as_tensor(z_out, dtype=torch.int32, device=q_tensor.device)
    q_int32 = q_tensor.to(torch.int32)
    
    # Determine bounds from LUT size
    lut_size = len(lut)
    # Assuming LUT is symmetric around z_out
    half_size = lut_size // 2
    
    q_min_32 = z_out_32 - half_size
    q_max_32 = z_out_32 + half_size - 1
    
    q_clamped = torch.clamp(q_int32, q_min_32, q_max_32)
    
    # Temporarily upcast to int64 ONLY to satisfy PyTorch's indexing constraints
    indices = q_clamped.to(torch.int64) - q_min_32.to(torch.int64)
    
    # Temporarily upcast lut to int64 for indexing since PyTorch lacks index_cpu for UInt32
    q_out = lut.to(torch.int64)[indices]
    return q_out.to(torch.uint32)

def generate_layer_gelu_lut(in_scale, in_zp, out_scale, out_zp):
    """
    Generates a localized integer-to-integer LUT for GELU mapped between [-10.0, 10.0].
    """
    in_zp_32 = torch.tensor(in_zp, dtype=torch.int32)
    q_min_bound = torch.round(torch.tensor(-10.0 / in_scale, dtype=torch.float32)).to(torch.int32) + in_zp_32
    q_max_bound = torch.round(torch.tensor(10.0 / in_scale, dtype=torch.float32)).to(torch.int32) + in_zp_32
    
    q_inputs = torch.arange(q_min_bound.item(), q_max_bound.item() + 1, dtype=torch.float64)
    
    f_inputs = in_scale * (q_inputs - in_zp)
    f_act = torch.nn.functional.gelu(f_inputs)
    
    q_act = torch.round(f_act / out_scale).to(torch.int32) + torch.tensor(out_zp, dtype=torch.int32)
    q_act = torch.clamp(q_act, 0, 4294967295)  # UInt32 range
    
    return q_min_bound.item(), q_max_bound.item(), q_act.to(torch.uint32)

def integer_max_pool2d(q_x, kernel_size=2, stride=2):
    """Max pooling works perfectly on int32 tensors."""
    return F.max_pool2d(q_x.to(torch.int32), kernel_size=kernel_size, stride=stride).to(q_x.dtype)

def integer_adaptive_max_pool2d(q_x, output_size=(7, 7)):
    """Pure integer adaptive max pool supporting 1x1 and 7x7 inputs."""
    h, w = q_x.shape[2:]
    out_h, out_w = output_size
    if h == out_h and w == out_w:
        return q_x
    if h == 1 and w == 1:
        return q_x.expand(-1, -1, out_h, out_w)
    raise NotImplementedError("Pure integer adaptive pool only supports 1x1 and 7x7 inputs.")