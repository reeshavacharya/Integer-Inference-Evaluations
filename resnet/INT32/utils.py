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

    if q_M0 == (1 << 31):
        q_M0 //= 2
        shift -= 1

    return int(q_M0), int(shift)


# ---------------------------------------------------------
# The Custom 128-Bit Accumulator Simulation
# ---------------------------------------------------------


def multiply_by_quantized_multiplier(int64_accumulator, q_M0, shift):
    """
    Vectorized 128-bit multiplication simulator for GPU/CPU PyTorch tensors.

    The accumulator can be up to ~63 bits (int64) and q_M0 is a 31-bit
    value (from frexp normalization). Their product occupies up to 94 bits,
    which requires a 128-bit register to hold safely. Since PyTorch has no
    128-bit integer tensor type, we simulate this with a 4-register split:

        A (63-bit) x B (31-bit)  ->  96-bit product, stored as:
            [ P_hi_hi (32b) | P_hi_lo (32b) | P_lo_hi (32b) | P_lo_lo (32b) ]

    All individual tensor operations stay within int64 range.

    The rounding bit (1 << total_shift-1) is injected into whichever
    register owns bit (total_shift-1), keeping operands bounded to 32 bits
    so no single addition can overflow int64.
    """
    # 1. Extract sign and work with the magnitude only (simplifies bit math)
    sign = torch.sign(int64_accumulator)
    A = torch.abs(int64_accumulator)  # up to 63 bits, fits int64
    B = int(q_M0)                     # 31-bit scalar, plain Python int

    # 2. Split A into two 32-bit halves
    A_lo = A & 0xFFFFFFFF   # bits  0-31
    A_hi = A >> 32           # bits 32-62  (at most 31 bits wide)

    # 3. Multiply each half by the 31-bit scalar B
    #    A_lo * B <= 0xFFFFFFFF * 0x7FFFFFFF ~ 2^63 - 2^32  -> fits int64
    #    A_hi * B <= 0x7FFFFFFF * 0x7FFFFFFF ~ 2^62          -> fits int64
    P_lo = A_lo * B   # bits  0-62, stored in int64
    P_hi = A_hi * B   # bits  0-61, conceptually sits at offset +32

    # 4. Resolve the 32-bit overlap between the two partial products
    #    P_lo occupies bits 0-62; its top 31 bits (32-62) overlap with P_hi.
    P_lo_upper = P_lo >> 32          # carry/overlap into the P_hi register
    P_lo_lower = P_lo & 0xFFFFFFFF  # bottom 32 bits, clean, fits int64

    # Combine: P_hi_combined now represents bits 32-93 of the full product,
    # stored as a value that fits comfortably in int64 (max ~2^62 + 2^31).
    P_hi_combined = P_hi + P_lo_upper

    # 5. Inject the rounding bit into the correct register.
    #
    #    total_shift = 31 + shift is the position of the binary point in the
    #    128-bit product register layout:
    #
    #      bit position in full product:  0         31        63        95
    #                                     |  P_lo_lo | P_lo_hi | P_hi_lo | ...
    #
    #    The rounding bit sits at position (total_shift - 1).
    #
    #    Case A: (total_shift - 1) >= 32  ->  the rounding bit lives inside
    #            P_hi_combined's domain (offset +32 from the base).
    #            We add 1 << (total_shift - 1 - 32) to P_hi_combined.
    #            Max addition: 1 << 30 (since total_shift <= 63).  Safe.
    #
    #    Case B: (total_shift - 1) < 32   ->  the rounding bit lives inside
    #            P_lo_lower. We add 1 << (total_shift - 1) which is at most
    #            1 << 30.  P_lo_lower <= 0xFFFFFFFF, so result <= ~2^31+2^30.
    #            Safe. Then propagate the carry to P_hi_combined.
    #
    total_shift = 31 + shift
    round_bit_pos = total_shift - 1  # position of the rounding bit

    if round_bit_pos >= 32:
        # Rounding bit falls inside P_hi_combined's register
        P_hi_combined = P_hi_combined + (1 << (round_bit_pos - 32))
    else:
        # Rounding bit falls inside P_lo_lower's register
        P_lo_lower = P_lo_lower + (1 << round_bit_pos)
        carry = P_lo_lower >> 32
        P_hi_combined = P_hi_combined + carry
        P_lo_lower = P_lo_lower & 0xFFFFFFFF

    # 6. Apply the final right-shift across the two-register representation.
    #    The full 96-bit value is: (P_hi_combined << 32) | P_lo_lower
    #    We right-shift by total_shift bits.
    if total_shift >= 32:
        # The entire result comes from P_hi_combined alone
        result = P_hi_combined >> (total_shift - 32)
    else:
        # Result straddles both registers
        result = (P_hi_combined << (32 - total_shift)) | (P_lo_lower >> total_shift)

    return result * sign


# ---------------------------------------------------------
# 2. Quantization / Arithmetic Logic
# ---------------------------------------------------------


def downscale_and_cast(int64_accumulator, q_M0, shift, z_out):
    scaled_accum = multiply_by_quantized_multiplier(int64_accumulator, q_M0, shift)
    q_out = scaled_accum + z_out
    q_out = torch.clamp(q_out, 0, 4294967295)
    return q_out.to(torch.uint32)


def quantize_tensor(tensor, scale, zero_point, dtype=torch.uint32):
    q_tensor = torch.round(tensor / scale) + zero_point
    if dtype == torch.uint32:
        q_tensor = torch.clamp(q_tensor, 0, 4294967295)
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


def integer_add(q1, z1, scale1, q2, z2, scale2, z_out, scale_out):
    x1 = q1.to(torch.int64) - z1
    x2 = q2.to(torch.int64) - z2

    M0_1, shift_1 = compute_integer_multiplier(scale1, 1.0, scale_out)
    M0_2, shift_2 = compute_integer_multiplier(scale2, 1.0, scale_out)

    val1 = multiply_by_quantized_multiplier(x1, M0_1, shift_1)
    val2 = multiply_by_quantized_multiplier(x2, M0_2, shift_2)

    q_out = val1 + val2 + z_out
    return torch.clamp(q_out, 0, 4294967295).to(torch.uint32)


def integer_global_avg_pool2d(q_in, z_in, scale_in, z_out, scale_out):
    N = q_in.size(2) * q_in.size(3)
    x = q_in.to(torch.int64) - z_in
    accum = x.sum(dim=(2, 3), keepdim=True)

    M0, shift = compute_integer_multiplier(scale_in, 1.0, scale_out * N)
    pooled = multiply_by_quantized_multiplier(accum, M0, shift)

    q_out = pooled + z_out
    return torch.clamp(q_out, 0, 4294967295).to(torch.uint32)


def quantized_relu(q_tensor, z_out):
    q_int64 = q_tensor.to(torch.int64)
    clamped = torch.clamp(q_int64, min=z_out)
    return clamped.to(torch.uint32)