import torch

# =======================================================
# 1. CORE HARDWARE ALU (Pure 32-bit Bitwise Logic)
# =======================================================

def _abs_32bit(x):
    """Computes absolute value using pure 32-bit Two's Complement logic."""
    mask = x >> 31
    return (x + mask) ^ mask

def _mulhu_32bit(a_uns, b_uns):
    """Core Unsigned 32x32 -> 64 Multiply High."""
    a_hi, a_lo = (a_uns >> 16) & 0xFFFF, a_uns & 0xFFFF
    b_hi, b_lo = (b_uns >> 16) & 0xFFFF, b_uns & 0xFFFF

    lo_lo = a_lo * b_lo
    hi_lo = a_hi * b_lo
    lo_hi = a_lo * b_hi
    hi_hi = a_hi * b_hi

    cross = (lo_lo >> 16) + hi_lo
    cross2 = (cross & 0xFFFF) + lo_hi
    
    msb = hi_hi + (cross >> 16) + (cross2 >> 16)
    return msb & 0xFFFFFFFF

a= torch.tensor(3, dtype=torch.uint32)
b= torch.tensor(4294967295, dtype=torch.uint32)
result = _mulhu_32bit(a, b)
print("Unsigned 32x32 -> 64 Multiply High Result:", result)

def _get_bit_length_32bit(x):
    """
    Pure 32-bit integer log2 (Find Highest Set Bit).
    Simulates a hardware Count Leading Zeros (CLZ) circuit.
    """
    x_32 = x.view(torch.int32)
    
    # 1. Bit Smearing: Propagate the highest '1' bit down to all lower bits
    x_32 = x_32 | (x_32 >> 1)
    x_32 = x_32 | (x_32 >> 2)
    x_32 = x_32 | (x_32 >> 4)
    x_32 = x_32 | (x_32 >> 8)
    x_32 = x_32 | (x_32 >> 16)
    
    # 2. Hardware Population Count (Popcount) to find the exact bit-length
    x_32 = x_32 - ((x_32 >> 1) & 0x55555555)
    x_32 = (x_32 & 0x33333333) + ((x_32 >> 2) & 0x33333333)
    x_32 = (x_32 + (x_32 >> 4)) & 0x0F0F0F0F
    x_32 = x_32 + (x_32 >> 8)
    x_32 = x_32 + (x_32 >> 16)
    
    return x_32 & 0x0000003F  # Maximum possible value is 32

def _adaptive_shift_64bit_unsigned(hi, lo):
    """
    Finds the highest active bit and dynamically shifts the magnitude 
    right using ONLY 32-bit integer registers.
    """
    overflow_bits = _get_bit_length_32bit(hi)

    # Safe shifting masks to prevent C++ backend panics on shift >= 32
    safe_shift = torch.where(overflow_bits > 0, overflow_bits - 1, torch.zeros_like(overflow_bits))
    
    logical_mask = torch.where(
        overflow_bits > 0,
        torch.tensor(0x7FFFFFFF, dtype=torch.int32, device=hi.device) >> safe_shift,
        torch.tensor(-1, dtype=torch.int32, device=hi.device)
    )

    safe_lo_shift = torch.where(overflow_bits > 0, overflow_bits, torch.zeros_like(overflow_bits))
    shifted_lo = (lo >> safe_lo_shift) & logical_mask

    safe_hi_shift = torch.where(overflow_bits > 0, 32 - overflow_bits, torch.zeros_like(overflow_bits))
    shifted_hi = torch.where(overflow_bits > 0, hi << safe_hi_shift, torch.zeros_like(hi))

    result = shifted_hi | shifted_lo
    return result.view(torch.int32), overflow_bits


# =======================================================
# 2. ADAPTIVE MULTIPLICATION OPERATIONS
# =======================================================

def adaptive_mul_uint32_uint32(a, b):
    """Unsigned * Unsigned -> Returns uint32"""
    # .view() preserves the raw bits identically without PyTorch clamping
    a_32, b_32 = a.view(torch.int32), b.view(torch.int32)
    
    lo = (a_32 * b_32) & 0xFFFFFFFF
    hi = _mulhu_32bit(a_32, b_32)
    
    mag_signed, dropped = _adaptive_shift_64bit_unsigned(hi, lo)
    
    # Return as native uint32 for PyTorch 2.3+
    return mag_signed.view(torch.uint32), dropped


def adaptive_mul_int32_int32(a, b):
    """Signed * Signed -> Returns int32 (Sign-Magnitude Architecture)"""
    a_32, b_32 = a.view(torch.int32), b.view(torch.int32)
    
    sign_a = a_32 < 0
    sign_b = b_32 < 0
    result_is_negative = sign_a ^ sign_b
    
    abs_a = _abs_32bit(a_32)
    abs_b = _abs_32bit(b_32)
    
    lo = (abs_a * abs_b) & 0xFFFFFFFF
    hi = _mulhu_32bit(abs_a, abs_b)
    
    mag, dropped = _adaptive_shift_64bit_unsigned(hi, lo)
    
    result = torch.where(result_is_negative, -mag, mag)
    return result, dropped


def adaptive_mul_uint32_int32(a_uns, b_sign):
    """Unsigned * Signed -> Returns int32"""
    a_32, b_32 = a_uns.view(torch.int32), b_sign.view(torch.int32)
    
    result_is_negative = b_32 < 0
    abs_b = _abs_32bit(b_32)
    
    # a_32 is already an unsigned magnitude, no abs() needed
    lo = (a_32 * abs_b) & 0xFFFFFFFF
    hi = _mulhu_32bit(a_32, abs_b)
    
    mag, dropped = _adaptive_shift_64bit_unsigned(hi, lo)
    
    result = torch.where(result_is_negative, -mag, mag)
    return result, dropped


# =======================================================
# 3. ADAPTIVE ADDITION / SUBTRACTION OPERATIONS
# =======================================================

def adaptive_add_uint32_uint32(a, b):
    """Unsigned + Unsigned -> Returns uint32"""
    a_32, b_32 = a.view(torch.int32), b.view(torch.int32)
    lsb = a_32 + b_32
    
    # Unsigned overflow: if both MSBs are 1, OR one MSB is 1 and result MSB is 0
    a_msb = a_32 < 0
    b_msb = b_32 < 0
    res_msb = lsb < 0
    overflow = (a_msb & b_msb) | ((a_msb | b_msb) & (~res_msb))
    
    # Logical shift right by 1 for unsigned keeping MSB
    msb = ((a_32 >> 1) & 0x7FFFFFFF) + ((b_32 >> 1) & 0x7FFFFFFF) + (a_32 & b_32 & 1)
    
    result = torch.where(overflow, msb, lsb)
    
    zero = torch.tensor(0, dtype=torch.int32, device=a.device)
    one = torch.tensor(1, dtype=torch.int32, device=a.device)
    dropped = torch.where(overflow, one, zero)
    
    return result.view(torch.uint32), dropped


def adaptive_add_int32_int32(a, b):
    """Signed + Signed -> Returns int32"""
    a_32, b_32 = a.view(torch.int32), b.view(torch.int32)
    lsb = a_32 + b_32
    
    a_neg = a_32 < 0
    b_neg = b_32 < 0
    res_neg = lsb < 0
    
    # Signed overflow: signs match but resulting sign flips
    overflow = (a_neg == b_neg) & (a_neg != res_neg)
    
    # Arithmetic right shift preserves the Two's Complement sign
    msb = (a_32 >> 1) + (b_32 >> 1) + (a_32 & b_32 & 1)
    
    result = torch.where(overflow, msb, lsb)
    
    zero = torch.tensor(0, dtype=torch.int32, device=a.device)
    one = torch.tensor(1, dtype=torch.int32, device=a.device)
    dropped = torch.where(overflow, one, zero)
    
    return result, dropped


def adaptive_sub_uint32_int32(a_uns, b_sign):
    """Unsigned - Signed -> Returns int32 (Zero-point Alignment)"""
    a_32, b_32 = a_uns.view(torch.int32), b_sign.view(torch.int32)
    
    b_negated = -b_32
    lsb = a_32 + b_negated
    
    a_pos = a_32 >= 0
    b_neg_pos = b_negated >= 0
    res_neg = lsb < 0
    
    overflow = (a_pos == b_neg_pos) & (a_pos != (~res_neg))
    
    msb = (a_32 >> 1) + (b_negated >> 1) + (a_32 & b_negated & 1)
    
    result = torch.where(overflow, msb, lsb)
    
    zero = torch.tensor(0, dtype=torch.int32, device=a_uns.device)
    one = torch.tensor(1, dtype=torch.int32, device=a_uns.device)
    dropped = torch.where(overflow, one, zero)
    
    return result, dropped