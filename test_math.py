import torch
import math

def compute_integer_multiplier(scale_w, scale_x, scale_out):
    M = (scale_w * scale_x) / scale_out
    m0, exponent = math.frexp(M)
    shift = -exponent
    q_M0 = round(m0 * (1 << 31))
    return q_M0, shift

def multiply_by_quantized_multiplier(A, q_M0, shift):
    A = torch.tensor([A], dtype=torch.int32)
    B = torch.tensor([q_M0], dtype=torch.int32)
    A_lo = A & 0xFFFF; A_hi = (A >> 16) & 0xFFFF
    B_lo = B & 0xFFFF; B_hi = (B >> 16) & 0xFFFF
    P0 = A_lo * B_lo; P1 = A_lo * B_hi; P2 = A_hi * B_lo; P3 = A_hi * B_hi
    R0 = P0 & 0xFFFF; carry1 = (P0 >> 16) & 0xFFFF
    S1 = carry1 + (P1 & 0xFFFF) + (P2 & 0xFFFF)
    R1 = S1 & 0xFFFF; carry2 = S1 >> 16
    S2 = carry2 + ((P1 >> 16) & 0xFFFF) + ((P2 >> 16) & 0xFFFF) + (P3 & 0xFFFF)
    R2 = S2 & 0xFFFF; carry3 = S2 >> 16
    S3 = carry3 + ((P3 >> 16) & 0xFFFF)
    R3 = S3 & 0xFFFF
    R_lo = R0 | (R1 << 16); R_hi = R2 | (R3 << 16)
    is_A_neg = (A < 0).to(torch.int32); is_B_neg = (B < 0).to(torch.int32)
    R_hi = R_hi - (is_A_neg * B); R_hi = R_hi - (is_B_neg * A)
    shift_tensor = torch.tensor([shift], dtype=torch.int32)
    total_shift = (31 + shift_tensor).to(torch.int32)
    round_shift = torch.clamp(total_shift - 1, min=0)
    one = torch.tensor(1, dtype=torch.int32); zero = torch.tensor(0, dtype=torch.int32)
    round_lo = torch.where(total_shift > 0, torch.where(round_shift < 32, one << round_shift, zero), zero)
    round_hi = torch.where(total_shift > 0, torch.where(round_shift >= 32, one << (round_shift - 32), zero), zero)
    c_round = ((R_lo & 0xFFFF) + (round_lo & 0xFFFF)) >> 16
    carry_round = (((R_lo >> 16) & 0xFFFF) + ((round_lo >> 16) & 0xFFFF) + c_round) >> 16
    R_lo = R_lo + round_lo; R_hi = R_hi + round_hi + carry_round
    shift_val = torch.clamp(total_shift, min=0)
    s_ge_32 = shift_val >= 32
    shift_hi = torch.clamp(shift_val - 32, min=0)
    shift_lo = torch.clamp(shift_val, max=31)
    thirty_two = torch.tensor(32, dtype=torch.int32)
    k = shift_lo
    mask = torch.where(k > 0, ~(torch.tensor(-1, dtype=torch.int32) << (thirty_two - k)), torch.tensor(-1, dtype=torch.int32))
    R_lo_part = (R_lo >> k) & mask
    R_hi_part = torch.where(k > 0, R_hi << (thirty_two - k), zero)
    res_lt_32 = R_hi_part | R_lo_part
    res_ge_32 = R_hi >> shift_hi
    return torch.where(s_ge_32, res_ge_32, res_lt_32).item()

A_val = 10000
M_float = 0.005
q_M0, shift = compute_integer_multiplier(1.0, M_float, 1.0)
ans = multiply_by_quantized_multiplier(A_val, q_M0, shift)
print(f"Int32 math: {ans}")
print(f"Float math: {A_val * M_float}")

A_val = -10000
ans = multiply_by_quantized_multiplier(A_val, q_M0, shift)
print(f"Int32 math: {ans}")
print(f"Float math: {A_val * M_float}")
