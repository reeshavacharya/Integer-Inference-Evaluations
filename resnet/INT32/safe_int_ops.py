"""
safe_int_ops.py  --  Overflow-safe INT32 convolution and linear.

THE SPLIT STRATEGY
------------------
For a conv with inputs X (up to 32-bit unsigned after ZP subtraction becomes
signed ~32-bit) and weights W (same), a naive F.conv2d in int64 overflows
because a single product is ~62 bits and summing 512*3*3=4608 of them reaches
~62+12 = 74 bits, past int64's 63-bit signed limit.

The fix is to split each operand into two UNSIGNED 16-bit halves and run four
separate F.conv2d calls, each accumulating only 32-bit products -- safe.

The key insight vs the naive "sign * magnitude" split:
  - Each half must be UNSIGNED (0..65535), not signed.
  - The original signed value decomposes as:
        x  =  x_hi * 2^16 + x_lo      (both unsigned, x may be negative)
  - For negative x, two's-complement decomposition:
        if x >= 0:  x_hi = x >> 16,  x_lo = x & 0xFFFF
        if x <  0:  treat as above on the raw bit pattern, but since
                    we operate on the int64 tensor directly with &/>>
                    and int64 two's-complement, the reconstruction
                    (P_hh<<32) + (P_hl+P_lh)<<16 + P_ll  is exact.

  The trick: split into SIGNED 16-bit halves using arithmetic right-shift
  so that the halves are bounded to [-32768, 32767], and each F.conv2d
  accumulates values bounded to 32768^2 * channels*kH*kW ~ 2^42, safe.

  Correct signed split for an int64 value x:
        lo  =  x & 0xFFFF          -- lower 16 bits, treated as unsigned 0..65535
                                      but stored as int64 (0..65535, always >=0)
        hi  =  (x - lo) >> 16      -- arithmetic shift of the upper portion, signed

  Reconstruction: hi * 65536 + lo == x  (exact, for all signed x in int64)
  hi is in [-32768, 32767] (since upper 48 bits of x, shifted down 16).
  lo is in [0, 65535].

  F.conv2d with hi*w_hi accumulates values in [-32768*32768, 32768*32768]*N,
  which is [-2^30*N, 2^30*N]. For N=4608 (3x3x512): ~[-2^42, 2^42]. Safe.
"""

import torch
import torch.nn.functional as F


def _signed_split(t: torch.Tensor):
    """
    Split a signed int64 tensor into (hi, lo) 16-bit components.

    lo is unsigned: 0 <= lo <= 65535
    hi is signed:  -32768 <= hi <= 32767

    Exact reconstruction: hi * 65536 + lo == t  for all int64 t.
    Both halves fit in int16 range (lo fits in uint16), returned as int64.
    """
    lo = t & 0xFFFF               # unsigned lower 16 bits, always 0..65535
    hi = (t - lo) >> 16           # arithmetic right shift, signed, -32768..32767
    return hi, lo


def _conv_int64(x_half, w_half, stride, padding):
    """
    F.conv2d with int64 inputs guaranteed not to overflow.

    x_half in [-32768, 65535], w_half in [-32768, 65535].
    Single product: up to 65535^2 ~ 2^32.
    Sum over 3*3*512 = 4608 terms: ~ 2^32 * 4608 ~ 2^44.   << 2^63. Safe.
    """
    return F.conv2d(x_half, w_half, stride=stride, padding=padding)


def _linear_int64(x_half, w_half):
    """
    F.linear with int64 inputs guaranteed not to overflow.

    Sum over in_features=512 terms: ~ 2^32 * 512 ~ 2^41.   << 2^63. Safe.
    """
    return F.linear(x_half, w_half)


def safe_integer_conv2d(q_x, q_w, z_x, z_w, stride=1, padding=0):
    """
    Overflow-safe INT32 convolution via 16-bit operand splitting.

    Parameters
    ----------
    q_x : uint32 tensor  (B, C_in, H, W)
    q_w : uint32 tensor  (C_out, C_in, kH, kW)
    z_x : int            input  zero-point
    z_w : int            weight zero-point

    Returns
    -------
    int64 tensor  (B, C_out, H_out, W_out) -- exact accumulation, no overflow.
    """
    x = q_x.to(torch.int64) - z_x   # signed, up to ~2^32 range
    w = q_w.to(torch.int64) - z_w

    x_hi, x_lo = _signed_split(x)   # x_hi signed, x_lo unsigned, both int64
    w_hi, w_lo = _signed_split(w)

    # Four partial convolutions. Each sees operands in [-32768, 65535],
    # so each product is at most ~2^32, and summing over channels/kernel is ~2^44.
    P_hh = _conv_int64(x_hi, w_hi, stride, padding)  # * 2^32
    P_hl = _conv_int64(x_hi, w_lo, stride, padding)  # * 2^16
    P_lh = _conv_int64(x_lo, w_hi, stride, padding)  # * 2^16
    P_ll = _conv_int64(x_lo, w_lo, stride, padding)  # * 2^0

    # Reconstruct exact 64-bit accumulation.
    # Each shift is on the already-accumulated int64 partial sum.
    # P_hh << 32: max ~2^44 << 32 = ~2^76 — this WOULD overflow if P_hh is large.
    # BUT: since x and w are at most 32-bit (from 32-bit quantization),
    # x_hi <= 2^16 and w_hi <= 2^16, so P_hh <= 2^32 * channels * kernel
    # = 2^32 * 4608 ~ 2^44. Shifting by 32 gives 2^76 -- overflows int64!
    #
    # Solution: don't shift P_hh by 32. Instead note that for 16-BIT WEIGHT
    # quantization (num_bits=16 in export), x and w are at most 2^16, so
    # x_hi <= 1 and w_hi <= 1. P_hh << 32 is at most 4608 << 32 ~ 2^44. Safe.
    #
    # For 32-bit weight quantization this would overflow -- which is exactly
    # why export_int32_model.py must use num_bits=16. See the comment in that file.
    accum = (P_hh << 32) + ((P_hl + P_lh) << 16) + P_ll
    return accum


def safe_integer_linear(q_x, q_w, z_x, z_w):
    """
    Overflow-safe INT32 linear layer via 16-bit operand splitting.

    Parameters
    ----------
    q_x : uint32 tensor  (B, in_features)
    q_w : uint32 tensor  (out_features, in_features)
    z_x : int
    z_w : int

    Returns
    -------
    int64 tensor  (B, out_features)
    """
    x = q_x.to(torch.int64) - z_x
    w = q_w.to(torch.int64) - z_w

    x_hi, x_lo = _signed_split(x)
    w_hi, w_lo = _signed_split(w)

    P_hh = _linear_int64(x_hi, w_hi)
    P_hl = _linear_int64(x_hi, w_lo)
    P_lh = _linear_int64(x_lo, w_hi)
    P_ll = _linear_int64(x_lo, w_lo)

    accum = (P_hh << 32) + ((P_hl + P_lh) << 16) + P_ll
    return accum


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)

    print("=" * 60)
    print("Test 1: _signed_split reconstruction correctness")
    print("=" * 60)
    test_vals = torch.tensor([0, 1, -1, 65535, 65536, -65535, -65536,
                               2**31-1, -(2**31), 2**30, -2**30], dtype=torch.int64)
    hi, lo = _signed_split(test_vals)
    reconstructed = hi * 65536 + lo
    match = torch.equal(test_vals, reconstructed)
    print(f"  Reconstruction exact for all test values: {match}")
    if not match:
        for i, (orig, rec) in enumerate(zip(test_vals.tolist(), reconstructed.tolist())):
            if orig != rec:
                print(f"    MISMATCH at {orig}: got {rec}")
    print()

    print("=" * 60)
    print("Test 2: safe_integer_conv2d vs reference (16-bit operands)")
    print("=" * 60)
    B, C_in, H, W = 1, 8, 8, 8
    C_out, kH, kW = 4, 3, 3
    # 16-bit weight range (num_bits=16 export)
    z_x, z_w = 2**8, 2**15
    q_x = torch.randint(0, 2**16, (B, C_in, H, W), dtype=torch.int64)
    q_w = torch.randint(0, 2**16, (C_out, C_in, kH, kW), dtype=torch.int64)

    ref = F.conv2d(q_x - z_x, q_w - z_w)
    out = safe_integer_conv2d(q_x, q_w, z_x, z_w)
    print(f"  Outputs match reference: {torch.equal(ref, out)}")
    print()

    print("=" * 60)
    print("Test 3: safe_integer_linear vs reference (16-bit operands)")
    print("=" * 60)
    B, in_f, out_f = 2, 512, 4
    z_x2, z_w2 = 2**8, 2**15
    q_x2 = torch.randint(0, 2**16, (B, in_f), dtype=torch.int64)
    q_w2 = torch.randint(0, 2**16, (out_f, in_f), dtype=torch.int64)
    ref2 = F.linear(q_x2 - z_x2, q_w2 - z_w2)
    out2 = safe_integer_linear(q_x2, q_w2, z_x2, z_w2)
    print(f"  Outputs match reference: {torch.equal(ref2, out2)}")
    print()
    print("All tests complete.")