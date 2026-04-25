import math

import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Parameter Calculation (Scale and Zero-Point)
# ---------------------------------------------------------

def get_quantization_params(tensor, num_bits=8):
    """
    Calculates the scale (S) and zero-point (Z) for a given float tensor.
    Maps to the [0, 255] range for 8-bit unsigned integers.
    """
    q_min, q_max = 0, (2**num_bits) - 1
    
    # Get actual min and max
    min_val, max_val = tensor.min().item(), tensor.max().item()
    
    # Nudge boundaries so that 0.0 is exactly representable [cite: 226]
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)
    
    # Calculate scale (S) [cite: 209, 217]
    scale = (max_val - min_val) / (q_max - q_min)
    if scale == 0:
        scale = 1e-8 # Prevent division by zero
        
    # Calculate zero-point (Z) and clamp to uint8 range [cite: 209-217]
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
    """
    Strict integer-only addition: q_out = Z_out + M1*(q1 - Z1) + M2*(q2 - Z2)
    Uses fixed-point multipliers to align scales without floating point math.
    """
    # 1. Shift inputs to zero-centered int32
    x1 = q1.to(torch.int32) - z1
    x2 = q2.to(torch.int32) - z2
    
    # 2. Get fixed-point multipliers for each branch
    M0_1, shift_1 = quantize_multiplier(scale1 / scale_out)
    M0_2, shift_2 = quantize_multiplier(scale2 / scale_out)
    
    # 3. Simulate hardware integer multiply-and-shift
    # Hardware logic: res = (x * M0) >> shift
    val1 = torch.round((x1.double() * M0_1) / (2 ** shift_1)).to(torch.int32)
    val2 = torch.round((x2.double() * M0_2) / (2 ** shift_2)).to(torch.int32)
    
    # 4. Add branches and apply output zero-point
    q_out = val1 + val2 + z_out
    
    # Cast back to uint8
    return torch.clamp(q_out, 0, 255).to(torch.uint8)


def integer_global_avg_pool2d(q_in, z_in, scale_in, z_out, scale_out):
    """
    Integer implementation of global average pooling.
    Accumulates in int32, multiplies by combined (Scale / N) multiplier, then downscales.
    """
    N = q_in.size(2) * q_in.size(3) # e.g., 7x7 spatial window = 49
    
    # 1. Shift to zero-centered int32 and sum all pixels
    x = q_in.to(torch.int32) - z_in
    accum = x.sum(dim=(2, 3), keepdim=True) # Accumulator is now int32
    
    # 2. Compute combined fixed-point multiplier
    # ratio = Scale_in / (Scale_out * N)
    ratio = scale_in / (scale_out * N)
    M0, shift = quantize_multiplier(ratio)
    
    # 3. Simulate integer multiply-and-shift
    pooled = torch.round((accum.double() * M0) / (2 ** shift)).to(torch.int32)
    
    # 4. Apply output zero-point and cast
    q_out = pooled + z_out
    return torch.clamp(q_out, 0, 255).to(torch.uint8)

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

def multiply_by_quantized_multiplier(int32_accumulator, q_M0, shift):
    """
    Executes the down-scaling entirely in integer arithmetic.
    """
    # 1. Cast to 64-bit integer to prevent overflow during multiplication
    accum_64 = int32_accumulator.to(torch.int64)

    # 2. Perform the integer multiplication
    raw_product = accum_64 * q_M0

    # 3. Calculate total right shift.
    # We shift by 31 bits to undo the (2^31) scaling applied to M0, 
    # plus the actual shift parameter 'n'.
    total_shift = 31 + shift

    # 4. Create the rounding factor: 2^(total_shift - 1).
    # This ensures "round-to-nearest" behavior instead of "round-to-zero"[cite: 190].
    rounding_factor = 1 << (total_shift - 1)
    
    # 5. Add rounding factor and perform the bitwise right shift
    result = (raw_product + rounding_factor) >> total_shift

    # Cast back to 32-bit integer
    return result.to(torch.int32)

def downscale_and_cast(int32_accumulator, q_M0, shift, z_out):
    """
    Scales the 32-bit accumulator down to the final scale, adds the output 
    zero-point, and performs a saturating cast to uint8 [cite: 188-191].
    """
    # 1. Pure integer fixed-point multiplication and shift
    scaled_accum = multiply_by_quantized_multiplier(int32_accumulator, q_M0, shift)

    # 2. Add the output zero-point
    q_out = scaled_accum + z_out

    # 3. Saturating cast to uint8 [cite: 191]
    q_out = torch.clamp(q_out, 0, 255)
    return q_out.to(torch.uint8)

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

def quantize_tensor(tensor, scale, zero_point, dtype=torch.uint8):
    """
    Converts a real float tensor (r) to a quantized integer tensor (q).
    q = round(r / S) + Z
    """
    q_tensor = torch.round(tensor / scale) + zero_point
    # Clamp based on dtype
    if dtype == torch.uint8:
        q_tensor = torch.clamp(q_tensor, 0, 255)
    elif dtype == torch.int32:
        # 32-bit clamping bounds
        q_tensor = torch.clamp(q_tensor, -2147483648, 2147483647)
        
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
    """
    Simulates the core integer matrix multiplication / accumulation [cite: 157, 171-173].
    Expects uint8 inputs, casts to int32, subtracts zero-points, and accumulates.
    """
    # Cast to int32 to prevent overflow during multiplication and accumulation
    x_int = q_x.to(torch.int32) - z_x
    w_int = q_w.to(torch.int32) - z_w
    
    # Perform standard linear operation (matmul) accumulating in int32
    return F.linear(x_int, w_int)

def integer_conv2d(q_x, q_w, z_x, z_w, stride=1, padding=0):
    """
    Simulates integer convolution. 
    Expects uint8 inputs, casts to int32, subtracts zero-points, and accumulates.
    """
    x_int = q_x.to(torch.int32) - z_x
    w_int = q_w.to(torch.int32) - z_w
    
    return F.conv2d(x_int, w_int, stride=stride, padding=padding)

def add_bias(int32_accumulator, q_bias):
    """
    Adds the 32-bit integer bias vector to the 32-bit accumulator.
    Handles broadcasting for both 4D (Conv2d) and 2D (Linear) accumulators.
    """
    bias_int32 = q_bias.to(torch.int32)
    
    # If the accumulator is from a Conv2d layer (N, C, H, W)
    if int32_accumulator.dim() == 4:
        # Reshape bias from (C,) to (1, C, 1, 1)
        bias_int32 = bias_int32.view(1, -1, 1, 1)
        
    return int32_accumulator + bias_int32

def quantized_relu(q_tensor, z_out):
    """
    Since q_tensor is uint8 [0, 255], an actual zero is represented by z_out.
    To apply ReLU, we clamp any value less than z_out to z_out [cite: 194-195].
    """
    return torch.clamp(q_tensor, min=z_out)