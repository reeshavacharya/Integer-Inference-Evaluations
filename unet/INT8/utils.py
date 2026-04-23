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

def compute_requantize_multiplier(scale_in, scale_target):
    """
    Offline calculation: Converts the floating-point ratio (S_in / S_target) 
    into a 32-bit integer M0 and a bit-shift.
    """
    # This is mathematically identical to your compute_integer_multiplier, 
    # just without the scale_w component.
    M_float = float(scale_in / scale_target)
    
    # Break into fractional part and exponent
    import math
    fraction, exponent = math.frexp(M_float)
    
    # Scale fraction up to int32 bounds
    M0 = int(round(fraction * (2 ** 31)))
    shift = 31 - exponent
    
    return M0, shift

def requantize_tensor(q_tensor, zp_in, zp_target, M0, shift):
    """
    Online calculation: Requantizes a tensor to a new scale using 100% integer math.
    """
    # 1. Cast to int64 to prevent overflow during the M0 multiplication!
    q_int64 = q_tensor.to(torch.int64) - zp_in
    
    # 2. Apply the M0 multiplier and rounding bit-shift
    q_scaled = (q_int64 * M0 + (1 << (shift - 1))) >> shift
    
    # 3. Add target zero point and safely clamp back to uint8
    q_out = torch.clamp(q_scaled + zp_target, 0, 255).to(torch.uint8)
    
    return q_out

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

def integer_conv_transpose2d(q_x, w_q, Z_in, Z_w, stride=2, padding=0, output_padding=0):
    """
    100% pure integer manual accumulation for Transposed 2D Convolution.
    Bypasses PyTorch's missing F.conv_transpose2d int32 backend entirely.
    """
    # 1. Normalize tuple/int arguments into separate Height and Width components
    stride_h = stride[0] if isinstance(stride, tuple) else stride
    stride_w = stride[1] if isinstance(stride, tuple) else stride
    
    pad_h = padding[0] if isinstance(padding, tuple) else padding
    pad_w = padding[1] if isinstance(padding, tuple) else padding
    
    out_pad_h = output_padding[0] if isinstance(output_padding, tuple) else output_padding
    out_pad_w = output_padding[1] if isinstance(output_padding, tuple) else output_padding

    # 2. Shift by Zero-Points and cast to int32
    x_int32 = q_x.to(torch.int32) - Z_in
    w_int32 = w_q.to(torch.int32) - Z_w
    
    B, in_C, H, W = x_int32.shape
    in_C_w, out_C, kH, kW = w_int32.shape
    
    # 3. Calculate the raw "canvas" dimensions before padding is cropped
    padded_H = (H - 1) * stride_h + kH
    padded_W = (W - 1) * stride_w + kW
    
    # 4. Create the blank int32 accumulator canvas
    device = x_int32.device
    acc_padded = torch.zeros((B, out_C, padded_H, padded_W), dtype=torch.int32, device=device)
    
    # Flatten spatial dimensions of the input for efficient matmul
    x_flat = x_int32.view(B, in_C, H * W).permute(0, 2, 1) 
    
    # 5. The Core Math: Iterate over the kernel's spatial dimensions
    for kh in range(kH):
        for kw in range(kW):
            w_slice = w_int32[:, :, kh, kw]
            dot_product = torch.matmul(x_flat, w_slice)
            dot_spatial = dot_product.permute(0, 2, 1).view(B, out_C, H, W)
            
            # Add the result to our canvas at the exact strided locations
            acc_padded[:, :, kh : kh + H*stride_h : stride_h, kw : kw + W*stride_w : stride_w] += dot_spatial
            
    # 6. Crop off the overlapping padding to match PyTorch output dimensions
    # If pad is 0, the slice goes up to the end.
    end_h = padded_H - pad_h if pad_h > 0 else padded_H
    end_w = padded_W - pad_w if pad_w > 0 else padded_W
    acc_cropped = acc_padded[:, :, pad_h : end_h, pad_w : end_w]
    
    # 7. Apply output_padding
    if out_pad_h > 0 or out_pad_w > 0:
        # F.pad format: (padding_left, padding_right, padding_top, padding_bottom)
        acc_cropped = F.pad(acc_cropped, (0, out_pad_w, 0, out_pad_h), value=0)
        
    return acc_cropped

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