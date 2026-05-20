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
    elif dtype == torch.int64:
        # Biases get full 64-bit bounds
        q_tensor = torch.clamp(q_tensor, -9223372036854775808, 9223372036854775807)
        
    return q_tensor.to(dtype)

# ---------------------------------------------------------
# 2. Integer Arithmetic Operations
# ---------------------------------------------------------
def integer_conv2d(q_x, q_w, z_x, z_w, stride=1, padding=0):
    # Upcast to int64. The sums here will natively overflow the hardware limits.
    x_int = q_x.to(torch.int64) - z_x
    w_int = q_w.to(torch.int64) - z_w
    return F.conv2d(x_int, w_int, stride=stride, padding=padding)

def integer_conv_transpose2d(q_x, q_w, z_x, z_w, stride=1, padding=0, output_padding=0):
    x_int = q_x.to(torch.int64) - z_x
    w_int = q_w.to(torch.int64) - z_w
    return F.conv_transpose2d(x_int, w_int, stride=stride, padding=padding, output_padding=output_padding)

def add_bias(int64_accumulator, q_bias):
    bias_int64 = q_bias.to(torch.int64)
    if int64_accumulator.dim() == 4:
        bias_int64 = bias_int64.view(1, -1, 1, 1)
    return int64_accumulator + bias_int64

def downscale_and_cast(int64_accum, q_M0, shift, z_out):
    # We execute the scaling in float64 to ensure the ONLY failure point is the MAC accumulator overflow.
    accum_float = int64_accum.to(torch.float64)
    M = (q_M0 / (1 << 31)) * (2.0 ** -shift)
    
    q_out = torch.round(accum_float * M) + z_out
    q_out = torch.clamp(q_out, 0, 4294967295)
    return q_out.to(torch.uint32)

def quantized_relu(q_tensor, z_out):
    # PyTorch lacks native torch.max support for uint32.
    # Upcast to int64 to perform the comparison safely, then cast back.
    q_tensor_int64 = q_tensor.to(torch.int64)
    z_out_tensor = torch.tensor(z_out, dtype=torch.int64, device=q_tensor.device)
    
    q_out_int64 = torch.max(q_tensor_int64, z_out_tensor)
    return q_out_int64.to(torch.uint32)

def requantize_tensor(q_tensor, z_old, z_new, q_M0, shift):
    q_float = q_tensor.to(torch.float64) - z_old
    M = (q_M0 / (1 << 31)) * (2.0 ** -shift)
    
    q_out = torch.round(q_float * M) + z_new
    q_out = torch.clamp(q_out, 0, 4294967295)
    return q_out.to(torch.uint32)