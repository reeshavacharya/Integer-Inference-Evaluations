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
    return scale_bias, 0

def compute_integer_multiplier(scale_w, scale_x, scale_out):
    M = (scale_w * scale_x) / scale_out
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
    x_int = q_x.to(torch.int64) - z_x
    w_int = q_w.to(torch.int64) - z_w
    return F.conv2d(x_int, w_int, stride=stride, padding=padding)

def integer_linear(q_x, q_w, z_x, z_w):
    x_int = q_x.to(torch.int64) - z_x
    w_int = q_w.to(torch.int64) - z_w
    return F.linear(x_int, w_int)

def add_bias(int64_accumulator, q_bias):
    bias_int64 = q_bias.to(torch.int64)
    if int64_accumulator.dim() == 4:
        bias_int64 = bias_int64.view(1, -1, 1, 1)
    return int64_accumulator + bias_int64

def downscale_and_cast(int64_accum, q_M0, shift, z_out):
    accum_float = int64_accum.to(torch.float64)
    M = (q_M0 / (1 << 31)) * (2.0 ** -shift)
    
    q_out = torch.round(accum_float * M) + z_out
    q_out = torch.clamp(q_out, 0, 4294967295)
    return q_out.to(torch.uint32)

def quantized_relu(q_tensor, z_out):
    q_tensor_int64 = q_tensor.to(torch.int64)
    z_out_tensor = torch.tensor(z_out, dtype=torch.int64, device=q_tensor.device)
    q_out_int64 = torch.max(q_tensor_int64, z_out_tensor)
    return q_out_int64.to(torch.uint32)

def integer_max_pool2d(q_x, kernel_size=2, stride=2):
    # Safe upcast for uint32 max pooling
    q_x_int64 = q_x.to(torch.int64)
    q_out = F.max_pool2d(q_x_int64.float(), kernel_size=kernel_size, stride=stride).to(torch.int64)
    return q_out.to(torch.uint32)

def integer_adaptive_avg_pool(q_x, z_in, s_in, z_out, s_out, output_size=(7, 7)):
    q_x_int64 = q_x.to(torch.int64)
    avg_float = F.adaptive_avg_pool2d(q_x_int64.float(), output_size)
    
    # Mathematical rescaling bridging features to classifier
    real_val = s_in * (avg_float - z_in)
    q_out = torch.round(real_val / s_out) + z_out
    q_out = torch.clamp(q_out, 0, 4294967295)
    return q_out.to(torch.uint32)