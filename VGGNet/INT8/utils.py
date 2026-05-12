import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Parameter Calculation
# ---------------------------------------------------------
def get_quantization_params(tensor, num_bits=8):
    """Calculates Absolute Min-Max scale (S) and zero-point (Z)."""
    q_min, q_max = 0, (2**num_bits) - 1
    min_val, max_val = tensor.min().item(), tensor.max().item()
    
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)
    
    scale = (max_val - min_val) / (q_max - q_min)
    if scale == 0: scale = 1e-8
        
    zero_point = round(q_min - (min_val / scale))
    zero_point = max(q_min, min(q_max, zero_point))
    
    return scale, zero_point

def get_bias_quantization_params(scale_w, scale_in):
    """Bias scale is strictly scale_w * scale_in. ZP is 0 for int32."""
    scale_bias = scale_w * scale_in
    return scale_bias, 0

def compute_integer_multiplier(scale_w, scale_in, scale_out):
    """Calculates the M0 and shift values for 32-bit accumulation downscaling."""
    M = (scale_w * scale_in) / scale_out
    if M <= 0:
        return 0, 0
    
    shift = 0
    while M < 0.5:
        M *= 2
        shift += 1
        
    q_M0 = round(M * (2**31))
    return q_M0, shift

def quantize_tensor(tensor, scale, zero_point, dtype=torch.uint8):
    """Casts floating point tensor to quantized integer format."""
    q_tensor = torch.round(tensor / scale) + zero_point
    if dtype == torch.uint8:
        q_tensor = torch.clamp(q_tensor, 0, 255)
    elif dtype == torch.int32:
        q_tensor = torch.clamp(q_tensor, -2147483648, 2147483647)
    return q_tensor.to(dtype)

# ---------------------------------------------------------
# 2. Integer Execution Functions
# ---------------------------------------------------------
def integer_conv2d(q_x, q_w, z_x, z_w, stride=1, padding=1):
    """Executes integer convolution accumulating in int32."""
    x_int = q_x.to(torch.int32) - z_x
    w_int = q_w.to(torch.int32) - z_w
    return F.conv2d(x_int, w_int, stride=stride, padding=padding)

def integer_linear(q_x, q_w, z_x, z_w):
    """Executes integer matrix multiplication accumulating in int32."""
    x_int = q_x.to(torch.int32) - z_x
    w_int = q_w.to(torch.int32) - z_w
    return F.linear(x_int, w_int)

def add_bias(int32_accumulator, q_bias):
    """Adds the 32-bit integer bias vector to the accumulator."""
    bias_int32 = q_bias.to(torch.int32)
    if int32_accumulator.dim() == 4:
        bias_int32 = bias_int32.view(1, -1, 1, 1)
    return int32_accumulator + bias_int32

def downscale_and_cast(int32_accum, q_M0, shift, zp_out):
    """Applies the bit-shift downscaling to return the tensor to 8-bit limits."""
    int64_accum = int32_accum.to(torch.int64)
    scaled = (int64_accum * q_M0) >> 31
    shifted = scaled >> shift
    
    q_out = shifted + zp_out
    q_out = torch.clamp(q_out, 0, 255)
    return q_out.to(torch.uint8)

def quantized_relu(q_x, z_x):
    """Integer ReLU simply clamps anything below the zero-point."""
    return torch.clamp(q_x, min=z_x)

def integer_max_pool2d(q_x, kernel_size=2, stride=2):
    """Max pooling runs flawlessly on uint8 with no scaling math."""
    # Convert uint8 to float temporarily JUST so PyTorch's max_pool2d accepts it, 
    # but no mathematical scaling is happening. Returns pure uint8.
    return F.max_pool2d(q_x.to(torch.float32), kernel_size=kernel_size, stride=stride).to(torch.uint8)

def integer_adaptive_avg_pool(q_x, z_in, s_in, z_out, s_out, output_size=(7,7)):
    """Handles VGG's bridging layer between features and classifier."""
    x_fp = s_in * (q_x.to(torch.float32) - z_in)
    pool_fp = F.adaptive_avg_pool2d(x_fp, output_size)
    return quantize_tensor(pool_fp, s_out, z_out, torch.uint8)