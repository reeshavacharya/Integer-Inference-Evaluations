import torch

INT32_MIN = -2147483648
INT32_MAX = 2147483647


def _as_int32(value, device):
    return torch.as_tensor(value, dtype=torch.int32, device=device)

def strict_integer_conv2d(q_x, q_w, z_x, z_w, stride=1, padding=0):
    z_x_32 = _as_int32(z_x, q_x.device)
    z_w_32 = _as_int32(z_w, q_w.device)
    # Modulo Math (native 32-bit arithmetic with wrapping)
    x = (q_x.to(torch.int32) - z_x_32).to(torch.int32)
    w = (q_w.to(torch.int32) - z_w_32).to(torch.int32)

    B, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape

    if padding > 0:
        x = torch.nn.functional.pad(x, (padding, padding, padding, padding))

    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1

    accum = torch.zeros((B, C_out, out_H, out_W), dtype=torch.int32, device=x.device)

    for kh in range(kH):
        h_start = kh
        h_end = h_start + out_H * stride
        for kw in range(kW):
            w_start = kw
            w_end = w_start + out_W * stride

            x_slice = x[:, :, h_start:h_end:stride, w_start:w_end:stride]
            w_slice = w[:, :, kh, kw].view(C_out, C_in, 1, 1)
            
            prod = x_slice.unsqueeze(1) * w_slice
            
            # Modulo Math (native 32-bit arithmetic with wrapping)
            accum = (accum + prod.sum(dim=2, dtype=torch.int32)).to(torch.int32)

    return accum


def strict_integer_linear(q_x, q_w, z_x, z_w):
    z_x_32 = _as_int32(z_x, q_x.device)
    z_w_32 = _as_int32(z_w, q_w.device)
    # Modulo Math (native 32-bit arithmetic with wrapping)
    x = (q_x.to(torch.int32) - z_x_32).to(torch.int32)
    w = (q_w.to(torch.int32) - z_w_32).to(torch.int32)

    B, in_f = x.shape
    out_f = w.shape[0]
    accum = torch.zeros((B, out_f), dtype=torch.int32, device=x.device)

    for i in range(in_f):
        x_val = x[:, i:i+1]
        w_val = w[:, i:i+1].transpose(0, 1)

        prod = x_val * w_val
        
        # Modulo Math
        accum = (accum + prod.to(torch.int32)).to(torch.int32)

    return accum


def strict_integer_max_pool2d(q_x, kernel_size=2, stride=2):
    """
    Pure 32-bit integer Max Pooling logic.
    Using purely view/reshaping and max, completely avoiding F.max_pool2d floating-point operations.
    """
    q_x_32 = q_x.to(torch.int32)
    B, C, H, W = q_x_32.shape
    assert H % kernel_size == 0 and W % kernel_size == 0, "Spatial dimensions must be perfectly divisible by kernel_size"
    assert kernel_size == 2 and stride == 2, "Only 2x2 stride 2 Max Pooling is strictly supported"
    
    # Dynamic reshape logic for sliding window 2x2:
    # 1. Split H into (H // 2, 2)
    # 2. Split W into (W // 2, 2)
    reshaped = q_x_32.view(B, C, H // kernel_size, kernel_size, W // kernel_size, kernel_size)
    
    # 3. Collapse the window dimensions (dim 3 and 5)
    return reshaped.amax(dim=(3, 5))
