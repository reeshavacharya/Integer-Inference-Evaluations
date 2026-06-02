import torch

INT32_MIN = -2147483648
INT32_MAX = 2147483647


def _as_int32(value, device):
    return torch.as_tensor(value, dtype=torch.int32, device=device)

def strict_integer_conv2d(q_x, q_w, z_x, z_w, stride=1, padding=0, clamp=True):
    z_x_32 = _as_int32(z_x, q_x.device)
    z_w_32 = _as_int32(z_w, q_w.device)
    if clamp:
        x = torch.clamp(q_x.to(torch.int64) - z_x_32.to(torch.int64), INT32_MIN, INT32_MAX).to(torch.int32)
        w = torch.clamp(q_w.to(torch.int64) - z_w_32.to(torch.int64), INT32_MIN, INT32_MAX).to(torch.int32)
    else:
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
            
            if clamp:
                # Saturation Math (prevent overflow by clamping accumulation)
                accum_64 = accum.to(torch.int64) + prod.sum(dim=2).to(torch.int64)
                accum = torch.clamp(accum_64, INT32_MIN, INT32_MAX).to(torch.int32)
            else:
                # Modulo Math (native 32-bit arithmetic with wrapping)
                accum = accum + prod.sum(dim=2, dtype=torch.int32)

    return accum


def strict_integer_linear(q_x, q_w, z_x, z_w, clamp=True):
    z_x_32 = _as_int32(z_x, q_x.device)
    z_w_32 = _as_int32(z_w, q_w.device)
    if clamp:
        x = torch.clamp(q_x.to(torch.int64) - z_x_32.to(torch.int64), INT32_MIN, INT32_MAX).to(torch.int32)
        w = torch.clamp(q_w.to(torch.int64) - z_w_32.to(torch.int64), INT32_MIN, INT32_MAX).to(torch.int32)
    else:
        x = (q_x.to(torch.int32) - z_x_32).to(torch.int32)
        w = (q_w.to(torch.int32) - z_w_32).to(torch.int32)

    B, in_f = x.shape
    out_f = w.shape[0]
    accum = torch.zeros((B, out_f), dtype=torch.int32, device=x.device)

    for i in range(in_f):
        x_val = x[:, i:i+1]
        w_val = w[:, i:i+1].transpose(0, 1)

        prod = x_val * w_val
        
        if clamp:
            # Saturation Math
            accum_64 = accum.to(torch.int64) + prod.to(torch.int64)
            accum = torch.clamp(accum_64, INT32_MIN, INT32_MAX).to(torch.int32)
        else:
            # Modulo Math
            accum = accum + prod.to(torch.int32)

    return accum