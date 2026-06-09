import torch

INT32_MIN = -2147483648
INT32_MAX = 2147483647

def _as_int32(value, device):
    return torch.as_tensor(value, dtype=torch.int32, device=device)

def strict_integer_conv2d(q_x, q_w, z_x, z_w, stride=1, padding=0):
    z_x_32 = _as_int32(z_x, q_x.device)
    z_w_32 = _as_int32(z_w, q_w.device)
    
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
            
            for c_out in range(C_out):
                w_c = w[c_out, :, kh, kw].view(1, C_in, 1, 1)
                prod_sum = (x_slice * w_c).sum(dim=1, dtype=torch.int32)
                accum[:, c_out, :, :] = (accum[:, c_out, :, :] + prod_sum).to(torch.int32)

    return accum

def strict_integer_conv_transpose2d(q_x, q_w, z_x, z_w, stride=1, padding=0, output_padding=0):
    z_x_32 = _as_int32(z_x, q_x.device)
    z_w_32 = _as_int32(z_w, q_w.device)
    
    x = (q_x.to(torch.int32) - z_x_32).to(torch.int32)
    w = (q_w.to(torch.int32) - z_w_32).to(torch.int32)

    B, C_in, H, W = x.shape
    C_in_w, C_out, kH, kW = w.shape

    out_H = (H - 1) * stride - 2 * padding + kH + output_padding
    out_W = (W - 1) * stride - 2 * padding + kW + output_padding

    accum = torch.zeros((B, C_out, out_H + 2 * padding, out_W + 2 * padding), dtype=torch.int32, device=x.device)

    for kh in range(kH):
        for kw in range(kW):
            x_slice = x
            
            h_start = kh
            h_end = kh + H * stride
            w_start = kw
            w_end = kw + W * stride
            
            for c_out in range(C_out):
                w_c = w[:, c_out, kh, kw].view(1, C_in_w, 1, 1)
                prod_sum = (x_slice * w_c).sum(dim=1, dtype=torch.int32)
                
                accum_slice = accum[:, c_out, h_start:h_end:stride, w_start:w_end:stride]
                accum[:, c_out, h_start:h_end:stride, w_start:w_end:stride] = (accum_slice + prod_sum).to(torch.int32)

    if padding > 0:
        accum = accum[:, :, padding:-padding, padding:-padding]

    return accum
