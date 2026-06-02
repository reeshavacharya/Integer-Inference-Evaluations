import os
import torch
import torch.nn.functional as F

def generate_q15_16_gelu_lut():
    print("Generating Q15.16 GELU Lookup Table...")
    
    # 1. Define Q15.16 scaling factor and bounds for [-10.0, 10.0]
    F_BITS = 16
    SCALE = 1 << F_BITS
    q_min = int(-10.0 * SCALE)  # -655360
    q_max = int(10.0 * SCALE)   #  655360

    # 2. Create the exact integer grid
    q_inputs = torch.arange(q_min, q_max + 1, dtype=torch.float64)

    # 3. Perform floating-point GELU offline
    f_inputs = q_inputs / SCALE
    f_outputs = F.gelu(f_inputs)
    f_outputs = torch.clamp(f_outputs, min=-10.0, max=10.0)

    # 4. Requantize back to strict Q15.16 integers
    q_outputs = torch.round(f_outputs * SCALE).to(torch.int32)

    # 5. Save the 1D Tensor to disk
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lut_path = os.path.join(this_dir, "gelu_q15_16_lut.pt")
    
    # Save the bounds alongside the table so the inference script knows how to index it
    torch.save({
        "lut": q_outputs,
        "q_min": q_min,
        "q_max": q_max
    }, lut_path)
    
    print(f"LUT successfully saved to: {lut_path}")
    print(f"Total entries: {len(q_outputs)} (Approx {q_outputs.element_size() * q_outputs.nelement() / 1e6:.2f} MB)")

if __name__ == "__main__":
    generate_q15_16_gelu_lut()