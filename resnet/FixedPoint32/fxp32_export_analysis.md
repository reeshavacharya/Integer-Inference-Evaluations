# FxP32 (Q15.16) Offline Export Pipeline — Feasibility Analysis

---

## 1. Is This Possible?

**Yes — with one critical caveat around the multiplication step.**

The concept is straightforward and mirrors the INT32 pipeline:

| Step | INT32 Pipeline | Proposed FxP32 Pipeline |
|------|---------------|------------------------|
| **Source** | Calibration ranges → scale/zp | Raw FP32 `.pth` model |
| **Export** | `export_int32_model.py` quantizes weights/biases to `uint8` in `uint32` containers | `export_fxp32_model.py` converts weights/biases to Q15.16 integers |
| **Artifact** | `_int32.pth` dict with `q_weight`, `q_bias`, `q_M0`, `shift` | `_fxp32.pth` dict with `q_weight`, `q_bias` (both Q15.16), LUTs |
| **Inference** | Load dict, run pure int32 forward pass | Load dict, run pure int32 forward pass |

The export step is actually **simpler** than INT32 because:
- No calibration is needed (no scale/zero-point computation)
- No `M0`/`shift` multiplier decomposition — the Q15.16 format is self-consistent
- BN folding is done once offline (already implemented in [fold_conv_bn_eval](file:///home/ree/Quantization-Examples/resnet/FixedPoint32/inference.py#L381-L404))
- LUT generation for GELU is already implemented in [lut.py](file:///home/ree/Quantization-Examples/resnet/FixedPoint32/lut.py)

The weight/bias export would look like:

```python
# Offline (in export_fxp32_model.py)
w_folded, b_folded = fold_conv_bn_eval(conv, bn)   # float
q_w = quantize_fixed_point(w_folded)                 # Q15.16 int32
q_bias = quantize_fixed_point(b_folded)              # Q15.16 int32
# Save to dict...
```

Then during inference, you load from the dict instead of calling `fold_conv_bn_eval` and `quantize_fixed_point` at runtime — eliminating all floating-point math from the forward pass.

---

## 2. The Critical Challenge: Multiplication in Strict 32-Bit

This is where the FxP32 pipeline diverges fundamentally from INT32, and where serious decisions must be made.

### 2.1 The Core Problem

When you multiply two Q15.16 values, the mathematical result is Q30.32 — a **64-bit number**:

```
   Q15.16  ×  Q15.16  =  Q30.32
   (32-bit)   (32-bit)   (64-bit)
```

You **must** right-shift by 16 bits to get back to Q15.16 format. The current implementation does exactly this:

```python
# Current code in utils.py (lines 51-56)
prod_64 = x_slice.unsqueeze(1).to(torch.int64) * w_slice.to(torch.int64)  # ← int64!
prod_trunc = (prod_64 + (1 << (f_bits - 1))) >> f_bits
prod_32 = prod_trunc.to(torch.int32)
```

> [!WARNING]
> **This is the float-point equivalent problem for FxP32.** The current implementation upcasts to `int64` for every single multiply-accumulate operation. This violates the strict 32-bit constraint.

### 2.2 Comparison: Why INT32 Doesn't Have This Problem

In the INT32 pipeline, the multiply-and-shift (`Accumulator × M0 >> shift`) happens **once per layer output pixel** — it's a final rescaling step. The actual convolution MAC loop (`x_val * w_val + accum`) stays safely in 32 bits because the operands are 8-bit values (max product: $255 × 255 = 65,025$).

In FxP32, **every single MAC** in the convolution involves multiplying two 32-bit values. There's no way around the 64-bit intermediate — it's fundamental to the format.

### 2.3 Can We Use Limb Decomposition?

Yes — the same 16-bit limb decomposition from [INT32/utils.py](file:///home/ree/Quantization-Examples/resnet/INT32/utils.py#L52-L135) can be applied. But the performance implications are severe:

| Operation | INT32 Pipeline | FxP32 Pipeline |
|-----------|---------------|----------------|
| Limb decomposition calls | **1× per output pixel** (only in `multiply_by_quantized_multiplier`) | **1× per MAC** (every `x * w` inside every conv kernel position) |
| Conv layer `3×3×512→512` | 1 limb-decomp call per output pixel | **4,608 limb-decomp calls** per output pixel |
| Total for ResNet-18 | ~100K limb decompositions | ~**460M** limb decompositions |

> [!IMPORTANT]
> Limb decomposition is technically correct but would make inference **~4,600× slower** compared to the INT32 pipeline for the same model, because every MAC becomes 4 cross-products + 4 carry propagations + 2 recombinations instead of a single `int32 × int32`.

### 2.4 The Addition Leak

There's also a secondary issue in [run_static_fixed_point_basic_block](file:///home/ree/Quantization-Examples/resnet/FixedPoint32/inference.py#L467-L469):

```python
sum_int64 = q_out2.to(torch.int64) + q_short.to(torch.int64)  # ← int64 upcast!
q_added = (sum_int64.to(torch.int32)).to(torch.int32)
```

This upcast during the residual addition is unnecessary — two Q15.16 values added together stay Q15.16 (with wrapping risk, not overflow to 64-bit). This can be trivially fixed to pure `int32` addition.

---

## 3. Summary of Challenges

| Challenge | Severity | Fixable? |
|-----------|----------|----------|
| **Multiplication requires 64-bit intermediate** | 🔴 Fundamental | Yes, via limb decomposition — but at ~4600× slowdown |
| **int64 addition leak in residual blocks** | 🟢 Trivial | Yes, replace with modulo int32 add |
| **BN folding at runtime** | 🟢 Trivial | Yes, move to export script |
| **Weight quantization at runtime** | 🟢 Trivial | Yes, move to export script |
| **GELU LUT already offline** | ✅ Done | Already implemented in `lut.py` |
| **Precision loss from Q15.16 range** | 🟡 Moderate | Inherent — Q15.16 can only represent ±32,767.999... |

---

## 4. ZK-Verifiable Inference: FxP32 vs INT32

### 4.1 Circuit Cost Comparison

| Metric | INT32 Pipeline | FxP32 Pipeline (with limb decomp) |
|--------|---------------|----------------------------------|
| **Multiply gate type** | Simple `int32 × int32` (8-bit operands, safe in 32-bit) | 4× cross-products + carry chain per MAC |
| **Constraints per MAC** | ~1 multiplication gate | ~12-16 multiplication + addition gates |
| **Total constraints (ResNet-18)** | ~100M | ~1.2B–1.6B |
| **Proof generation time** | Baseline | ~12-16× slower |
| **Proof size** | Baseline | ~2-3× larger (more committed polynomials) |

### 4.2 Determinism

Both pipelines are fully deterministic — they use only integer arithmetic. From a correctness standpoint, both produce bit-exact reproducible results. ✅

### 4.3 Precision vs. Circuit Cost Trade-off

The FxP32 pipeline preserves more precision than INT32:
- **FxP32**: Weights retain full dynamic range of the trained model (clamped to ±32K)
- **INT32**: Weights are quantized to 8-bit (256 levels), with precision loss from scale/zero-point rounding

However, the precision benefit comes at an enormous ZK circuit cost. For most practical ZK inference use cases (where proof generation time dominates), the INT32 pipeline provides a far better accuracy-per-constraint ratio.

### 4.4 Verdict

> [!IMPORTANT]
> **INT32 is the superior choice for ZK-verifiable inference.** It achieves strong accuracy with ~12-16× fewer constraints than a strict-32-bit FxP32 pipeline. The FxP32 pipeline's main advantage — no calibration step — is offset by its fundamental need for limb decomposition on every MAC operation.

If the FxP32 pipeline is pursued, it should be considered a **research/comparison baseline**, not the production ZK target.

---

## 5. Recommendation

If you want to proceed with `export_fxp32_model.py`, the implementation is clean and straightforward:

1. **Export script**: Fold BN, quantize weights/biases to Q15.16, save GELU LUT, write `_fxp32.pth`
2. **Inference script**: Load dict, apply limb-decomposed MAC, fix the `int64` addition leak
3. **Benchmark**: Compare accuracy loss against INT32 and FP32 baselines

The main decision point is whether the limb decomposition slowdown is acceptable for your use case, or whether you'd prefer to allow `int64` intermediates during FxP32 inference (which breaks strict 32-bit but is mathematically cleaner and faster).
