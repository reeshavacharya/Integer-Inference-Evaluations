# FixedPoint32 Pipeline Audit: Strict Q15.16 Compliance

## Requirements Under Test

1. **All computation must use Q15.16** — 16 integer bits (signed), 16 fractional bits
2. **Only 32-bit registers** — no int64, no int16, no int8, no float at inference time
3. **Quantization is offline** — weights/biases converted before inference begins; inputs quantized at ingress

---

## File-by-File Audit

### [utils.py](file:///home/ree/Quantization-Examples/resnet/FixedPoint32/utils.py)

#### ✅ `quantize_fixed_point` (L14-L18) — Correct
```python
q_tensor = torch.round(tensor.to(torch.float64) * (2 ** f_bits))
q_tensor = torch.clamp(q_tensor, INT32_MIN, INT32_MAX)
return q_tensor.to(torch.int32)
```
This is an **offline** operation (pre-inference). Using `float64` here is acceptable — it's the quantization step that converts FP32 model weights into Q15.16 integers before inference starts.

#### ✅ `dequantize_fixed_point` (L20-L22) — Correct
Only used post-inference for error measurement / logit interpretation. Not part of the forward pass.

---

#### 🔴 Violation 1: `execute_and_shift_conv2d` (L52-L56) — Uses int64

```python
# Line 52: ALU Multiply (Q15.16 * Q15.16 = Q31.32 internal wire)
prod_64 = x_slice.unsqueeze(1).to(torch.int64) * w_slice.to(torch.int64)

# Line 55: Immediate Truncation
prod_trunc = (prod_64 + (1 << (f_bits - 1))) >> f_bits
prod_32 = prod_trunc.to(torch.int32)
```

**Problem:** The multiplication widens both operands to `int64` before multiplying. The product `Q15.16 × Q15.16` produces a `Q31.32` result which requires 64 bits. The rounding addition and right-shift also operate on `int64`. This violates the "only 32-bit registers" constraint.

**Why it exists:** This is mathematically necessary. When you multiply two Q15.16 numbers, the exact result has 32 fractional bits and 32 integer bits — requiring 64 bits total. There is no way to compute the exact product in 32 bits.

**Fix (to meet strict 32-bit):** Use the same **16-bit limb decomposition** as the INT32 pipeline's `multiply_by_quantized_multiplier`. Decompose each 32-bit operand into two 16-bit halves, compute four 32-bit partial products, and reassemble:

```python
# Decompose A (Q15.16) and B (Q15.16) into 16-bit limbs
A_lo = A & 0xFFFF;  A_hi = (A >> 16) & 0xFFFF
B_lo = B & 0xFFFF;  B_hi = (B >> 16) & 0xFFFF

# Four 32-bit partial products (each fits in int32)
P0 = A_lo * B_lo    # bits [0..31]
P1 = A_lo * B_hi    # bits [16..47]
P2 = A_hi * B_lo    # bits [16..47]
P3 = A_hi * B_hi    # bits [32..63]

# Reassemble with carries, then right-shift by 16 (f_bits)
# to truncate back to Q15.16
```

> [!WARNING]
> This limb decomposition must be applied to **every single MAC operation** (multiply-accumulate). For a 3×3 conv with 64 input channels, that's `9 × 64 = 576` limb decompositions per output pixel. This is the fundamental performance cost of strict 32-bit compliance.

---

#### 🔴 Violation 2: `execute_and_shift_linear` (L76-L80) — Same int64 issue

```python
prod_64 = x_val.to(torch.int64) * w_val.to(torch.int64)
prod_trunc = (prod_64 + (1 << (f_bits - 1))) >> f_bits
prod_32 = prod_trunc.to(torch.int32)
```

**Same problem and same fix** as Violation 1. The FC layer multiply also uses `int64`.

---

#### ✅ `add_bias` (L88-L95) — Correct
```python
return q_accum + bias_int32
```
Pure `int32 + int32` modulo arithmetic. No violation.

#### ✅ `fixed_point_max_pool2d` (L98-L100) — Correct
```python
return torch.amax(q_in.to(torch.int32), dim=(2, 3), keepdim=True)
```
Pure integer comparison. No arithmetic, no bit-width change.

#### ✅ `fixed_point_relu` (L102-L105) — Correct
```python
return torch.clamp(q_tensor, min=0)
```
In Q15.16, float 0.0 maps to integer 0. This is a pure 32-bit comparison.

---

#### 🟡 Violation 3: `fixed_point_gelu_lut` (L117) — Uses int64 for indexing

```python
indices = q_clamped.to(torch.int64) - q_min
```

**Problem:** The cast to `int64` is used because PyTorch's tensor indexing requires `int64` indices. The actual LUT lookup is a memory fetch, not arithmetic. However, this still means an `int64` register is touched during inference.

**Fix:** This is a **PyTorch framework limitation**, not a mathematical one. On real hardware or in a ZK circuit, indexing is just pointer arithmetic on the 32-bit value. For strict compliance in Python, you could work around this with:
```python
indices = (q_clamped - q_min).to(torch.int64)  # Cast AFTER subtraction
```
This keeps the subtraction in `int32` and only casts for PyTorch's indexing requirement. The actual subtraction stays 32-bit.

> [!NOTE]
> This violation is cosmetic in a hardware/ZK context. LUT lookup is a memory read, not computation.

---

### [inference.py](file:///home/ree/Quantization-Examples/resnet/FixedPoint32/inference.py)

#### 🔴 Violation 4: `run_static_fixed_point_basic_block` (L467-L469) — Addition uses int64

```python
# Integer addition with 64-bit intermediate then downcast to int32
sum_int64 = q_out2.to(torch.int64) + q_short.to(torch.int64)
q_added = (sum_int64.to(torch.int32)).to(torch.int32)
```

**Problem:** The residual addition widens both operands to `int64` before adding, then truncates back. This is a 64-bit operation.

**Why it exists:** To avoid overflow when adding two Q15.16 values whose sum might exceed `INT32_MAX`. For example, if `q_out2 = 1.5 billion` and `q_short = 1.5 billion`, the true sum is `3 billion` which overflows `int32` but fits in `int64`.

**Fix:** The `error.py` FxP32 path already does this correctly:
```python
# error.py line 342:
q_added = (q_out2 + q_short).to(torch.int32)
```
This is pure 32-bit modulo addition. If overflow occurs, it wraps — which is the accepted behavior in the strict 32-bit pipeline (the user has acknowledged overflow risk).

**Change required in [inference.py:L467-L469](file:///home/ree/Quantization-Examples/resnet/FixedPoint32/inference.py#L467-L469):**
```diff
-    # Integer addition with 64-bit intermediate then downcast to int32
-    sum_int64 = q_out2.to(torch.int64) + q_short.to(torch.int64)
-    q_added = (sum_int64.to(torch.int32)).to(torch.int32)
+    # Strict 32-bit modulo addition (wraps on overflow)
+    q_added = (q_out2.to(torch.int32) + q_short.to(torch.int32)).to(torch.int32)
```

---

#### 🔴 Violation 5: Runtime weight quantization — Uses float during inference

```python
# inference.py line 429-433:
def run_static_fixed_point_conv_block(q_input, conv, bn, ...):
    w_folded, b_folded = fold_conv_bn_eval(conv, bn)    # float math
    q_w = quantize_fixed_point(w_folded)                  # float → int
    q_bias = quantize_fixed_point(b_folded)               # float → int
```

**Problem:** Every time a conv block runs, it:
1. Folds BN into weights using **float arithmetic** (`fold_conv_bn_eval` — division, sqrt, multiplication)
2. Quantizes the result to Q15.16 using **float64 arithmetic** (`quantize_fixed_point`)

This means **floating-point computation happens during inference**, not just offline.

**Why it exists:** The FxP32 pipeline has no offline export step (unlike INT32's `export_int32_model.py`). It dynamically converts the FP32 model on every forward pass.

**Fix:** Create an **offline export module** (`export_fxp32_model.py`) that:
1. Loads the FP32 `.pth` file
2. Folds BN into weights (float, offline)
3. Quantizes all weights and biases to Q15.16 `int32` tensors
4. Saves a dictionary of pure-integer tensors to disk
5. At inference time, load the pre-quantized dictionary — no float math needed

This is exactly what `export_int32_model.py` does for the INT32 pipeline.

---

### [lut.py](file:///home/ree/Quantization-Examples/resnet/FixedPoint32/lut.py)

#### ✅ Fully Correct — Offline Only
The LUT generation uses `float64` extensively, but it runs **once, offline**, producing a pure `int32` lookup table saved to disk. At inference time, only the integer table is loaded. No violation.

---

## Summary of Violations

| # | Location | Violation | Severity | Fix Difficulty |
|---|----------|-----------|----------|----------------|
| 1 | `utils.py:L52` | Conv MAC uses `int64` | 🔴 Critical | Hard — requires limb decomposition per-MAC |
| 2 | `utils.py:L76` | Linear MAC uses `int64` | 🔴 Critical | Hard — same limb decomposition |
| 3 | `utils.py:L117` | LUT indexing uses `int64` | 🟡 Minor | Trivial — PyTorch limitation, not real math |
| 4 | `inference.py:L467-469` | Addition uses `int64` | 🔴 Critical | Trivial — change to modulo `int32` add |
| 5 | `inference.py:L429-433` | Runtime BN fold uses float | 🔴 Critical | Medium — create offline export module |

---

## Proposed Fix Strategy

### Phase 1: Quick Wins (no accuracy impact)

1. **Fix Violation 4** — Change the residual addition from `int64` to `int32` modulo. This is a 3-line change in [inference.py:L467-469](file:///home/ree/Quantization-Examples/resnet/FixedPoint32/inference.py#L467-L469).

2. **Fix Violation 3** — Move the `int64` cast after the subtraction in `fixed_point_gelu_lut`. 1-line change.

### Phase 2: Offline Export (medium effort)

3. **Fix Violation 5** — Create `export_fxp32_model.py` that pre-quantizes all weights/biases to Q15.16 and saves them. Modify `inference.py` to load the pre-quantized dictionary instead of doing runtime BN folding. This follows the exact same pattern as the INT32 pipeline.

### Phase 3: Strict 32-bit MAC (high effort, significant performance cost)

4. **Fix Violations 1 & 2** — Replace the `int64` multiply-then-truncate with 16-bit limb decomposition. This is the same technique used in `INT32/utils.py:multiply_by_quantized_multiplier`, adapted for Q15.16 truncation instead of the asymmetric quantization downscale.

The limb decomposition produces the correct upper 32 bits of the 64-bit product, then right-shifts by 16 — all using only 32-bit registers.

> [!IMPORTANT]
> **Performance impact of Phase 3:** Every MAC becomes ~12 multiplications + ~20 additions + carry propagation instead of 1 multiplication + 1 shift. For ResNet-18 on CIFAR10, this means:
> - Current: ~36M MAC operations per image
> - After: ~36M × 12 = **~430M multiplications** per image
> - Estimated slowdown: **10-15×**

### What stays the same
- `quantize_fixed_point` and `dequantize_fixed_point` — offline/post-inference only
- `add_bias` — already pure int32
- `fixed_point_relu` — already pure int32
- `fixed_point_max_pool2d` — already pure int32
- `lut.py` — already offline-only
