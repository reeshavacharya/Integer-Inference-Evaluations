# Integer & Fixed Point Inference on Medical Datasets

This implementation studies how trained neural networks can be executed without standard floating-point arithmetic at inference time. The codebase compares three execution modes across classification and segmentation workloads:

- floating-point inference
- integer-only INT8 inference
- static 64-bit fixed-point inference

The experiments cover three model families:

- `lenet/`: compact image classification models
- `resnet/`: deeper residual image classification models
- `unet/`: image segmentation models

The medical-facing datasets in this experiment include Brain MRI, CHEST, several Multi-Cancer subsets, Skin-Lesion, Brain-MRI-Seg, and BUSI. Some scripts also include MNIST, CIFAR10, and Flood as reference workloads. Training scripts download datasets into `./data/` when needed, then build deterministic `80/10/10` train/validation/test splits.

## Experiment Overview

The main question in this experiment is whether the same trained models can be executed with:

1. standard floating-point arithmetic
2. 8-bit integer-only arithmetic
3. signed 64-bit fixed-point arithmetic

The floating-point path is the reference implementation. The INT8 and fixed-point paths reuse the trained floating-point weights, convert them into integer representations, and then reproduce inference with integer math.

At a high level:

- training is always done in floating point
- quantized execution is simulated after training
- inference scripts select a random sample from the deterministic test split
- benchmark scripts run full-dataset or partial-dataset evaluations with explicit flags

## Integer-Only Inference

The INT8 flow in `lenet/INT8`, `resnet/INT8`, and `unet/INT8` uses quantized activations and weights with integer accumulation. The goal is to replace floating-point inference with operations on `uint8` values, `int32` accumulators, and precomputed integer rescaling terms.

### 1. Calibration

Before integer inference, the code runs a floating-point forward pass and records activation ranges with forward hooks. For each relevant layer, it stores:

- input minimum and maximum
- output minimum and maximum

These ranges are then converted into affine quantization parameters.

For an 8-bit tensor with integer range:

$$
q_{\min}=0,\quad q_{\max}=255
$$

and real-valued tensor range:

$$
r_{\min},\quad r_{\max}
$$

the scale and zero-point are computed as:

$$
S = \frac{r_{\max}-r_{\min}}{q_{\max}-q_{\min}}
$$

$$
Z = \operatorname{round}\left(q_{\min} - \frac{r_{\min}}{S}\right)
$$

with `Z` clamped to `[0,255]`, and the range is nudged so that `0.0` is representable.

The tensor quantization equation is:

$$
q = \operatorname{round}\left(\frac{r}{S}\right) + Z
$$

The implementation applies this to:

- input activations
- layer weights
- per-layer output activations

Bias terms are stored as signed 32-bit integers using:

$$
S_{\text{bias}} = S_w \cdot S_x,\qquad Z_{\text{bias}} = 0
$$

so that bias addition is compatible with the integer accumulator domain.

### 2. Integer Convolution / Linear Execution

Once an input tensor and a layer weight tensor have been quantized, convolution and linear layers are evaluated as integer dot products:

$$
\text{acc} = \sum (q_x - Z_x)(q_w - Z_w)
$$

where:

- \(q_x\) is the quantized activation
- \(q_w\) is the quantized weight
- \(Z_x\) and \(Z_w\) are the corresponding zero-points

The accumulator lives in signed 32-bit space. Then quantized bias is added:

$$
\text{acc}_{\text{bias}} = \text{acc} + q_b
$$

### 3. Integer Requantization

Each layer output must be converted from the accumulator scale back into the next layer's activation scale. The floating-point rescaling factor is:

$$
M = \frac{S_w \cdot S_x}{S_{\text{out}}}
$$

Instead of multiplying by $M$ in floating point, the code decomposes it into:

$$
M \approx M_0 \cdot 2^{-n}
$$

where:

- \(M_0\) is an integer multiplier
- \(n\) is a right-shift amount

The output is then produced entirely with integer arithmetic:

$$
q_{\text{out}} =
\operatorname{clip}_{[0,255]}
\left(
\left(
\frac{\text{acc}_{\text{bias}} \cdot M_0}{2^{31+n}}
\right)_{\text{rounded}}
 + Z_{\text{out}}
\right)
$$

In the implementation, rounding is handled by adding a power-of-two offset before the right shift.

### 4. Integer Activations and Pooling

ReLU is applied without dequantizing. Since real zero maps to the output zero-point, quantized ReLU becomes:

$$
q' = \max(q, Z_{\text{out}})
$$

For LeNet-style average pooling, the implementation uses pure integer pooling:

$$
\operatorname{avg}_{2\times2}(x) = \frac{x_1 + x_2 + x_3 + x_4}{4}
$$

implemented with integer summation and a right shift, including a small rounding term.

For ResNet, the integer path also handles:

- residual additions in integer space
- integer global average pooling

For U-Net, the integer path additionally handles:

- max-pooling
- concatenation of skip connections
- transposed convolution in integer space

## 64-Bit Fixed-Point Inference

The `FixedPoint64` paths use a signed `int64` representation instead of affine INT8 quantization. The format used in the utilities is a static Q31.32 layout:

$$
\hat{x} = \operatorname{round}(x \cdot 2^{32})
$$

That means:

- 31 bits for the signed integer part
- 32 bits for the fractional part

Dequantization is simply:

$$
x \approx \frac{\hat{x}}{2^{32}}
$$

### 1. Why Fixed-Point?

This path keeps the execution integer-based while preserving more precision than INT8. It is useful when:

- INT8 is too coarse
- deterministic integer arithmetic is desired
- one wants to inspect bit-growth and overflow headroom explicitly

### 2. The Overflow Problem

If two Q31.32 values are multiplied directly, the raw product carries roughly 64 fractional bits before rescaling. For deep networks, that can overflow a signed 64-bit container during accumulation.

To control this, the implementation uses pre-truncation before multiplication.

### 3. Pre-Truncation Strategy

For a tensor with `F_BITS = 32`, the code splits the fractional shift across activations and weights:

$$
s_x = 16,\qquad s_w = 16
$$

Then it rounds and right-shifts each operand before multiplication:

$$
\hat{x}' = \left\lfloor \frac{\hat{x} + 2^{s_x-1}}{2^{s_x}} \right\rfloor
$$

$$
\hat{w}' = \left\lfloor \frac{\hat{w} + 2^{s_w-1}}{2^{s_w}} \right\rfloor
$$

Convolution or matrix multiplication is then performed on the truncated integers:

$$
\hat{y} = \hat{x}' * \hat{w}'
$$

This keeps the result in a practical `int64` range while still landing back in an effective Q31.32-style scale after the multiply.

### 4. Bias, ReLU, and Pooling

Bias is quantized into the same Q31.32 domain and added directly:

$$
\hat{y}_{\text{bias}} = \hat{y} + \hat{b}
$$

ReLU is:

$$
\hat{y}' = \max(\hat{y}, 0)
$$

Average pooling is implemented in integer form:

$$
\hat{p} = \left\lfloor \frac{\hat{x}_1 + \hat{x}_2 + \hat{x}_3 + \hat{x}_4 + 2}{4} \right\rfloor
$$

For U-Net and ResNet, the fixed-point path extends the same idea to:

- deep convolution stacks
- skip connections
- transposed convolution or residual additions
- final dequantization for metric computation

### 5. What the Fixed-Point Scripts Report

The 64-bit inference scripts print extra diagnostics such as:

- maximum accumulator bit-length
- remaining headroom inside signed 64-bit arithmetic
- truncation remainder / precision loss indicators

Those logs help estimate whether the fixed-point format is numerically safe for the network being executed.

## Installation

The root `requirements.txt` lists the core Python dependencies used by the project:

- `numpy`
- `torch`
- `torchvision`

Create a virtual environment and install them from the implementation root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you later want to leave the environment:

```bash
deactivate
```

## Implementation Layout

```text
Quantization-Examples/
├── lenet/
│   ├── lenet5.py
│   ├── benchmark.py
│   ├── INT8/inference.py
│   └── FixedPoint64/inference.py
├── resnet/
│   ├── resnet18.py
│   ├── benchmark.py
│   ├── INT8/inference.py
│   └── FixedPoint64/inference.py
├── unet/
│   ├── u_net.py
│   ├── benchmark.py
│   ├── INT8/inference.py
│   └── FixedPoint64/inference.py
└── requirements.txt
```

## Running LeNet Experiments

Run commands from the implementation root.

### 1. Train First

LeNet checkpoints are created by `lenet/lenet5.py`. Train the datasets you want before running random-sample inference or benchmarks.

Train the reference datasets:

```bash
python lenet/lenet5.py --train_data MNIST
python lenet/lenet5.py --train_data CIFAR10
python lenet/lenet5.py --train_data Brain-MRI
python lenet/lenet5.py --train_data CHEST
```

Train all Multi-Cancer subsets in one run:

```bash
python lenet/lenet5.py --train_data Multi-Cancer
```

Or train a single Multi-Cancer subset directly:

```bash
python lenet/lenet5.py --train_data Brain-Cancer
python lenet/lenet5.py --train_data Breast-Cancer
python lenet/lenet5.py --train_data Cervical-Cancer
python lenet/lenet5.py --train_data Kidney-Cancer
python lenet/lenet5.py --train_data Lung-And-Colon-Cancer
python lenet/lenet5.py --train_data Lymphoma-Cancer
python lenet/lenet5.py --train_data Oral-Cancer
```

### 2. Random Test-Sample Checks

The inference scripts draw a random sample from the deterministic 10% test split and compare execution modes.

INT8 path:

```bash
python lenet/INT8/inference.py --infer Brain-MRI
python lenet/INT8/inference.py --infer Brain-MRI --int
python lenet/INT8/inference.py --infer Brain-MRI --floating-point
```

64-bit fixed-point path:

```bash
python lenet/FixedPoint64/inference.py --infer Brain-MRI
python lenet/FixedPoint64/inference.py --infer Brain-MRI --fixed-point
python lenet/FixedPoint64/inference.py --infer Brain-MRI --floating-point
```

Other valid `--infer` values are:

- `MNIST`
- `CIFAR10`
- `Brain-MRI`
- `CHEST`
- `Brain-Cancer`
- `Breast-Cancer`
- `Cervical-Cancer`
- `Kidney-Cancer`
- `Lung-And-Colon-Cancer`
- `Lymphoma-Cancer`
- `Oral-Cancer`

### 3. Benchmarks

Use `lenet/benchmark.py` for dataset-level comparisons.

Benchmark all trained LeNet datasets and all modes:

```bash
python lenet/benchmark.py
```

Benchmark one dataset only:

```bash
python lenet/benchmark.py --bench Brain-MRI
```

Benchmark a limited number of test samples:

```bash
python lenet/benchmark.py --bench Brain-MRI --num_data 100
```

Benchmark one execution mode only:

```bash
python lenet/benchmark.py --bench Brain-MRI --mode int
python lenet/benchmark.py --bench Brain-MRI --mode fixed-point
python lenet/benchmark.py --bench Brain-MRI --mode floating-point
```

## Running U-Net Experiments

### 1. Train First

U-Net checkpoints are created per dataset with `unet/u_net.py`.

```bash
python unet/u_net.py --train_data Skin-Lesion
python unet/u_net.py --train_data Flood
python unet/u_net.py --train_data Brain-MRI-Seg
python unet/u_net.py --train_data BUSI
```

### 2. Random Test-Sample Checks

INT8 segmentation inference:

```bash
python unet/INT8/inference.py --infer Skin-Lesion
python unet/INT8/inference.py --infer Skin-Lesion --int
python unet/INT8/inference.py --infer Skin-Lesion --floating-point
```

64-bit fixed-point segmentation inference:

```bash
python unet/FixedPoint64/inference.py --infer Skin-Lesion
python unet/FixedPoint64/inference.py --infer Skin-Lesion --fixed-point
python unet/FixedPoint64/inference.py --infer Skin-Lesion --floating-point
```

Other valid `--infer` values are:

- `Skin-Lesion`
- `Flood`
- `Brain-MRI-Seg`
- `BUSI`

### 3. Benchmarks

Use the unified benchmark runner:

```bash
python unet/benchmark.py
```

Benchmark one dataset only:

```bash
python unet/benchmark.py --bench Brain-MRI-Seg
```

Benchmark a subset of the test split:

```bash
python unet/benchmark.py --bench Brain-MRI-Seg --num_data 25
```

Benchmark one execution mode only:

```bash
python unet/benchmark.py --bench Brain-MRI-Seg --mode int
python unet/benchmark.py --bench Brain-MRI-Seg --mode fixed-point
python unet/benchmark.py --bench Brain-MRI-Seg --mode floating-point
```

## Running ResNet Experiments

### 1. Train First

ResNet checkpoints are created by `resnet/resnet18.py`.

Train the reference datasets:

```bash
python resnet/resnet18.py --train_data MNIST --in_channels 1
python resnet/resnet18.py --train_data CIFAR10 --in_channels 3
python resnet/resnet18.py --train_data Brain-MRI --in_channels 1
python resnet/resnet18.py --train_data CHEST --in_channels 1
```

Train all Multi-Cancer subsets in one run:

```bash
python resnet/resnet18.py --train_data Multi-Cancer
```

Or train a single Multi-Cancer subset directly:

```bash
python resnet/resnet18.py --train_data Brain-Cancer
python resnet/resnet18.py --train_data Breast-Cancer
python resnet/resnet18.py --train_data Cervical-Cancer
python resnet/resnet18.py --train_data Kidney-Cancer
python resnet/resnet18.py --train_data Lung-And-Colon-Cancer
python resnet/resnet18.py --train_data Lymphoma-Cancer
python resnet/resnet18.py --train_data Oral-Cancer
```

### 2. Random Test-Sample Checks

INT8 path:

```bash
python resnet/INT8/inference.py --infer Brain-MRI
python resnet/INT8/inference.py --infer Brain-MRI --int
python resnet/INT8/inference.py --infer Brain-MRI --floating-point
```

64-bit fixed-point path:

```bash
python resnet/FixedPoint64/inference.py --infer Brain-MRI
python resnet/FixedPoint64/inference.py --infer Brain-MRI --fixed-point
python resnet/FixedPoint64/inference.py --infer Brain-MRI --floating-point
```

Other valid `--infer` values are:

- `MNIST`
- `CIFAR10`
- `Brain-MRI`
- `CHEST`
- `Brain-Cancer`
- `Breast-Cancer`
- `Cervical-Cancer`
- `Kidney-Cancer`
- `Lung-And-Colon-Cancer`
- `Lymphoma-Cancer`
- `Oral-Cancer`

### 3. Benchmarks

Use `resnet/benchmark.py` for dataset-level comparisons.

Benchmark everything:

```bash
python resnet/benchmark.py
```

Benchmark one dataset:

```bash
python resnet/benchmark.py --bench Brain-MRI
```

Benchmark a fixed number of test images:

```bash
python resnet/benchmark.py --bench Brain-MRI --num_data 100
```

Benchmark a single execution mode:

```bash
python resnet/benchmark.py --bench Brain-MRI --mode int
python resnet/benchmark.py --bench Brain-MRI --mode fixed-point
python resnet/benchmark.py --bench Brain-MRI --mode floating-point
```

## Notes

- All random-sample inference scripts pull examples from the deterministic 10% test split, not from the training split.
- Classification benchmarks report accuracy.
- U-Net benchmarks report Dice, IoU, pixel accuracy, and F1.
- Several training scripts auto-download datasets into `./data/` if they are not present.
- Multi-Cancer training creates separate checkpoints per cancer subset.
