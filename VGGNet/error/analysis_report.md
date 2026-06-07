# VGGNet Error Pipeline Analysis Report

This report analyzes the floating point vs. integer quantization divergence captured across the VGG19 networks for various activations (ReLU, GELU, LeakyReLU) and datasets.

---

## 1. ReLU Activation Analysis

### Dataset: CIFAR10
- **Trend**: Error accumulates steadily across convolutions, with mild peaks.
- **Classifiers**: `classifier_0_pre_act` sees a large spike in MAE, immediately corrected by `classifier_0_relu`. Final `classifier_6` produces stable error metrics (MaxAE ~7.69) with no integer overflow.

### Dataset: MNIST & OrganAMNIST
- **Trend**: Exhibits very stable and low quantization noise due to the simpler nature of the grayscale features.
- **Classifiers**: The classifier accumulation spikes are much smaller here compared to complex RGB datasets.

### Dataset: BloodMNIST
- **Trend**: Massive error discontinuity at the final layer.
- **Classifiers**: Max Absolute Error explodes to `836,485,971` strictly at `classifier_6`. This is heavily decoupled from normal quantization noise.

### Dataset: OCTMNIST, Brain_MRI, PneumoniaMNIST
- **Trend**: Consistent sawtooth SQNR patterns.
- **Classifiers**: Error compounds primarily in the massive 25,088-input inner product of `classifier_0`, heavily mitigated by the ReLU pass.

---

## 2. GELU Activation Analysis

### Dataset: CIFAR10 & BloodMNIST
- **Trend**: GELU lacks the "hard zeroing" capability of ReLU. Negative values are smoothed, not deleted.
- **Classifiers**: The error drop at `classifier_0_gelu` is much softer than ReLU because the noise residing in the negative domain is only attenuated, not eliminated.

### Dataset: MNIST, OrganAMNIST, PneumoniaMNIST
- **Trend**: Shows slightly smoother error accumulation lines across the feature blocks compared to ReLU.

### Dataset: OCTMNIST & Brain_MRI
- **Trend**: Since GELU uses a Look-Up Table (LUT) in INT32 mode, there is minor discrete step-noise added, but it prevents the large magnitude spikes seen in pure arithmetic activations.

---

## 3. LeakyReLU Activation Analysis

### All Datasets
- **Trend**: LeakyReLU retains the negative distribution multiplied by a small slope (e.g. 0.01 or 1.0 depending on the `INT32` configuration).
- **Classifiers**: Because negative values are preserved, the SQNR bounce-back at activation layers is noticeably weaker than in ReLU. The negative noise continues to propagate.

---

## 4. Common Trends & Answers

### Why does the negative number percentage not rise at all except at the very end in ReLU?
ReLU mathematically functions as `max(0, x)`. In quantized terms, it rigidly clamps any value below the zero-point to the zero-point. As a result, the quantized percentage of negative numbers is strictly `0%` for every single ReLU post-activation layer. The only reason it rises at the very end is because `classifier_6` is the final logit output layer in VGG19 and it **does not** have an activation layer attached to it. 

### Why is SQNR rising and dropping at each subsequent layer?
This creates a consistent **sawtooth** pattern caused by the alternation of operations:
1. **Drop (Pre-Act):** Convolution and Linear layers accumulate tens of thousands of MAC (Multiply-Accumulate) operations. The downscaling (`M0 / shift`) introduces truncation and rounding noise. This causes SQNR to drastically drop.
2. **Rise (Post-Act):** Activations like ReLU act as noise filters. By zeroing out all values below the threshold, ReLU deletes a massive chunk of the quantization noise residing in the negative domain. This artificially "cleans" the tensor, causing the SQNR to bounce back up.

### Why do the classifiers have the highest error spikes, which drop after ReLU?
The VGG19 `classifier_0` is a massive `Linear(25088, 4096)` layer. Computing a single output neuron requires accumulating **25,088** quantized products. Every product carries a tiny rounding error; summing 25k of them causes the quantization drift to compound heavily, manifesting as a massive Absolute Error spike.
The error drops heavily at `classifier_0_relu` because nearly half of those wild accumulations fall into the negative domain and are instantly clipped to 0, completely erasing their contribution to the Absolute Error metric.

### Why does Max Absolute Error rise for `classifier_6` *only* in BloodMNIST?
This is caused by an **Unsigned Integer Underflow Evaluation Glitch**. 
The INT32 pipeline casts final outputs to `torch.uint32`. The `classifier_6` layer outputs raw logits (which can naturally be negative). For BloodMNIST, one of the logit outputs fell below `0` (e.g., `-1`). In unsigned 32-bit arithmetic, `-1` wraps around to `4,294,967,295`. 
When `error.py` evaluates the error, it casts this to float: `4294967295.0`, and subtracts the zero point. This registers as an 836-million margin Absolute Error! The other datasets simply didn't output a logit negative enough to wrap around 0, thereby masking the bug.

### Are the Error Metrics fair and correctly computed?
The mathematical formulas in `LayerMetrics` (SQNR, Cosine Similarity, MAE) are **correctly computed and fair**. 
However, there is a minor flaw in how `error.py` handles datatype casting for the error evaluation trace:
`dq_tensor = s_out * (q_tensor.to(torch.float64) - z_out)`
If `q_tensor` is an unsigned integer (`uint32`) but secretly holds a wrapped-around two's-complement negative value, `to(torch.float64)` parses it as 4.29 billion instead of a small negative number. It should be cast to an `int32` bit-representation before converting to float to prevent these false underflow explosions in `classifier_6`.
