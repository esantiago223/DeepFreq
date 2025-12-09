# DeepFreq Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [What is DeepFreq?](#what-is-deepfreq)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Understanding the Architecture](#understanding-the-architecture)
6. [Using Pre-trained Models](#using-pre-trained-models)
7. [Generating Custom Datasets](#generating-custom-datasets)
8. [Training from Scratch](#training-from-scratch)
9. [Testing and Evaluation](#testing-and-evaluation)
10. [Example Use Cases](#example-use-cases)

---

## Introduction

This tutorial will guide you through using DeepFreq, a data-driven approach for estimating sinusoid frequencies in noisy signals. Whether you're a researcher, signal processing engineer, or student, this guide will help you get started with DeepFreq.

## What is DeepFreq?

DeepFreq is a deep learning-based system for frequency estimation from noisy sinusoidal signals. Traditional methods like MUSIC or periodogram-based approaches often struggle with:
- Low signal-to-noise ratios (SNR)
- Closely spaced frequencies
- Varying number of frequency components

DeepFreq addresses these challenges using a two-stage neural network architecture:

1. **Frequency Representation Module (FR)**: Transforms the input signal into a frequency representation
2. **Frequency Counting Module (FC)**: Estimates the number of frequency components present

### Key Features
- Robust to noise (works well even at low SNR)
- Handles closely spaced frequencies
- Automatically estimates the number of frequencies
- Pre-trained models available for immediate use

## Installation

### Requirements

DeepFreq requires Python 3.6+ and the following packages:

```bash
pip install torch>=1.1.0
pip install scipy>=1.1.0
pip install tensorboard>=1.14.0
```

Or simply install from the requirements file:

```bash
pip install -r requirements.txt
```

**Important Note on PyTorch Versions:**
- The pre-trained **Frequency Counting (FC)** module has compatibility issues with PyTorch 1.8+
- The pre-trained **Frequency Representation (FR)** module works with all PyTorch versions
- If you need the FC module to work, use PyTorch 1.7 or earlier:
  ```bash
  pip install torch==1.7.1
  ```
- Alternatively, you can retrain the FC module with your current PyTorch version

### Additional Dependencies for Testing

For full benchmarking capabilities (including CBLasso comparisons), you may need:
- MATLAB Python API
- CVX optimization library

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/esantiago223/DeepFreq.git
cd DeepFreq
```

### 2. Run the Example Notebook

The fastest way to see DeepFreq in action is through the provided Jupyter notebook:

```bash
jupyter notebook example_notebook.ipynb
```

This notebook demonstrates:
- Loading pre-trained models
- Processing sample signals
- Visualizing frequency estimation results

### 3. Use the Simple Example Script

For a quick Python script example, run:

```bash
python simple_example.py
```

This script shows how to:
- Load pre-trained DeepFreq models
- Generate a test signal with multiple frequencies
- Estimate frequencies from noisy observations
- Visualize the results

## Understanding the Architecture

### Frequency Representation Module

The FR module takes a noisy signal as input and produces a frequency representation (similar to a power spectrum). Two architectures are available:

1. **PSnet**: Simpler architecture using linear layers and 1D convolutions
2. **FrequencyRepresentationModule**: More sophisticated architecture with upsampling

Key parameters:
- `signal_dim`: Dimension of the input signal (e.g., 50 samples)
- `fr_size`: Size of the frequency representation (e.g., 1000)
- `n_filters`: Number of convolutional filters (e.g., 8)
- `n_layers`: Number of convolutional layers (e.g., 3)

### Frequency Counting Module

The FC module takes the frequency representation and estimates how many frequency components are present.

Two modes:
- **Regression**: Outputs a continuous estimate of the number of frequencies
- **Classification**: Classifies into discrete frequency count bins

## Using Pre-trained Models

Pre-trained models are located in the `pretrained_models/` directory:

```
pretrained_models/
├── DeepFreq/
│   ├── frequency_representation_module.pth
│   └── frequency_counting_module.pth
└── PSnet/
    ├── psnet.pth
    └── frequency_counting_psnet.pth
```

### Loading Models

```python
import torch
import numpy as np
from modules import FrequencyRepresentationModule, FrequencyCountingModule
import util

# Specify device (use 'cpu' if CUDA is not available)
device = torch.device('cpu')  # or torch.device('cuda') if GPU is available

# Load frequency representation module
fr_module, _, _, _, _ = util.load('pretrained_models/DeepFreq/frequency_representation_module.pth', 'fr', device=device)
fr_module.eval()

# Load frequency counting module
fc_module, _, _, _, _ = util.load('pretrained_models/DeepFreq/frequency_counting_module.pth', 'fc', device=device)
fc_module.eval()

# Create frequency grid
xgrid = np.linspace(-0.5, 0.5, fr_module.fr_size, endpoint=False)
```

### Processing a Signal

```python
import torch
from data.fr import find_freq

# Your noisy signal (shape: [batch_size, 2, signal_dim])
# Dimension 1 has size 2 for [real_part, imaginary_part]
noisy_signal = torch.tensor(your_signal, dtype=torch.float32)

# Get frequency representation
with torch.no_grad():
    freq_representation = fr_module(noisy_signal)

    # Estimate number of frequencies
    num_freqs = fc_module(freq_representation)
    num_freqs = torch.round(num_freqs).int()

# Extract frequency estimates
freq_estimates = find_freq(freq_representation.numpy(), num_freqs.numpy(), xgrid)
```

## Generating Custom Datasets

Use `generate_dataset.py` to create synthetic test data:

```bash
python generate_dataset.py \
    --output_dir my_dataset/ \
    --n_test 1000 \
    --signal_dimension 50 \
    --minimum_separation 1.0 \
    --max_freq 10 \
    --dB 0 5 10 15 20 25 30 35 40 45 50
```

### Parameters Explained

- `--output_dir`: Where to save the generated data
- `--n_test`: Number of test signals to generate
- `--signal_dimension`: Length of each signal (number of samples)
- `--minimum_separation`: Minimum spacing between frequencies (normalized by 1/signal_dim)
- `--max_freq`: Maximum number of frequency components per signal
- `--dB`: List of SNR levels (in dB) to generate

### Output Files

The script generates:
- `infdB.npy`: Clean signals (no noise)
- `XdB.npy`: Noisy signals at X dB SNR
- `f.npy`: True frequency values
- `data.args`: JSON file with generation parameters

## Training from Scratch

### Two-Stage Training Process

DeepFreq is trained in two stages:

1. **Stage 1**: Train the Frequency Representation module
2. **Stage 2**: Freeze FR module, train the Frequency Counting module

### Basic Training Command

```bash
python train.py \
    --n_training 200000 \
    --n_epochs_fr 200 \
    --n_epochs_fc 100 \
    --output_dir checkpoints/my_experiment \
    --signal_dim 50 \
    --fr_size 1000 \
    --max_num_freq 10
```

### Key Training Parameters

**Data parameters:**
- `--n_training`: Number of training samples per epoch
- `--n_validation`: Number of validation samples
- `--signal_dim`: Signal dimension (default: 50)
- `--max_num_freq`: Maximum number of frequencies (default: 10)

**FR Module parameters:**
- `--fr_module_type`: Either 'fr' or 'psnet'
- `--fr_size`: Size of frequency representation
- `--fr_n_filters`: Number of filters
- `--fr_n_layers`: Number of layers
- `--n_epochs_fr`: Epochs to train FR module

**FC Module parameters:**
- `--fc_module_type`: Either 'regression' or 'classification'
- `--fc_n_filters`: Number of filters
- `--fc_n_layers`: Number of layers
- `--n_epochs_fc`: Epochs to train FC module

**Training settings:**
- `--snr`: Signal-to-noise ratio for training (default: 100, which is ~20dB)
- `--lr_fr`: Learning rate for FR module (default: 0.001)
- `--lr_fc`: Learning rate for FC module (default: 0.001)

### Monitoring Training

Training progress is logged using TensorBoard:

```bash
tensorboard --logdir checkpoints/my_experiment
```

Metrics tracked:
- FR L2 training loss
- FR L2 validation loss
- Frequency Non-detection Rate (FNR)
- FC accuracy

## Testing and Evaluation

Evaluate trained models on test datasets:

```bash
python test.py \
    --data_dir test_dataset/ \
    --output_dir results/ \
    --fr_path pretrained_models/DeepFreq/frequency_representation_module.pth \
    --fc_path pretrained_models/DeepFreq/frequency_counting_module.pth \
    --overwrite
```

### Comparison with Baselines

The test script compares DeepFreq against:
- MUSIC algorithm (with AIC/MDL model order selection)
- Periodogram
- CBLasso
- PSnet baseline

### Evaluation Metrics

- **False Non-detection Rate (FNR)**: Percentage of true frequencies not detected
- **Chamfer Distance**: Average minimum distance between estimated and true frequencies
- **Counting Accuracy**: Percentage of signals where the number of frequencies is correctly estimated

### Output

Results are saved in the output directory:
- Performance metrics as JSON files
- Comparison plots (if matplotlib/seaborn available)
- Per-SNR performance breakdown

## Example Use Cases

### Use Case 1: Single Signal Analysis

```python
import torch
import numpy as np
from util import load
from data.fr import find_freq

# Load model
device = torch.device('cpu')
fr_module, _, _, _, _ = load('pretrained_models/DeepFreq/frequency_representation_module.pth', 'fr', device=device)
fr_module.eval()

# Your signal (50 samples, complex valued)
# signal shape: [1, 2, 50] where dimension 1 contains [real_part, imag_part]
signal = torch.tensor(your_signal).unsqueeze(0)

# Estimate
with torch.no_grad():
    freq_rep = fr_module(signal)

# Find peaks in frequency representation
xgrid = np.linspace(-0.5, 0.5, fr_module.fr_size, endpoint=False)
estimated_freqs = find_freq(freq_rep.numpy(), num_frequencies=3, xgrid=xgrid)

print(f"Estimated frequencies: {estimated_freqs[0]}")
```

### Use Case 2: Batch Processing

```python
# Process multiple signals at once
signals = torch.tensor(your_signals)  # Shape: [batch_size, 2, 50]

with torch.no_grad():
    freq_reps = fr_module(signals)
    num_freqs = fc_module(freq_reps)
    num_freqs = torch.round(num_freqs).int().numpy()

# Extract frequencies for each signal
all_estimates = []
for i in range(len(signals)):
    freqs = find_freq(freq_reps[i:i+1].numpy(), num_freqs[i:i+1], xgrid)
    all_estimates.append(freqs[0])
```

### Use Case 3: Real-world Signal Processing

For real-world applications:

1. **Normalize your signal**: Scale to similar amplitude range as training data
2. **Convert to complex form**: If your signal is real-valued, construct complex signal
3. **Match signal dimension**: Interpolate/downsample to match trained model (typically 50 samples)
4. **Add proper formatting**: Reshape to [batch, 2, signal_dim] format

```python
from scipy import signal as scipy_signal

def preprocess_real_signal(real_signal, target_length=50):
    """Preprocess real-world signal for DeepFreq"""

    # Resample to target length
    if len(real_signal) != target_length:
        real_signal = scipy_signal.resample(real_signal, target_length)

    # Normalize
    real_signal = real_signal / np.max(np.abs(real_signal))

    # Create complex representation (Hilbert transform)
    analytic_signal = scipy_signal.hilbert(real_signal)

    # Format for DeepFreq: [2, signal_dim] with [real_part, imag_part]
    formatted = np.stack([analytic_signal.real, analytic_signal.imag], axis=0)

    return torch.tensor(formatted, dtype=torch.float32).unsqueeze(0)
```

## Tips and Best Practices

1. **Signal Preprocessing**: Normalize your signals to have similar amplitude distributions as training data
2. **Frequency Range**: DeepFreq assumes normalized frequencies in [-0.5, 0.5]
3. **Minimum Separation**: Models work best when frequencies are separated by at least 1/signal_dim
4. **SNR Considerations**: Pre-trained models are trained on SNR around 20dB but generalize well to other SNRs
5. **Batch Size**: Use batching for faster processing of multiple signals

## Troubleshooting

### Common Issues

**Issue**: RuntimeError about CUDA device when loading models
- **Solution**: The pre-trained models were saved on CUDA. If you don't have a GPU, load with CPU device:
  ```python
  device = torch.device('cpu')
  fr_module, _, _, _, _ = util.load(model_path, 'fr', device=device)
  ```

**Issue**: Low accuracy on real-world signals
- **Solution**: Ensure proper normalization and signal preprocessing
- Check that frequency range matches model assumptions

**Issue**: Training diverges or produces NaN
- **Solution**: Reduce learning rate, check data generation parameters

**Issue**: Out of memory during training
- **Solution**: Reduce batch size (`--batch_size` parameter) or use smaller models

**Issue**: ValueError about unpacking values in noise function
- **Solution**: Ensure your signal has shape `[batch_size, 2, signal_dim]` not `[batch_size, 2*signal_dim]`

**Issue**: FC module dimension mismatch (mat1 and mat2 shapes cannot be multiplied)
- **Solution**: This is a PyTorch version compatibility issue. The pre-trained FC models were trained with PyTorch 1.1-1.7 where padding behavior was different. With newer versions (1.8+), the convolutional layers with circular padding produce different output sizes.
- **Workaround**: Either use the known number of frequencies, or retrain the FC module with your PyTorch version, or use PyTorch 1.7 or earlier.
- The FR (Frequency Representation) module works fine across all versions.

## Further Reading

- [Paper: Data-Driven Estimation of Sinusoid Frequencies (NeurIPS 2019)](https://arxiv.org/abs/1906.00823)
- [Project webpage](https://sreyas-mohan.github.io/DeepFreq/)
- [Example notebook](example_notebook.ipynb)

## Citation

If you use DeepFreq in your research, please cite:

```bibtex
@inproceedings{izacard2019deepfreq,
    title={Data-Driven Estimation of Sinusoid Frequencies},
    author={Izacard, Gautier and Mohan, Sreyas and Fernandez-Granda, Carlos},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2019},
    pages = {5127--5137},
    volume = {32},
}
```

---

For questions or issues, please visit the [GitHub repository](https://github.com/esantiago223/DeepFreq).
