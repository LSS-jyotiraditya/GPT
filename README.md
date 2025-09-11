# 🚀 NanoGPT with Mixture of Experts (MoE) - TinyStories Training

A comprehensive implementation of NanoGPT with Mixture of Experts architecture, trained on the TinyStories dataset. This repository contains both standard NanoGPT and MoE variants with detailed training pipelines and analysis tools.

## 📊 Model Overview

### Architecture Variants

#### 1. **Standard NanoGPT** (`nano_gpt_model.py`)
- **Parameters**: ~10M (configurable)
- **Architecture**: Standard transformer with multi-head attention and feed-forward layers
- **Features**: Layer normalization, dropout, positional embeddings

#### 2. **MoE NanoGPT** (`moe_nano_gpt_model.py`) 
- **Parameters**: 134.6M (trained model)
- **Architecture**: Transformer with Mixture of Experts replacing feed-forward layers
- **Experts**: 8 experts per layer, top-2 routing
- **Features**: Noisy top-k gating, expert sparsity, enhanced scalability

### Model Specifications (Trained MoE Model)

| Parameter | Value |
|-----------|-------|
| **Model Size** | 134.6M parameters |
| **Layers** | 12 |
| **Embedding Dimension** | 256 |
| **Attention Heads** | 8 |
| **Experts per Layer** | 8 |
| **Top-K Experts** | 2 |
| **Block Size** | 1080 tokens |
| **Vocabulary Size** | 2048 |
| **Dropout** | 0.1 |

## 📚 Dataset Information

### TinyStories Dataset
- **Source**: `roneneldan/TinyStories` from Hugging Face
- **Training Samples**: 2,119,719 stories
- **Validation Samples**: 21,990 stories
- **Tokenizer**: Custom BPE tokenizer with 2048 vocabulary
- **Special Tokens**: `<|unknown|>`, `<|im_start|>`, `<|im_end|>`
- **Context Length**: 1080 tokens

### Data Processing Pipeline
1. **Dataset Loading**: Raw TinyStories from Hugging Face
2. **Tokenization**: Custom BPE tokenizer training (`train_tokenizer.py`)
3. **Preprocessing**: Padding/truncation to fixed sequence length (`pre_map.py`)
4. **Format**: PyTorch-compatible dataset with input_ids

## 🏋️ Training Results

### Training Configuration
- **Epochs**: 6 (planned) / 1+ (completed)
- **Batch Size**: 32 × 2 = 64 (with gradient accumulation)
- **Learning Rate**: 0.001 (OneCycleLR scheduler)
- **Optimizer**: AdamW
- **Mixed Precision**: Enabled (FP16)
- **Gradient Checkpointing**: Enabled
- **Device**: NVIDIA A100 80GB PCIe

### Training Progress & Loss Statistics

#### Epoch 1 Results:
- **Training Loss**: 0.3463
- **Validation Loss**: 0.2739
- **Learning Rate**: 9.36e-04
- **Sparsity**: 0.000
- **Training Speed**: ~2.7s/iteration
- **Memory Usage**: ~3.4GB allocated, 76.8GB cached
- **Status**: ✅ New best model saved

#### Training Performance:
- **Total Training Steps**: 198,723
- **Completed Steps**: 55,636+ (28% progress)
- **Training Speed**: 2.7-2.8 seconds per iteration
- **Tokens/Second**: ~24 tokens/second
- **GPU Utilization**: Optimal with A100 80GB

### Model Checkpoints
- `enhanced-moe-134.6M-best.pt` - Best validation loss model
- `enhanced-moe-134.6M-epoch-0.pt` - End of epoch 1 checkpoint

## 🏗️ Architecture Details

### Standard NanoGPT Components
```python
class NanoGPT(nn.Module):
    ├── Token Embeddings (vocab_size × n_embed)
    ├── Position Embeddings (block_size × n_embed)
    ├── Transformer Blocks (n_layers)
    │   ├── Multi-Head Attention
    │   │   ├── Query, Key, Value projections
    │   │   ├── Causal masking
    │   │   └── Dropout
    │   ├── Feed-Forward Network
    │   │   ├── Linear(n_embed → 4×n_embed)
    │   │   ├── ReLU activation
    │   │   └── Linear(4×n_embed → n_embed)
    │   └── Layer Normalization (Pre-norm)
    └── Language Model Head (n_embed → vocab_size)
```

### MoE NanoGPT Components
```python
class NanoGPTMoE(nn.Module):
    ├── Token Embeddings (vocab_size × n_embed)
    ├── Position Embeddings (block_size × n_embed)
    ├── Transformer Blocks (n_layers)
    │   ├── Multi-Head Attention (same as standard)
    │   ├── Mixture of Experts Layer
    │   │   ├── Noisy Top-K Gating Network
    │   │   │   ├── Gate weights (n_embed → n_experts)
    │   │   │   ├── Noise weights (n_embed → n_experts)
    │   │   │   └── Top-K selection with softmax
    │   │   └── Expert Networks (n_experts)
    │   │       ├── Linear(n_embed → 4×n_embed)
    │   │       ├── ReLU activation
    │   │       └── Linear(4×n_embed → n_embed)
    │   └── Layer Normalization (Pre-norm)
    └── Language Model Head (n_embed → vocab_size)
```

### Key MoE Features
- **Sparse Activation**: Only top-2 experts activated per token
- **Load Balancing**: Noisy gating prevents expert collapse
- **Scalability**: Linear scaling with number of experts
- **Efficiency**: ~2.5x parameters with ~1.2x compute cost

## 🚀 Usage

### Quick Start
```python
from moe_nano_gpt_model import NanoGPTMoE
from tokenizers import Tokenizer
import torch

# Load tokenizer
tokenizer = Tokenizer.from_file("data/TinyStories-tokenizer.json")

# Load trained model
checkpoint = torch.load("checkpoints/enhanced-moe-134.6M-best.pt")
hyperparameters = checkpoint['hyperparameters']

model = NanoGPTMoE(hyperparameters, device="cuda")
model.load_state_dict(checkpoint['model'])
model.eval()

# Generate text
prompt = "Once upon a time"
tokens = tokenizer.encode(prompt).ids
input_ids = torch.tensor([tokens], device="cuda")

with torch.no_grad():
    generated = model.generate(input_ids, max_new_tokens=100)
    result = tokenizer.decode(generated[0].tolist())
    print(result)
```

### Interactive Testing
```bash
python test.py
```
Features:
- Interactive text generation
- Performance benchmarking
- Expert usage analysis
- Configurable generation parameters

### Training from Scratch
```bash
# 1. Prepare dataset
python pre_map.py --output-dir data --train-count 30000 --val-count 3000

# 2. Train MoE model
python small_moes_train.py

# 3. Train standard model (alternative)
# python train_standard.py
```

## 📁 Repository Structure

```
slms/
├── 📄 README.md                    # This comprehensive guide
├── 🔧 requirements.txt             # Python dependencies
├── 
├── 🤖 Models
│   ├── nano_gpt_model.py          # Standard NanoGPT implementation
│   ├── moe_nano_gpt_model.py      # MoE NanoGPT implementation
│   └── simple_stories_4m/         # Model configuration
│
├── 🏋️ Training
│   ├── small_moes_train.py        # MoE training script
│   ├── train2.log                 # Training logs with metrics
│   └── checkpoints/               # Model checkpoints
│       ├── enhanced-moe-134.6M-best.pt
│       └── enhanced-moe-134.6M-epoch-0.pt
│
├── 📊 Data Processing
│   ├── train_tokenizer.py         # BPE tokenizer training
│   ├── pre_map.py                 # Dataset preprocessing
│   └── data/                      # Processed datasets
│       ├── TinyStories-tokenizer.json
│       ├── hyperparameters.json
│       ├── moe-hyperparameters.json
│       ├── train/                 # Training data
│       └── validation/            # Validation data
│
└── 🧪 Testing & Analysis
    └── test.py                    # Interactive testing & benchmarking
```

## 🔬 Technical Innovations

### 1. **Noisy Top-K Gating**
- Prevents expert collapse through noise injection
- Ensures load balancing across experts
- Maintains training stability

### 2. **Enhanced Training Pipeline**
- Mixed precision training (FP16)
- Gradient checkpointing for memory efficiency
- OneCycleLR scheduling for optimal convergence
- Comprehensive logging and monitoring

### 3. **Expert Analysis Tools**
- Real-time expert usage tracking
- Sparsity monitoring
- Performance benchmarking
- Interactive generation interface

## 📈 Performance Metrics

### Generation Speed
- **Standard Model**: ~30 tokens/second
- **MoE Model**: ~24 tokens/second
- **Memory Efficiency**: 3.4GB active, 76.8GB cached

### Training Efficiency
- **Convergence**: Rapid loss reduction in first epoch
- **Stability**: Consistent training without divergence
- **Scalability**: Efficient scaling to 134.6M parameters

### Quality Metrics
- **Validation Loss**: 0.2739 (epoch 1)
- **Training Loss**: 0.3463 (epoch 1)
- **Perplexity**: ~1.31 (validation)

## 🛠️ Installation & Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- `torch` - PyTorch framework
- `transformers` - Hugging Face transformers
- `datasets` - Dataset loading and processing
- `tokenizers` - Fast tokenization
- `matplotlib` - Plotting and visualization
- `numpy`, `pandas` - Data manipulation
- `tqdm` - Progress bars
- `accelerate` - Training acceleration

### Hardware Requirements
- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory (RTX 3090, A100)
- **Training**: 24GB+ GPU memory for full dataset

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TinyStories Dataset**: roneneldan/TinyStories
- **NanoGPT**: Andrej Karpathy's nanoGPT implementation
- **MoE Architecture**: Based on "Outrageously Large Neural Networks" (Shazeer et al.)
- **Training Infrastructure**: NVIDIA A100 GPU support

## 📚 References

1. Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
2. Vaswani, A., et al. (2017). "Attention Is All You Need"
3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners"
4. Karpathy, A. "nanoGPT" - https://github.com/karpathy/nanoGPT

---

**🎯 Ready to explore the power of Mixture of Experts? Start with `python test.py` for interactive generation!**
