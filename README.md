# Understanding Transformers and GPT from Scratch

A hands-on learning journey to deeply understand how transformers work and how they're used to build Large Language Models like GPT.

## What You'll Learn

- How RNNs and LSTMs work (and their limitations)
- The attention mechanism that changed everything
- The transformer architecture, component by component
- How to implement transformers from scratch (NumPy and PyTorch)
- How GPT differs from the original transformer
- How to train your own tiny language model

## Getting Started

### Setup

#### Using uv (Recommended - Fast!)

```bash
# Create virtual environment with uv
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (much faster than pip!)
uv pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Then open `notebooks/00_introduction_and_setup.ipynb` to begin!

#### Alternative: Using pip

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Then open `notebooks/00_introduction_and_setup.ipynb` to begin!

## Learning Path

Work through the notebooks in order:

### Phase 1: Foundations & History
- **00_introduction_and_setup.ipynb** - Start here
- **01_pre_transformer_era.ipynb** - RNNs, LSTMs, and why we needed transformers

### Phase 2: The Attention Revolution
- **02_attention_mechanism.ipynb** - The key innovation
- **03_transformer_overview.ipynb** - Architecture overview

### Phase 3: Building Blocks
- **04_self_attention_from_scratch.ipynb** - Query, Key, Value explained
- **05_multi_head_attention.ipynb** - Parallel attention mechanisms
- **06_positional_encoding.ipynb** - Adding sequence order
- **07_feed_forward_networks.ipynb** - The FFN component
- **08_layer_norm_and_residuals.ipynb** - Training stability

### Phase 4: Complete Transformer
- **09_complete_transformer.ipynb** - Putting it all together

### Phase 5: From Transformer to GPT
- **10_gpt_architecture.ipynb** - Decoder-only transformers
- **11_tokenization.ipynb** - Text to tokens
- **12_training_a_tiny_gpt.ipynb** - Train your own language model

## Philosophy

- **Build to understand**: Implement components from scratch before using libraries
- **Visualize everything**: See what's happening inside the model
- **Balanced math**: Key equations explained intuitively, not just presented
- **Hands-on**: Every notebook has code you can run and modify

## Project Structure

```
ai-learning/
├── notebooks/          # Jupyter notebooks (work through in order)
├── src/               # Shared utility functions
├── data/              # Sample datasets for experiments
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Prerequisites

- Python basics (functions, classes, loops)
- Basic linear algebra (matrix multiplication, vectors)
- Familiarity with NumPy is helpful but not required
- No deep learning experience needed!

## References

Key papers you'll understand by the end:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Improving Language Understanding by Generative Pre-Training" (GPT-1)
- "Language Models are Unsupervised Multitask Learners" (GPT-2)

## Tips

- Take your time with each notebook
- Run all cells and experiment with parameters
- Complete the exercises before moving forward
- Visualizations are there to build intuition - study them!
- Don't just copy code - understand why each line exists

## Future Topics

After completing the core curriculum, explore:
- Vision Transformers (ViT)
- BERT and masked language modeling
- Modern optimizations (Flash Attention)
- Fine-tuning techniques (LoRA, PEFT)
- Scaling laws and emergent abilities

Happy learning!
