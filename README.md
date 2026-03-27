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

### Jupyter Kernel And GPU Check

If you are using VS Code or Jupyter and want GPU acceleration:

1. Create the environment and install dependencies.
2. Select the `Python (.venv gpt-under-the-hood)` kernel in the notebook UI.
3. Restart the notebook kernel after switching.
4. Re-run the first setup cell.

To verify CUDA from the command line:

```powershell
.\.venv\Scripts\python.exe gpu_check.py
```

To verify CUDA inside a notebook:

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

If `torch.cuda.is_available()` is `False`, the most common causes are:
- The notebook is still using a different kernel/interpreter
- A CPU-only PyTorch build is installed in the active environment
- The kernel was not restarted after changing the environment

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
- **01_pre_transformer_era_solutions.ipynb** - Reference solutions and extended experiments for notebook 01

### Phase 2: The Attention Revolution
- **02_attention_mechanism.ipynb** - The key innovation
- **03_transformer_architecture.ipynb** - Transformer building blocks and positional encoding

### Phase 3: From Transformer To GPT
- **04_gpt_architecture.ipynb** - Decoder-only transformers and causal attention
- **05_tokenization.ipynb** - Character-level tokenization and the motivation for BPE

### Phase 4: Training And Fine-Tuning
- **06_training_complete_model.ipynb** - Train a small decoder-only language model on Tiny Shakespeare
- **07_instruction_finetuning.ipynb** - Fine-tune the base model on Shakespeare Q&A pairs

## Current Repo State

This repository started as a notebook-first learning project and now also includes shared Python modules for the later pipeline stages.

- The early notebooks remain the main learning path.
- The later training flow has been partially refactored into reusable code under `src/`.
- Notebook 06 and notebook 07 now share model, tokenizer, and inference code.

## Philosophy

- **Build to understand**: Implement components from scratch before using libraries
- **Visualize everything**: See what's happening inside the model
- **Balanced math**: Key equations explained intuitively, not just presented
- **Hands-on**: Every notebook has code you can run and modify

## Project Structure

```
ai-learning/
├── notebooks/                 # Main learning notebooks
├── src/
│   ├── model.py              # Shared MiniGPT model components
│   ├── tokenizer.py          # Character and educational BPE tokenizers
│   ├── inference.py          # Shared generation helpers
│   └── utils.py              # Plotting and notebook utilities
├── scripts/
│   └── clean_qa_dataset.py   # Cleans and splits Shakespeare Q&A data
├── data/
│   ├── shakespeare.txt
│   ├── shakespeare_qa.json
│   ├── shakespeare_qa_cleaned.json
│   ├── shakespeare_qa_train.json
│   ├── shakespeare_qa_val.json
│   └── shakespeare_qa_cleaning_report.json
├── models/                   # Saved notebook checkpoints
├── gpu_check.py              # Small CUDA environment check
├── test_notebook_07.py       # Smoke test for notebook 07 concepts
├── requirements.txt
└── README.md
```

## Notebook 6 MiniGPT Architecture

The default `MiniGPT` configuration used in notebook 6 is a small decoder-only transformer with:

- `vocab_size = 69`
- `d_model = 256`
- `num_heads = 8`
- `num_layers = 6`
- `d_ff = 1024`
- `max_seq_len = 128`
- `dropout = 0.1`
- Total trainable parameters: `4,774,469`

High-level structure:

```text
Token Embedding
  -> Sinusoidal Positional Encoding
  -> Dropout
  -> 6 x GPTBlock
      -> Pre-Norm Causal Self-Attention + Residual
      -> Pre-Norm Feed-Forward Network + Residual
  -> Final LayerNorm
  -> Output Projection to Vocabulary Logits
```

Parameter breakdown:

| Component | Shape / Count | Parameters |
|-----------|---------------|-----------:|
| Token embedding | `69 x 256` | 17,664 |
| 1 GPT block | attention + FFN + 2 layer norms | 789,760 |
| 6 GPT blocks | `6 x 789,760` | 4,738,560 |
| Final layer norm | `256` weights + `256` bias | 512 |
| Output projection | `256 x 69` + bias | 17,733 |
| Total |  | 4,774,469 |

Per-block detail:

| Subcomponent | Parameters |
|--------------|-----------:|
| Q projection `Linear(256, 256)` | 65,792 |
| K projection `Linear(256, 256)` | 65,792 |
| V projection `Linear(256, 256)` | 65,792 |
| Output projection `Linear(256, 256)` | 65,792 |
| Feed-forward `Linear(256, 1024)` | 263,168 |
| Feed-forward `Linear(1024, 256)` | 262,400 |
| LayerNorms (2 total) | 1,024 |
| Total per GPT block | 789,760 |

Notes:

- Positional encoding is sinusoidal, so it adds no trainable parameters.
- The notebook does not tie input embedding weights with the output projection.
- The repo now includes both a character tokenizer and a lightweight educational BPE tokenizer in `src/tokenizer.py`.
- The exact vocabulary size and total parameter count change if you switch tokenizer type or model dimensions.

## Notebook 6 And 7 Workflow

The later notebooks now form a connected mini training pipeline:

1. **Notebook 06** trains a base MiniGPT language model on Tiny Shakespeare.
2. The base checkpoint is saved under `models/`.
3. **Notebook 07** loads that checkpoint and fine-tunes it on instruction-style Q&A data.
4. The Q&A dataset can be cleaned and split using `scripts/clean_qa_dataset.py`.

This makes the repo useful in two ways:

- as a notebook-based learning path
- as a small reusable codebase for experimenting with tokenization, pretraining, and instruction fine-tuning

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
